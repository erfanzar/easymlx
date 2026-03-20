# Copyright 2026 The EASYDEL / EASYMLX Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RWKV7 MLX model implementation for serving and inference.

This is a linear-attention / recurrent model. It does *not* use standard
transformer KV caches. The ``cache_views`` argument accepted by the EasyMLX
base-class interface is simply ignored.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule

from .rwkv7_configuration import Rwkv7Config

CacheView = TransformerCacheView | PageCacheView


# ---------------------------------------------------------------------------
# Helper functions (mirrors upstream mlx-lm rwkv7.py)
# ---------------------------------------------------------------------------


@partial(mx.compile, shapeless=True)
def _addcmul(x, y, z):
    """Fused add-multiply: ``x + y * z``.

    Args:
        x: Base tensor.
        y: Multiplier tensor.
        z: Multiplicand tensor.

    Returns:
        Result tensor ``x + y * z``.
    """
    return x + y * z


@partial(mx.compile, shapeless=True)
def _l2_norm(x):
    """L2-normalize along the last axis with epsilon clamping.

    Args:
        x: Input tensor.

    Returns:
        L2-normalized tensor.
    """
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-7)


@mx.compile
def _wkv7_step_ops(r, w, k, v, a, b, state):
    """Single WKV7 recurrence step (compiled).

    Computes one timestep of the RWKV7 linear attention recurrence:
    ``state = state * w + v @ k^T + (state @ a) @ b^T``,
    ``y = state @ r``.

    Args:
        r: Receptance tensor of shape ``(B, H, D)``.
        w: Decay tensor of shape ``(B, H, D)``.
        k: Key tensor of shape ``(B, H, D)``.
        v: Value tensor of shape ``(B, H, D)``.
        a: A-factor tensor of shape ``(B, H, D)``.
        b: B-factor tensor of shape ``(B, H, D)``.
        state: Recurrent state of shape ``(B, H, D, D)``.

    Returns:
        Tuple of (output ``(B, H, D, 1)``, updated state ``(B, H, D, D)``).
    """
    sab = (state @ a[..., None]) @ b[..., None, :]
    state = state * w[:, :, None, :] + v[..., None] @ k[..., None, :] + sab
    y = state @ r[..., None]
    return y, state


# ---------------------------------------------------------------------------
# Metal kernel for WKV7 (GPU acceleration)
# ---------------------------------------------------------------------------


def _make_wkv7_kernel():
    """Create Metal GPU kernel for accelerated WKV7 recurrence.

    Returns:
        A compiled Metal kernel callable, or None if Metal is unavailable.
    """
    if not mx.metal.is_available():
        return None
    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / H;
        auto h_idx = n % H;
        constexpr int n_per_t = D / 32;

        auto r_ = r + b_idx * T * H * D + h_idx * D;
        auto w_ = w + b_idx * T * H * D + h_idx * D;
        auto k_ = k + b_idx * T * H * D + h_idx * D;
        auto v_ = v + b_idx * T * H * D + h_idx * D;
        auto a_ = a + b_idx * T * H * D + h_idx * D;
        auto b_ = b + b_idx * T * H * D + h_idx * D;
        y += b_idx * T * H * D + h_idx * D;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in  + (n * D + dv_idx) * D;
        auto o_state = state_out + (n * D + dv_idx) * D;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          float sa = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            sa += state[i] * a_[s_idx];
            state[i] = state[i] * w_[s_idx];
          }
          sa = simd_sum(sa);

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * v_[dv_idx] + sa * b_[s_idx];
            out += state[i] * r_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          r_ += H * D;
          w_ += H * D;
          k_ += H * D;
          v_ += H * D;
          a_ += H * D;
          b_ += H * D;
          y  += H * D;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    """
    inputs = ["r", "w", "k", "v", "a", "b", "state_in", "T"]
    return mx.fast.metal_kernel(
        name="wkv7_kernel",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


_wkv7_kernel = _make_wkv7_kernel()


def _wkv7_kernel_call(
    r: mx.array,
    w: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
):
    """Invoke the Metal WKV7 kernel.

    Args:
        r: Receptance of shape ``(B, T, H, D)``.
        w: Decay of shape ``(B, T, H, D)``.
        k: Key of shape ``(B, T, H, D)``.
        v: Value of shape ``(B, T, H, D)``.
        a: A-factor of shape ``(B, T, H, D)``.
        b: B-factor of shape ``(B, T, H, D)``.
        state: Recurrent state of shape ``(B*H, D, D)``.

    Returns:
        Tuple of (output ``(B, T, H, D)``, updated state).
    """
    B, T, H, D = r.shape
    input_dtype = r.dtype
    return _wkv7_kernel(
        inputs=[r, w, k, v, a, b, state, T],
        template=[
            ("InT", input_dtype),
            ("H", H),
            ("D", D),
        ],
        grid=(32, D, B * H),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, H, D), state.shape],
        output_dtypes=[input_dtype, input_dtype],
    )


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class LayerNormPerHead(nn.Module):
    """Per-head LayerNorm for RWKV7 output normalization.

    Applies weight-less LayerNorm independently to each head, then scales
    and biases with per-head learnable parameters.

    Attributes:
        weight: Learnable scale of shape ``(num_heads, head_dim)``.
        bias: Learnable bias of shape ``(num_heads, head_dim)``.
        eps: Epsilon for numerical stability.

    Example:
        >>> norm = LayerNormPerHead(64, 32, eps=64e-5)
        >>> out = norm(x)  # x: (B, T, H, D)
    """

    def __init__(self, head_dim: int, num_heads: int, eps: float):
        """Initialize per-head LayerNorm.

        Args:
            head_dim: Per-head dimensionality.
            num_heads: Number of heads.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.weight = mx.zeros((num_heads, head_dim))
        self.bias = mx.zeros((num_heads, head_dim))
        self.eps = eps

    def __call__(self, x):
        """Apply per-head LayerNorm.

        Args:
            x: Input tensor with heads in the second-to-last dimension.

        Returns:
            Normalized and scaled tensor.
        """
        return self.weight * mx.fast.layer_norm(x, None, None, self.eps) + self.bias


class LoRA(nn.Module):
    """Low-rank adapter for RWKV7 projections.

    Implements a two-layer low-rank projection with an optional
    activation function between the layers.

    Attributes:
        input_dim: Input dimensionality.
        output_dim: Output dimensionality.
        low_rank_dim: Bottleneck dimensionality.
        bias: Whether the output layer uses bias.
        activation: Activation function module.
        lora: List of [down_proj, activation, up_proj] modules.

    Example:
        >>> lora = LoRA(2048, 2048, 64, activation="tanh")
        >>> out = lora(x)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool | None = True,
        activation: str | None = "tanh",
    ):
        """Initialize LoRA adapter.

        Args:
            input_dim: Input dimensionality.
            output_dim: Output dimensionality.
            low_rank_dim: Bottleneck dimensionality.
            bias: Whether the output layer uses bias.
            activation: Activation function name. One of ``"tanh"``,
                ``"sigmoid"``, ``"relu"``, or None for identity.

        Raises:
            ValueError: If an unsupported activation type is provided.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        if activation is None:
            self.activation = nn.Identity()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation}.")

        self.lora = [
            nn.Linear(self.input_dim, self.low_rank_dim, bias=False),
            self.activation,
            nn.Linear(self.low_rank_dim, self.output_dim, bias=self.bias),
        ]

    def __call__(self, x) -> mx.array:
        """Apply low-rank projection with activation.

        Args:
            x: Input tensor of shape ``(*lead, input_dim)``.

        Returns:
            Output tensor of shape ``(*lead, output_dim)``.
        """
        return self.lora[2](self.lora[1](self.lora[0](x)))


class TokenShift(nn.Module):
    """Token shift operator for RWKV7.

    Shifts the input sequence by one position to the right, using the
    cached previous token as the first element. This enables the model
    to compute differences between adjacent tokens.

    Example:
        >>> shift = TokenShift()
        >>> x_prev = shift(x, state)  # Returns shifted version of x
    """

    def __call__(self, x, state):
        """Compute token-shifted version of the input.

        Args:
            x: Input tensor of shape ``(B, L, D)``.
            state: Previous last-token cache of shape ``(B, 1, D)``
                or None (zeros used on first call).

        Returns:
            Shifted tensor of shape ``(B, L, D)`` where position 0
            contains the cached state and positions 1..L-1 contain
            positions 0..L-2 of the input.
        """
        B, L, D = x.shape
        if state is None:
            state = mx.zeros((B, 1, D), x.dtype)
        if L == 1:
            return state
        else:
            return mx.concatenate([state, x[:, :-1, :]], axis=1)


class Rwkv7ChannelMixing(nn.Module):
    """Channel mixing (FFN) block for RWKV7.

    Applies token shift followed by a two-layer MLP with squared ReLU
    activation.

    Attributes:
        key: Down projection to intermediate size.
        value: Up projection back to hidden size.
        x_k: Learnable token-shift interpolation weight.
        token_shift: Token shift operator.

    Example:
        >>> ffn = Rwkv7ChannelMixing(config)
        >>> out = ffn(x, cache)
    """

    def __init__(self, config: Rwkv7Config):
        """Initialize RWKV7 channel mixing block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        hidden_dim = config.hidden_size
        intermediate_size = config.intermediate_size

        self.key = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_dim, bias=False)
        self.x_k = mx.zeros((hidden_dim,))
        self.token_shift = TokenShift()

    def __call__(self, x, cache) -> mx.array:
        """Apply channel mixing with token shift.

        Args:
            x: Input tensor of shape ``(B, L, D)``.
            cache: List of cache tensors ``[shift_cache, state_cache,
                ffn_shift_cache]`` or None.

        Returns:
            Output tensor of shape ``(B, L, D)``.
        """
        state = cache[2] if cache is not None else None
        x_prev = self.token_shift(x, state)
        xx = _addcmul(x, x_prev - x, self.x_k)
        if cache is not None:
            cache[2] = x[:, -1:, :]
        return self.value(nn.relu2(self.key(xx)))


class Rwkv7TimeMixing(nn.Module):
    """Time mixing (linear attention with WKV recurrence) block for RWKV7.

    Implements the RWKV7 linear attention mechanism with:
    - Token shift for temporal context
    - Low-rank projections for decay (w), value interpolation (v),
      attention gates (a), and output gates (g)
    - WKV7 recurrence with Metal GPU acceleration when available
    - Per-head LayerNorm on the output

    Attributes:
        layer_idx: Index of this layer in the stack.
        hidden_size: Model hidden dimensionality.
        head_dim: Per-head dimensionality.
        num_heads: Number of attention heads.
        token_shift: Token shift operator.
        x_r, x_w, x_k, x_v, x_a, x_g: Token shift interpolation weights.
        k_k, k_a, r_k: Per-head scaling parameters.
        r_proj, k_proj, v_proj, o_proj: Linear projections.
        g_norm: Per-head LayerNorm.
        w_lora: LoRA for decay computation.
        v_lora: LoRA for value interpolation (layers > 0 only).
        a_lora: LoRA for attention gate.
        g_lora: LoRA for output gate.

    Example:
        >>> tm = Rwkv7TimeMixing(config, layer_idx=0)
        >>> out, v_first = tm(x, v_first=None, cache=None)
    """

    def __init__(self, config: Rwkv7Config, layer_idx: int):
        """Initialize RWKV7 time mixing block.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer. Layer 0 initializes
                ``v_first``; subsequent layers use value interpolation.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_size
        self.num_heads = config.num_heads
        self.a_low_rank_dim = config.a_low_rank_dim
        self.v_low_rank_dim = config.v_low_rank_dim
        self.gate_low_rank_dim = config.gate_low_rank_dim
        self.decay_low_rank_dim = config.decay_low_rank_dim

        self.token_shift = TokenShift()

        self.x_r = mx.zeros((1, 1, self.hidden_size))
        self.x_w = mx.zeros((1, 1, self.hidden_size))
        self.x_k = mx.zeros((1, 1, self.hidden_size))
        self.x_v = mx.zeros((1, 1, self.hidden_size))
        self.x_a = mx.zeros((1, 1, self.hidden_size))
        self.x_g = mx.zeros((1, 1, self.hidden_size))

        self.k_k = mx.zeros((self.num_heads, self.head_dim))
        self.k_a = mx.zeros((self.num_heads, self.head_dim))
        self.r_k = mx.zeros((self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.g_norm = LayerNormPerHead(self.head_dim, self.num_heads, eps=64e-5)

        self.w_lora = LoRA(
            self.hidden_size,
            self.hidden_size,
            low_rank_dim=self.decay_low_rank_dim,
            activation="tanh",
        )

        if self.layer_idx > 0:
            self.v_lora = LoRA(
                self.hidden_size,
                self.hidden_size,
                low_rank_dim=self.v_low_rank_dim,
                activation=None,
            )

        self.a_lora = LoRA(
            self.hidden_size,
            self.hidden_size,
            low_rank_dim=self.a_low_rank_dim,
            activation=None,
        )

        self.g_lora = LoRA(
            self.hidden_size,
            self.hidden_size,
            low_rank_dim=self.gate_low_rank_dim,
            activation="sigmoid",
            bias=False,
        )

    def _wkv7(self, r, w, k, v, a, b, state):
        """Run WKV7 recurrence over the sequence.

        Uses the Metal kernel if available, otherwise falls back to a
        Python loop over timesteps.

        Args:
            r: Receptance of shape ``(B, L, H, D)``.
            w: Decay of shape ``(B, L, H, D)``.
            k: Key of shape ``(B, L, H, D)``.
            v: Value of shape ``(B, L, H, D)``.
            a: A-factor of shape ``(B, L, H, D)``.
            b: B-factor of shape ``(B, L, H, D)``.
            state: Recurrent state of shape ``(B, H, D, D)`` or None.

        Returns:
            Tuple of (output ``(B, L, H, D)``, updated state).
        """
        B, L, _, _ = r.shape
        if state is None:
            state = mx.zeros((B, self.num_heads, self.head_dim, self.head_dim), dtype=r.dtype)

        if (
            _wkv7_kernel is not None
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and self.head_dim >= 32
        ):
            return _wkv7_kernel_call(r, w, k, v, a, b, state)
        else:
            ys = []
            for t in range(L):
                y, state = _wkv7_step_ops(r[:, t], w[:, t], k[:, t], v[:, t], a[:, t], b[:, t], state)
                ys.append(y)
            y = mx.stack(ys, axis=1).astype(r.dtype)
            return y, state

    def __call__(self, x, v_first, cache):
        """Apply RWKV7 time mixing with WKV recurrence.

        Args:
            x: Input tensor of shape ``(B, L, D)``.
            v_first: First-layer value tensor for value interpolation,
                or None (will be set on layer 0).
            cache: List of cache tensors ``[shift_cache, state_cache,
                ffn_shift_cache]`` or None.

        Returns:
            Tuple of (output ``(B, L, D)``, v_first tensor).
        """
        if cache is None:
            token_shift_cache, state_cache = None, None
        else:
            token_shift_cache, state_cache = cache[0], cache[1]

        B, L, D = x.shape
        x_prev = self.token_shift(x, token_shift_cache)
        xx = x_prev - x

        xr = _addcmul(x, xx, self.x_r)
        xw = _addcmul(x, xx, self.x_w)
        xk = _addcmul(x, xx, self.x_k)
        xv = _addcmul(x, xx, self.x_v)
        xa = _addcmul(x, xx, self.x_a)
        xg = _addcmul(x, xx, self.x_g)

        key = self.k_proj(xk).reshape(B, L, self.num_heads, self.head_dim)
        value = self.v_proj(xv).reshape(B, L, self.num_heads, self.head_dim)
        receptance = self.r_proj(xr).reshape(B, L, self.num_heads, self.head_dim)
        iclr = mx.sigmoid(self.a_lora(xa)).reshape(B, L, self.num_heads, self.head_dim)
        gate = self.g_lora(xg)

        if self.layer_idx == 0:
            v_first = value
        else:
            vv = mx.sigmoid(self.v_lora(xv)).reshape(B, L, self.num_heads, self.head_dim)
            value = _addcmul(value, v_first - value, vv)

        decay = mx.sigmoid(self.w_lora(xw).reshape(B, L, self.num_heads, self.head_dim)).astype(mx.float32)
        decay = mx.exp(-0.606531 * decay).astype(receptance.dtype)
        kk = _l2_norm(key * self.k_k)
        key = key * (1 + (iclr - 1) * self.k_a)
        a = -kk
        b = kk * iclr

        out, new_state_cache = self._wkv7(receptance, decay, key, value, a, b, state_cache)
        out = self.g_norm(out.reshape(B, L, self.num_heads, self.head_dim))
        out = (out + (receptance * key * self.r_k).sum(axis=-1, keepdims=True) * value).reshape([B, L, D])

        if cache is not None:
            cache[0] = x[:, -1:, :]
            cache[1] = new_state_cache

        out = self.o_proj(out * gate)
        return out, v_first


class Rwkv7Layer(nn.Module):
    """Single RWKV7 layer combining time mixing and channel mixing.

    Layer 0 includes an additional pre-norm (``pre_norm``) applied
    before time mixing.

    Attributes:
        layer_idx: Index of this layer.
        pre_norm: LayerNorm applied only at layer 0.
        attn: Time mixing (linear attention) module.
        ffn: Channel mixing (FFN) module.
        attn_norm: Pre-attention LayerNorm.
        ffn_norm: Pre-FFN LayerNorm.

    Example:
        >>> layer = Rwkv7Layer(config, layer_idx=0)
        >>> out, v_first = layer(x, v_first=None, cache=None)
    """

    def __init__(self, config: Rwkv7Config, layer_idx: int):
        """Initialize RWKV7 layer.

        Args:
            config: Model configuration.
            layer_idx: Index of this layer. Layer 0 gets an extra
                pre-norm.
        """
        super().__init__()
        self.layer_idx = layer_idx
        if self.layer_idx == 0:
            self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Rwkv7TimeMixing(config, layer_idx=self.layer_idx)
        self.ffn = Rwkv7ChannelMixing(config)
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, x, v_first, cache):
        """Run one RWKV7 layer.

        Args:
            x: Input tensor of shape ``(B, L, D)``.
            v_first: First-layer value tensor or None.
            cache: Cache list or None.

        Returns:
            Tuple of (output ``(B, L, D)``, v_first tensor).
        """
        if self.layer_idx == 0:
            x = self.pre_norm(x)

        h, v_first = self.attn(self.attn_norm(x), v_first, cache)
        h = x + h
        out = h + self.ffn(self.ffn_norm(h), cache)
        return out, v_first


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


@register_module(task_type=TaskType.BASE_MODULE, config=Rwkv7Config, model_type="rwkv7")
class Rwkv7Model(EasyMLXBaseModule):
    """Base RWKV7 recurrent model.

    This is a linear-attention model that manages its own internal recurrent
    state. The ``cache_views`` parameter from the standard EasyMLX interface
    is ignored -- RWKV7 does not use transformer KV caches.
    """

    config_class = Rwkv7Config

    def __init__(self, config: Rwkv7Config):
        """Initialize the base RWKV7 model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Rwkv7Layer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the forward pass through all RWKV7 layers.

        Note: ``cache_views``, ``cache_metadata``, and ``attention_mask``
        are ignored because RWKV7 uses internal recurrent state instead
        of transformer KV caches.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or
                ``(seq_len,)``.
            attention_mask: Ignored (accepted for API compatibility).
            input_embeddings: Pre-computed embeddings.
            cache_views: Ignored (accepted for API compatibility).
            cache_metadata: Ignored (accepted for API compatibility).

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``
            after final LayerNorm.
        """
        # cache_views is ignored for RWKV7 -- state is managed internally.
        del cache_views, cache_metadata, attention_mask

        if input_embeddings is not None:
            hidden_states = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1:
                input_ids = input_ids[None, :]
            hidden_states = self.embed_tokens(input_ids)

        v_first = None
        for layer in self.layers:
            hidden_states, v_first = layer(hidden_states, v_first, cache=None)

        return self.norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Reshape per-head parameters that arrive flattened from upstream.

        Parameters named ``k_k``, ``k_a``, or ``g_norm`` that arrive as
        1-D tensors of shape ``(num_heads * head_dim,)`` are reshaped to
        ``(num_heads, head_dim)``. Rotary inv_freq buffers are removed.

        Args:
            weights: Raw weight dictionary from a checkpoint.

        Returns:
            Sanitized weight dictionary with reshaped per-head parameters.
        """
        num_heads = self.config.num_heads
        head_dim = self.config.head_size
        sanitized = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue
            if "k_k" in k or "k_a" in k or "g_norm" in k:
                if v.ndim == 1 and v.shape[0] == num_heads * head_dim:
                    v = v.reshape(num_heads, head_dim)
            sanitized[k] = v
        return sanitized


# ---------------------------------------------------------------------------
# Causal LM wrapper
# ---------------------------------------------------------------------------


@register_module(task_type=TaskType.CAUSAL_LM, config=Rwkv7Config, model_type="rwkv7")
class Rwkv7ForCausalLM(BaseCausalLMModule[Rwkv7Model, Rwkv7Config]):
    """RWKV7 causal language model.

    Wraps ``Rwkv7Model`` and adds an LM head for next-token prediction.
    Uses internal recurrent state rather than transformer KV caches.

    Attributes:
        config_class: Associated configuration class (``Rwkv7Config``).

    Example:
        >>> model = Rwkv7ForCausalLM(config)
        >>> logits = model(input_ids)
    """

    config_class = Rwkv7Config

    def __init__(self, config: Rwkv7Config):
        """Initialize the RWKV7 causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Rwkv7Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights by delegating to the base model.

        Chains the parent class sanitization with the base model's
        per-head parameter reshaping.

        Args:
            weights: Raw weight dictionary from a checkpoint.

        Returns:
            Sanitized weight dictionary.
        """
        weights = super().sanitize(weights)
        return self.base_model.sanitize(weights)


__all__ = ("Rwkv7ForCausalLM", "Rwkv7Model")
