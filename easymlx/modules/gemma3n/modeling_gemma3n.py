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

"""Gemma3N MLX implementation (serving/inference only).

Gemma3N is a complex architecture with:
  - AltUp (Alternating Updates) for multi-stream processing
  - Laurel (Learned Augmented Residual Layer) blocks
  - Per-layer token and model-projected inputs
  - KV sharing across later layers
  - Sliding + full attention pattern
  - Activation sparsity (gelu_topk)
  - Final logit softcapping

This wraps a VLM model, stripping vision/audio towers for text-only inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import (
    PageCacheView,
    PageMetadata,
    TransformerCacheView,
)
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.layers.attention import AttentionPerformer, build_attention_mask
from easymlx.layers.rotary import get_rope
from easymlx.modules._base import BaseCausalLMModule

from .gemma3n_configuration import Gemma3NConfig

CacheView = TransformerCacheView | PageCacheView


@dataclass
class Gemma3NTextConfig:
    """Internal text config parsed from the ``text_config`` dict.

    Contains all architecture-specific parameters for the Gemma3N text
    backbone, including AltUp multi-stream settings, Laurel rank,
    sliding window attention patterns, per-layer intermediate sizes,
    activation sparsity patterns, and KV sharing.

    Attributes:
        hidden_size: Hidden dimensionality. Defaults to 1536.
        num_hidden_layers: Number of decoder layers. Defaults to 26.
        intermediate_size: MLP intermediate size, or per-layer list.
            Defaults to 6144.
        num_attention_heads: Number of query heads. Defaults to 8.
        head_dim: Per-head dimensionality. Defaults to 256.
        rms_norm_eps: RMSNorm epsilon. Defaults to 1e-6.
        vocab_size: Vocabulary size. Defaults to 262144.
        num_key_value_heads: Number of KV heads. Defaults to 4.
        num_kv_shared_layers: Number of trailing layers that share KV
            projections. Defaults to 0.
        vocab_size_per_layer_input: Vocabulary size for per-layer token
            embeddings. Defaults to 262144.
        sliding_window: Sliding window size for local layers.
            Defaults to 512.
        max_position_embeddings: Maximum sequence length. Defaults to 32768.
        rope_local_base_freq: RoPE base for sliding layers.
            Defaults to 10000.0.
        rope_theta: RoPE base for full attention layers.
            Defaults to 1000000.0.
        final_logit_softcapping: Softcap for output logits, or None.
            Defaults to None.
        layer_types: Per-layer type list (``"sliding_attention"`` or
            ``"full_attention"``). Defaults to None.
        activation_sparsity_pattern: Per-layer sparsity fractions.
            Defaults to None.
        hidden_size_per_layer_input: Dimensionality of per-layer inputs.
            Defaults to 256.
        altup_num_inputs: Number of AltUp streams. Defaults to 2.
        altup_coef_clip: Clipping bound for AltUp coefficients.
            Defaults to None.
        altup_correct_scale: Whether to scale corrected outputs.
            Defaults to False.
        altup_active_idx: Index of the active AltUp stream.
            Defaults to 0.
        laurel_rank: Rank of Laurel low-rank projection.
            Defaults to 64.
        rope_scaling: Optional RoPE scaling config. Defaults to None.
    """

    hidden_size: int = 1536
    num_hidden_layers: int = 26
    intermediate_size: int | list[int] = 6144
    num_attention_heads: int = 8
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    num_key_value_heads: int = 4
    num_kv_shared_layers: int = 0
    vocab_size_per_layer_input: int = 262144
    sliding_window: int = 512
    max_position_embeddings: int = 32768
    rope_local_base_freq: float = 10_000.0
    rope_theta: float = 1_000_000.0
    final_logit_softcapping: float | None = None
    layer_types: list[str] | None = None
    activation_sparsity_pattern: list[float] | None = None
    hidden_size_per_layer_input: int = 256
    altup_num_inputs: int = 2
    altup_coef_clip: float | None = None
    altup_correct_scale: bool = False
    altup_active_idx: int = 0
    laurel_rank: int = 64
    rope_scaling: dict | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Gemma3NTextConfig:
        """Create a config from a dictionary, ignoring unknown keys.

        Args:
            d: Dictionary of config parameters.

        Returns:
            A ``Gemma3NTextConfig`` instance.
        """
        import dataclasses

        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


class RMSNoScale(nn.Module):
    """RMSNorm without learnable weights.

    Normalizes by RMS but does not apply any learned scaling. Used for
    value normalization in Gemma3N attention.

    Attributes:
        eps: Epsilon for numerical stability.

    Args:
        eps: Epsilon for numerical stability. Defaults to 1e-5.
    """

    def __init__(self, eps: float = 1e-5):
        """Initialize weight-free RMSNorm.

        Args:
            eps: Epsilon for numerical stability. Defaults to 1e-5.
        """
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization without scaling.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor of the same shape.
        """
        return mx.fast.rms_norm(x, None, self.eps)


class Gemma3NLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer (LAUREL).

    A low-rank residual block that projects down to ``laurel_rank``
    dimensions and back up to ``hidden_size``, then adds the normalized
    result as a residual: ``x + norm(right(left(x)))``.

    Attributes:
        linear_left: Down-projection to ``laurel_rank``.
        linear_right: Up-projection back to ``hidden_size``.
        post_laurel_norm: RMSNorm applied to the up-projected output.

    Example::

        >>> block = Gemma3NLaurelBlock(Gemma3NTextConfig(hidden_size=64, laurel_rank=16))
        >>> out = block(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: Gemma3NTextConfig):
        """Initialize LAUREL block.

        Args:
            config: Text config with ``hidden_size`` and ``laurel_rank``.
        """
        super().__init__()
        self.linear_left = nn.Linear(config.hidden_size, config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(config.laurel_rank, config.hidden_size, bias=False)
        self.post_laurel_norm = nn.RMSNorm(dims=config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply LAUREL residual.

        Args:
            x: Input tensor of shape ``[..., hidden_size]``.

        Returns:
            ``x + norm(right(left(x)))``.
        """
        laurel_x = self.linear_left(x)
        laurel_x = self.linear_right(laurel_x)
        normed_laurel_x = self.post_laurel_norm(laurel_x)
        return x + normed_laurel_x


class Gemma3NAttention(nn.Module):
    """Gemma3N attention with Q/K/V normalization and optional KV sharing.

    Applies per-head RMSNorm to queries and keys, and weight-free
    RMSNorm to values. Supports both sliding-window and full attention
    with separate RoPE base frequencies. Later layers can share KV
    projections for memory efficiency.

    Attributes:
        is_sliding: Whether this layer uses sliding window attention.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dimensionality.
        is_kv_shared_layer: Whether KV projections are shared.
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
        q_norm: Per-head RMSNorm for queries.
        k_norm: Per-head RMSNorm for keys.
        v_norm: Weight-free RMSNorm for values.
        rope: Rotary embedding (local or global).
        attention_performer: Attention backend.

    Example::

        >>> attn = Gemma3NAttention(
        ...     Gemma3NTextConfig(
        ...         hidden_size=64, num_attention_heads=4,
        ...         layer_types=["sliding_attention"] * 4,
        ...     ),
        ...     layer_idx=0, is_kv_shared_layer=False,
        ... )
        >>> attn.is_sliding
        True
    """

    def __init__(self, config: Gemma3NTextConfig, layer_idx: int, is_kv_shared_layer: bool):
        """Initialize Gemma3N attention.

        Args:
            config: Text config.
            layer_idx: Zero-based layer index for determining attention type.
            is_kv_shared_layer: Whether this layer shares KV projections
                with another layer.
        """
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.is_kv_shared_layer = is_kv_shared_layer

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(dims=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(dims=config.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNoScale(eps=config.rms_norm_eps)

        rope_base = config.rope_local_base_freq if self.is_sliding else config.rope_theta
        self.rope = get_rope(
            dims=self.head_dim,
            base=rope_base,
            traditional=False,
        )
        self.attention_performer = AttentionPerformer(scale=1.0)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Compute attention with Q/K/V normalization and RoPE.

        Args:
            hidden_states: Input of shape ``[batch, seq_len, hidden_size]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata for paged attention.

        Returns:
            Output of shape ``[batch, seq_len, hidden_size]``.
        """
        lead = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).reshape(*lead, self.num_heads, self.head_dim)
        q = self.q_norm(q)

        if not self.is_kv_shared_layer:
            k = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
            k = self.k_norm(k)
            v = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
            v = self.v_norm(v)
        else:
            k = self.k_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
            k = self.k_norm(k)
            v = self.v_proj(hidden_states).reshape(*lead, self.num_kv_heads, self.head_dim)
            v = self.v_norm(v)

        attn = self.attention_performer(
            q,
            k,
            v,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            rope=self.rope,
        )
        return self.o_proj(attn.reshape(*lead, -1))


@partial(mx.compile, shapeless=True)
def gelu_topk(inputs: mx.array, std_multiplier: mx.array) -> mx.array:
    """Activation sparsity: GELU with top-k thresholding.

    Computes a threshold as ``mean + std * std_multiplier`` per position,
    zeroes activations below the threshold, then applies GELU approximate.

    Args:
        inputs: Pre-activation tensor from the gate projection.
        std_multiplier: Scalar controlling the sparsity cutoff,
            derived from the desired sparsity fraction via the inverse
            error function.

    Returns:
        Sparse GELU activations of the same shape as ``inputs``.
    """
    inputs_mean = mx.mean(inputs, axis=-1, keepdims=True)
    inputs_std = mx.std(inputs, axis=-1, keepdims=True)
    cutoff_x = inputs_mean + inputs_std * std_multiplier.astype(inputs_std.dtype)
    return nn.gelu_approx(mx.maximum(0, inputs - cutoff_x))


class Gemma3NMLP(nn.Module):
    """Gated MLP with optional activation sparsity for Gemma3N.

    Supports per-layer intermediate sizes and activation sparsity
    via ``gelu_topk``. When ``activation_sparsity`` > 0, only
    activations above a threshold are kept, implementing learned
    structured sparsity.

    Attributes:
        gate_proj: Gating projection.
        up_proj: Value projection.
        down_proj: Output projection.
        activation_sparsity: Sparsity fraction (0 = dense).

    Example::

        >>> mlp = Gemma3NMLP(Gemma3NTextConfig(hidden_size=64, intermediate_size=128))
        >>> out = mlp(mx.zeros((1, 8, 64)))
        >>> out.shape
        [1, 8, 64]
    """

    def __init__(self, config: Gemma3NTextConfig, layer_idx: int = 0):
        """Initialize Gemma3N MLP.

        Args:
            config: Text config with intermediate size and sparsity.
            layer_idx: Layer index for per-layer intermediate size
                and sparsity pattern lookup. Defaults to 0.
        """
        super().__init__()
        intermediate_size = (
            config.intermediate_size[layer_idx]
            if isinstance(config.intermediate_size, list)
            else config.intermediate_size
        )
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

        if config.activation_sparsity_pattern is not None:
            self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]
        else:
            self.activation_sparsity = 0.0
        if self.activation_sparsity > 0:
            self._std_multiplier = math.sqrt(2.0) * mx.erfinv(2 * self.activation_sparsity - 1)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply gated MLP with optional activation sparsity.

        Args:
            x: Input of shape ``[..., hidden_size]``.

        Returns:
            Output of the same shape.
        """
        gate_proj = self.gate_proj(x)
        if self.activation_sparsity > 0.0:
            activations = gelu_topk(gate_proj, self._std_multiplier)
        else:
            activations = nn.gelu_approx(gate_proj)
        return self.down_proj(activations * self.up_proj(x))


class Gemma3NAltUp(nn.Module):
    """Alternating Updates (AltUp) module for multi-stream processing.

    Maintains multiple parallel hidden state streams and uses
    router-based modality coefficients to predict and correct across
    streams. The ``predict`` method creates per-stream predictions,
    and ``correct`` adjusts all streams based on the activated output.

    Attributes:
        config: Text configuration.
        correct_output_scale: Optional scale for corrected outputs.
        correction_coefs: Linear layer for correction coefficients.
        prediction_coefs: Linear layer for prediction coefficients.
        modality_router: Linear router for computing modality weights.
        router_norm: RMSNorm applied before routing.

    Example::

        >>> altup = Gemma3NAltUp(Gemma3NTextConfig(hidden_size=64, altup_num_inputs=2))
    """

    def __init__(self, config: Gemma3NTextConfig):
        """Initialize AltUp module.

        Args:
            config: Text config with AltUp parameters.
        """
        super().__init__()
        self.config = config
        self.correct_output_scale = mx.zeros((config.hidden_size,))
        self.correction_coefs = nn.Linear(config.altup_num_inputs, config.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(config.altup_num_inputs, config.altup_num_inputs**2, bias=False)
        self.modality_router = nn.Linear(config.hidden_size, config.altup_num_inputs, bias=False)
        self.router_norm = nn.RMSNorm(dims=config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def _project_coefficients(layer: nn.Linear, x: mx.array, *, clip: float | None) -> mx.array:
        """Project coefficient inputs without mutating the layer weights.

        The original implementation permanently upcast the coefficient
        weights to float32 inside the forward path. That widens the
        resident parameter tensors and slows subsequent matmuls. Keep the
        layer weights in their native dtype, cast the inputs to match, and
        only materialize a clipped view when requested.
        """
        weight = layer.weight
        if x.dtype != weight.dtype:
            x = x.astype(weight.dtype)
        if clip is not None:
            weight = mx.clip(weight, -clip, clip)
        output = x @ weight.T
        bias = getattr(layer, "bias", None)
        if bias is not None:
            output = output + bias
        return output

    def compute_router_modalities(self, x: mx.array) -> mx.array:
        """Compute modality routing weights via tanh-bounded linear.

        Args:
            x: Active stream hidden states.

        Returns:
            Modality weights of shape ``[..., altup_num_inputs]``.
        """
        router_inputs = self.router_norm(x) * (self.config.hidden_size**-1.0)
        router_dtype = self.modality_router.weight.dtype
        if router_inputs.dtype != router_dtype:
            router_inputs = router_inputs.astype(router_dtype)
        routed = self.modality_router(router_inputs)
        if routed.dtype != mx.float32:
            routed = routed.astype(mx.float32)
        return mx.tanh(routed)

    def predict(self, x: mx.array) -> mx.array:
        """Generate per-stream predictions from all streams.

        Uses router modalities from the active stream to compute
        cross-stream prediction coefficients, then applies them as
        linear combinations with residual.

        Args:
            x: Stacked streams of shape
                ``[num_inputs, batch, seq_len, hidden_size]``.

        Returns:
            Predictions of the same shape as ``x``.
        """
        modalities = self.compute_router_modalities(x[self.config.altup_active_idx])
        all_coefs = (
            self._project_coefficients(
                self.prediction_coefs,
                modalities,
                clip=self.config.altup_coef_clip,
            )
            .astype(mx.float32)
            .reshape(
                *modalities.shape[:-1],
                self.config.altup_num_inputs,
                self.config.altup_num_inputs,
            )
            .transpose(0, 1, 3, 2)
        )
        x_up = x.astype(mx.float32)
        x_permuted = x_up.transpose(1, 2, 3, 0)
        predictions = mx.matmul(x_permuted, all_coefs)
        predictions = predictions.transpose(3, 0, 1, 2)
        predictions += x_up
        return predictions.astype(x.dtype)

    def correct(self, predictions: mx.array, activated: mx.array) -> mx.array:
        """Correct all stream predictions based on the activated output.

        Computes the innovation (difference between activated output and
        active prediction) and distributes it across all streams using
        router-based correction coefficients.

        Args:
            predictions: Per-stream predictions of shape
                ``[num_inputs, batch, seq_len, hidden_size]``.
            activated: Activated output from the main decoder of shape
                ``[batch, seq_len, hidden_size]``.

        Returns:
            Corrected predictions of the same shape as ``predictions``.
        """
        modalities = self.compute_router_modalities(activated)
        all_coefs = (
            self._project_coefficients(
                self.correction_coefs,
                modalities,
                clip=self.config.altup_coef_clip,
            ).astype(mx.float32)
            + 1.0
        )
        active_x = predictions[self.config.altup_active_idx]
        innovation = activated - active_x
        all_coefs = all_coefs.moveaxis(2, 0)
        corrected = innovation[None] * all_coefs[..., None]
        corrected += predictions
        return corrected.astype(activated.dtype)


class Gemma3NDecoderLayer(nn.Module):
    """Decoder layer with AltUp, Laurel, and per-layer inputs.

    Each layer performs: (1) AltUp predict across streams,
    (2) Laurel augmentation on the active prediction,
    (3) attention + residual, (4) MLP + residual with Laurel mixing,
    (5) AltUp correct across streams, (6) per-layer input gating.

    Attributes:
        config: Text configuration.
        self_attn: Gemma3N attention with Q/K/V norms.
        mlp: MLP with optional activation sparsity.
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm after attention.
        pre_feedforward_layernorm: RMSNorm before MLP.
        post_feedforward_layernorm: RMSNorm after MLP.
        is_sliding: Whether this layer uses sliding window.
        altup: AltUp multi-stream module.
        laurel: LAUREL residual block.
        per_layer_input_gate: Gating projection for per-layer inputs.
        per_layer_projection: Projection from per-layer input dim.
        post_per_layer_input_norm: RMSNorm after per-layer input.

    Example::

        >>> layer = Gemma3NDecoderLayer(
        ...     Gemma3NTextConfig(hidden_size=64, layer_types=["sliding_attention"]),
        ...     layer_idx=0, is_kv_shared_layer=False,
        ... )
    """

    def __init__(self, config: Gemma3NTextConfig, layer_idx: int, is_kv_shared_layer: bool):
        """Initialize Gemma3N decoder layer.

        Args:
            config: Text config.
            layer_idx: Zero-based layer index.
            is_kv_shared_layer: Whether KV projections are shared.
        """
        super().__init__()
        self.config = config
        self.self_attn = Gemma3NAttention(config, layer_idx, is_kv_shared_layer)
        self.mlp = Gemma3NMLP(config, layer_idx=layer_idx)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_sliding = self.self_attn.is_sliding

        self.altup = Gemma3NAltUp(config)
        self.laurel = Gemma3NLaurelBlock(config)
        self.per_layer_input_gate = nn.Linear(config.hidden_size, config.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(config.hidden_size_per_layer_input, config.hidden_size, bias=False)
        self.post_per_layer_input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | str | None = None,
        cache_view: CacheView | None = None,
        cache_metadata: PageMetadata | None = None,
        per_layer_input: mx.array | None = None,
    ) -> mx.array:
        """Forward pass with AltUp, Laurel, attention, MLP, and per-layer inputs.

        Args:
            x: Multi-stream hidden states of shape
                ``[num_inputs, batch, seq_len, hidden_size]``.
            mask: Attention mask.
            cache_view: Optional KV cache.
            cache_metadata: Page metadata.
            per_layer_input: Per-layer input of shape
                ``[batch, seq_len, hidden_size_per_layer_input]``
                or None.

        Returns:
            Corrected multi-stream hidden states.
        """
        predictions = self.altup.predict(x)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        attn = self.self_attn(
            active_prediction_normed,
            mask=mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        attn = self.post_attention_layernorm(attn)
        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) * (2.0**-0.5)

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[self.config.altup_active_idx]
        if self.config.altup_correct_scale:
            first_prediction = first_prediction * self.altup.correct_output_scale

        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = nn.gelu_approx(first_prediction)

        if per_layer_input is not None:
            first_prediction = mx.multiply(first_prediction, per_layer_input)

        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)

        corrected_predictions[1:] = corrected_predictions[1:] + first_prediction

        return corrected_predictions


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap: float, x: mx.array) -> mx.array:
    """Apply softcapping to logits: ``tanh(x / softcap) * softcap``.

    Args:
        softcap: Capping value.
        x: Logit tensor.

    Returns:
        Softcapped logits bounded to ``[-softcap, softcap]``.
    """
    out = mx.tanh(x / softcap)
    return out * softcap


@register_module(task_type=TaskType.BASE_MODULE, config=Gemma3NConfig, model_type="gemma3n")
class Gemma3NModel(EasyMLXBaseModule):
    """Base Gemma3N language model with AltUp multi-stream processing.

    Implements the full Gemma3N text backbone with:
    - AltUp multi-stream embedding and unembedding projections
    - Per-layer token embeddings and model-projected inputs
    - Sliding window + full attention alternation
    - KV sharing across later layers
    - Magnitude normalization for auxiliary streams

    Attributes:
        embed_tokens: Token embedding layer.
        layers: Stack of ``Gemma3NDecoderLayer`` instances.
        embed_tokens_per_layer: Per-layer token embedding.
        per_layer_model_projection: Projection from embeddings to
            per-layer input space.
        altup_projections: Projections for auxiliary AltUp streams.
        altup_unembed_projections: Unembedding for auxiliary streams.
        norm: Final RMSNorm.
        sliding_window: Sliding window size.

    Example::

        >>> model = Gemma3NModel(Gemma3NConfig(text_config={
        ...     "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2,
        ...     "layer_types": ["sliding_attention", "full_attention"],
        ... }))
    """

    config_class = Gemma3NConfig

    def __init__(self, config: Gemma3NConfig):
        """Initialize Gemma3N base model.

        Args:
            config: Wrapper config whose ``text_config`` dict is parsed
                into ``Gemma3NTextConfig``.
        """
        super().__init__(config)
        self.config = config
        text_config = Gemma3NTextConfig.from_dict(config.text_config)
        self._text_config = text_config

        self.hidden_size = text_config.hidden_size
        self.num_hidden_layers = text_config.num_hidden_layers
        self.vocab_size_per_layer_input = text_config.vocab_size_per_layer_input
        self.hidden_size_per_layer_input = text_config.hidden_size_per_layer_input
        self.final_logit_softcapping = text_config.final_logit_softcapping
        self.first_kv_shared_layer_idx = text_config.num_hidden_layers - text_config.num_kv_shared_layers

        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = [
            Gemma3NDecoderLayer(
                config=text_config,
                layer_idx=i,
                is_kv_shared_layer=i >= self.first_kv_shared_layer_idx,
            )
            for i in range(text_config.num_hidden_layers)
        ]

        self.embed_tokens_per_layer = nn.Embedding(
            text_config.vocab_size_per_layer_input,
            text_config.num_hidden_layers * text_config.hidden_size_per_layer_input,
        )
        self.per_layer_model_projection = nn.Linear(
            text_config.hidden_size,
            text_config.num_hidden_layers * text_config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = nn.RMSNorm(
            dims=text_config.hidden_size_per_layer_input,
            eps=text_config.rms_norm_eps,
        )

        self.altup_projections = [
            nn.Linear(text_config.hidden_size, text_config.hidden_size, bias=False)
            for _ in range(1, text_config.altup_num_inputs)
        ]
        self.altup_unembed_projections = [
            nn.Linear(text_config.hidden_size, text_config.hidden_size, bias=False)
            for _ in range(1, text_config.altup_num_inputs)
        ]

        self.norm = nn.RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

        if text_config.layer_types is not None:
            sliding_indices = [i for i, lt in enumerate(text_config.layer_types) if lt == "sliding_attention"]
            full_indices = [i for i, lt in enumerate(text_config.layer_types) if lt == "full_attention"]
            self.first_sliding_idx = sliding_indices[0] if sliding_indices else 0
            self.first_full_idx = full_indices[0] if full_indices else 0
        else:
            self.first_sliding_idx = 0
            self.first_full_idx = 0
        self.sliding_window = text_config.sliding_window

    def _get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        """Compute per-layer token embeddings.

        Looks up tokens in the per-layer embedding table (masking OOV
        tokens to zero) and reshapes to per-layer slices.

        Args:
            input_ids: Token ids of shape ``[batch, seq_len]``.

        Returns:
            Per-layer inputs of shape
            ``[batch, seq_len, num_layers, hidden_size_per_layer_input]``.
        """
        per_layer_inputs_mask = input_ids < self.vocab_size_per_layer_input
        tokens = mx.where(per_layer_inputs_mask, input_ids, mx.zeros_like(input_ids))
        result = self.embed_tokens_per_layer(tokens) * (self.hidden_size_per_layer_input**0.5)
        return result.reshape(
            *input_ids.shape,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def _project_per_layer_inputs(
        self,
        inputs_embeds: mx.array,
        per_layer_inputs: mx.array,
    ) -> mx.array:
        """Combine model-projected inputs with per-layer token embeddings.

        Projects the full embedding through a linear layer, normalizes,
        and averages with per-layer token inputs.

        Args:
            inputs_embeds: Token embeddings of shape
                ``[batch, seq_len, hidden_size]``.
            per_layer_inputs: Per-layer token embeddings of shape
                ``[batch, seq_len, num_layers, hidden_size_per_layer_input]``.

        Returns:
            Combined per-layer inputs of the same shape as
            ``per_layer_inputs``.
        """
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * (self.hidden_size**-0.5)
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        return (per_layer_projection + per_layer_inputs) * (2.0**-0.5)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Forward pass with AltUp multi-stream processing.

        Embeds tokens, projects to AltUp streams with magnitude
        normalization, processes through decoder layers with per-layer
        inputs, unembeds auxiliary streams, and averages all streams.

        Args:
            input_ids: Token ids of shape ``[batch, seq_len]``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Page metadata for paged attention.

        Returns:
            Normalized hidden states of shape
            ``[batch, seq_len, hidden_size]``.

        Raises:
            ValueError: If ``cache_views`` length does not match layers.
        """
        if cache_views is not None and len(cache_views) != len(self.layers):
            raise ValueError("cache_views length must match number of layers.")

        input_ids = mx.array(input_ids, dtype=mx.int32)
        if input_ids.ndim == 1 and cache_metadata is None:
            input_ids = input_ids[None, :]

        if input_embeddings is not None:
            h = mx.array(input_embeddings)
        else:
            h = self.embed_tokens(input_ids) * (self.hidden_size**0.5)

        per_layer_inputs = self._get_per_layer_inputs(input_ids)
        per_layer_inputs = self._project_per_layer_inputs(h, per_layer_inputs)

        mask: mx.array | str | None = None
        sliding_mask: mx.array | str | None = None
        if h.ndim == 3:
            batch_size, seq_len = h.shape[:2]
            if not (cache_views is not None and seq_len == 1):
                attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
                mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)
                if any(layer.is_sliding for layer in self.layers):
                    sliding_mask = build_attention_mask(
                        attention_mask_arr,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        window_size=self.sliding_window,
                    )

        h0 = h
        target_magnitude = mx.mean(h0**2, axis=-1, keepdims=True) ** 0.5
        h_list = [h0]
        h_list.extend([proj(h0) for proj in self.altup_projections])
        h = mx.stack(h_list, axis=0)
        mags = mx.mean(h[1:] ** 2, axis=-1, keepdims=True) ** 0.5
        h = mx.concatenate(
            [h[:1], h[1:] * (target_magnitude / mx.maximum(mags, mx.finfo(h0.dtype).min))],
            axis=0,
        )

        for i, layer in enumerate(self.layers):
            layer_cache = None if cache_views is None else cache_views[i]
            layer_metadata = cache_metadata
            if layer_metadata is not None and layer.is_sliding:
                layer_metadata = cache_metadata.with_sliding_window(self.sliding_window)
            layer_mask = sliding_mask if layer.is_sliding else mask

            pli = per_layer_inputs[:, :, i, :] if per_layer_inputs.ndim == 4 else None

            h = layer(
                h,
                mask=layer_mask,
                cache_view=layer_cache,
                cache_metadata=layer_metadata,
                per_layer_input=pli,
            )

        target_magnitude = mx.mean(h[0] ** 2, axis=-1, keepdims=True) ** 0.5
        for i, proj in enumerate(self.altup_unembed_projections):
            h = mx.concatenate(
                [h[: i + 1], proj(h[i + 1])[None], h[i + 2 :]],
                axis=0,
            )
        mags = mx.mean(h[1:] ** 2, axis=-1, keepdims=True) ** 0.5
        h = mx.concatenate(
            [h[:1], h[1:] * (target_magnitude / mx.maximum(mags, mx.finfo(h0.dtype).min))],
            axis=0,
        )
        h = mx.mean(h, axis=0)

        return self.norm(h)


@register_module(task_type=TaskType.CAUSAL_LM, config=Gemma3NConfig, model_type="gemma3n")
class Gemma3NForCausalLM(BaseCausalLMModule[Gemma3NModel, Gemma3NConfig]):
    """Gemma3N causal language model with final logit softcapping.

    Wraps ``Gemma3NModel`` and applies optional ``final_logit_softcapping``
    from the text config. Vision, audio, and rotary embedding weights
    are stripped during sanitization for text-only inference.

    Attributes:
        config_class: ``Gemma3NConfig``.

    Example::

        >>> model = Gemma3NForCausalLM(Gemma3NConfig(text_config={
        ...     "vocab_size": 256, "hidden_size": 64,
        ... }))
    """

    config_class = Gemma3NConfig

    def __init__(self, config: Gemma3NConfig):
        super().__init__(
            config=config,
            base_model_class=Gemma3NModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Project to vocabulary logits with optional softcapping.

        Args:
            hidden_states: Transformer output of shape
                ``[batch, seq_len, hidden_size]``.

        Returns:
            Logits of shape ``[batch, seq_len, vocab_size]``.
        """
        logits = super().compute_lm_logits(hidden_states)
        text_config = Gemma3NTextConfig.from_dict(self.config.text_config)
        if text_config.final_logit_softcapping is not None:
            logits = logit_softcap(text_config.final_logit_softcapping, logits)
        return logits

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Strip vision, audio, and rotary embedding weights.

        Removes keys containing ``vision_tower``, ``audio_tower``,
        ``embed_audio``, ``embed_vision``, ``rotary_emb.inv_freq``,
        and ``rope.inv_freq`` for text-only inference.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with text-only keys.
        """
        weights = super().sanitize(weights)
        weights = {
            key: value
            for key, value in weights.items()
            if "vision_tower" not in key
            and "audio_tower" not in key
            and "embed_audio" not in key
            and "embed_vision" not in key
            and "rotary_emb.inv_freq" not in key
            and "rope.inv_freq" not in key
        }
        return weights


__all__ = ("Gemma3NForCausalLM", "Gemma3NModel")
