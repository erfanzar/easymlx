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

"""Mamba2 MLX model implementation for serving and inference.

Mamba2 extends the original Mamba with grouped SSM heads, a gated RMS
normalization layer, and a restructured projection scheme.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule

from .mamba2_configuration import Mamba2Config

CacheView = TransformerCacheView | PageCacheView


def _swiglu(gate: mx.array, x: mx.array) -> mx.array:
    """SwiGLU activation: silu(gate) * x."""
    return nn.silu(gate) * x


class MambaRMSNormGated(nn.Module):
    """RMS normalization with optional SwiGLU gating (Mamba2-style).

    When a gate tensor is provided, applies ``silu(gate) * x`` before
    RMS normalization. Used as the output normalization in Mamba2 blocks.

    Attributes:
        eps: Epsilon for numerical stability.
        weight: Learnable scale parameter.

    Example:
        >>> norm = MambaRMSNormGated(hidden_size=64)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """Initialize gated RMSNorm.

        Args:
            hidden_size: Feature dimension.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array | None = None) -> mx.array:
        """Apply optional SwiGLU gating followed by RMS normalization.

        Args:
            hidden_states: Input tensor.
            gate: Optional gate tensor for SwiGLU gating. If ``None``,
                only RMSNorm is applied.

        Returns:
            Normalized (and optionally gated) tensor.
        """
        if gate is not None:
            hidden_states = _swiglu(gate, hidden_states)
        return mx.fast.rms_norm(hidden_states, self.weight, self.eps)


class Mamba2Block(nn.Module):
    """Single Mamba2 block with grouped SSM heads.

    Projects input into gate, convolution input, and time-step delta.
    After depthwise convolution, extracts B, C state matrices and runs
    a sequential selective scan with grouped heads.

    Attributes:
        layer_idx: Index of this block in the model stack.
        num_heads: Number of SSM heads.
        hidden_size: Model hidden dimension.
        ssm_state_size: SSM state dimensionality.
        conv_kernel_size: Convolution kernel size.
        intermediate_size: ``num_heads * head_dim``.
        n_groups: Number of head groups.
        head_dim: Per-head dimensionality.
        time_step_limit: Clamping bounds for time steps.
        conv1d: Depthwise 1D convolution.
        in_proj: Input projection.
        dt_bias: Learned bias for time-step delta.
        A_log: Log-space SSM state transition matrix.
        D: Skip connection scalar per head.
        norm: Gated RMSNorm for output.
        out_proj: Output projection.

    Example:
        >>> config = Mamba2Config(hidden_size=64, num_heads=4, head_dim=16)
        >>> block = Mamba2Block(config, layer_idx=0)
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        """Initialize a Mamba2 block.

        Args:
            config: Model configuration.
            layer_idx: Index of this block in the model stack.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.num_heads * config.head_dim
        self.use_conv_bias = config.use_conv_bias
        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.time_step_limit = config.time_step_limit
        self.heads_per_group = self.num_heads // self.n_groups
        self.use_bias = config.use_bias

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=config.conv_kernel,
            padding=0,
            groups=self.conv_dim,
            bias=config.use_conv_bias,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.use_bias)

        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)

        self.norm = MambaRMSNormGated(self.intermediate_size, eps=config.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def _conv(self, conv_input: mx.array) -> mx.array:
        """Apply depthwise conv1d with zero-padding and SiLU activation.

        Args:
            conv_input: Tensor of shape ``(batch, seq_len, conv_dim)``.

        Returns:
            Activated convolution output of the same shape.
        """
        padded_input = mx.pad(
            conv_input,
            [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)],
        )
        conv_output = self.conv1d(padded_input)
        return nn.silu(conv_output)

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
    ) -> mx.array:
        """Run the selective scan with grouped SSM heads.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, intermediate_size)``.
            B: State input matrix of shape ``(batch, seq_len, n_groups * state_size)``.
            C: State output matrix of shape ``(batch, seq_len, n_groups * state_size)``.
            dt: Time-step delta of shape ``(batch, seq_len, num_heads)``.

        Returns:
            SSM output of shape ``(batch, seq_len, intermediate_size)``.
        """
        batch_size, seq_len, _ = hidden_states.shape

        hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)

        # Clamp and apply dt bias
        A = -mx.exp(self.A_log)
        dt = dt + self.dt_bias
        dt = mx.clip(dt, *self.time_step_limit)
        dt = nn.softplus(dt)

        # Expand B and C for grouped heads
        if self.n_groups < self.num_heads:
            B = mx.repeat(B, self.heads_per_group, axis=2)
            C = mx.repeat(C, self.heads_per_group, axis=2)

        # Sequential scan
        state = None
        y_parts = []
        for t in range(seq_len):
            dt_t = dt[:, t, :, None, None]  # (B, nheads, 1, 1)
            x_t = hidden_states[:, t, :, :, None]  # (B, nheads, head_dim, 1)
            B_t = B[:, t, :, None, :]  # (B, nheads, 1, state_size)
            C_t = C[:, t, :, :, None]  # (B, nheads, state_size, 1)

            dA = mx.exp(dt_t * A[:, None, None])
            dBx = dt_t * (x_t @ B_t)  # (B, nheads, head_dim, state_size)

            if state is not None:
                state = state * dA + dBx
            else:
                state = dBx

            y_t = (state @ C_t).squeeze(-1)  # (B, nheads, head_dim)
            y_t = y_t + self.D[:, None] * hidden_states[:, t]
            y_parts.append(y_t)

        y = mx.stack(y_parts, axis=1)  # (B, T, nheads, head_dim)
        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Run the forward pass through the Mamba2 block.

        Projects input, applies convolution to extract x/B/C, runs the
        grouped SSM, applies gated normalization, and projects output.

        Args:
            hidden_states: Input tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        projected = self.in_proj(hidden_states)
        gate, conv_input, dt = mx.split(
            projected,
            [
                self.intermediate_size,
                self.intermediate_size + self.conv_dim,
            ],
            axis=-1,
        )

        conv_output = self._conv(conv_input)
        x, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )

        y = self._ssm(x, B, C, dt)
        y = self.norm(y, gate)
        return self.out_proj(y)


class Mamba2ResidualBlock(nn.Module):
    """Residual block wrapping a Mamba2Block with pre-normalization.

    Applies RMSNorm before the Mamba2Block and adds a residual connection.

    Attributes:
        mixer: The underlying ``Mamba2Block``.
        norm: Pre-block RMSNorm.

    Example:
        >>> config = Mamba2Config(hidden_size=64)
        >>> block = Mamba2ResidualBlock(config, layer_idx=0)
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        """Initialize the residual block.

        Args:
            config: Model configuration.
            layer_idx: Index of this block in the model stack.
        """
        super().__init__()
        self.mixer = Mamba2Block(config, layer_idx)
        self.norm = nn.RMSNorm(config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply norm, mixer, and residual connection.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        return self.mixer(self.norm(x)) + x


@register_module(task_type=TaskType.BASE_MODULE, config=Mamba2Config, model_type="mamba2")
class Mamba2Model(EasyMLXBaseModule):
    """Base Mamba2 SSM model (no LM head).

    Pure SSM model with grouped heads. Does not use standard KV caches.
    """

    config_class = Mamba2Config

    def __init__(self, config: Mamba2Config):
        """Initialize the base Mamba2 model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Mamba2ResidualBlock(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the forward pass through the Mamba2 SSM backbone.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Unused; accepted for API compatibility.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Unused; accepted for API compatibility.
            cache_metadata: Paged attention metadata (for API compatibility).

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.
        """
        if input_embeddings is not None:
            hidden_states = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1 and cache_metadata is None:
                input_ids = input_ids[None, :]
            hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Transpose conv1d weights from upstream format and remove tied head.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        sanitized = {}
        for k, v in weights.items():
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                sanitized[k] = v.moveaxis(2, 1)
            else:
                sanitized[k] = v
        if getattr(self.config, "tie_word_embeddings", True):
            sanitized.pop("lm_head.weight", None)
        return sanitized


@register_module(task_type=TaskType.CAUSAL_LM, config=Mamba2Config, model_type="mamba2")
class Mamba2ForCausalLM(BaseCausalLMModule[Mamba2Model, Mamba2Config]):
    """Mamba2 causal language model with an LM head.

    Wraps ``Mamba2Model`` (grouped-head SSM) with a language modeling head.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = Mamba2Config(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
        >>> model = Mamba2ForCausalLM(config)
    """

    config_class = Mamba2Config

    def __init__(self, config: Mamba2Config):
        """Initialize the Mamba2 causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Mamba2Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Delegate sanitization to the base model.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        return self.base_model.sanitize(weights)


__all__ = ("Mamba2ForCausalLM", "Mamba2Model")
