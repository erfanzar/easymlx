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

"""Mamba MLX model implementation for serving and inference.

Pure selective state space model (SSM) -- no attention layers.
Maintains conv state and SSM state instead of KV caches.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule

from .mamba_configuration import MambaConfig

CacheView = TransformerCacheView | PageCacheView


def _swiglu(gate: mx.array, x: mx.array) -> mx.array:
    """SwiGLU activation: silu(gate) * x."""
    return nn.silu(gate) * x


class MambaBlock(nn.Module):
    """Single Mamba selective state space block.

    Performs: ``in_proj -> conv1d -> SSM selective scan -> out_proj``
    with a gated residual path through SwiGLU. This is a pure SSM block
    with NO attention mechanism.

    Attributes:
        hidden_size: Model hidden dimension (d_model).
        ssm_state_size: SSM state dimensionality (d_state).
        conv_kernel_size: Convolution kernel size (d_conv).
        intermediate_size: SSM intermediate dimensionality (d_inner).
        time_step_rank: Rank for the delta projection.
        use_conv_bias: Whether conv1d uses bias.
        use_bcdt_rms: Whether to apply RMS norm to B, C, dt.
        in_proj: Projects input to ``2 * intermediate_size`` (x and gate).
        conv1d: Depthwise 1D convolution.
        x_proj: Projects to ``time_step_rank + 2 * state_size`` (dt, B, C).
        dt_proj: Projects delta from rank space to intermediate size.
        A_log: Log-space SSM state transition matrix.
        D: Skip connection scalar per channel.
        out_proj: Output projection.

    Example:
        >>> config = MambaConfig(hidden_size=64, intermediate_size=128, state_size=16)
        >>> block = MambaBlock(config)
    """

    def __init__(self, config: MambaConfig):
        """Initialize a Mamba SSM block.

        Args:
            config: Model configuration with SSM hyperparameters.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        self.use_conv_bias = config.use_conv_bias
        self.use_bcdt_rms = config.use_bcdt_rms

        if self.use_bcdt_rms:
            self._mixer_rms_eps = config.mixer_rms_eps

        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            bias=self.use_conv_bias,
            padding=0,
        )

        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + 2 * self.ssm_state_size,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        A = mx.repeat(
            mx.arange(1.0, self.ssm_state_size + 1.0).reshape([1, self.ssm_state_size]),
            repeats=self.intermediate_size,
            axis=0,
        )
        self.A_log = mx.log(A)
        self.D = mx.ones([self.intermediate_size])

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def _mixer_norm(self, x: mx.array) -> mx.array:
        """Apply RMS normalization for mixer (falcon_mamba compatibility).

        Args:
            x: Input tensor.

        Returns:
            RMS-normalized tensor.
        """
        return mx.fast.rms_norm(x, mx.ones(x.shape[-1], x.dtype), eps=self._mixer_rms_eps)

    def ssm_step(self, x: mx.array, A: mx.array, state: mx.array | None = None) -> tuple[mx.array, mx.array]:
        """Execute a single SSM selective scan step.

        Args:
            x: Input tensor of shape ``(batch, intermediate_size)``.
            A: Negative-exponentiated state transition matrix.
            state: Previous SSM state of shape
                ``(batch, intermediate_size, state_size)``, or ``None``.

        Returns:
            Tuple of ``(output, new_state)`` where output has shape
            ``(batch, intermediate_size)`` and new_state has shape
            ``(batch, intermediate_size, state_size)``.
        """
        D = self.D
        deltaBC = self.x_proj(x)
        delta, B, C = mx.split(
            deltaBC,
            [self.time_step_rank, self.time_step_rank + self.ssm_state_size],
            axis=-1,
        )
        if self.use_bcdt_rms:
            delta = self._mixer_norm(delta)
            B = self._mixer_norm(B)
            C = self._mixer_norm(C)
        delta = nn.softplus(self.dt_proj(delta))
        new_state = mx.expand_dims(delta * x, -1) * mx.expand_dims(B, 1)
        if state is not None:
            new_state += state * mx.exp(mx.expand_dims(delta, -1) * A)
        y = (new_state @ mx.expand_dims(C, -1)).squeeze(2)
        y = y + D * x
        return y, new_state

    def _process_sequence(
        self,
        x: mx.array,
        conv_cache: mx.array | None,
        state_cache: mx.array | None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """Process a full sequence through the Mamba block.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            conv_cache: Previous convolution cache, or ``None``.
            state_cache: Previous SSM state cache, or ``None``.

        Returns:
            Tuple of ``(output, (new_conv_cache, new_state_cache))``.
        """
        _B, T, _D = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.split(indices_or_sections=2, axis=-1)

        K = self.conv_kernel_size
        if conv_cache is not None:
            x_full = mx.concatenate([conv_cache, x_branch], axis=1)
        else:
            x_full = mx.pad(x_branch, [(0, 0), (K - 1, 0), (0, 0)])

        conv_out = self.conv1d(x_full)
        new_conv_cache = x_full[:, -(K - 1) :, :]
        x_branch = nn.silu(conv_out)

        A = -mx.exp(self.A_log)
        current_state = state_cache
        y_parts = []
        for t in range(T):
            y_t, current_state = self.ssm_step(x_branch[:, t], A, current_state)
            y_parts.append(y_t)
        y = mx.stack(y_parts, axis=1)

        output = self.out_proj(_swiglu(z, y))
        return output, (new_conv_cache, current_state)

    def __call__(
        self,
        x: mx.array,
        conv_cache: mx.array | None = None,
        state_cache: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Run the Mamba block forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            conv_cache: Previous convolution state cache, or ``None``.
            state_cache: Previous SSM state cache, or ``None``.

        Returns:
            Tuple of ``(output, new_conv_cache, new_state_cache)``.
        """
        output, (new_conv_cache, new_state_cache) = self._process_sequence(x, conv_cache, state_cache)
        return output, new_conv_cache, new_state_cache


class MambaResidualBlock(nn.Module):
    """Residual block wrapping a MambaBlock with pre-normalization.

    Applies RMSNorm before the MambaBlock and adds a residual connection.

    Attributes:
        mixer: The underlying ``MambaBlock``.
        norm: Pre-block RMSNorm.

    Example:
        >>> config = MambaConfig(hidden_size=64)
        >>> block = MambaResidualBlock(config)
    """

    def __init__(self, config: MambaConfig):
        """Initialize the residual block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = nn.RMSNorm(config.hidden_size)

    def __call__(
        self,
        x: mx.array,
        conv_cache: mx.array | None = None,
        state_cache: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Apply norm, mixer, and residual connection.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            conv_cache: Previous convolution cache, or ``None``.
            state_cache: Previous SSM state cache, or ``None``.

        Returns:
            Tuple of ``(output, new_conv_cache, new_state_cache)``.
        """
        output, new_conv, new_state = self.mixer(self.norm(x), conv_cache, state_cache)
        return output + x, new_conv, new_state


@register_module(task_type=TaskType.BASE_MODULE, config=MambaConfig, model_type="mamba")
class MambaModel(EasyMLXBaseModule):
    """Base Mamba SSM model (no LM head).

    This is a pure SSM model with no attention layers. It does not use
    standard KV caches. The ``cache_views`` parameter is accepted for
    API compatibility but SSM state is managed internally during generation.
    """

    config_class = MambaConfig

    def __init__(self, config: MambaConfig):
        """Initialize the base Mamba model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [MambaResidualBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the forward pass through the Mamba SSM backbone.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Unused; accepted for API compatibility.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Unused; accepted for API compatibility (SSM state
                is managed internally).
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
            hidden_states, _, _ = layer(hidden_states)

        return self.norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Transpose conv1d weights from upstream format and remove tied head.

        Conv1d weights are transposed from ``(out, in, kernel)`` to
        ``(out, kernel, in)`` when the last dimension is not 1. If
        ``tie_word_embeddings`` is enabled, the ``lm_head.weight`` key
        is removed.

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


@register_module(task_type=TaskType.CAUSAL_LM, config=MambaConfig, model_type="mamba")
class MambaForCausalLM(BaseCausalLMModule[MambaModel, MambaConfig]):
    """Mamba causal language model with an LM head.

    Wraps ``MambaModel`` (pure SSM, no attention) with a language
    modeling head. Supports tied input/output embeddings (default).

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = MambaConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
        >>> model = MambaForCausalLM(config)
    """

    config_class = MambaConfig

    def __init__(self, config: MambaConfig):
        """Initialize the Mamba causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=MambaModel,
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


__all__ = ("MambaForCausalLM", "MambaModel")
