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

"""Mamba2 configuration for EasyMLX inference."""

from __future__ import annotations

import math

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("mamba2")
class Mamba2Config(EasyMLXBaseConfig):
    """Configuration for the Mamba2 selective state space model.

    Mamba2 extends Mamba with grouped heads and a different SSM
    computation strategy.

    Attributes:
        model_type: Identifier string (``"mamba2"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        num_heads: Number of SSM heads.
        head_dim: Per-head dimensionality.
        intermediate_size: SSM intermediate dimensionality.
        state_size: SSM state dimensionality.
        num_hidden_layers: Number of residual blocks.
        conv_kernel: 1D convolution kernel size.
        n_groups: Number of head groups.
        use_bias: Whether linear projections use bias.
        use_conv_bias: Whether the conv1d layer uses bias.
        time_step_rank: Rank for delta projection, or ``"auto"``.
        time_step_limit: Clamping bounds for time steps.
        layer_norm_epsilon: Epsilon for layer normalization.
        tie_word_embeddings: Whether to tie input/output embeddings.
    """

    model_type = "mamba2"

    def __init__(
        self,
        *,
        vocab_size: int = 32768,
        hidden_size: int = 768,
        num_heads: int = 48,
        head_dim: int = 16,
        state_size: int = 128,
        num_hidden_layers: int = 32,
        conv_kernel: int = 4,
        n_groups: int = 1,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        time_step_rank: int | str = "auto",
        time_step_limit: tuple[float, float] = (0.0, float("inf")),
        layer_norm_epsilon: float = 1e-5,
        tie_word_embeddings: bool = True,
        ssm_state_size: int | None = None,
        max_position_embeddings: int = 2056,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.intermediate_size = self.num_heads * self.head_dim
        self.state_size = int(state_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = 0
        self.num_key_value_heads = 0
        self.conv_kernel = int(conv_kernel)
        self.n_groups = int(n_groups)
        self.use_bias = bool(use_bias)
        self.use_conv_bias = bool(use_conv_bias)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.max_position_embeddings = int(max_position_embeddings)

        if time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)
        else:
            self.time_step_rank = int(time_step_rank)

        if isinstance(time_step_limit, (list, tuple)):
            self.time_step_limit = tuple(float(x) for x in time_step_limit)
        else:
            self.time_step_limit = (0.0, float("inf"))

        self.ssm_state_size = int(ssm_state_size) if ssm_state_size is not None else self.state_size

        """Initialize Mamba2 configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden states.
            num_heads: Number of SSM heads.
            head_dim: Per-head dimensionality.
            state_size: SSM state dimensionality.
            num_hidden_layers: Number of residual blocks.
            conv_kernel: 1D convolution kernel size.
            n_groups: Number of head groups.
            use_bias: Whether linear projections use bias.
            use_conv_bias: Whether the conv1d layer uses bias.
            time_step_rank: Rank for delta projection. ``"auto"`` computes
                ``ceil(hidden_size / 16)``.
            time_step_limit: ``(min, max)`` clamping bounds for time steps.
            layer_norm_epsilon: Epsilon for layer normalization.
            tie_word_embeddings: Whether to tie input/output embeddings.
            ssm_state_size: Override for SSM state size (defaults to ``state_size``).
            max_position_embeddings: Maximum sequence length.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID(s).
            bos_token_id: Beginning-of-sequence token ID.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("Mamba2Config",)
