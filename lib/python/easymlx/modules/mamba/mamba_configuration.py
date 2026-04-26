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

"""Mamba configuration for EasyMLX inference."""

from __future__ import annotations

import math

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("mamba")
class MambaConfig(EasyMLXBaseConfig):
    """Configuration for the Mamba selective state space model.

    Attributes:
        model_type: Identifier string (``"mamba"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states (d_model).
        intermediate_size: SSM intermediate dimensionality (d_inner).
        state_size: SSM state dimensionality (d_state).
        num_hidden_layers: Number of Mamba residual blocks.
        conv_kernel: 1D convolution kernel size (d_conv).
        use_bias: Whether linear projections use bias.
        use_conv_bias: Whether the conv1d layer uses bias.
        time_step_rank: Rank for delta projection, or ``"auto"``.
        tie_word_embeddings: Whether to tie input/output embeddings.
        use_bcdt_rms: Whether to apply RMS normalization to B, C, dt
            (used by falcon_mamba).
        mixer_rms_eps: Epsilon for mixer RMS normalization.
    """

    model_type = "mamba"

    def __init__(
        self,
        *,
        vocab_size: int = 50280,
        hidden_size: int = 768,
        intermediate_size: int = 1536,
        state_size: int = 16,
        num_hidden_layers: int = 32,
        conv_kernel: int = 4,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        time_step_rank: int | str = "auto",
        tie_word_embeddings: bool = True,
        use_bcdt_rms: bool = False,
        mixer_rms_eps: float = 1e-6,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.state_size = int(state_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = 0
        self.num_key_value_heads = 0
        self.conv_kernel = int(conv_kernel)
        self.use_bias = bool(use_bias)
        self.use_conv_bias = bool(use_conv_bias)
        self.use_bcdt_rms = bool(use_bcdt_rms)
        self.mixer_rms_eps = float(mixer_rms_eps)

        if time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)
        else:
            self.time_step_rank = int(time_step_rank)

        """Initialize Mamba configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden states (d_model).
            intermediate_size: SSM intermediate dimensionality (d_inner).
            state_size: SSM state dimensionality (d_state).
            num_hidden_layers: Number of Mamba residual blocks.
            conv_kernel: 1D convolution kernel size (d_conv).
            use_bias: Whether linear projections use bias.
            use_conv_bias: Whether the conv1d layer uses bias.
            time_step_rank: Rank for delta projection. ``"auto"`` computes
                ``ceil(hidden_size / 16)``.
            tie_word_embeddings: Whether to tie input/output embeddings.
            use_bcdt_rms: Whether to apply RMS normalization to B, C, dt.
            mixer_rms_eps: Epsilon for mixer RMS normalization.
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


__all__ = ("MambaConfig",)
