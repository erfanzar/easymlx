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

"""PLaMo2 configuration (serving/inference only).

PLaMo2 is a hybrid Attention + Mamba architecture where some layers use
standard attention and others use Mamba SSM blocks.
"""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("plamo2")
class Plamo2Config(EasyMLXBaseConfig):
    """Configuration for the PLaMo2 hybrid transformer model.

    Attributes:
        model_type: Identifier string (``"plamo2"``).
        hidden_size_per_head: Per-head dimensionality.
        attention_window_size: Attention window size.
        mamba_d_state: Mamba SSM state dimension.
        mamba_d_conv: Mamba convolution kernel size.
        mamba_num_heads: Number of Mamba heads.
        mamba_step: Step size for Mamba layer placement.
        mamba_chunk_size: Chunk size for Mamba processing.
        mamba_enabled: Whether Mamba layers are enabled.
        full_attention_idx: Indices of layers with full attention.
    """

    model_type = "plamo2"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 13312,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        hidden_size_per_head: int = 128,
        max_position_embeddings: int = 2048,
        attention_window_size: int = 2048,
        full_attention_idx: list[int] | None = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_traditional: bool = False,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = True,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_num_heads: int = 64,
        mamba_step: int = 2,
        mamba_chunk_size: int = 256,
        mamba_enabled: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize PLaMo2 configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden states.
            intermediate_size: SwiGLU MLP intermediate dimensionality.
            num_hidden_layers: Number of decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of KV heads for GQA. Defaults
                to 4 if None.
            hidden_size_per_head: Per-head dimensionality.
            max_position_embeddings: Maximum sequence length.
            attention_window_size: Window size for attention layers.
            full_attention_idx: Indices of layers that use full (global)
                attention instead of windowed attention.
            rms_norm_eps: Epsilon for RMSNorm.
            rope_theta: Base frequency for RoPE.
            rope_traditional: Whether to use the traditional RoPE layout.
            attention_bias: Whether attention projections use bias.
            mlp_bias: Whether MLP projections use bias.
            tie_word_embeddings: Whether to tie input/output embeddings.
            mamba_d_state: Mamba SSM state dimension.
            mamba_d_conv: Mamba convolution kernel size.
            mamba_num_heads: Number of Mamba heads.
            mamba_step: Step size for Mamba layer placement.
            mamba_chunk_size: Chunk size for Mamba processing.
            mamba_enabled: Whether Mamba layers are enabled.
            pad_token_id: Token ID used for padding.
            eos_token_id: Token ID(s) for end of sequence.
            bos_token_id: Token ID for beginning of sequence.
            **kwargs: Additional keyword arguments forwarded to
                ``EasyMLXBaseConfig``.
        """
        if num_key_value_heads is None:
            num_key_value_heads = 4
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.hidden_size_per_head = int(hidden_size_per_head)
        self.head_dim = int(hidden_size_per_head)
        self.max_position_embeddings = int(max_position_embeddings)
        self.attention_window_size = int(attention_window_size)
        self.full_attention_idx = full_attention_idx
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rope_traditional = bool(rope_traditional)
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)
        self.mamba_d_state = int(mamba_d_state)
        self.mamba_d_conv = int(mamba_d_conv)
        self.mamba_num_heads = int(mamba_num_heads)
        self.mamba_step = int(mamba_step)
        self.mamba_chunk_size = int(mamba_chunk_size)
        self.mamba_enabled = bool(mamba_enabled)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "Plamo2Config"
