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

"""Phi3Small configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("phi3small")
class Phi3SmallConfig(EasyMLXBaseConfig):
    """Configuration for the Phi3Small transformer model.

    Phi3Small uses a dense architecture with GeGELU activation,
    block-sparse attention patterns, and muP scaling for embeddings
    and attention.

    Attributes:
        model_type: Identifier string (``"phi3small"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        ff_intermediate_size: MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        dense_attention_every_n_layers: How often to use dense attention.
        gegelu_limit: Clamp limit for GeGELU activation.
        gegelu_pad_to_256: Whether to pad GeGELU to multiple of 256.
        mup_attn_multiplier: muP attention scaling multiplier.
        mup_embedding_multiplier: muP embedding scaling multiplier.
        mup_use_scaling: Whether to use muP scaling.
        mup_width_multiplier: muP width multiplier for logits scaling.
        layer_norm_epsilon: LayerNorm epsilon.
        rope_embedding_base: RoPE base frequency.
        rope_position_scale: RoPE position scale.
        blocksparse_block_size: Block size for block-sparse attention.
        blocksparse_num_local_blocks: Number of local blocks.
        blocksparse_vert_stride: Vertical stride for block-sparse attention.
    """

    model_type = "phi3small"

    def __init__(
        self,
        *,
        vocab_size: int = 100352,
        hidden_size: int = 4096,
        ff_intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        dense_attention_every_n_layers: int = 2,
        gegelu_limit: float = 20.0,
        gegelu_pad_to_256: bool = True,
        mup_attn_multiplier: float = 1.0,
        mup_embedding_multiplier: float = 10.0,
        mup_use_scaling: bool = True,
        mup_width_multiplier: float = 8.0,
        layer_norm_epsilon: float = 1e-5,
        rope_embedding_base: float = 1000000.0,
        rope_position_scale: float = 1.0,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        blocksparse_block_size: int = 64,
        blocksparse_num_local_blocks: int = 16,
        blocksparse_vert_stride: int = 8,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize Phi3Small configuration.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Dimensionality of hidden states.
            ff_intermediate_size: MLP intermediate dimensionality.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of KV heads for GQA.
            dense_attention_every_n_layers: Interval for placing layers
                with block-sparse attention pattern (layers at indices
                divisible by this value use block-sparse attention).
            gegelu_limit: Clamp limit for GeGELU activation values.
            gegelu_pad_to_256: Whether to pad GeGELU intermediate size
                to a multiple of 256 (not used at runtime).
            mup_attn_multiplier: muP scaling multiplier for attention
                logits normalization.
            mup_embedding_multiplier: muP scaling factor applied to
                embeddings after lookup.
            mup_use_scaling: Whether to enable muP-style scaling for
                attention and embeddings.
            mup_width_multiplier: muP width multiplier used to scale
                output logits.
            layer_norm_epsilon: Epsilon for LayerNorm.
            rope_embedding_base: Base frequency for RoPE.
            rope_position_scale: Position scale factor for RoPE.
            max_position_embeddings: Maximum sequence length.
            head_dim: Per-head dimensionality. If None, computed as
                ``hidden_size // num_attention_heads``.
            blocksparse_block_size: Block size for block-sparse attention.
            blocksparse_num_local_blocks: Number of local blocks in
                block-sparse attention.
            blocksparse_vert_stride: Vertical stride for block-sparse
                attention.
            tie_word_embeddings: Whether to tie input/output embeddings.
            pad_token_id: Token ID used for padding.
            eos_token_id: Token ID(s) for end of sequence.
            bos_token_id: Token ID for beginning of sequence.
            **kwargs: Additional keyword arguments forwarded to
                ``EasyMLXBaseConfig``.
        """
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.ff_intermediate_size = int(ff_intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.dense_attention_every_n_layers = int(dense_attention_every_n_layers)
        self.gegelu_limit = float(gegelu_limit)
        self.gegelu_pad_to_256 = bool(gegelu_pad_to_256)
        self.mup_attn_multiplier = float(mup_attn_multiplier)
        self.mup_embedding_multiplier = float(mup_embedding_multiplier)
        self.mup_use_scaling = bool(mup_use_scaling)
        self.mup_width_multiplier = float(mup_width_multiplier)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.rope_embedding_base = float(rope_embedding_base)
        self.rope_position_scale = float(rope_position_scale)
        self.max_position_embeddings = int(max_position_embeddings)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.blocksparse_block_size = int(blocksparse_block_size)
        self.blocksparse_num_local_blocks = int(blocksparse_num_local_blocks)
        self.blocksparse_vert_stride = int(blocksparse_vert_stride)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "Phi3SmallConfig"
