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

"""MiniMax configuration for EasyMLX inference.

MiniMax is a Mixture-of-Experts model with sigmoid gating, QK normalization,
and e_score_correction_bias for expert routing. Uses SwitchGLU for expert
computation.
"""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("minimax")
class MiniMaxConfig(EasyMLXBaseConfig):
    """Configuration for the MiniMax MoE language model.

    Attributes:
        model_type: Identifier string (``"minimax"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: Per-expert MLP intermediate dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality, or None to auto-derive.
        max_position_embeddings: Maximum sequence length.
        num_experts_per_tok: Number of experts activated per token.
        num_local_experts: Total number of routing experts.
        shared_intermediate_size: Shared expert intermediate size.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        rotary_dim: Number of dimensions for rotary embeddings.
        scoring_func: Routing scoring function (``"sigmoid"``).
        use_qk_norm: Whether to apply QK normalization.
        tie_word_embeddings: Whether to tie input/output embeddings.
    """

    model_type = "minimax"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int = 4096,
        num_experts_per_tok: int = 2,
        num_local_experts: int = 8,
        shared_intermediate_size: int = 0,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rotary_dim: int | None = None,
        scoring_func: str = "sigmoid",
        use_qk_norm: bool = True,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize MiniMax configuration.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimensionality of hidden states.
            intermediate_size (int): Per-expert MLP intermediate dimensionality.
            num_hidden_layers (int): Number of transformer decoder layers.
            num_attention_heads (int): Number of attention heads.
            num_key_value_heads (int | None): Number of KV heads for GQA.
                Defaults to ``num_attention_heads``.
            head_dim (int | None): Per-head dimensionality. Auto-derived from
                ``hidden_size // num_attention_heads`` when None.
            max_position_embeddings (int): Maximum sequence length.
            num_experts_per_tok (int): Number of experts activated per token.
            num_local_experts (int): Total number of routing experts.
            shared_intermediate_size (int): Shared expert intermediate size.
            rms_norm_eps (float): RMSNorm epsilon.
            rope_theta (float): RoPE base frequency.
            rotary_dim (int | None): Number of dimensions for rotary embeddings.
            scoring_func (str): Routing scoring function (``"sigmoid"``).
            use_qk_norm (bool): Whether to apply QK normalization.
            tie_word_embeddings (bool): Whether to tie input/output embeddings.
            pad_token_id (int | None): Padding token ID.
            eos_token_id (int | list[int] | None): End-of-sequence token ID(s).
            bos_token_id (int | None): Beginning-of-sequence token ID.
            **kwargs: Additional arguments passed to ``EasyMLXBaseConfig``.
        """
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.max_position_embeddings = int(max_position_embeddings)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.num_local_experts = int(num_local_experts)
        self.shared_intermediate_size = int(shared_intermediate_size)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.rotary_dim = (
            int(rotary_dim)
            if rotary_dim is not None
            else (self.head_dim or (self.hidden_size // self.num_attention_heads))
        )
        self.scoring_func = str(scoring_func)
        self.use_qk_norm = bool(use_qk_norm)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "MiniMaxConfig"
