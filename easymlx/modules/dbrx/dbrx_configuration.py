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

"""DBRX configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("dbrx")
class DBRXConfig(EasyMLXBaseConfig):
    """Configuration for the DBRX transformer model.

    DBRX is a Mixture-of-Experts model with NormAttnNorm structure
    (two LayerNorms per layer), SwiGLU MoE experts, fused QKV
    projection with QKV value clipping, and top-k expert routing.
    Registered under model type ``"dbrx"``.

    Args:
        vocab_size: Number of tokens in the vocabulary. Defaults to 100352.
        d_model: Dimensionality of hidden representations. Also aliased
            as ``hidden_size``. Defaults to 6144.
        n_layers: Number of transformer decoder layers. Defaults to 40.
        n_heads: Number of attention heads. Defaults to 48.
        ffn_config: Dictionary configuring the MoE FFN layer with keys:
            ``ffn_hidden_size`` (expert intermediate dim, default 10752),
            ``moe_num_experts`` (total experts, default 16),
            ``moe_top_k`` (experts per token, default 4),
            ``moe_jitter_eps`` (jitter epsilon, default 0.0).
        attn_config: Dictionary configuring attention with keys:
            ``kv_n_heads`` (KV heads for GQA, default 8),
            ``clip_qkv`` (QKV clipping value, default 8.0),
            ``rope_theta`` (RoPE base frequency, default 500000.0).
        tie_word_embeddings: Whether to tie input/output embeddings. Defaults to ``False``.

    Attributes:
        model_type: Identifier string (``"dbrx"``).
        hidden_size: Alias for ``d_model``.

    Example::

        >>> config = DBRXConfig(d_model=4096, n_layers=24)
        >>> config.model_type
        'dbrx'
    """

    model_type = "dbrx"

    def __init__(
        self,
        *,
        vocab_size: int = 100352,
        d_model: int = 6144,
        n_layers: int = 40,
        n_heads: int = 48,
        ffn_config: dict[str, tp.Any] | None = None,
        attn_config: dict[str, tp.Any] | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize the DBRX configuration.

        See class docstring for full parameter documentation.
        """
        if ffn_config is None:
            ffn_config = {
                "ffn_hidden_size": 10752,
                "moe_num_experts": 16,
                "moe_top_k": 4,
                "moe_jitter_eps": 0.0,
            }
        if attn_config is None:
            attn_config = {
                "kv_n_heads": 8,
                "clip_qkv": 8.0,
                "rope_theta": 500000.0,
            }

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.hidden_size = int(d_model)  # alias for BaseCausalLMModule compatibility
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ffn_config = dict(ffn_config)
        self.attn_config = dict(attn_config)
        self.num_hidden_layers = self.n_layers
        self.num_attention_heads = self.n_heads
        self.num_key_value_heads = self.attn_config.get("kv_n_heads", 8)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "DBRXConfig"
