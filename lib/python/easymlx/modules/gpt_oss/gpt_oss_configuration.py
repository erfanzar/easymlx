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

"""GPT-OSS configuration (serving/inference only).

This module defines the configuration class for the GPT-OSS model, which
uses a Mixture-of-Experts architecture with sliding window attention and
YaRN RoPE scaling.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("gpt_oss")
class GptOssConfig(EasyMLXBaseConfig):
    """Configuration for the GPT-OSS Mixture-of-Experts model.

    Defines hyperparameters for the GPT-OSS architecture including
    sliding window attention, MoE routing, and activation clamping.

    Attributes:
        model_type: Identifier string (``"gpt_oss"``).
        num_hidden_layers: Number of transformer decoder layers.
        num_local_experts: Total number of local MoE experts.
        vocab_size: Vocabulary size.
        hidden_size: Hidden state dimensionality.
        intermediate_size: MoE expert intermediate size.
        head_dim: Per-head dimensionality.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        sliding_window: Sliding window size for local attention layers.
        rope_theta: RoPE base frequency.
        hidden_act: Activation function name.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: RMSNorm epsilon.
        rope_scaling: RoPE scaling configuration (defaults to YaRN).
        attention_dropout: Attention dropout rate.
        num_experts_per_tok: Experts activated per token.
        router_aux_loss_coef: Router auxiliary loss coefficient.
        output_router_logits: Whether to output router logits.
        use_cache: Whether to use KV caching.
        mlp_activations_limit: Clamping limit for MLP activations.
        attention_bias: Whether attention includes bias (always True).
        layer_types: Per-layer attention type specification.
    """

    model_type = "gpt_oss"

    def __init__(
        self,
        *,
        num_hidden_layers: int = 36,
        num_local_experts: int = 128,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        head_dim: int | None = 64,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = 8,
        sliding_window: int = 128,
        rope_theta: float = 150000.0,
        tie_word_embeddings: bool = False,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        rope_scaling: dict[str, tp.Any] | None = None,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 4,
        router_aux_loss_coef: float = 0.9,
        output_router_logits: bool = False,
        use_cache: bool = True,
        layer_types: list["str"] | None = None,
        mlp_activations_limit: float = 7.0,
        pad_token_id: int | None = None,
        eos_token_id: int | list["int"] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initializes a GPT-OSS configuration.

        Args:
            num_hidden_layers: Number of decoder layers. Defaults to 36.
            num_local_experts: Total MoE experts. Defaults to 128.
            vocab_size: Vocabulary size. Defaults to 201088.
            hidden_size: Hidden dim. Defaults to 2880.
            intermediate_size: Expert intermediate size. Defaults to 2880.
            head_dim: Per-head dim. Defaults to 64.
            num_attention_heads: Attention heads. Defaults to 64.
            num_key_value_heads: KV heads. Defaults to 8.
            sliding_window: Sliding window size. Defaults to 128.
            rope_theta: RoPE base frequency. Defaults to 150000.0.
            tie_word_embeddings: Tie embeddings. Defaults to False.
            hidden_act: Activation function. Defaults to ``"silu"``.
            initializer_range: Init range. Defaults to 0.02.
            max_position_embeddings: Max seq length. Defaults to 131072.
            rms_norm_eps: RMSNorm epsilon. Defaults to 1e-5.
            rope_scaling: RoPE scaling config. Defaults to YaRN with
                factor 32.
            attention_dropout: Attention dropout. Defaults to 0.0.
            num_experts_per_tok: Experts per token. Defaults to 4.
            router_aux_loss_coef: Router loss coefficient. Defaults to 0.9.
            output_router_logits: Output router logits. Defaults to False.
            use_cache: Enable KV caching. Defaults to True.
            layer_types: Per-layer attention types. Defaults to alternating
                sliding/full attention.
            mlp_activations_limit: MLP activation clamp. Defaults to 7.0.
            pad_token_id: Padding token ID. Defaults to None.
            eos_token_id: EOS token ID(s). Defaults to None.
            bos_token_id: BOS token ID. Defaults to None.
            **kwargs: Additional keyword arguments for the base class.
        """
        if rope_scaling is None:
            rope_scaling = {
                "rope_type": "yarn",
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
            }
        if "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling = dict(rope_scaling)
            rope_scaling["rope_type"] = rope_scaling["type"]

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_hidden_layers = int(num_hidden_layers)
        self.num_local_experts = int(num_local_experts)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.head_dim = int(head_dim) if head_dim is not None else self.hidden_size // num_attention_heads
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.sliding_window = int(sliding_window)
        self.rope_theta = float(rope_theta)
        self.hidden_act = str(hidden_act)
        self.initializer_range = float(initializer_range)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_scaling = rope_scaling
        self.attention_dropout = float(attention_dropout)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.router_aux_loss_coef = float(router_aux_loss_coef)
        self.output_router_logits = bool(output_router_logits)
        self.use_cache = bool(use_cache)
        self.mlp_activations_limit = float(mlp_activations_limit)
        self.attention_bias = True

        if layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "GptOssConfig"
