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

"""GraniteMoeHybrid configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("granitemoehybrid")
class GraniteMoeHybridConfig(EasyMLXBaseConfig):
    """Configuration for the GraniteMoeHybrid transformer model.

    GraniteMoeHybrid is a Mamba2+Attention hybrid architecture with
    optional MoE. Each layer can be either ``"mamba"`` or ``"attention"``,
    specified via ``layer_types``. When ``num_local_experts`` is set,
    MoE routing with SwitchGLU experts and a shared MLP is used;
    otherwise dense SwiGLU MLPs are used. Retains Granite's custom
    scaling multipliers for embeddings, attention, residuals, and logits.
    Registered as model type ``"granitemoehybrid"``.

    Attributes:
        model_type: Identifier string (``"granitemoehybrid"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: MoE/dense MLP intermediate dimensionality.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        layer_types: Per-layer list of ``"mamba"`` or ``"attention"``.
        embedding_multiplier: Multiplier for token embeddings.
        attention_multiplier: Multiplier for attention scale.
        residual_multiplier: Multiplier for residual connections.
        logits_scaling: Divisor applied to output logits.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        max_position_embeddings: Maximum sequence length.
        attention_bias: Whether attention includes bias.
        mlp_bias: Whether MLP projections include bias.
        num_local_experts: Number of MoE experts (None for dense).
        num_experts_per_tok: Experts activated per token.
        shared_intermediate_size: Intermediate size for shared MLP.
        mamba_n_heads: Number of Mamba SSM heads.
        mamba_d_head: Mamba head dimension.
        mamba_d_state: Mamba SSM state size.
        mamba_d_conv: Mamba conv kernel size.
        mamba_n_groups: Number of Mamba groups.
        mamba_proj_bias: Whether Mamba projections use bias.
        mamba_conv_bias: Whether Mamba conv uses bias.
        position_embedding_type: ``"rope"`` or ``"nope"``.
        time_step_limit: Mamba time step limits.

    Args:
        vocab_size: Vocabulary size. Defaults to 32000.
        hidden_size: Hidden dimensionality. Defaults to 4096.
        intermediate_size: MLP/MoE intermediate size. Defaults to 11008.
        num_hidden_layers: Number of layers. Defaults to 32.
        num_attention_heads: Number of heads. Defaults to 32.
        num_key_value_heads: KV heads. Defaults to ``num_attention_heads``.
        layer_types: Per-layer type list. Defaults to all ``"attention"``.
        embedding_multiplier: Embedding scale. Defaults to 1.0.
        attention_multiplier: Attention scale. Defaults to 1.0.
        residual_multiplier: Residual scale. Defaults to 1.0.
        logits_scaling: Logit divisor. Defaults to 1.0.
        rms_norm_eps: RMSNorm epsilon. Defaults to 1e-5.
        rope_theta: RoPE base. Defaults to 10000.0.
        max_position_embeddings: Max seq length. Defaults to 2048.
        attention_bias: Attention bias. Defaults to False.
        mlp_bias: MLP bias. Defaults to False.
        num_local_experts: Total experts (None=dense). Defaults to None.
        num_experts_per_tok: Active experts. Defaults to None.
        shared_intermediate_size: Shared MLP size. Defaults to None.
        mamba_n_heads: Mamba heads. Defaults to None.
        mamba_d_head: Mamba head dim. Defaults to None.
        mamba_d_state: SSM state size. Defaults to None.
        mamba_d_conv: Conv kernel size. Defaults to None.
        mamba_n_groups: Mamba groups. Defaults to None.
        mamba_proj_bias: Mamba projection bias. Defaults to None.
        mamba_conv_bias: Mamba conv bias. Defaults to None.
        position_embedding_type: ``"rope"`` or ``"nope"``.
            Defaults to ``"rope"``.
        time_step_limit: Mamba time step bounds. Defaults to (0.001, 100.0).
        tie_word_embeddings: Tie embeddings. Defaults to True.

    Example::

        >>> config = GraniteMoeHybridConfig(
        ...     layer_types=["attention", "mamba", "attention"],
        ...     num_local_experts=4,
        ... )
        >>> config.use_moe
        True
    """

    model_type = "granitemoehybrid"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        layer_types: list[str] | None = None,
        embedding_multiplier: float = 1.0,
        attention_multiplier: float = 1.0,
        residual_multiplier: float = 1.0,
        logits_scaling: float = 1.0,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        num_local_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        shared_intermediate_size: int | None = None,
        mamba_n_heads: int | None = None,
        mamba_d_head: int | None = None,
        mamba_d_state: int | None = None,
        mamba_d_conv: int | None = None,
        mamba_n_groups: int | None = None,
        mamba_proj_bias: bool | None = None,
        mamba_conv_bias: bool | None = None,
        position_embedding_type: str = "rope",
        time_step_limit: tuple[float, float] = (0.001, 100.0),
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if layer_types is None:
            layer_types = ["attention"] * num_hidden_layers

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.layer_types = list(layer_types)
        self.embedding_multiplier = float(embedding_multiplier)
        self.attention_multiplier = float(attention_multiplier)
        self.residual_multiplier = float(residual_multiplier)
        self.logits_scaling = float(logits_scaling)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.max_position_embeddings = int(max_position_embeddings)
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)
        self.num_local_experts = int(num_local_experts) if num_local_experts is not None else None
        self.num_experts_per_tok = int(num_experts_per_tok) if num_experts_per_tok is not None else None
        self.shared_intermediate_size = int(shared_intermediate_size) if shared_intermediate_size is not None else None
        self.mamba_n_heads = int(mamba_n_heads) if mamba_n_heads is not None else None
        self.mamba_d_head = int(mamba_d_head) if mamba_d_head is not None else None
        self.mamba_d_state = int(mamba_d_state) if mamba_d_state is not None else None
        self.mamba_d_conv = int(mamba_d_conv) if mamba_d_conv is not None else None
        self.mamba_n_groups = int(mamba_n_groups) if mamba_n_groups is not None else None
        self.mamba_proj_bias = bool(mamba_proj_bias) if mamba_proj_bias is not None else None
        self.mamba_conv_bias = bool(mamba_conv_bias) if mamba_conv_bias is not None else None
        self.position_embedding_type = str(position_embedding_type)
        self.time_step_limit = tuple(time_step_limit)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def use_moe(self) -> bool:
        """Whether the model uses Mixture-of-Experts.

        Returns:
            True if ``num_local_experts`` is set and nonzero.
        """
        return bool(self.num_local_experts)


__all__ = "GraniteMoeHybridConfig"
