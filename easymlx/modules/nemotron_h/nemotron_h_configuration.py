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

"""Nemotron-H configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("nemotron_h")
class NemotronHConfig(EasyMLXBaseConfig):
    """Configuration for the Nemotron-H hybrid transformer model.

    Nemotron-H interleaves Mamba2 SSM blocks (``"M"``), attention blocks
    (``"*"``), dense MLP blocks (``"-"``), and MoE blocks (``"E"``)
    according to the ``hybrid_override_pattern``.

    Attributes:
        model_type: Identifier string (``"nemotron_h"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        intermediate_size: Dense MLP intermediate dimensionality.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads for GQA.
        head_dim: Per-head dimensionality.
        mamba_num_heads: Number of Mamba SSM heads.
        mamba_head_dim: Per-head dimension for Mamba SSM.
        ssm_state_size: SSM state dimension.
        conv_kernel: 1D convolution kernel size for Mamba.
        n_groups: Number of groups for Mamba B/C projections.
        hybrid_override_pattern: Per-layer block type pattern.
        moe_intermediate_size: Per-expert intermediate dimensionality.
        n_routed_experts: Total number of routed experts.
        num_experts_per_tok: Number of experts per token.
    """

    model_type = "nemotron_h"

    def __init__(
        self,
        *,
        vocab_size: int = 131072,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        max_position_embeddings: int = 4096,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        attention_bias: bool = False,
        mamba_num_heads: int = 64,
        mamba_head_dim: int = 64,
        mamba_proj_bias: bool = False,
        ssm_state_size: int = 128,
        conv_kernel: int = 4,
        n_groups: int = 8,
        mlp_bias: bool = False,
        layer_norm_epsilon: float = 1e-5,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        hybrid_override_pattern: list[str] | None = None,
        moe_intermediate_size: int | None = None,
        moe_shared_expert_intermediate_size: int | None = None,
        n_group: int | None = None,
        n_routed_experts: int | None = None,
        n_shared_experts: int | None = None,
        topk_group: int | None = None,
        num_experts_per_tok: int | None = None,
        norm_topk_prob: bool | None = None,
        routed_scaling_factor: float | None = None,
        time_step_limit: tuple[float, float] | None = None,
        time_step_min: float | None = None,
        time_step_max: float | None = None,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize Nemotron-H configuration.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimensionality of hidden states.
            intermediate_size (int): Dense MLP intermediate dimensionality.
            num_hidden_layers (int): Number of decoder layers.
            max_position_embeddings (int): Maximum sequence length.
            num_attention_heads (int): Number of attention heads.
            num_key_value_heads (int | None): Number of KV heads for GQA.
            head_dim (int | None): Per-head dimensionality.
            attention_bias (bool): Whether attention includes bias.
            mamba_num_heads (int): Number of Mamba SSM heads.
            mamba_head_dim (int): Per-head dimension for Mamba SSM.
            mamba_proj_bias (bool): Whether Mamba projections include bias.
            ssm_state_size (int): SSM state dimension.
            conv_kernel (int): 1D convolution kernel size for Mamba.
            n_groups (int): Number of groups for Mamba B/C projections.
            mlp_bias (bool): Whether MLP projections include bias.
            layer_norm_epsilon (float): LayerNorm epsilon.
            use_bias (bool): General bias flag.
            use_conv_bias (bool): Whether convolution includes bias.
            hybrid_override_pattern (list[str] | None): Per-layer block type
                pattern using ``"M"`` (Mamba), ``"*"`` (attention),
                ``"-"`` (MLP), ``"E"`` (MoE).
            moe_intermediate_size (int | None): Per-expert intermediate dim.
            moe_shared_expert_intermediate_size (int | None): Shared expert dim.
            n_group (int | None): MoE group count for topk.
            n_routed_experts (int | None): Total number of routed experts.
            n_shared_experts (int | None): Number of shared experts.
            topk_group (int | None): MoE topk group count.
            num_experts_per_tok (int | None): Experts activated per token.
            norm_topk_prob (bool | None): Whether to normalize top-k probs.
            routed_scaling_factor (float | None): Routed expert scaling factor.
            time_step_limit (tuple[float, float] | None): Mamba time step range.
            time_step_min (float | None): Minimum time step (legacy).
            time_step_max (float | None): Maximum time step (legacy).
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
        self.max_position_embeddings = int(max_position_embeddings)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim) if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.attention_bias = bool(attention_bias)
        self.mamba_num_heads = int(mamba_num_heads)
        self.mamba_head_dim = int(mamba_head_dim)
        self.mamba_proj_bias = bool(mamba_proj_bias)
        self.ssm_state_size = int(ssm_state_size)
        self.conv_kernel = int(conv_kernel)
        self.n_groups = int(n_groups)
        self.mlp_bias = bool(mlp_bias)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.rms_norm_eps = self.layer_norm_epsilon
        self.use_bias = bool(use_bias)
        self.use_conv_bias = bool(use_conv_bias)
        self.hybrid_override_pattern = hybrid_override_pattern or ["*", "-"] * (self.num_hidden_layers // 2)
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.n_group = n_group
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

        if time_step_limit is None and time_step_min is not None and time_step_max is not None:
            self.time_step_limit = (float(time_step_min), float(time_step_max))
        else:
            self.time_step_limit = time_step_limit

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("NemotronHConfig",)
