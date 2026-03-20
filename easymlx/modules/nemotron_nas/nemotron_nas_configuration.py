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

"""NemotronNAS configuration for EasyMLX inference."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@dataclass(frozen=True)
class AttentionBlockConfig:
    """Per-layer attention configuration for NemotronNAS.

    Attributes:
        no_op: If True, skip attention entirely.
        replace_with_linear: If True, replace attention with a linear layer.
        n_heads_in_group: GQA group size (num_heads / n_heads_in_group = kv_heads).
    """

    no_op: bool = False
    replace_with_linear: bool = False
    n_heads_in_group: int | None = None

    def __post_init__(self):
        """Validate attention block configuration.

        Raises:
            ValueError: If ``n_heads_in_group`` is not specified for active
                (non-no_op, non-linear-replacement) attention blocks.
        """
        if self.no_op or self.replace_with_linear:
            object.__setattr__(self, "n_heads_in_group", None)
        elif not self.no_op and self.n_heads_in_group is None:
            raise ValueError("n_heads_in_group must be specified for active attention blocks")


@dataclass(frozen=True)
class FFNBlockConfig:
    """Per-layer FFN configuration for NemotronNAS.

    Attributes:
        no_op: If True, skip FFN entirely.
        replace_with_linear: If True, replace FFN with a linear layer.
        ffn_mult: Multiplier to compute intermediate_size from hidden_size.
    """

    no_op: bool = False
    replace_with_linear: bool = False
    ffn_mult: float | None = None

    def __post_init__(self):
        """Validate FFN block configuration.

        Raises:
            ValueError: If ``ffn_mult`` is not specified for active
                (non-no_op, non-linear-replacement) FFN blocks.
        """
        if self.no_op or self.replace_with_linear:
            object.__setattr__(self, "ffn_mult", None)
        elif not self.no_op and self.ffn_mult is None:
            raise ValueError("ffn_mult must be specified for active FFN blocks")
        elif self.ffn_mult is not None:
            object.__setattr__(self, "ffn_mult", round(self.ffn_mult, 6))


@dataclass(frozen=True)
class BlockConfig:
    """Per-layer block configuration combining attention and FFN configs."""

    attention: AttentionBlockConfig
    ffn: FFNBlockConfig

    @classmethod
    def from_dict(cls, data: dict) -> BlockConfig:
        """Create a BlockConfig from a dictionary.

        Args:
            data (dict): Dictionary with optional ``"attention"`` and ``"ffn"``
                sub-dictionaries.

        Returns:
            BlockConfig: Parsed block configuration.
        """
        attn_conf = AttentionBlockConfig(**data.get("attention", {}))
        ffn_conf = FFNBlockConfig(**data.get("ffn", {}))
        return cls(attention=attn_conf, ffn=ffn_conf)


def _find_multiple(n: int, k: int) -> int:
    """Find smallest multiple of k that is >= n.

    Args:
        n (int): The value to round up.
        k (int): The multiple to align to.

    Returns:
        int: Smallest multiple of k that is >= n.
    """
    if n % k == 0:
        return n
    return n + k - (n % k)


def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
    """Calculate intermediate size based on multiplier, rounded to multiple of 256.

    Uses the formula ``round_up(2 * ffn_mult * n_embd / 3, 256)``.

    Args:
        ffn_mult (float): FFN expansion multiplier.
        n_embd (int): Hidden size (embedding dimension).

    Returns:
        int: Intermediate size rounded to nearest multiple of 256.
    """
    intermediate_size = int(2 * ffn_mult * n_embd / 3)
    return _find_multiple(intermediate_size, 256)


@register_config("nemotron-nas")
class NemotronNASConfig(EasyMLXBaseConfig):
    """Configuration for the NemotronNAS transformer model.

    NemotronNAS uses heterogeneous per-layer architecture where each
    layer can have different attention and FFN configurations. Some
    layers can be no-op or replaced with linear layers.

    Attributes:
        model_type: Identifier string (``"nemotron-nas"``).
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of hidden states.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads.
        rms_norm_eps: RMSNorm epsilon.
        block_configs: Per-layer block configuration list.
        hidden_act: Activation function name.
        attention_bias: Whether attention includes bias.
        mlp_bias: Whether MLP projections include bias.
        rope_theta: RoPE base frequency.
        rope_scaling: Optional RoPE scaling configuration.
        max_position_embeddings: Maximum sequence length.
    """

    model_type = "nemotron-nas"

    def __init__(
        self,
        *,
        vocab_size: int = 128256,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        rms_norm_eps: float = 1e-5,
        block_configs: list[dict | BlockConfig] | None = None,
        hidden_act: str = "silu",
        attention_bias: bool = False,
        mlp_bias: bool = False,
        rope_theta: float = 500000.0,
        rope_scaling: dict[str, tp.Any] | None = None,
        max_position_embeddings: int = 131072,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize NemotronNAS configuration.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimensionality of hidden states.
            num_hidden_layers (int): Number of transformer decoder layers.
            num_attention_heads (int): Number of attention heads.
            rms_norm_eps (float): RMSNorm epsilon.
            block_configs (list[dict | BlockConfig] | None): Per-layer block
                configurations. Each entry specifies attention and FFN behavior
                (active, no-op, or linear replacement).
            hidden_act (str): Activation function name (``"silu"``, etc.).
            attention_bias (bool): Whether attention projections include bias.
            mlp_bias (bool): Whether MLP projections include bias.
            rope_theta (float): RoPE base frequency.
            rope_scaling (dict[str, Any] | None): RoPE scaling configuration.
            max_position_embeddings (int): Maximum sequence length.
            tie_word_embeddings (bool): Whether to tie input/output embeddings.
            pad_token_id (int | None): Padding token ID.
            eos_token_id (int | list[int] | None): End-of-sequence token ID(s).
            bos_token_id (int | None): Beginning-of-sequence token ID.
            **kwargs: Additional arguments passed to ``EasyMLXBaseConfig``.
        """
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.rms_norm_eps = float(rms_norm_eps)
        self.hidden_act = str(hidden_act)
        self.attention_bias = bool(attention_bias)
        self.mlp_bias = bool(mlp_bias)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = int(max_position_embeddings)

        # Parse block_configs
        if block_configs is None:
            # Default: all standard layers with full attention and ffn_mult=4
            block_configs = [
                BlockConfig(
                    attention=AttentionBlockConfig(n_heads_in_group=1),
                    ffn=FFNBlockConfig(ffn_mult=4.0),
                )
                for _ in range(num_hidden_layers)
            ]
        else:
            parsed = []
            for bc in block_configs:
                if isinstance(bc, dict):
                    parsed.append(BlockConfig.from_dict(bc))
                else:
                    parsed.append(bc)
            block_configs = parsed

        self.block_configs = block_configs

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = "NemotronNASConfig"
