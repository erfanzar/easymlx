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

"""Kimi VL MLX model implementation for serving and inference.

Kimi VL is a vision-language model whose text backbone is DeepSeek V3.
This module exposes only the text backbone as a CAUSAL_LM; the vision
tower and multi-modal projector weights are stripped during sanitize.
"""

from __future__ import annotations

import mlx.core as mx

from easymlx.infra import TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule
from easymlx.modules.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Model

from .kimi_vl_configuration import KimiVLConfig


@register_module(task_type=TaskType.BASE_MODULE, config=KimiVLConfig, model_type="kimi_vl")
class KimiVLModel(DeepseekV3Model):
    """Kimi VL base model (text backbone only).

    Upstream Kimi VL uses a DeepSeek V3 text backbone.  We reuse the
    full DeepSeek V3 model and only override sanitize to strip
    vision-related weights and remap the ``language_model`` prefix.
    """

    config_class = KimiVLConfig

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Strip vision tower weights and remap language_model prefix.

        Removes all keys containing ``vision_tower`` or
        ``multi_modal_projector``, then strips the ``language_model.``
        prefix from remaining keys so they align with the DeepSeek V3
        model structure. Finally delegates to the parent ``sanitize``.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Sanitized weight dictionary with vision keys removed and
            language_model prefix stripped.
        """

        def _keep(key: str) -> bool:
            """Return True if *key* is not a vision-related weight."""
            return "vision_tower" not in key and "multi_modal_projector" not in key

        remapped: dict[str, mx.array] = {}
        for key, value in weights.items():
            if not _keep(key):
                continue
            if key.startswith("language_model."):
                key = key[len("language_model.") :]
            remapped[key] = value

        return super().sanitize(remapped)


@register_module(task_type=TaskType.CAUSAL_LM, config=KimiVLConfig, model_type="kimi_vl")
class KimiVLForCausalLM(BaseCausalLMModule[KimiVLModel, KimiVLConfig]):
    """Kimi VL causal language model (text backbone only).

    Wraps ``KimiVLModel`` (a DeepSeek V3 text backbone) with an LM head.
    Vision tower and multi-modal projector weights are stripped during
    sanitization.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = KimiVLConfig(vocab_size=1000, hidden_size=256,
        ...     num_hidden_layers=4, num_attention_heads=8)
        >>> model = KimiVLForCausalLM(config)
    """

    config_class = KimiVLConfig

    def __init__(self, config: KimiVLConfig):
        """Initialize KimiVLForCausalLM.

        Args:
            config: Kimi VL model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=KimiVLModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Strip vision tower weights, remap language_model prefix, then apply base sanitize.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Sanitized weight dictionary.
        """

        def _keep(key: str) -> bool:
            """Return True if *key* is not a vision-related weight."""
            return "vision_tower" not in key and "multi_modal_projector" not in key

        remapped: dict[str, mx.array] = {}
        for key, value in weights.items():
            if not _keep(key):
                continue
            if key.startswith("language_model."):
                key = key[len("language_model.") :]
            remapped[key] = value

        return super().sanitize(remapped)


__all__ = ("KimiVLForCausalLM", "KimiVLModel")
