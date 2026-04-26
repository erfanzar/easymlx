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

"""Kimi K2.5 MLX model implementation for serving and inference.

Wraps the DeepSeek V3 backbone, following the upstream Kimi K2.5
checkpoint layout where the text model is nested under ``language_model``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import (
    PageCacheView,
    PageMetadata,
    TransformerCacheView,
)
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule
from easymlx.modules.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Model

from .kimi_k25_configuration import KimiK25Config

CacheView = TransformerCacheView | PageCacheView


@register_module(task_type=TaskType.BASE_MODULE, config=KimiK25Config, model_type="kimi_k25")
class KimiK25Model(EasyMLXBaseModule):
    """Base Kimi K2.5 transformer model wrapping DeepSeek V3.

    Attributes:
        config_class: The associated configuration class (``KimiK25Config``).
        language_model: Inner DeepSeek V3 model.
    """

    config_class = KimiK25Config

    def __init__(self, config: KimiK25Config):
        """Initialize KimiK25Model.

        Args:
            config: Kimi K2.5 model configuration.
        """
        super().__init__(config)
        self.language_model = DeepseekV3Model(config.text_config)

    @property
    def embed_tokens(self) -> nn.Embedding:
        """Token embedding layer from the inner model."""
        return self.language_model.embed_tokens

    @property
    def layers(self):
        """Decoder layers from the inner model."""
        return self.language_model.layers

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the Kimi K2.5 forward pass by delegating to the inner DeepSeek V3 model.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.
        """
        return self.language_model(
            input_ids,
            attention_mask=attention_mask,
            input_embeddings=input_embeddings,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remap upstream checkpoint keys for Kimi K2.5.

        Performs the following transformations:
        1. Strips vision-related keys (``vision_tower``, ``vision_model``,
           ``multi_modal_projector``, ``mm_projector``).
        2. Remaps ``language_model.*`` prefix to expose the inner model
           weights.
        3. Delegates MoE expert stacking to the inner DeepSeek V3 model's
           ``sanitize`` method.
        4. Removes rotary embedding inv_freq buffers.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Sanitized weight dictionary ready for model loading.
        """

        vision_prefixes = ("vision_tower.", "vision_model.", "multi_modal_projector.", "mm_projector.")
        weights = {key: value for key, value in weights.items() if not any(key.startswith(p) for p in vision_prefixes)}

        lm_prefix = "language_model."
        lm_weights = {}
        other_weights = {}
        for key, value in weights.items():
            if key.startswith(lm_prefix):
                lm_weights[key[len(lm_prefix) :]] = value
            else:
                other_weights[key] = value

        if lm_weights:
            model_prefix = "model."
            inner_weights = {}
            head_weights = {}
            for key, value in lm_weights.items():
                if key.startswith(model_prefix):
                    inner_weights[key] = value
                else:
                    head_weights[key] = value

            if hasattr(self.language_model, "sanitize"):
                inner_weights = self.language_model.sanitize(inner_weights)

            sanitized = {}
            for k, v in inner_weights.items():
                sanitized[f"{lm_prefix}{k}"] = v
            for k, v in head_weights.items():
                sanitized[f"{lm_prefix}{k}"] = v
            sanitized.update(other_weights)
        else:
            sanitized = weights

        return {
            key: value
            for key, value in sanitized.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=KimiK25Config, model_type="kimi_k25")
class KimiK25ForCausalLM(BaseCausalLMModule[KimiK25Model, KimiK25Config]):
    """Kimi K2.5 model with a causal language modeling head.

    Attributes:
        config_class: The associated configuration class (``KimiK25Config``).
    """

    config_class = KimiK25Config

    def __init__(self, config: KimiK25Config):
        """Initialize KimiK25ForCausalLM.

        Args:
            config: Kimi K2.5 model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=KimiK25Model,
            tie_word_embeddings=False,
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Delegate sanitization to the inner KimiK25Model, then filter rotary buffers.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Sanitized weight dictionary.
        """

        base_model = self.base_model
        if hasattr(base_model, "sanitize"):
            weights = base_model.sanitize(weights)

        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


__all__ = ("KimiK25ForCausalLM", "KimiK25Model")
