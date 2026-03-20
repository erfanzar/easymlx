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

"""Pixtral compatibility wrapper built on the existing Llama stack."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule

from ..llama import LlamaModel
from .pixtral_configuration import PixtralConfig


def _rename_prefix(weights: dict[str, mx.array], *, old: str, new: str) -> dict[str, mx.array]:
    """Rename weight keys by replacing an old prefix with a new one.

    Args:
        weights: Weight dictionary.
        old: Prefix to match and replace.
        new: Replacement prefix.

    Returns:
        New weight dictionary with renamed keys.
    """
    return {(new + key[len(old) :]) if key.startswith(old) else key: value for key, value in weights.items()}


@register_module(task_type=TaskType.BASE_MODULE, config=PixtralConfig, model_type="pixtral")
class PixtralModel(EasyMLXBaseModule):
    """Pixtral base model that delegates to the Llama transformer stack.

    This is a text-only wrapper. Vision tower and multimodal projector
    weights are stripped during sanitization.

    Attributes:
        config_class: Associated configuration class (``PixtralConfig``).
        text_config: Resolved Llama text configuration.
        model: Inner ``LlamaModel`` transformer stack.

    Example:
        >>> model = PixtralModel(config)
        >>> hidden = model(input_ids)
    """

    config_class = PixtralConfig

    def __init__(self, config: PixtralConfig):
        """Initialize the Pixtral base model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.text_config = config.get_text_config()
        self.model = LlamaModel(self.text_config)

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list | None = None,
        cache_metadata=None,
    ) -> mx.array:
        """Forward pass delegated to the inner Llama model.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or
                ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)``.
        """
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            input_embeddings=input_embeddings,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
        )

    def get_embedding(self) -> nn.Embedding:
        """Return the token embedding layer from the inner model.

        Returns:
            The ``nn.Embedding`` module.
        """
        return self.model.embed_tokens

    @property
    def layers(self):
        """Return the decoder layer list from the inner Llama model."""
        return self.model.layers

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Drop vision tower and multimodal projector weights, remap prefixes.

        Removes all keys containing ``vision_tower`` or
        ``multi_modal_projector`` and remaps the
        ``language_model.model.`` prefix to ``model.``.

        Args:
            weights: Raw weight dictionary from a checkpoint.

        Returns:
            Filtered and remapped weight dictionary.
        """
        weights = {
            key: value
            for key, value in weights.items()
            if "vision_tower" not in key and "multi_modal_projector" not in key
        }
        return _rename_prefix(weights, old="language_model.model.", new="model.")


@register_module(task_type=TaskType.CAUSAL_LM, config=PixtralConfig, model_type="pixtral")
class PixtralForCausalLM(BaseCausalLMModule[PixtralModel, PixtralConfig]):
    """Pixtral causal LM wrapper with Llama-backed text generation.

    Attributes:
        config_class: Associated configuration class (``PixtralConfig``).

    Example:
        >>> model = PixtralForCausalLM(config)
        >>> logits = model(input_ids)
    """

    config_class = PixtralConfig

    def __init__(self, config: PixtralConfig):
        """Initialize the Pixtral causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=PixtralModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights for the causal LM wrapper.

        Remaps ``language_model.lm_head`` to ``lm_head``, drops vision
        weights, and remaps ``language_model.model.`` to ``model.model.``.

        Args:
            weights: Raw weight dictionary from a checkpoint.

        Returns:
            Filtered and remapped weight dictionary.
        """
        weights = super().sanitize(weights)
        if "language_model.lm_head.weight" in weights:
            weights["lm_head.weight"] = weights.pop("language_model.lm_head.weight")
        weights = {
            key: value
            for key, value in weights.items()
            if "vision_tower" not in key and "multi_modal_projector" not in key
        }
        return _rename_prefix(weights, old="language_model.model.", new="model.model.")


__all__ = ("PixtralForCausalLM", "PixtralModel")
