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

"""Mistral3 wrapper models for EasyMLX inference."""

from __future__ import annotations

import mlx.core as mx

from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule

from ..llama import LlamaModel
from ..ministral3 import Ministral3Model
from .mistral3_configuration import Mistral3Config


def _rename_prefix(weights: dict[str, mx.array], *, old: str, new: str) -> dict[str, mx.array]:
    """Rename weight key prefixes in a checkpoint dict.

    Args:
        weights: Checkpoint weight dict.
        old: The prefix to replace.
        new: The replacement prefix.

    Returns:
        New dict with matching prefixes replaced.
    """
    return {(new + key[len(old) :]) if key.startswith(old) else key: value for key, value in weights.items()}


@register_module(task_type=TaskType.BASE_MODULE, config=Mistral3Config, model_type="mistral3")
class Mistral3Model(EasyMLXBaseModule):
    """Mistral3 text model wrapper that delegates to a Ministral3 or Llama backbone.

    This wrapper instantiates the appropriate text backbone based on the
    ``text_config.model_type`` and delegates all forward-pass computation
    to it. Vision-related weights (vision_tower, multi_modal_projector)
    are filtered during sanitization since this module handles text only.

    Attributes:
        config_class: Associated configuration class (``Mistral3Config``).
        language_model: The underlying ``Ministral3Model`` or ``LlamaModel``.

    Example:
        >>> config = Mistral3Config(text_config={"model_type": "ministral3"})
        >>> model = Mistral3Model(config)
    """

    config_class = Mistral3Config

    def __init__(self, config: Mistral3Config):
        """Initialize the Mistral3 text wrapper.

        Args:
            config: Mistral3 configuration. The text backbone type is
                determined by ``config.get_text_config().model_type``.
        """
        super().__init__(config)
        text_config = config.get_text_config()
        if text_config.model_type == "ministral3":
            self.language_model = Ministral3Model(text_config)
        else:
            self.language_model = LlamaModel(text_config)

    @property
    def layers(self):
        """Return the decoder layers from the underlying language model."""
        return self.language_model.layers

    def get_embedding(self):
        """Return the token embedding layer.

        Returns:
            The ``nn.Embedding`` module from the underlying language model.
        """
        return self.language_model.embed_tokens

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list | None = None,
        cache_metadata=None,
    ) -> mx.array:
        """Run the forward pass through the text backbone.

        Delegates entirely to the underlying ``Ministral3Model`` or
        ``LlamaModel``.

        Args:
            input_ids: Token IDs.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged-attention metadata.

        Returns:
            Hidden states from the language model.
        """
        return self.language_model(
            input_ids,
            attention_mask=attention_mask,
            input_embeddings=input_embeddings,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remove vision-related and rotary buffer weights.

        Filters out keys containing ``vision_tower``,
        ``multi_modal_projector``, ``rotary_emb.inv_freq``, and
        ``rope.inv_freq``.

        Args:
            weights: Raw checkpoint weight dict.

        Returns:
            Cleaned weight dict with vision and rotary keys removed.
        """
        weights = {
            key: value
            for key, value in weights.items()
            if "vision_tower" not in key
            and "multi_modal_projector" not in key
            and "rotary_emb.inv_freq" not in key
            and "rope.inv_freq" not in key
        }
        return _rename_prefix(weights, old="language_model.", new="language_model.")


@register_module(task_type=TaskType.CAUSAL_LM, config=Mistral3Config, model_type="mistral3")
class Mistral3ForCausalLM(BaseCausalLMModule[Mistral3Model, Mistral3Config]):
    """Mistral3 causal language model wrapper with an LM head.

    Wraps ``Mistral3Model`` and provides weight sanitization that remaps
    upstream ``language_model.`` prefixed weights to the nested
    ``model.language_model.`` path and extracts ``lm_head.weight``.

    Attributes:
        config_class: Associated configuration class (``Mistral3Config``).

    Example:
        >>> config = Mistral3Config(text_config={"model_type": "ministral3"})
        >>> model = Mistral3ForCausalLM(config)
    """

    config_class = Mistral3Config

    def __init__(self, config: Mistral3Config):
        """Initialize the Mistral3 causal LM wrapper.

        Args:
            config: Mistral3 model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Mistral3Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remap upstream weight prefixes for the Mistral3 wrapper layout.

        Performs the following transformations:
        1. Calls ``super().sanitize()`` to filter rotary/vision keys.
        2. Moves ``language_model.lm_head.weight`` to ``lm_head.weight``.
        3. Remaps ``language_model.`` prefix to ``model.language_model.``.
        4. Delegates to ``self.model.sanitize()`` for further cleanup.

        Args:
            weights: Raw checkpoint weight dict.

        Returns:
            Cleaned and remapped weight dict.
        """
        weights = super().sanitize(weights)
        if "language_model.lm_head.weight" in weights:
            weights["lm_head.weight"] = weights.pop("language_model.lm_head.weight")
        weights = _rename_prefix(weights, old="language_model.", new="model.language_model.")
        return self.model.sanitize(weights)


__all__ = ("Mistral3ForCausalLM", "Mistral3Model")
