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

"""Gemma3 VLM wrapper for EasyMLX inference.

Wraps a Gemma3Text backbone for text-only inference.
Vision components are stripped during sanitization.
"""

from __future__ import annotations

import mlx.core as mx

from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule

from ..gemma3_text import Gemma3TextModel
from .gemma3_configuration import Gemma3Config


def _rename_prefix(weights: dict[str, mx.array], *, old: str, new: str) -> dict[str, mx.array]:
    """Rename weight key prefixes from ``old`` to ``new``.

    Args:
        weights: Dictionary of weight name to array mappings.
        old: Prefix to replace.
        new: Replacement prefix.

    Returns:
        New dictionary with renamed keys.
    """
    return {(new + key[len(old) :]) if key.startswith(old) else key: value for key, value in weights.items()}


@register_module(task_type=TaskType.BASE_MODULE, config=Gemma3Config, model_type="gemma3")
class Gemma3Model(EasyMLXBaseModule):
    """Text wrapper that delegates to a ``Gemma3TextModel`` language model.

    This model strips vision/multimodal components and wraps the
    Gemma3Text backbone for text-only inference with sliding window
    and full attention alternation.

    Attributes:
        language_model: The underlying ``Gemma3TextModel`` instance.

    Example::

        >>> model = Gemma3Model(Gemma3Config(vocab_size=256))
        >>> h = model(mx.array([[1, 2, 3]]))
    """

    config_class = Gemma3Config

    def __init__(self, config: Gemma3Config):
        """Initialize Gemma3 text wrapper.

        Args:
            config: VLM configuration; the text backbone config is
                extracted via ``config.get_text_config()``.
        """
        super().__init__(config)
        text_config = config.get_text_config()
        self.language_model = Gemma3TextModel(text_config)

    @property
    def layers(self):
        """Return the decoder layer stack from the text backbone."""
        return self.language_model.layers

    def get_embedding(self):
        """Return the token embedding layer from the text backbone."""
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
        """Forward pass delegating to the Gemma3Text backbone.

        Args:
            input_ids: Token ids of shape ``[batch, seq_len]``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings.
            cache_views: Per-layer KV cache views.
            cache_metadata: Page metadata for paged attention.

        Returns:
            Hidden states from the text backbone.
        """
        return self.language_model(
            input_ids,
            attention_mask=attention_mask,
            input_embeddings=input_embeddings,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights by removing vision/multimodal and rotary keys.

        Strips ``vision_tower``, ``multi_modal_projector``, and
        ``rotary_emb.inv_freq`` / ``rope.inv_freq`` keys from the
        weight dictionary.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary with text-only keys.
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


@register_module(task_type=TaskType.CAUSAL_LM, config=Gemma3Config, model_type="gemma3")
class Gemma3ForCausalLM(BaseCausalLMModule[Gemma3Model, Gemma3Config]):
    """Gemma3 causal language model wrapper.

    Wraps ``Gemma3Model`` and adds a linear LM head. During sanitization,
    remaps ``language_model.lm_head.weight`` to the top-level LM head
    and delegates remaining weight cleanup to the base model.

    Attributes:
        config_class: ``Gemma3Config``.

    Example::

        >>> model = Gemma3ForCausalLM(Gemma3Config(vocab_size=256))
        >>> logits = model(mx.array([[1, 2, 3]]))
    """

    config_class = Gemma3Config

    def __init__(self, config: Gemma3Config):
        super().__init__(
            config=config,
            base_model_class=Gemma3Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize VLM weights for text-only causal LM inference.

        Moves ``language_model.lm_head.weight`` to ``lm_head.weight``,
        remaps ``language_model.`` prefixes to ``model.language_model.``,
        and delegates vision/rotary stripping to the base model.

        Args:
            weights: Raw checkpoint weight dictionary.

        Returns:
            Cleaned weight dictionary.
        """
        weights = super().sanitize(weights)
        if "language_model.lm_head.weight" in weights:
            weights["lm_head.weight"] = weights.pop("language_model.lm_head.weight")
        weights = _rename_prefix(weights, old="language_model.", new="model.language_model.")
        return self.model.sanitize(weights)


__all__ = ("Gemma3ForCausalLM", "Gemma3Model")
