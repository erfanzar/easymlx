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

"""Llama4 text-only wrappers for MLX inference.

This module provides text-only EasyMLX wrappers around the Llama4 text
decoder stack, exposed under the HuggingFace ``llama4_text`` model type.
Weight sanitization remaps HuggingFace ``model.*`` prefixes to
``language_model.*`` and drops rotary embedding inverse-frequency buffers.
"""

from __future__ import annotations

import mlx.core as mx

from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule
from easymlx.modules.llama4.modeling_llama4 import Llama4TextModel as _Llama4TextModel

from .llama4_text_configuration import Llama4TextConfig


def _rename_prefix(weights: dict[str, mx.array], *, old: str, new: str) -> dict[str, mx.array]:
    """Rename weight keys by replacing a prefix string.

    Args:
        weights: Dictionary mapping parameter names to weight arrays.
        old: The prefix to match and replace.
        new: The replacement prefix.

    Returns:
        A new dictionary with matching prefixes replaced.
    """
    return {(new + key[len(old) :]) if key.startswith(old) else key: value for key, value in weights.items()}


@register_module(task_type=TaskType.BASE_MODULE, config=Llama4TextConfig, model_type="llama4_text")
class Llama4TextModel(EasyMLXBaseModule):
    """Thin EasyMLX wrapper around the Llama4 text decoder stack.

    Delegates all forward-pass logic to the inner ``Llama4TextModel`` from
    the ``llama4`` module, and provides weight sanitization for HuggingFace
    checkpoint compatibility.

    Attributes:
        config_class: The associated configuration class (``Llama4TextConfig``).
        language_model: The underlying Llama4 text decoder transformer stack.

    Example::

        config = Llama4TextConfig(hidden_size=2048, num_hidden_layers=16)
        model = Llama4TextModel(config)
    """

    config_class = Llama4TextConfig

    def __init__(self, config: Llama4TextConfig):
        """Initialize the text-only Llama4 base model.

        Args:
            config: Llama4 text-only configuration. ``finalize()`` is called
                to resolve derived attributes.
        """
        super().__init__(config)
        config.finalize()
        self.language_model = _Llama4TextModel(config)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list | None = None,
        cache_metadata=None,
    ) -> mx.array:
        """Run the text decoder forward pass.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings; overrides ``input_ids``
                embedding lookup when provided.
            cache_views: Per-layer KV cache views for autoregressive decoding.
            cache_metadata: Paged-cache metadata.

        Returns:
            Hidden states of shape ``(batch, seq_len, hidden_size)`` or
            ``(total_tokens, hidden_size)`` in paged mode.
        """
        return self.language_model(
            input_ids,
            attention_mask=attention_mask,
            input_embeddings=input_embeddings,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
        )

    @property
    def embed_tokens(self):
        """Return the token embedding layer from the language model."""
        return self.language_model.embed_tokens

    @property
    def layers(self):
        """Return the list of decoder layers from the language model."""
        return self.language_model.layers

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize HuggingFace weights for the text-only Llama4 model.

        Removes rotary embedding inverse-frequency buffers and renames the
        ``model.`` prefix to ``language_model.`` to match the wrapper layout.

        Args:
            weights: Raw weight dictionary from a HuggingFace checkpoint.

        Returns:
            Sanitized weight dictionary with renamed keys and dropped buffers.
        """
        weights = {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }
        return _rename_prefix(weights, old="model.", new="language_model.")


@register_module(task_type=TaskType.CAUSAL_LM, config=Llama4TextConfig, model_type="llama4_text")
class Llama4TextForCausalLM(BaseCausalLMModule[Llama4TextModel, Llama4TextConfig]):
    """Llama4 text-only model with a causal language modeling head.

    Wraps ``Llama4TextModel`` and adds vocabulary projection to produce
    next-token logits. Weight sanitization handles HuggingFace checkpoint
    key remapping including ``output.weight`` to ``lm_head.weight``.

    Attributes:
        config_class: The associated configuration class (``Llama4TextConfig``).
    """

    config_class = Llama4TextConfig

    def __init__(self, config: Llama4TextConfig):
        """Initialize the causal language model.

        Args:
            config: Llama4 text-only configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Llama4TextModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize HuggingFace weights for the causal LM wrapper.

        Applies the base sanitization, renames ``output.weight`` to
        ``lm_head.weight``, drops rotary embedding inverse-frequency
        buffers, and remaps ``model.*`` keys to ``model.language_model.*``.

        Args:
            weights: Raw weight dictionary from a HuggingFace checkpoint.

        Returns:
            Sanitized weight dictionary ready for loading.
        """
        weights = super().sanitize(weights)
        if "output.weight" in weights:
            weights["lm_head.weight"] = weights.pop("output.weight")
        weights = {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }
        return _rename_prefix(weights, old="model.", new="model.language_model.")


__all__ = ("Llama4TextForCausalLM", "Llama4TextModel")
