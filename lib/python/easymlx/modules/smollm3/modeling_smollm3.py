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

"""SmolLM-3 MLX implementation.

This module provides a compatibility-first thin wrapper around the existing
Llama stack. It keeps the Llama attention/MLP/cache machinery intact and only
replaces the RoPE module on layers that should run without rotary embeddings.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule

from ..llama import LlamaModel
from .smollm3_configuration import SmolLM3Config

CacheView = TransformerCacheView | PageCacheView


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


class SmolLM3NoPE(nn.Module):
    """No-op rotary positional embedding for SmolLM-3 no-RoPE layers.

    This module replaces the standard RoPE module on layers where rotary
    embeddings are disabled. It returns the input unchanged, preserving
    the API contract expected by the attention module.

    Example:
        >>> nope = SmolLM3NoPE()
        >>> x = mx.ones((1, 4, 10, 64))
        >>> assert (nope(x) == x).all()
    """

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """Return the input unchanged (identity function).

        Args:
            x: Input tensor of any shape.
            offset: Positional offset (ignored).

        Returns:
            The input tensor ``x`` unmodified.
        """
        del offset
        return x


def _resolve_no_rope_layers(config: SmolLM3Config) -> list[int]:
    """Return a normalized 0/1 mask indicating which layers use RoPE.

    A value of ``1`` means the layer keeps RoPE enabled; ``0`` means
    RoPE is replaced with ``SmolLM3NoPE``. If ``no_rope_layers`` is
    not set on the config, the mask is derived from
    ``no_rope_layer_interval``.

    Args:
        config: SmolLM-3 model configuration.

    Returns:
        A list of ``int`` (0 or 1) with length ``num_hidden_layers``.
    """
    layers = getattr(config, "no_rope_layers", None)
    if layers is None:
        interval = int(getattr(config, "no_rope_layer_interval", 4))
        return [int((idx + 1) % interval != 0) for idx in range(config.num_hidden_layers)]
    return [int(bool(flag)) for flag in layers]


@register_module(task_type=TaskType.BASE_MODULE, config=SmolLM3Config, model_type="smollm3")
class SmolLM3Model(EasyMLXBaseModule):
    """Base SmolLM-3 transformer model.

    A thin wrapper around the Llama model that selectively replaces the
    RoPE module with ``SmolLM3NoPE`` on designated layers. All other
    components (attention, MLP, caching) are inherited from the Llama
    implementation.

    Attributes:
        config_class: Associated configuration class (``SmolLM3Config``).
        model: Underlying ``LlamaModel`` instance.
        no_rope_layers: Per-layer mask indicating RoPE usage (``1`` = RoPE,
            ``0`` = NoPE).

    Example:
        >>> config = SmolLM3Config(hidden_size=2048, num_hidden_layers=4)
        >>> model = SmolLM3Model(config)
        >>> out = model(mx.array([[1, 2, 3]]))
    """

    config_class = SmolLM3Config

    def __init__(self, config: SmolLM3Config):
        """Initialize the SmolLM-3 model.

        Creates a ``LlamaModel`` and patches layers that should not use
        RoPE by replacing their ``rope`` attribute with ``SmolLM3NoPE``.

        Args:
            config: SmolLM-3 model configuration.
        """
        super().__init__(config)
        self.model = LlamaModel(config)
        self.no_rope_layers = _resolve_no_rope_layers(config)
        for layer, use_rope in zip(self.model.layers, self.no_rope_layers, strict=False):
            if not use_rope:
                layer.self_attn.rope = SmolLM3NoPE()

    @property
    def layers(self):
        """Return the decoder layers from the underlying Llama model."""
        return self.model.layers

    def get_embedding(self):
        """Return the token embedding layer.

        Returns:
            The ``nn.Embedding`` module from the underlying Llama model.
        """
        return self.model.embed_tokens

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the forward pass through the SmolLM-3 model.

        Delegates entirely to the underlying ``LlamaModel``, which
        handles embedding, attention (with or without RoPE per layer),
        and final normalization.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings. When provided,
                ``input_ids`` is ignored.
            cache_views: Per-layer KV cache views for autoregressive decoding.
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

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Remove non-persistent rotary embedding buffers from checkpoint weights.

        Filters out ``rotary_emb.inv_freq`` and ``rope.inv_freq`` keys
        that are stored in some upstream checkpoints but recomputed at init.

        Args:
            weights: Raw checkpoint weight dict.

        Returns:
            Cleaned weight dict with rotary buffers removed.
        """
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key and "rope.inv_freq" not in key
        }


@register_module(task_type=TaskType.CAUSAL_LM, config=SmolLM3Config, model_type="smollm3")
class SmolLM3ForCausalLM(BaseCausalLMModule[SmolLM3Model, SmolLM3Config]):
    """SmolLM-3 causal language model with LM head.

    Wraps ``SmolLM3Model`` and adds a linear language-model head for
    next-token prediction. Supports weight tying between the embedding
    layer and the LM head.

    Attributes:
        config_class: Associated configuration class (``SmolLM3Config``).

    Example:
        >>> config = SmolLM3Config(hidden_size=2048, num_hidden_layers=4)
        >>> model = SmolLM3ForCausalLM(config)
        >>> output = model(mx.array([[1, 2, 3]]))
    """

    config_class = SmolLM3Config

    def __init__(self, config: SmolLM3Config):
        """Initialize the causal LM wrapper.

        Args:
            config: SmolLM-3 model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=SmolLM3Model,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights and remap ``model.`` prefix to ``model.model.``.

        The upstream SmolLM-3 checkpoint stores the base model weights under
        ``model.`` but this wrapper nests it under ``model.model.`` (the
        ``SmolLM3Model.model`` ``LlamaModel``).

        Args:
            weights: Raw checkpoint weight dict.

        Returns:
            Cleaned and re-prefixed weight dict.
        """
        weights = super().sanitize(weights)
        return _rename_prefix(weights, old="model.", new="model.model.")


Smollm3NoPE = SmolLM3NoPE
Smollm3Model = SmolLM3Model
Smollm3ForCausalLM = SmolLM3ForCausalLM

__all__ = (
    "SmolLM3ForCausalLM",
    "SmolLM3Model",
    "SmolLM3NoPE",
    "Smollm3ForCausalLM",
    "Smollm3Model",
    "Smollm3NoPE",
)
