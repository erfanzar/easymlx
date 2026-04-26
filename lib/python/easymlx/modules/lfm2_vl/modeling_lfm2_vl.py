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

"""LFM2-VL MLX model implementation for serving and inference.

LFM2-VL wraps the LFM2 text model. The vision tower and projector are
stripped at load time via sanitize, so this module only handles text
generation through the underlying LFM2 backbone.
"""

from __future__ import annotations

import mlx.core as mx

from easymlx.caching import PageCacheView, PageMetadata, TransformerCacheView
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules._base import BaseCausalLMModule
from easymlx.modules.lfm2 import Lfm2Config, Lfm2Model

from .lfm2_vl_configuration import Lfm2VlConfig

CacheView = TransformerCacheView | PageCacheView


@register_module(task_type=TaskType.BASE_MODULE, config=Lfm2VlConfig, model_type="lfm2-vl")
class Lfm2VlModel(EasyMLXBaseModule):
    """Base LFM2-VL model providing text-only inference through an LFM2 backbone.

    The vision tower and multi-modal projector are stripped during weight
    sanitization, so this module only handles text generation.

    Attributes:
        config_class: Associated configuration class.
        language_model: The underlying ``Lfm2Model`` text backbone.

    Example:
        >>> config = Lfm2VlConfig(text_config={"vocab_size": 1000, "hidden_size": 64})
        >>> model = Lfm2VlModel(config)
    """

    config_class = Lfm2VlConfig

    def __init__(self, config: Lfm2VlConfig):
        """Initialize the LFM2-VL model.

        Args:
            config: Model configuration containing nested text config.
        """
        super().__init__(config)
        text_cfg = config.text_config if isinstance(config.text_config, Lfm2Config) else Lfm2Config(**config.text_config)
        self.language_model = Lfm2Model(text_cfg)

    @property
    def layers(self):
        """Access the decoder layers of the underlying text model.

        Returns:
            List of decoder layers from the ``Lfm2Model``.
        """
        return self.language_model.layers

    @property
    def embed_tokens(self):
        """Access the token embedding table of the underlying text model.

        Returns:
            The ``nn.Embedding`` module from the ``Lfm2Model``.
        """
        return self.language_model.embed_tokens

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the forward pass through the LFM2 text backbone.

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` or ``(seq_len,)``.
            attention_mask: Optional attention mask.
            input_embeddings: Pre-computed embeddings (overrides ``input_ids``).
            cache_views: Per-layer KV cache views.
            cache_metadata: Paged attention metadata.

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
        """Strip vision tower and multi-modal projector weights.

        Removes all weights prefixed with ``vision_tower.`` and
        ``multi_modal_projector.``, and transposes conv1d weights.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary with only text model weights.
        """

        sanitized = {}
        for k, v in weights.items():
            if k.startswith("vision_tower.") or k.startswith("multi_modal_projector."):
                continue
            if "conv.weight" in k:
                if v.shape[-1] > v.shape[1]:
                    v = v.transpose(0, 2, 1)
            sanitized[k] = v
        return sanitized


@register_module(task_type=TaskType.CAUSAL_LM, config=Lfm2VlConfig, model_type="lfm2-vl")
class Lfm2VlForCausalLM(BaseCausalLMModule[Lfm2VlModel, Lfm2VlConfig]):
    """LFM2-VL causal language model with an LM head.

    Wraps ``Lfm2VlModel`` (text-only backbone) with a language modeling head.

    Attributes:
        config_class: Associated configuration class.

    Example:
        >>> config = Lfm2VlConfig(text_config={"vocab_size": 1000, "hidden_size": 64})
        >>> model = Lfm2VlForCausalLM(config)
    """

    config_class = Lfm2VlConfig

    def __init__(self, config: Lfm2VlConfig):
        """Initialize the LFM2-VL causal LM.

        Args:
            config: Model configuration.
        """
        super().__init__(
            config=config,
            base_model_class=Lfm2VlModel,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights via base model and parent class.

        Args:
            weights: Dictionary of parameter name to weight tensor.

        Returns:
            Sanitized weight dictionary.
        """
        weights = self.base_model.sanitize(weights)
        return super().sanitize(weights)


__all__ = ("Lfm2VlForCausalLM", "Lfm2VlModel")
