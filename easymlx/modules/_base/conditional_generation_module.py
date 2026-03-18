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

"""Generic base class for Conditional Generation (encoder-decoder) tasks on MLX.

This module provides ``BaseConditionalGenerationModule``, the foundation for
encoder-decoder and sequence-to-sequence models such as T5, BART, and
vision-language models. It wraps a base model with an LM head and supports
weight tying, logit capping, and flexible input forwarding.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.infra.base_module import EasyMLXBaseModule
from easymlx.infra.modeling_outputs import CausalLMOutput

ModelT = tp.TypeVar("ModelT", bound=nn.Module)
ConfigT = tp.TypeVar("ConfigT", bound=EasyMLXBaseConfig)


class BaseConditionalGenerationModule[ModelT: nn.Module, ConfigT: EasyMLXBaseConfig](EasyMLXBaseModule):
    """Base class for encoder-decoder / conditional generation models on MLX.

    Wraps a base model and adds an LM head that projects decoder hidden states
    to vocabulary logits. Supports weight tying and logit capping.
    """

    def __init__(
        self,
        config: ConfigT,
        base_model: ModelT | None = None,
        base_model_class: type[ModelT] | None = None,
        base_model_name: str = "model",
        *,
        tie_word_embeddings: bool = False,
        logit_cap: float | None = None,
        lm_head_name: str = "lm_head",
        lm_head_bias: bool = False,
        create_lm_head: bool = True,
    ):
        """Initialize the conditional generation module.

        Args:
            config: Model configuration containing hyperparameters.
            base_model: Pre-instantiated base model instance. If provided,
                ``base_model_class`` is ignored.
            base_model_class: Base model class to instantiate. Required if
                ``base_model`` is not provided.
            base_model_name: Attribute name under which to store the base
                model. Defaults to ``"model"``.
            tie_word_embeddings: Whether to share weights between the input
                embedding layer and the LM head. Defaults to False.
            logit_cap: Maximum absolute value for output logits via tanh
                scaling. If None, no capping is applied. Defaults to None.
            lm_head_name: Attribute name for the LM head linear layer.
                Defaults to ``"lm_head"``.
            lm_head_bias: Whether to include bias in the LM head.
                Defaults to False.
            create_lm_head: Whether to create an LM head layer. Set to False
                when the base model already contains its own LM head.
                Defaults to True.

        Raises:
            ValueError: If neither ``base_model`` nor ``base_model_class``
                is provided.
        """
        super().__init__(config=config)
        self._base_model_name = base_model_name
        self._lm_head_name = lm_head_name
        self._tie_word_embeddings = tie_word_embeddings
        self._logit_cap = logit_cap

        if base_model is not None:
            setattr(self, base_model_name, base_model)
        elif base_model_class is not None:
            setattr(self, base_model_name, base_model_class(config))
        else:
            raise ValueError("Either base_model or base_model_class must be provided.")

        if create_lm_head and not tie_word_embeddings:
            text_config = getattr(config, "text_config", config)
            hidden_size = getattr(text_config, "hidden_size", config.hidden_size)
            vocab_size = getattr(text_config, "vocab_size", config.vocab_size)
            lm_head = nn.Linear(hidden_size, vocab_size, bias=lm_head_bias)
            setattr(self, lm_head_name, lm_head)

    @property
    def base_model(self) -> ModelT:
        """Access the base model via the configured attribute name.

        Returns:
            The underlying base model instance.
        """
        return getattr(self, self._base_model_name)

    def get_embedding(self) -> nn.Embedding:
        """Return the input embedding layer from the base model.

        Searches for the embedding layer using common attribute names
        and method conventions across different model architectures.

        Returns:
            The input embedding module.

        Raises:
            AttributeError: If no embedding layer can be found.
        """
        model = self.base_model
        for attr in ("get_embedding", "get_input_embeddings"):
            fn = getattr(model, attr, None)
            if callable(fn):
                return fn()
        for attr in ("embed_tokens", "wte", "word_embeddings", "embeddings", "shared"):
            if hasattr(model, attr):
                return getattr(model, attr)
        raise AttributeError("Cannot find embedding layer on base model.")

    def apply_lm_head(self, hidden_states: mx.array) -> mx.array:
        """Project hidden states to vocabulary logits.

        Uses the LM head linear layer, or the embedding layer's ``as_linear``
        method if weight tying is enabled. Applies logit capping if configured.

        Args:
            hidden_states: Hidden state tensor of shape
                ``(..., hidden_size)``.

        Returns:
            Logits tensor of shape ``(..., vocab_size)``.
        """
        if self._tie_word_embeddings:
            embedding = self.get_embedding()
            logits = embedding.as_linear(hidden_states)
        else:
            lm_head = self.get_task_head()
            logits = lm_head(hidden_states)
        if self._logit_cap is not None:
            cap = self._logit_cap
            logits = cap * mx.tanh(logits / cap)
        return logits

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        decoder_input_ids: mx.array | None = None,
        decoder_attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        cache: tp.Any | None = None,
        inputs_embeds: mx.array | None = None,
        apply_lm_head: bool = True,
        **kwargs,
    ) -> CausalLMOutput:
        """Forward pass through the conditional generation model.

        Runs the base model with the provided inputs and optionally applies
        the LM head to produce vocabulary logits.

        Args:
            input_ids: Encoder input token IDs of shape
                ``(batch_size, seq_len)``.
            attention_mask: Encoder attention mask of shape
                ``(batch_size, seq_len)``.
            position_ids: Position indices for positional embeddings.
            decoder_input_ids: Decoder input token IDs.
            decoder_attention_mask: Decoder attention mask.
            pixel_values: Input images for vision-language models.
            cache: Cached key-value states for incremental decoding.
            inputs_embeds: Pre-computed input embeddings, used instead of
                ``input_ids`` if provided.
            apply_lm_head: Whether to project hidden states to logits.
                Defaults to True.
            **kwargs: Additional keyword arguments forwarded to the base model.

        Returns:
            ``CausalLMOutput`` containing logits and optional cache.
        """
        base_kwargs: dict[str, tp.Any] = {}
        if inputs_embeds is not None:
            base_kwargs["inputs_embeds"] = inputs_embeds
        elif input_ids is not None:
            base_kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            base_kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            base_kwargs["position_ids"] = position_ids
        if decoder_input_ids is not None:
            base_kwargs["decoder_input_ids"] = decoder_input_ids
        if decoder_attention_mask is not None:
            base_kwargs["decoder_attention_mask"] = decoder_attention_mask
        if pixel_values is not None:
            base_kwargs["pixel_values"] = pixel_values
        if cache is not None:
            base_kwargs["cache"] = cache
        base_kwargs.update(kwargs)

        outputs = self.base_model(**base_kwargs)

        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            out_cache = outputs[1] if len(outputs) > 1 else None
        else:
            hidden_states = getattr(outputs, "last_hidden_state", outputs)
            out_cache = getattr(outputs, "cache", getattr(outputs, "past_key_values", None))

        logits = None
        if apply_lm_head:
            logits = self.apply_lm_head(hidden_states)

        return CausalLMOutput(logits=logits, cache=out_cache)

    def get_task_head(self) -> nn.Module:
        """Return the LM head module used for projecting to vocabulary logits.

        Returns:
            The LM head ``nn.Module``.

        Raises:
            AttributeError: If no LM head can be found on this module or
                the base model.
        """
        if hasattr(self, self._lm_head_name):
            return getattr(self, self._lm_head_name)
        if hasattr(self.base_model, "get_lm_head"):
            return self.base_model.get_lm_head()
        raise AttributeError(f"{self.__class__.__name__} has no LM head named '{self._lm_head_name}'.")

    def get_lm_head(self) -> nn.Module:
        """Return the LM head module. Alias for ``get_task_head``.

        Returns:
            The LM head ``nn.Module``.
        """
        return self.get_task_head()

    def get_encoder(self) -> nn.Module:
        """Return the encoder part of the model.

        Returns:
            The encoder ``nn.Module``.

        Raises:
            NotImplementedError: If the base model does not have an encoder.
        """
        if hasattr(self.base_model, "get_encoder"):
            return self.base_model.get_encoder()
        raise NotImplementedError("This model does not have an encoder.")
