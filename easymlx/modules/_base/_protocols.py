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

"""Protocol definitions for base model interfaces.

This module defines structural types (Protocols) that base models must conform to
in order to be used with the generic task modules. Protocols provide runtime type
checking and better IDE support through Python's typing system.

Protocols in this module define the expected interfaces for different model
architectures:
    - BaseModelProtocol: Standard decoder-only transformers (GPT, LLaMA, etc.)
    - EncoderDecoderProtocol: Encoder-decoder models (T5, BART, etc.)
    - VisionModelProtocol: Vision models (ViT, CLIP vision, etc.)
    - VisionLanguageProtocol: Multimodal VLM models (LLaVA, Qwen2-VL, etc.)
"""

from __future__ import annotations

import typing as tp
from typing import Any, Protocol, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from easymlx.infra.modeling_outputs import BaseModelOutput


@runtime_checkable
class BaseModelProtocol(Protocol):
    """Protocol defining the expected interface for decoder-only base models.

    Any model that implements these methods can be used with the generic task
    modules. This includes decoder-only autoregressive models like GPT, LLaMA,
    Mistral, Qwen, etc.

    Attributes:
        config (Any): Model configuration object containing hyperparameters
            like hidden_size, num_layers, vocab_size, etc.
    """

    config: Any

    def __call__(
        self,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        past_key_values: tp.Any | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the base model.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            inputs_embeds: Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_dim).
            attention_mask: Mask of shape (batch_size, sequence_length) where
                1 indicates valid positions and 0 indicates padding.
            position_ids: Position indices of shape (batch_size, sequence_length).
            past_key_values: Cached key-value states for efficient generation.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput containing last_hidden_state and optional extras.
        """
        ...

    def get_embedding(self) -> nn.Module:
        """Return the input embedding layer of the model."""
        ...

    def get_decoder(self) -> nn.Module:
        """Return the decoder (transformer layers) part of the model."""
        ...


@runtime_checkable
class EncoderDecoderProtocol(BaseModelProtocol, Protocol):
    """Protocol for encoder-decoder models (e.g., T5, BART).

    Extends BaseModelProtocol with encoder-specific methods for models that
    have separate encoder and decoder components.
    """

    def get_encoder(self) -> nn.Module:
        """Return the encoder part of the model."""
        ...


@runtime_checkable
class VisionModelProtocol(Protocol):
    """Protocol for vision models (e.g., ViT, CLIP vision encoder).

    Defines the interface for models that process image inputs and produce
    visual representations.

    Attributes:
        config (Any): Model configuration containing vision-specific parameters.
    """

    config: Any

    def __call__(
        self,
        pixel_values: mx.array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the vision model.

        Args:
            pixel_values: Input images of shape (batch_size, channels, height, width).
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput containing last_hidden_state and optional extras.
        """
        ...

    def get_embedding(self) -> nn.Module:
        """Return the patch embedding/projection layer."""
        ...


@runtime_checkable
class VisionLanguageProtocol(Protocol):
    """Protocol for vision-language models (e.g., LLaVA, Qwen2-VL).

    Defines the interface for multimodal models that process both image and
    text inputs.

    Attributes:
        config (Any): Model configuration containing both vision and language parameters.
    """

    config: Any

    def __call__(
        self,
        input_ids: mx.array | None = None,
        pixel_values: mx.array | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        past_key_values: tp.Any | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the vision-language model.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            pixel_values: Input images of shape (batch_size, channels, height, width).
            attention_mask: Mask of shape (batch_size, sequence_length).
            position_ids: Position indices for positional embeddings.
            past_key_values: Cached key-value states for generation.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.

        Returns:
            BaseModelOutput with language model hidden states after processing
            the combined vision-language sequence.
        """
        ...

    def get_vision_tower(self) -> nn.Module:
        """Return the vision encoder component."""
        ...

    def get_language_model(self) -> nn.Module:
        """Return the language model component."""
        ...
