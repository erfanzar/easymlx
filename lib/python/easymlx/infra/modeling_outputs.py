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

"""Model output dataclasses for easymlx.

These mirror the lightweight, dictionary-like output structures commonly used
in HuggingFace and EasyDeL, but are MLX-typed. Every output class inherits
from :class:`transformers.utils.generic.ModelOutput` so that it can be used
interchangeably with HuggingFace utilities.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import mlx.core as mx
from transformers.utils.generic import ModelOutput


@dataclass
class CausalLMOutput(ModelOutput):
    """Output for causal language models.

    Attributes:
        logits: Prediction scores of the language modelling head with
            shape ``[batch, seq_len, vocab_size]``.
        cache: Optional model-specific cache object for incremental
            decoding. The concrete type depends on the model and cache
            backend in use.
        hidden_states: Optional tuple of hidden-state feature tensors.
    """

    logits: mx.array
    cache: tp.Any | None = None
    hidden_states: tuple[mx.array, ...] | None = None


@dataclass
class GenerateOutput(ModelOutput):
    """Common base output for ``generate()``-style APIs.

    Attributes:
        sequences: Generated token IDs with shape
            ``[batch, prompt_len + generated_len]``.
        scores: Optional tuple of per-step logit arrays, each of shape
            ``[batch, vocab_size]``. Present only when
            ``output_scores=True``.
    """

    sequences: mx.array
    scores: tuple[mx.array, ...] | None = None


@dataclass
class GreedySearchOutput(GenerateOutput):
    """Output returned by ``generate()`` when ``do_sample=False``.

    Inherits all attributes from :class:`GenerateOutput`.
    """


@dataclass
class SampleOutput(GenerateOutput):
    """Output returned by ``generate()`` when ``do_sample=True``.

    Inherits all attributes from :class:`GenerateOutput`.
    """


@dataclass
class BaseModelOutput(ModelOutput):
    """Output for base (backbone) models without a task-specific head.

    Attributes:
        last_hidden_state: Final layer hidden states with shape
            ``[batch, seq_len, hidden_size]``.
        hidden_states: Optional tuple of per-layer hidden states,
            including the embedding output.
        attentions: Optional tuple of per-layer attention weight arrays.
        past_key_values: Optional cache object for incremental decoding.
    """

    last_hidden_state: mx.array
    hidden_states: tuple[mx.array, ...] | None = None
    attentions: tuple[mx.array, ...] | None = None
    past_key_values: tp.Any | None = None


@dataclass
class MoeCausalLMOutput(CausalLMOutput):
    """Output for Mixture-of-Experts causal language models.

    Extends :class:`CausalLMOutput` with auxiliary router loss
    information used for load-balancing during training.

    Attributes:
        aux_loss: Aggregated auxiliary loss for router load balancing.
        router_logits: Per-layer router logits for expert selection.
        all_router_losses: Per-layer router auxiliary losses.
    """

    aux_loss: mx.array | None = None
    router_logits: tuple[mx.array, ...] | None = None
    all_router_losses: list[mx.array] | tuple[mx.array, ...] | None = None


@dataclass
class SequenceClassifierOutput(ModelOutput):
    """Output for sequence classification tasks.

    Attributes:
        logits: Classification logits with shape
            ``[batch, num_labels]``.
        hidden_states: Optional tuple of per-layer hidden states.
        attentions: Optional tuple of per-layer attention weight arrays.
        aux_loss: Optional MoE auxiliary loss (when using an MoE
            backbone).
    """

    logits: mx.array
    hidden_states: tuple[mx.array, ...] | None = None
    attentions: tuple[mx.array, ...] | None = None
    aux_loss: mx.array | None = None


@dataclass
class TokenClassifierOutput(ModelOutput):
    """Output for token classification tasks (e.g. NER).

    Attributes:
        logits: Per-token classification logits with shape
            ``[batch, seq_len, num_labels]``.
        hidden_states: Optional tuple of per-layer hidden states.
        attentions: Optional tuple of per-layer attention weight arrays.
    """

    logits: mx.array
    hidden_states: tuple[mx.array, ...] | None = None
    attentions: tuple[mx.array, ...] | None = None


@dataclass
class QuestionAnsweringOutput(ModelOutput):
    """Output for extractive question answering tasks.

    Attributes:
        start_logits: Start position logits with shape
            ``[batch, seq_len]``.
        end_logits: End position logits with shape
            ``[batch, seq_len]``.
        hidden_states: Optional tuple of per-layer hidden states.
        attentions: Optional tuple of per-layer attention weight arrays.
    """

    start_logits: mx.array
    end_logits: mx.array
    hidden_states: tuple[mx.array, ...] | None = None
    attentions: tuple[mx.array, ...] | None = None


@dataclass
class EmbeddingOutput(ModelOutput):
    """Output for embedding / retrieval models.

    Attributes:
        embeddings: Sentence-level embeddings with shape
            ``[batch, hidden_size]``.
        hidden_states: Optional tuple of per-layer hidden states.
        attentions: Optional tuple of per-layer attention weight arrays.
    """

    embeddings: mx.array
    hidden_states: tuple[mx.array, ...] | None = None
    attentions: tuple[mx.array, ...] | None = None


@dataclass
class VLMCausalLMOutput(CausalLMOutput):
    """Output for vision-language causal language models.

    Extends :class:`CausalLMOutput` with image hidden states produced
    by the vision encoder.

    Attributes:
        image_hidden_states: Hidden states from the vision encoder,
            or ``None`` if no images were provided.
    """

    image_hidden_states: mx.array | None = None


@dataclass
class ImageClassifierOutput(ModelOutput):
    """Output for image classification tasks.

    Attributes:
        logits: Classification logits with shape
            ``[batch, num_labels]``.
        hidden_states: Optional tuple of per-layer hidden states.
        attentions: Optional tuple of per-layer attention weight arrays.
    """

    logits: mx.array
    hidden_states: tuple[mx.array, ...] | None = None
    attentions: tuple[mx.array, ...] | None = None
