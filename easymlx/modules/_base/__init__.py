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

"""Base module exports mirroring EasyDeL's module layout."""

from __future__ import annotations

from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.infra.base_module import EasyMLXBaseModule

from ._auto_mapper import (
    AUTO_MODEL_FACTORY_REGISTRY,
    create_causal_lm_class,
    create_conditional_generation_class,
    create_embedding_class,
    create_image_classification_class,
    create_question_answering_class,
    create_sequence_classification_class,
    create_task_model_class,
    create_token_classification_class,
)
from ._base_task_module import BaseTaskModule
from ._features import (
    GradientCheckpointingFeature,
    LogitCapFeature,
    RouterAuxLossFeature,
    SequenceLengthPoolingFeature,
    TieEmbeddingsFeature,
)
from ._protocols import (
    BaseModelProtocol,
    EncoderDecoderProtocol,
    VisionLanguageProtocol,
    VisionModelProtocol,
)
from .causal_lm_module import BaseCausalLMModule
from .conditional_generation_module import BaseConditionalGenerationModule
from .embedding_module import BaseEmbeddingModule
from .image_classification_module import BaseImageClassificationModule
from .question_answering_module import BaseQuestionAnsweringModule
from .sequence_classification_module import BaseSequenceClassificationModule
from .token_classification_module import BaseTokenClassificationModule
from .vision_language_module import BaseVisionLanguageModule

__all__ = (
    "AUTO_MODEL_FACTORY_REGISTRY",
    "BaseCausalLMModule",
    "BaseConditionalGenerationModule",
    "BaseEmbeddingModule",
    "BaseImageClassificationModule",
    "BaseModelProtocol",
    "BaseQuestionAnsweringModule",
    "BaseSequenceClassificationModule",
    "BaseTaskModule",
    "BaseTokenClassificationModule",
    "BaseVisionLanguageModule",
    "EasyMLXBaseConfig",
    "EasyMLXBaseModule",
    "EncoderDecoderProtocol",
    "GradientCheckpointingFeature",
    "LogitCapFeature",
    "RouterAuxLossFeature",
    "SequenceLengthPoolingFeature",
    "TieEmbeddingsFeature",
    "VisionLanguageProtocol",
    "VisionModelProtocol",
    "create_causal_lm_class",
    "create_conditional_generation_class",
    "create_embedding_class",
    "create_image_classification_class",
    "create_question_answering_class",
    "create_sequence_classification_class",
    "create_task_model_class",
    "create_token_classification_class",
)
