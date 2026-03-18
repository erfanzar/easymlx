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

# pyright:reportUnusedImport=none
# pyright:reportImportCycles=none

"""EasyMLX: MLX-native framework for efficient inference on Apple Silicon.

Provides lazy-loaded access to all EasyMLX modules, model implementations,
inference engines, and utilities. Mirrors the EasyDeL project structure
but optimized for Apple Silicon via MLX with eager, mutable semantics.
"""

__version__ = "0.0.1"

import sys as _sys
import typing as _tp

from .utils import LazyModule as _LazyModule
from .utils import is_package_available as _is_package_available

_import_structure = {
    "utils": [
        "LazyModule",
        "Registry",
        "StateDictConverter",
        "TensorConverter",
        "ModelConverter",
        "is_package_available",
    ],
    "inference": [
        "eSurge",
        "eSurgeApiServer",
        "SamplingParams",
    ],
    "infra": [
        "EasyMLXBaseConfig",
        "EasyMLXBaseModule",
    ],
    "infra.errors": [
        "EasyMLXRuntimeError",
        "EasyMLXSyntaxRuntimeError",
        "EasyMLXTimerError",
    ],
    "infra.factory": [
        "ConfigType",
        "TaskType",
        "register_config",
        "register_module",
    ],
    "infra.modeling_outputs": [
        "CausalLMOutput",
        "BaseModelOutput",
        "MoeCausalLMOutput",
        "SequenceClassifierOutput",
        "TokenClassifierOutput",
        "QuestionAnsweringOutput",
        "EmbeddingOutput",
        "VLMCausalLMOutput",
        "ImageClassifierOutput",
    ],
    "layers": [],
    "layers.attention": [
        "AttentionMechanisms",
        "FlexibleAttentionModule",
        "AttentionPerformer",
    ],
    "layers.moe": [],
    "operations": [
        "AttentionOutput",
        "OperationImpl",
        "OperationRegistry",
    ],
    "modules": [],
    "modules.auto": [
        "AutoEasyMLXConfig",
        "AutoEasyMLXModel",
        "AutoEasyMLXModelForCausalLM",
        "AutoEasyMLXModelForSequenceClassification",
        "AutoEasyMLXModelForTokenClassification",
        "AutoEasyMLXModelForQuestionAnswering",
        "AutoEasyMLXModelForImageTextToText",
        "AutoEasyMLXModelForEmbedding",
        "AutoEasyMLXModelForImageClassification",
        "get_modules_by_type",
    ],
    "modules.glm": [
        "GlmConfig",
        "GlmForCausalLM",
        "GlmModel",
    ],
    "modules.glm4": [
        "Glm4Config",
        "Glm4ForCausalLM",
        "Glm4Model",
    ],
    "modules.glm46v": [
        "Glm46VConfig",
        "Glm46VForConditionalGeneration",
        "Glm46VModel",
    ],
    "modules.glm4_moe": [
        "Glm4MoeConfig",
        "Glm4MoeForCausalLM",
        "Glm4MoeModel",
    ],
    "modules.glm4_moe_lite": [
        "Glm4MoeLiteConfig",
        "Glm4MoeLiteForCausalLM",
        "Glm4MoeLiteModel",
    ],
    "modules.glm4v": [
        "Glm4VConfig",
        "Glm4VForConditionalGeneration",
        "Glm4VModel",
        "Glm4VTextConfig",
        "Glm4VVisionConfig",
    ],
    "modules.glm4v_moe": [
        "Glm4VMoeConfig",
        "Glm4VMoeForConditionalGeneration",
        "Glm4VMoeModel",
        "Glm4VMoeTextConfig",
        "Glm4VMoeVisionConfig",
    ],
    "modules.gpt_oss": [
        "GptOssConfig",
        "GptOssForCausalLM",
        "GptOssModel",
    ],
    "modules.llama": [
        "LlamaConfig",
        "LlamaForCausalLM",
        "LlamaModel",
    ],
    "modules.llama4": [
        "Llama4Config",
        "Llama4ForConditionalGeneration",
        "Llama4Model",
        "Llama4VisionConfig",
        "Llama4TextConfig",
    ],
    "modules.qwen": [
        "QwenConfig",
        "QwenForCausalLM",
        "QwenModel",
    ],
    "modules.qwen2": [
        "Qwen2Config",
        "Qwen2ForCausalLM",
        "Qwen2Model",
    ],
    "modules.qwen2_moe": [
        "Qwen2MoeConfig",
        "Qwen2MoeForCausalLM",
        "Qwen2MoeModel",
    ],
    "modules.qwen2_vl": [
        "Qwen2VLConfig",
        "Qwen2VLForConditionalGeneration",
        "Qwen2VLModel",
        "Qwen2VLVisionConfig",
        "Qwen2VLTextConfig",
    ],
    "modules.qwen3": [
        "Qwen3Config",
        "Qwen3ForCausalLM",
        "Qwen3Model",
    ],
    "modules.qwen3_moe": [
        "Qwen3MoeConfig",
        "Qwen3MoeForCausalLM",
        "Qwen3MoeModel",
    ],
    "modules.qwen3_next": [
        "Qwen3NextConfig",
        "Qwen3NextForCausalLM",
        "Qwen3NextModel",
    ],
    "modules.qwen3_vl": [
        "Qwen3VLConfig",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLModel",
        "Qwen3VLVisionConfig",
        "Qwen3VLTextConfig",
    ],
    "modules.qwen3_vl_moe": [
        "Qwen3VLMoeConfig",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3VLMoeModel",
        "Qwen3VLMoeVisionConfig",
        "Qwen3VLMoeTextConfig",
    ],
    "modules.qwen3_omni_moe": [
        "Qwen3OmniMoeConfig",
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3OmniMoeModel",
    ],
    "modules.qwen3_5": [
        "Qwen3_5Config",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5Model",
        "Qwen3_5TextConfig",
        "Qwen3_5VisionConfig",
    ],
    "modules.qwen3_5_moe": [
        "Qwen3_5MoeConfig",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeModel",
        "Qwen3_5MoeTextConfig",
        "Qwen3_5MoeVisionConfig",
    ],
    "operations.kernels.gated_delta_rule": [
        "GatedDeltaRuleOp",
        "GatedDeltaRuleOutput",
    ],
}


if _tp.TYPE_CHECKING:
    from .inference import SamplingParams, eSurge, eSurgeApiServer
    from .infra import EasyMLXBaseConfig, EasyMLXBaseModule
    from .infra.errors import EasyMLXRuntimeError, EasyMLXSyntaxRuntimeError, EasyMLXTimerError
    from .infra.factory import ConfigType, TaskType, register_config, register_module
    from .infra.modeling_outputs import (
        BaseModelOutput,
        CausalLMOutput,
        EmbeddingOutput,
        ImageClassifierOutput,
        MoeCausalLMOutput,
        QuestionAnsweringOutput,
        SequenceClassifierOutput,
        TokenClassifierOutput,
        VLMCausalLMOutput,
    )
    from .layers.attention import AttentionMechanisms, AttentionPerformer, FlexibleAttentionModule
    from .modules.auto import (
        AutoEasyMLXConfig,
        AutoEasyMLXModel,
        AutoEasyMLXModelForCausalLM,
        AutoEasyMLXModelForEmbedding,
        AutoEasyMLXModelForImageClassification,
        AutoEasyMLXModelForImageTextToText,
        AutoEasyMLXModelForQuestionAnswering,
        AutoEasyMLXModelForSequenceClassification,
        AutoEasyMLXModelForTokenClassification,
        get_modules_by_type,
    )
    from .modules.glm import GlmConfig, GlmForCausalLM, GlmModel
    from .modules.glm4 import Glm4Config, Glm4ForCausalLM, Glm4Model
    from .modules.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM, Glm4MoeModel
    from .modules.glm4_moe_lite import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM, Glm4MoeLiteModel
    from .modules.glm4v import Glm4VConfig, Glm4VForConditionalGeneration, Glm4VModel, Glm4VTextConfig, Glm4VVisionConfig
    from .modules.glm4v_moe import (
        Glm4VMoeConfig,
        Glm4VMoeForConditionalGeneration,
        Glm4VMoeModel,
        Glm4VMoeTextConfig,
        Glm4VMoeVisionConfig,
    )
    from .modules.glm46v import Glm46VConfig, Glm46VForConditionalGeneration, Glm46VModel
    from .modules.gpt_oss import GptOssConfig, GptOssForCausalLM, GptOssModel
    from .modules.llama import LlamaConfig, LlamaForCausalLM, LlamaModel
    from .modules.llama4 import (
        Llama4Config,
        Llama4ForConditionalGeneration,
        Llama4Model,
        Llama4TextConfig,
        Llama4VisionConfig,
    )
    from .modules.qwen import QwenConfig, QwenForCausalLM, QwenModel
    from .modules.qwen2 import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
    from .modules.qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM, Qwen2MoeModel
    from .modules.qwen2_vl import (
        Qwen2VLConfig,
        Qwen2VLForConditionalGeneration,
        Qwen2VLModel,
        Qwen2VLTextConfig,
        Qwen2VLVisionConfig,
    )
    from .modules.qwen3 import Qwen3Config, Qwen3ForCausalLM, Qwen3Model
    from .modules.qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )
    from .modules.qwen3_5_moe import (
        Qwen3_5MoeConfig,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeModel,
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeVisionConfig,
    )
    from .modules.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3MoeModel
    from .modules.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM, Qwen3NextModel
    from .modules.qwen3_omni_moe import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeModel
    from .modules.qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLModel,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )
    from .modules.qwen3_vl_moe import (
        Qwen3VLMoeConfig,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3VLMoeModel,
        Qwen3VLMoeTextConfig,
        Qwen3VLMoeVisionConfig,
    )
    from .operations import AttentionOutput, OperationImpl, OperationRegistry
    from .operations.kernels.gated_delta_rule import GatedDeltaRuleOp, GatedDeltaRuleOutput
    from .utils import LazyModule, ModelConverter, Registry, StateDictConverter, TensorConverter, is_package_available
else:
    _sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
