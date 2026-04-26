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
    "infra.etils": [
        "LayerwiseQuantizationConfig",
        "QuantizationConfig",
        "QuantizationMode",
        "QuantizationRule",
        "QuantizationSpec",
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
    "modules.helium": [
        "HeliumConfig",
        "HeliumForCausalLM",
        "HeliumModel",
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
    "modules.llama4_text": [
        "Llama4TextConfig",
        "Llama4TextForCausalLM",
        "Llama4TextModel",
    ],
    "modules.ministral3": [
        "Ministral3Config",
        "Ministral3ForCausalLM",
        "Ministral3Model",
    ],
    "modules.mistral3": [
        "Mistral3Config",
        "Mistral3ForCausalLM",
        "Mistral3Model",
    ],
    "modules.pixtral": [
        "PixtralConfig",
        "PixtralForCausalLM",
        "PixtralModel",
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
    "modules.afm7": ["Afm7Config", "Afm7ForCausalLM", "Afm7Model"],
    "modules.afmoe": ["AfmoeConfig", "AfmoeForCausalLM", "AfmoeModel"],
    "modules.apertus": ["ApertusConfig", "ApertusForCausalLM", "ApertusModel"],
    "modules.baichuan_m1": ["BaichuanM1Config", "BaichuanM1ForCausalLM", "BaichuanM1Model"],
    "modules.bailing_moe": ["BailingMoeConfig", "BailingMoeForCausalLM", "BailingMoeModel"],
    "modules.bailing_moe_linear": ["BailingMoeLinearConfig", "BailingMoeLinearForCausalLM", "BailingMoeLinearModel"],
    "modules.bitnet": ["BitNetConfig", "BitNetForCausalLM", "BitNetModel"],
    "modules.cohere": ["CohereConfig", "CohereForCausalLM", "CohereModel"],
    "modules.cohere2": ["Cohere2Config", "Cohere2ForCausalLM", "Cohere2Model"],
    "modules.dbrx": ["DBRXConfig", "DBRXForCausalLM", "DBRXModel"],
    "modules.deepseek": ["DeepseekConfig", "DeepseekForCausalLM", "DeepseekModel"],
    "modules.deepseek_v2": ["DeepseekV2Config", "DeepseekV2ForCausalLM", "DeepseekV2Model"],
    "modules.deepseek_v3": ["DeepseekV3Config", "DeepseekV3ForCausalLM", "DeepseekV3Model"],
    "modules.deepseek_v32": ["DeepseekV32Config", "DeepseekV32ForCausalLM", "DeepseekV32Model"],
    "modules.dots1": ["Dots1Config", "Dots1ForCausalLM", "Dots1Model"],
    "modules.ernie4_5": ["Ernie45Config", "Ernie45ForCausalLM", "Ernie45Model"],
    "modules.ernie4_5_moe": ["Ernie45MoeConfig", "Ernie45MoeForCausalLM", "Ernie45MoeModel"],
    "modules.exaone": ["ExaoneConfig", "ExaoneForCausalLM", "ExaoneModel"],
    "modules.exaone4": ["Exaone4Config", "Exaone4ForCausalLM", "Exaone4Model"],
    "modules.exaone_moe": ["ExaoneMoeConfig", "ExaoneMoeForCausalLM", "ExaoneMoeModel"],
    "modules.flux2_klein": [
        "AutoencoderKLFlux2",
        "AutoencoderKLFlux2Config",
        "FlowMatchEulerDiscreteScheduler",
        "FlowMatchEulerDiscreteSchedulerConfig",
        "Flux2KleinConfig",
        "Flux2KleinPipeline",
        "Flux2KleinPipelineOutput",
        "Flux2Transformer2DModel",
        "Flux2TransformerConfig",
    ],
    "modules.falcon_h1": ["FalconH1Config", "FalconH1ForCausalLM", "FalconH1Model"],
    "modules.gemma": ["GemmaConfig", "GemmaForCausalLM", "GemmaModel"],
    "modules.gemma2": ["Gemma2Config", "Gemma2ForCausalLM", "Gemma2Model"],
    "modules.gemma3": ["Gemma3Config", "Gemma3ForCausalLM", "Gemma3Model"],
    "modules.gemma3_text": ["Gemma3TextConfig", "Gemma3TextForCausalLM", "Gemma3TextModel"],
    "modules.gemma3n": ["Gemma3NConfig", "Gemma3NForCausalLM", "Gemma3NModel"],
    "modules.gpt2": ["GPT2Config", "GPT2ForCausalLM", "GPT2Model"],
    "modules.gpt_bigcode": ["GPTBigCodeConfig", "GPTBigCodeForCausalLM", "GPTBigCodeModel"],
    "modules.gpt_neox": ["GPTNeoXConfig", "GPTNeoXForCausalLM", "GPTNeoXModel"],
    "modules.granite": ["GraniteConfig", "GraniteForCausalLM", "GraniteModel"],
    "modules.granitemoe": ["GraniteMoeConfig", "GraniteMoeForCausalLM", "GraniteMoeModel"],
    "modules.granitemoehybrid": ["GraniteMoeHybridConfig", "GraniteMoeHybridForCausalLM", "GraniteMoeHybridModel"],
    "modules.hunyuan": ["HunyuanConfig", "HunyuanForCausalLM", "HunyuanModel"],
    "modules.hunyuan_v1_dense": ["HunyuanV1DenseConfig", "HunyuanV1DenseForCausalLM", "HunyuanV1DenseModel"],
    "modules.internlm2": ["InternLM2Config", "InternLM2ForCausalLM", "InternLM2Model"],
    "modules.internlm3": ["InternLM3Config", "InternLM3ForCausalLM", "InternLM3Model"],
    "modules.iquestloopcoder": ["IQuestLoopCoderConfig", "IQuestLoopCoderForCausalLM", "IQuestLoopCoderModel"],
    "modules.jamba": ["JambaConfig", "JambaForCausalLM", "JambaModel"],
    "modules.kimi_k25": ["KimiK25Config", "KimiK25ForCausalLM", "KimiK25Model"],
    "modules.kimi_linear": ["KimiLinearConfig", "KimiLinearForCausalLM", "KimiLinearModel"],
    "modules.kimi_vl": ["KimiVLConfig", "KimiVLForCausalLM", "KimiVLModel"],
    "modules.klear": ["KlearConfig", "KlearForCausalLM", "KlearModel"],
    "modules.lfm2": ["Lfm2Config", "Lfm2ForCausalLM", "Lfm2Model"],
    "modules.lfm2_moe": ["Lfm2MoeConfig", "Lfm2MoeForCausalLM", "Lfm2MoeModel"],
    "modules.lfm2_vl": ["Lfm2VlConfig", "Lfm2VlForCausalLM", "Lfm2VlModel"],
    "modules.lille_130m": ["Lille130mConfig", "Lille130mForCausalLM", "Lille130mModel"],
    "modules.longcat_flash": ["LongcatFlashConfig", "LongcatFlashForCausalLM", "LongcatFlashModel"],
    "modules.longcat_flash_ngram": ["LongcatFlashNgramConfig", "LongcatFlashNgramForCausalLM", "LongcatFlashNgramModel"],
    "modules.mamba": ["MambaConfig", "MambaForCausalLM", "MambaModel"],
    "modules.mamba2": ["Mamba2Config", "Mamba2ForCausalLM", "Mamba2Model"],
    "modules.minicpm": ["MiniCPMConfig", "MiniCPMForCausalLM", "MiniCPMModel"],
    "modules.minicpm3": ["MiniCPM3Config", "MiniCPM3ForCausalLM", "MiniCPM3Model"],
    "modules.mimo": ["MiMoConfig", "MiMoForCausalLM", "MiMoModel"],
    "modules.mimo_v2_flash": ["MiMoV2FlashConfig", "MiMoV2FlashForCausalLM", "MiMoV2FlashModel"],
    "modules.minimax": ["MiniMaxConfig", "MiniMaxForCausalLM", "MiniMaxModel"],
    "modules.mixtral": ["MixtralConfig", "MixtralForCausalLM", "MixtralModel"],
    "modules.nanochat": ["NanochatConfig", "NanochatForCausalLM", "NanochatModel"],
    "modules.nemotron": ["NemotronConfig", "NemotronForCausalLM", "NemotronModel"],
    "modules.nemotron_h": ["NemotronHConfig", "NemotronHForCausalLM", "NemotronHModel"],
    "modules.nemotron_nas": ["NemotronNASConfig", "NemotronNASForCausalLM", "NemotronNASModel"],
    "modules.olmo": ["OlmoConfig", "OlmoForCausalLM", "OlmoModel"],
    "modules.olmo2": ["Olmo2Config", "Olmo2ForCausalLM", "Olmo2Model"],
    "modules.olmo3": ["OLMo3Config", "OLMo3ForCausalLM", "OLMo3Model"],
    "modules.olmoe": ["OlmoeConfig", "OlmoeForCausalLM", "OlmoeModel"],
    "modules.openelm": ["OpenELMConfig", "OpenELMForCausalLM", "OpenELMModel"],
    "modules.phi": ["PhiConfig", "PhiForCausalLM", "PhiModel"],
    "modules.phi3": ["Phi3Config", "Phi3ForCausalLM", "Phi3Model"],
    "modules.phi3small": ["Phi3SmallConfig", "Phi3SmallForCausalLM", "Phi3SmallModel"],
    "modules.phimoe": ["PhimoeConfig", "PhimoeForCausalLM", "PhimoeModel"],
    "modules.phixtral": ["PhixtralConfig", "PhixtralForCausalLM", "PhixtralModel"],
    "modules.plamo": ["PlamoConfig", "PlamoForCausalLM", "PlamoModel"],
    "modules.plamo2": ["Plamo2Config", "Plamo2ForCausalLM", "Plamo2Model"],
    "modules.recurrent_gemma": ["RecurrentGemmaConfig", "RecurrentGemmaForCausalLM", "RecurrentGemmaModel"],
    "modules.rwkv7": ["Rwkv7Config", "Rwkv7ForCausalLM", "Rwkv7Model"],
    "modules.seed_oss": ["SeedOssConfig", "SeedOssForCausalLM", "SeedOssModel"],
    "modules.solar_open": [
        "SolarOpenConfig",
        "SolarOpenForCausalLM",
        "SolarOpenModel",
    ],
    "modules.smollm3": [
        "SmolLM3Config",
        "SmolLM3ForCausalLM",
        "SmolLM3Model",
        "SmolLM3NoPE",
        "Smollm3Config",
        "Smollm3ForCausalLM",
        "Smollm3Model",
        "Smollm3NoPE",
    ],
    "modules.stablelm": ["StableLMConfig", "StableLMForCausalLM", "StableLMModel"],
    "modules.starcoder2": ["Starcoder2Config", "Starcoder2ForCausalLM", "Starcoder2Model"],
    "modules.step3p5": ["Step3p5Config", "Step3p5ForCausalLM", "Step3p5Model"],
    "modules.telechat3": [
        "Telechat3Config",
        "Telechat3ForCausalLM",
        "Telechat3Model",
    ],
    "modules.youtu_llm": ["YouTuLLMConfig", "YouTuLLMForCausalLM", "YouTuLLMModel"],
    "operations.kernels.gated_delta_rule": [
        "GatedDeltaRuleOp",
        "GatedDeltaRuleOutput",
    ],
}


if _tp.TYPE_CHECKING:
    from .inference import SamplingParams, eSurge, eSurgeApiServer
    from .infra import EasyMLXBaseConfig, EasyMLXBaseModule
    from .infra.errors import EasyMLXRuntimeError, EasyMLXSyntaxRuntimeError, EasyMLXTimerError
    from .infra.etils import (
        LayerwiseQuantizationConfig,
        QuantizationConfig,
        QuantizationMode,
        QuantizationRule,
        QuantizationSpec,
    )
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
    from .modules.afm7 import Afm7Config, Afm7ForCausalLM, Afm7Model
    from .modules.afmoe import AfmoeConfig, AfmoeForCausalLM, AfmoeModel
    from .modules.apertus import ApertusConfig, ApertusForCausalLM, ApertusModel
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
    from .modules.baichuan_m1 import BaichuanM1Config, BaichuanM1ForCausalLM, BaichuanM1Model
    from .modules.bailing_moe import BailingMoeConfig, BailingMoeForCausalLM, BailingMoeModel
    from .modules.bailing_moe_linear import BailingMoeLinearConfig, BailingMoeLinearForCausalLM, BailingMoeLinearModel
    from .modules.bitnet import BitNetConfig, BitNetForCausalLM, BitNetModel
    from .modules.cohere import CohereConfig, CohereForCausalLM, CohereModel
    from .modules.cohere2 import Cohere2Config, Cohere2ForCausalLM, Cohere2Model
    from .modules.dbrx import DBRXConfig, DBRXForCausalLM, DBRXModel
    from .modules.deepseek import DeepseekConfig, DeepseekForCausalLM, DeepseekModel
    from .modules.deepseek_v2 import DeepseekV2Config, DeepseekV2ForCausalLM, DeepseekV2Model
    from .modules.deepseek_v3 import DeepseekV3Config, DeepseekV3ForCausalLM, DeepseekV3Model
    from .modules.deepseek_v32 import DeepseekV32Config, DeepseekV32ForCausalLM, DeepseekV32Model
    from .modules.dots1 import Dots1Config, Dots1ForCausalLM, Dots1Model
    from .modules.ernie4_5 import Ernie45Config, Ernie45ForCausalLM, Ernie45Model
    from .modules.ernie4_5_moe import Ernie45MoeConfig, Ernie45MoeForCausalLM, Ernie45MoeModel
    from .modules.exaone import ExaoneConfig, ExaoneForCausalLM, ExaoneModel
    from .modules.exaone4 import Exaone4Config, Exaone4ForCausalLM, Exaone4Model
    from .modules.exaone_moe import ExaoneMoeConfig, ExaoneMoeForCausalLM, ExaoneMoeModel
    from .modules.falcon_h1 import FalconH1Config, FalconH1ForCausalLM, FalconH1Model
    from .modules.flux2_klein import (
        AutoencoderKLFlux2,
        AutoencoderKLFlux2Config,
        FlowMatchEulerDiscreteScheduler,
        FlowMatchEulerDiscreteSchedulerConfig,
        Flux2KleinConfig,
        Flux2KleinPipeline,
        Flux2KleinPipelineOutput,
        Flux2Transformer2DModel,
        Flux2TransformerConfig,
    )
    from .modules.gemma import GemmaConfig, GemmaForCausalLM, GemmaModel
    from .modules.gemma2 import Gemma2Config, Gemma2ForCausalLM, Gemma2Model
    from .modules.gemma3 import Gemma3Config, Gemma3ForCausalLM, Gemma3Model
    from .modules.gemma3_text import Gemma3TextConfig, Gemma3TextForCausalLM, Gemma3TextModel
    from .modules.gemma3n import Gemma3NConfig, Gemma3NForCausalLM, Gemma3NModel
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
    from .modules.gpt2 import GPT2Config, GPT2ForCausalLM, GPT2Model
    from .modules.gpt_bigcode import GPTBigCodeConfig, GPTBigCodeForCausalLM, GPTBigCodeModel
    from .modules.gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM, GPTNeoXModel
    from .modules.gpt_oss import GptOssConfig, GptOssForCausalLM, GptOssModel
    from .modules.granite import GraniteConfig, GraniteForCausalLM, GraniteModel
    from .modules.granitemoe import GraniteMoeConfig, GraniteMoeForCausalLM, GraniteMoeModel
    from .modules.granitemoehybrid import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM, GraniteMoeHybridModel
    from .modules.helium import HeliumConfig, HeliumForCausalLM, HeliumModel
    from .modules.hunyuan import HunyuanConfig, HunyuanForCausalLM, HunyuanModel
    from .modules.hunyuan_v1_dense import HunyuanV1DenseConfig, HunyuanV1DenseForCausalLM, HunyuanV1DenseModel
    from .modules.internlm2 import InternLM2Config, InternLM2ForCausalLM, InternLM2Model
    from .modules.internlm3 import InternLM3Config, InternLM3ForCausalLM, InternLM3Model
    from .modules.iquestloopcoder import IQuestLoopCoderConfig, IQuestLoopCoderForCausalLM, IQuestLoopCoderModel
    from .modules.jamba import JambaConfig, JambaForCausalLM, JambaModel
    from .modules.kimi_k25 import KimiK25Config, KimiK25ForCausalLM, KimiK25Model
    from .modules.kimi_linear import KimiLinearConfig, KimiLinearForCausalLM, KimiLinearModel
    from .modules.kimi_vl import KimiVLConfig, KimiVLForCausalLM, KimiVLModel
    from .modules.klear import KlearConfig, KlearForCausalLM, KlearModel
    from .modules.lfm2 import Lfm2Config, Lfm2ForCausalLM, Lfm2Model
    from .modules.lfm2_moe import Lfm2MoeConfig, Lfm2MoeForCausalLM, Lfm2MoeModel
    from .modules.lfm2_vl import Lfm2VlConfig, Lfm2VlForCausalLM, Lfm2VlModel
    from .modules.lille_130m import Lille130mConfig, Lille130mForCausalLM, Lille130mModel
    from .modules.llama import LlamaConfig, LlamaForCausalLM, LlamaModel
    from .modules.llama4 import (
        Llama4Config,
        Llama4ForConditionalGeneration,
        Llama4Model,
        Llama4TextConfig,
        Llama4VisionConfig,
    )
    from .modules.llama4_text import Llama4TextForCausalLM, Llama4TextModel
    from .modules.longcat_flash import LongcatFlashConfig, LongcatFlashForCausalLM, LongcatFlashModel
    from .modules.longcat_flash_ngram import (
        LongcatFlashNgramConfig,
        LongcatFlashNgramForCausalLM,
        LongcatFlashNgramModel,
    )
    from .modules.mamba import MambaConfig, MambaForCausalLM, MambaModel
    from .modules.mamba2 import Mamba2Config, Mamba2ForCausalLM, Mamba2Model
    from .modules.mimo import MiMoConfig, MiMoForCausalLM, MiMoModel
    from .modules.mimo_v2_flash import MiMoV2FlashConfig, MiMoV2FlashForCausalLM, MiMoV2FlashModel
    from .modules.minicpm import MiniCPMConfig, MiniCPMForCausalLM, MiniCPMModel
    from .modules.minicpm3 import MiniCPM3Config, MiniCPM3ForCausalLM, MiniCPM3Model
    from .modules.minimax import MiniMaxConfig, MiniMaxForCausalLM, MiniMaxModel
    from .modules.ministral3 import Ministral3Config, Ministral3ForCausalLM, Ministral3Model
    from .modules.mistral3 import Mistral3Config, Mistral3ForCausalLM, Mistral3Model
    from .modules.mixtral import MixtralConfig, MixtralForCausalLM, MixtralModel
    from .modules.nanochat import NanochatConfig, NanochatForCausalLM, NanochatModel
    from .modules.nemotron import NemotronConfig, NemotronForCausalLM, NemotronModel
    from .modules.nemotron_h import NemotronHConfig, NemotronHForCausalLM, NemotronHModel
    from .modules.nemotron_nas import NemotronNASConfig, NemotronNASForCausalLM, NemotronNASModel
    from .modules.olmo import OlmoConfig, OlmoForCausalLM, OlmoModel
    from .modules.olmo2 import Olmo2Config, Olmo2ForCausalLM, Olmo2Model
    from .modules.olmo3 import OLMo3Config, OLMo3ForCausalLM, OLMo3Model
    from .modules.olmoe import OlmoeConfig, OlmoeForCausalLM, OlmoeModel
    from .modules.openelm import OpenELMConfig, OpenELMForCausalLM, OpenELMModel
    from .modules.phi import PhiConfig, PhiForCausalLM, PhiModel
    from .modules.phi3 import Phi3Config, Phi3ForCausalLM, Phi3Model
    from .modules.phi3small import Phi3SmallConfig, Phi3SmallForCausalLM, Phi3SmallModel
    from .modules.phimoe import PhimoeConfig, PhimoeForCausalLM, PhimoeModel
    from .modules.phixtral import PhixtralConfig, PhixtralForCausalLM, PhixtralModel
    from .modules.pixtral import PixtralConfig, PixtralForCausalLM, PixtralModel
    from .modules.plamo import PlamoConfig, PlamoForCausalLM, PlamoModel
    from .modules.plamo2 import Plamo2Config, Plamo2ForCausalLM, Plamo2Model
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
    from .modules.recurrent_gemma import RecurrentGemmaConfig, RecurrentGemmaForCausalLM, RecurrentGemmaModel
    from .modules.rwkv7 import Rwkv7Config, Rwkv7ForCausalLM, Rwkv7Model
    from .modules.seed_oss import SeedOssConfig, SeedOssForCausalLM, SeedOssModel
    from .modules.smollm3 import (
        SmolLM3Config,
        Smollm3Config,
        SmolLM3ForCausalLM,
        Smollm3ForCausalLM,
        SmolLM3Model,
        Smollm3Model,
        SmolLM3NoPE,
        Smollm3NoPE,
    )
    from .modules.solar_open import SolarOpenConfig, SolarOpenForCausalLM, SolarOpenModel
    from .modules.stablelm import StableLMConfig, StableLMForCausalLM, StableLMModel
    from .modules.starcoder2 import Starcoder2Config, Starcoder2ForCausalLM, Starcoder2Model
    from .modules.step3p5 import Step3p5Config, Step3p5ForCausalLM, Step3p5Model
    from .modules.telechat3 import Telechat3Config, Telechat3ForCausalLM, Telechat3Model
    from .modules.youtu_llm import YouTuLLMConfig, YouTuLLMForCausalLM, YouTuLLMModel
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
