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

"""Regression tests for newly added model-family public API wiring."""

from __future__ import annotations

import mlx.core as mx
import pytest

import easymlx
from easymlx.modules.auto import AutoEasyMLXModel, AutoEasyMLXModelForCausalLM
from easymlx.modules.cohere import CohereConfig, CohereForCausalLM, CohereModel
from easymlx.modules.gemma import GemmaConfig, GemmaForCausalLM, GemmaModel
from easymlx.modules.gemma2 import Gemma2Config, Gemma2ForCausalLM, Gemma2Model
from easymlx.modules.gpt2 import GPT2Config, GPT2ForCausalLM, GPT2Model
from easymlx.modules.granite import GraniteConfig, GraniteForCausalLM, GraniteModel
from easymlx.modules.helium import HeliumConfig, HeliumForCausalLM, HeliumModel
from easymlx.modules.internlm2 import InternLM2Config, InternLM2ForCausalLM, InternLM2Model
from easymlx.modules.llama4_text import Llama4TextConfig, Llama4TextForCausalLM, Llama4TextModel
from easymlx.modules.ministral3 import Ministral3Config, Ministral3ForCausalLM, Ministral3Model
from easymlx.modules.mistral3 import Mistral3Config, Mistral3ForCausalLM, Mistral3Model
from easymlx.modules.mixtral import MixtralConfig, MixtralForCausalLM, MixtralModel
from easymlx.modules.phi import PhiConfig, PhiForCausalLM, PhiModel
from easymlx.modules.phi3 import Phi3Config, Phi3ForCausalLM, Phi3Model
from easymlx.modules.pixtral import PixtralConfig, PixtralForCausalLM, PixtralModel
from easymlx.modules.smollm3 import SmolLM3Config, SmolLM3ForCausalLM, SmolLM3Model
from easymlx.modules.solar_open import SolarOpenConfig, SolarOpenForCausalLM, SolarOpenModel
from easymlx.modules.starcoder2 import Starcoder2Config, Starcoder2ForCausalLM, Starcoder2Model
from easymlx.modules.telechat3 import Telechat3Config, Telechat3ForCausalLM, Telechat3Model


@pytest.mark.parametrize(
    ("config", "base_cls", "causal_cls"),
    [
        pytest.param(
            SmolLM3Config(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
            ),
            SmolLM3Model,
            SmolLM3ForCausalLM,
            id="smollm3",
        ),
        pytest.param(
            Llama4TextConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                intermediate_size_mlp=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
                num_local_experts=4,
                num_experts_per_tok=1,
                interleave_moe_layer_step=2,
                no_rope_layer_interval=2,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            ),
            Llama4TextModel,
            Llama4TextForCausalLM,
            id="llama4_text",
        ),
        pytest.param(
            HeliumConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
                tie_word_embeddings=False,
            ),
            HeliumModel,
            HeliumForCausalLM,
            id="helium",
        ),
        pytest.param(
            Ministral3Config(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
                sliding_window=8,
                layer_types=["sliding_attention", "full_attention"],
                rope_parameters={
                    "rope_theta": 10000.0,
                    "original_max_position_embeddings": 128,
                    "llama_4_scaling_beta": 0.1,
                },
            ),
            Ministral3Model,
            Ministral3ForCausalLM,
            id="ministral3",
        ),
        pytest.param(
            Mistral3Config(
                text_config={
                    "model_type": "ministral3",
                    "vocab_size": 128,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 8,
                    "max_position_embeddings": 128,
                    "sliding_window": 8,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "rope_parameters": {
                        "rope_theta": 10000.0,
                        "original_max_position_embeddings": 128,
                        "llama_4_scaling_beta": 0.1,
                    },
                }
            ),
            Mistral3Model,
            Mistral3ForCausalLM,
            id="mistral3",
        ),
        pytest.param(
            PixtralConfig(
                text_config={
                    "vocab_size": 128,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 8,
                    "max_position_embeddings": 128,
                    "tie_word_embeddings": False,
                }
            ),
            PixtralModel,
            PixtralForCausalLM,
            id="pixtral",
        ),
        pytest.param(
            SolarOpenConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
                moe_intermediate_size=64,
                num_experts_per_tok=2,
                n_shared_experts=1,
                n_routed_experts=4,
                first_k_dense_replace=1,
            ),
            SolarOpenModel,
            SolarOpenForCausalLM,
            id="solar_open",
        ),
        pytest.param(
            Telechat3Config(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                max_position_embeddings=128,
                num_attention_heads=4,
                num_hidden_layers=2,
                num_key_value_heads=2,
                head_dim=8,
                tie_word_embeddings=False,
            ),
            Telechat3Model,
            Telechat3ForCausalLM,
            id="telechat3",
        ),
        pytest.param(
            GemmaConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
            ),
            GemmaModel,
            GemmaForCausalLM,
            id="gemma",
        ),
        pytest.param(
            Gemma2Config(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
            ),
            Gemma2Model,
            Gemma2ForCausalLM,
            id="gemma2",
        ),
        pytest.param(
            PhiConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=128,
            ),
            PhiModel,
            PhiForCausalLM,
            id="phi",
        ),
        pytest.param(
            Phi3Config(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
            ),
            Phi3Model,
            Phi3ForCausalLM,
            id="phi3",
        ),
        pytest.param(
            GPT2Config(
                n_embd=32,
                n_head=4,
                n_layer=2,
                vocab_size=128,
                n_positions=128,
            ),
            GPT2Model,
            GPT2ForCausalLM,
            id="gpt2",
        ),
        pytest.param(
            Starcoder2Config(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
            ),
            Starcoder2Model,
            Starcoder2ForCausalLM,
            id="starcoder2",
        ),
        pytest.param(
            CohereConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
            ),
            CohereModel,
            CohereForCausalLM,
            id="cohere",
        ),
        pytest.param(
            GraniteConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
            ),
            GraniteModel,
            GraniteForCausalLM,
            id="granite",
        ),
        pytest.param(
            InternLM2Config(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
            ),
            InternLM2Model,
            InternLM2ForCausalLM,
            id="internlm2",
        ),
        pytest.param(
            MixtralConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_local_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=128,
            ),
            MixtralModel,
            MixtralForCausalLM,
            id="mixtral",
        ),
    ],
)
def test_auto_models_resolve_new_families(config, base_cls, causal_cls):
    base_model = AutoEasyMLXModel.from_config(config)
    causal_model = AutoEasyMLXModelForCausalLM.from_config(config)

    assert isinstance(base_model, base_cls)
    assert isinstance(causal_model, causal_cls)


def test_top_level_exports_include_new_families():
    assert easymlx.SmolLM3Config is SmolLM3Config
    assert easymlx.SmolLM3ForCausalLM is SmolLM3ForCausalLM
    assert easymlx.HeliumConfig is HeliumConfig
    assert easymlx.HeliumForCausalLM is HeliumForCausalLM
    assert easymlx.Llama4TextConfig is Llama4TextConfig
    assert easymlx.Llama4TextForCausalLM is Llama4TextForCausalLM
    assert easymlx.Ministral3Config is Ministral3Config
    assert easymlx.Ministral3ForCausalLM is Ministral3ForCausalLM
    assert easymlx.Mistral3Config is Mistral3Config
    assert easymlx.Mistral3ForCausalLM is Mistral3ForCausalLM
    assert easymlx.PixtralConfig is PixtralConfig
    assert easymlx.PixtralForCausalLM is PixtralForCausalLM
    assert easymlx.SolarOpenConfig is SolarOpenConfig
    assert easymlx.SolarOpenForCausalLM is SolarOpenForCausalLM
    assert easymlx.Telechat3Config is Telechat3Config
    assert easymlx.Telechat3ForCausalLM is Telechat3ForCausalLM
    # New families
    assert easymlx.GemmaConfig is GemmaConfig
    assert easymlx.GemmaForCausalLM is GemmaForCausalLM
    assert easymlx.Gemma2Config is Gemma2Config
    assert easymlx.Gemma2ForCausalLM is Gemma2ForCausalLM
    assert easymlx.PhiConfig is PhiConfig
    assert easymlx.PhiForCausalLM is PhiForCausalLM
    assert easymlx.Phi3Config is Phi3Config
    assert easymlx.Phi3ForCausalLM is Phi3ForCausalLM
    assert easymlx.GPT2Config is GPT2Config
    assert easymlx.GPT2ForCausalLM is GPT2ForCausalLM
    assert easymlx.Starcoder2Config is Starcoder2Config
    assert easymlx.Starcoder2ForCausalLM is Starcoder2ForCausalLM
    assert easymlx.CohereConfig is CohereConfig
    assert easymlx.CohereForCausalLM is CohereForCausalLM
    assert easymlx.GraniteConfig is GraniteConfig
    assert easymlx.GraniteForCausalLM is GraniteForCausalLM
    assert easymlx.InternLM2Config is InternLM2Config
    assert easymlx.InternLM2ForCausalLM is InternLM2ForCausalLM
    assert easymlx.MixtralConfig is MixtralConfig
    assert easymlx.MixtralForCausalLM is MixtralForCausalLM


def test_llama4_text_model_sanitize_filters_rope_buffers():
    model = Llama4TextModel(
        Llama4TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            intermediate_size_mlp=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            num_local_experts=4,
            num_experts_per_tok=1,
            interleave_moe_layer_step=2,
            no_rope_layer_interval=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )

    sanitized = model.sanitize(
        {
            "model.layers.0.self_attn.rope.inv_freq": mx.ones((8,)),
            "model.layers.0.self_attn.q_proj.weight": mx.ones((32, 32)),
            "rotary_emb.inv_freq": mx.ones((8,)),
        }
    )

    assert "model.layers.0.self_attn.rope.inv_freq" not in sanitized
    assert "rotary_emb.inv_freq" not in sanitized
    assert "language_model.layers.0.self_attn.q_proj.weight" in sanitized
