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

import mlx.core as mx

from easymlx.modules.llama4 import Llama4Config, Llama4ForConditionalGeneration
from easymlx.modules.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM
from easymlx.modules.qwen3_omni_moe import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration


def test_llama4_forward_shapes():
    config = Llama4Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 64,
            "intermediate_size_mlp": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "interleave_moe_layer_step": 2,
            "no_rope_layer_interval": 2,
        },
        vision_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "vision_output_dim": 64,
            "image_size": 8,
            "patch_size": 4,
        },
        image_token_index=120,
    )
    model = Llama4ForConditionalGeneration(config)
    num_patches = (config.vision_config["image_size"] // config.vision_config["patch_size"]) ** 2
    input_ids = mx.array([[1] + [config.image_token_id] * num_patches], dtype=mx.int32)
    pixel_values = mx.random.uniform(shape=(1, 3, 8, 8))
    logits = model(input_ids, pixel_values=pixel_values, return_dict=False)
    assert logits.shape == (1, input_ids.shape[1], config.text_config["vocab_size"])


def test_qwen3_next_forward_shapes():
    config = Qwen3NextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention", "linear_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    model = Qwen3NextForCausalLM(config)
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    logits = model(input_ids, return_dict=False)
    assert logits.shape == (1, input_ids.shape[1], config.vocab_size)


def test_qwen3_omni_moe_forward_shapes():
    config = Qwen3OmniMoeConfig(
        thinker_config={
            "audio_config": {},
            "vision_config": {
                "hidden_size": 16,
                "num_heads": 4,
                "depth": 1,
                "intermediate_size": 32,
                "out_hidden_size": 32,
                "image_size": 8,
                "patch_size": 4,
            },
            "text_config": {
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
            },
            "image_token_id": 120,
            "video_token_id": 121,
        }
    )
    model = Qwen3OmniMoeForConditionalGeneration(config)
    vision_cfg = config.thinker_config["vision_config"]
    num_patches = (vision_cfg["image_size"] // vision_cfg["patch_size"]) ** 2
    input_ids = mx.array([[1] + [config.thinker_config["image_token_id"]] * num_patches], dtype=mx.int32)
    pixel_values = mx.random.uniform(shape=(1, 3, 8, 8))
    logits = model(input_ids, pixel_values=pixel_values, return_dict=False)
    assert logits.shape == (1, input_ids.shape[1], config.thinker_config["text_config"]["vocab_size"])
