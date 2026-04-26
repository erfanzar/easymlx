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

"""Tests for analytical FLOPs estimation."""

import pytest
from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.modules.glm4_moe_lite import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM
from easymlx.modules.glm4v import Glm4VConfig, Glm4VForConditionalGeneration
from easymlx.modules.llama import LlamaConfig, LlamaForCausalLM
from easymlx.modules.qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM
from easymlx.modules.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from easymlx.modules.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM
from easymlx.modules.qwen3_omni_moe import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration


def test_dense_llama_get_flops_matches_manual_formula():
    """Dense decoder-only FLOPs should match the closed-form estimate."""
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)

    batch_size = 2
    sequence_length = 4
    actual = model.get_flops(batch_size=batch_size, sequence_length=sequence_length)

    tokens = batch_size * sequence_length
    hidden_size = 16
    intermediate_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 4
    vocab_size = 100

    attention_proj = (
        2
        * tokens
        * (
            hidden_size * (num_heads * head_dim)
            + hidden_size * (num_kv_heads * head_dim)
            + hidden_size * (num_kv_heads * head_dim)
            + (num_heads * head_dim) * hidden_size
        )
    )
    attention_matmul = 2 * batch_size * num_heads * sequence_length * sequence_length * (head_dim + head_dim)
    mlp = 6 * tokens * hidden_size * intermediate_size
    lm_head = 2 * tokens * hidden_size * vocab_size
    expected = config.num_hidden_layers * (attention_proj + attention_matmul + mlp) + lm_head

    assert actual == expected


def test_get_flops_rejects_unsupported_task_types():
    class DummySequenceClassificationModel(EasyMLXBaseModule):
        _model_task = TaskType.SEQUENCE_CLASSIFICATION.value

        def __call__(self, *args, **kwargs):
            del args, kwargs
            raise NotImplementedError

    model = DummySequenceClassificationModel(
        LlamaConfig(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
    )

    with pytest.raises(NotImplementedError, match="not supported for task type"):
        model.get_flops(batch_size=2, sequence_length=4)


def test_qwen3_next_get_flops_matches_manual_formula():
    """Hybrid Qwen3-Next FLOPs should account for both full and linear layers."""
    config = Qwen3NextConfig(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        mlp_only_layers=[0, 1, 2, 3],
        tie_word_embeddings=False,
    )
    model = Qwen3NextForCausalLM(config)

    batch_size = 2
    sequence_length = 4
    actual = model.get_flops(batch_size=batch_size, sequence_length=sequence_length)

    tokens = batch_size * sequence_length
    hidden_size = config.hidden_size
    full_q_proj_dim = 2 * config.num_attention_heads * config.head_dim
    full_kv_proj_dim = config.num_key_value_heads * config.head_dim
    full_attention_proj = (
        2
        * tokens
        * (
            hidden_size * full_q_proj_dim
            + hidden_size * full_kv_proj_dim
            + hidden_size * full_kv_proj_dim
            + (config.num_attention_heads * config.head_dim) * hidden_size
        )
    )
    full_attention_matmul = (
        2
        * batch_size
        * config.num_attention_heads
        * sequence_length
        * sequence_length
        * (config.head_dim + config.head_dim)
    )

    linear_value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    linear_inner_dim = 2 * config.linear_num_key_heads * config.linear_key_head_dim + linear_value_dim
    linear_attention_proj = (
        2
        * tokens
        * (
            hidden_size * linear_inner_dim
            + hidden_size * linear_value_dim
            + hidden_size * config.linear_num_value_heads
            + hidden_size * config.linear_num_value_heads
            + linear_value_dim * hidden_size
        )
    )
    linear_conv = 2 * batch_size * sequence_length * linear_inner_dim * config.linear_conv_kernel_dim
    linear_gdr = (
        batch_size
        * sequence_length
        * config.linear_num_value_heads
        * (7 * config.linear_key_head_dim * config.linear_value_head_dim)
    )

    mlp = 6 * tokens * config.hidden_size * config.intermediate_size
    lm_head = 2 * tokens * config.hidden_size * config.vocab_size
    expected = 2 * (full_attention_proj + full_attention_matmul + mlp)
    expected += 2 * (linear_attention_proj + linear_conv + linear_gdr + mlp)
    expected += lm_head

    assert actual == expected


def test_qwen2_vl_get_flops_requires_vision_shape_when_not_in_config(small_model_config):
    """VLM FLOPs should request explicit vision dimensions when config defaults are absent."""
    config = Qwen2VLConfig(
        text_config={
            "vocab_size": small_model_config["vocab_size"],
            "hidden_size": small_model_config["hidden_size"],
            "intermediate_size": small_model_config["intermediate_size"],
            "num_hidden_layers": small_model_config["num_hidden_layers"],
            "num_attention_heads": small_model_config["num_attention_heads"],
            "num_key_value_heads": small_model_config["num_key_value_heads"],
            "max_position_embeddings": small_model_config["max_position_embeddings"],
        },
        vision_config={
            "depth": 1,
            "embed_dim": 32,
            "hidden_size": 64,
            "num_heads": 4,
            "patch_size": 4,
            "in_channels": 3,
        },
        image_token_id=small_model_config["vocab_size"] - 1,
        video_token_id=small_model_config["vocab_size"] - 2,
    )
    model = Qwen2VLForConditionalGeneration(config)

    with pytest.raises(ValueError, match="Vision FLOPs require"):
        model.get_flops(batch_size=2, sequence_length=16, use_vision=True)

    text_only = model.get_flops(batch_size=2, sequence_length=16)
    with_vision = model.get_flops(batch_size=2, sequence_length=16, use_vision=True, image_size=8)
    assert with_vision > text_only


@pytest.mark.parametrize(
    ("builder", "flop_kwargs"),
    [
        (
            lambda sm: Qwen2MoeForCausalLM(
                Qwen2MoeConfig(
                    vocab_size=sm["vocab_size"],
                    hidden_size=sm["hidden_size"],
                    intermediate_size=sm["intermediate_size"],
                    num_hidden_layers=sm["num_hidden_layers"],
                    num_attention_heads=sm["num_attention_heads"],
                    num_key_value_heads=sm["num_key_value_heads"],
                    moe_intermediate_size=32,
                    shared_expert_intermediate_size=64,
                    num_experts=sm["num_experts"],
                    num_experts_per_tok=sm["num_experts_per_tok"],
                )
            ),
            {},
        ),
        (
            lambda sm: Glm4MoeLiteForCausalLM(
                Glm4MoeLiteConfig(
                    vocab_size=sm["vocab_size"],
                    hidden_size=32,
                    intermediate_size=64,
                    moe_intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    n_shared_experts=1,
                    n_routed_experts=4,
                    routed_scaling_factor=1.0,
                    kv_lora_rank=8,
                    q_lora_rank=None,
                    qk_rope_head_dim=4,
                    qk_nope_head_dim=4,
                    v_head_dim=8,
                    n_group=1,
                    topk_group=1,
                    num_experts_per_tok=2,
                    max_position_embeddings=sm["max_position_embeddings"],
                    mlp_layer_types=["dense", "dense"],
                )
            ),
            {},
        ),
        (
            lambda sm: Glm4VForConditionalGeneration(
                Glm4VConfig(
                    text_config={
                        "vocab_size": sm["vocab_size"],
                        "hidden_size": sm["hidden_size"],
                        "intermediate_size": sm["intermediate_size"],
                        "num_hidden_layers": sm["num_hidden_layers"],
                        "num_attention_heads": sm["num_attention_heads"],
                        "num_key_value_heads": sm["num_key_value_heads"],
                        "max_position_embeddings": sm["max_position_embeddings"],
                        "partial_rotary_factor": 0.5,
                    },
                    vision_config={
                        "depth": 1,
                        "hidden_size": 16,
                        "intermediate_size": 32,
                        "num_heads": 4,
                        "patch_size": 4,
                        "image_size": 8,
                        "in_channels": 3,
                        "out_hidden_size": sm["hidden_size"],
                        "spatial_merge_size": 2,
                        "temporal_patch_size": 2,
                    },
                    image_token_id=sm["vocab_size"] - 1,
                    video_token_id=sm["vocab_size"] - 2,
                    hidden_size=sm["hidden_size"],
                )
            ),
            {"use_vision": True, "image_size": 8, "vision_frames": 2},
        ),
        (
            lambda sm: Qwen3OmniMoeForConditionalGeneration(
                Qwen3OmniMoeConfig(
                    thinker_config={
                        "audio_config": {},
                        "vision_config": {
                            "hidden_size": 16,
                            "intermediate_size": 32,
                            "num_heads": 4,
                            "depth": 1,
                            "out_hidden_size": sm["hidden_size"],
                            "patch_size": 4,
                            "in_channels": 3,
                        },
                        "text_config": {
                            "vocab_size": sm["vocab_size"],
                            "hidden_size": sm["hidden_size"],
                            "intermediate_size": sm["intermediate_size"],
                            "num_hidden_layers": sm["num_hidden_layers"],
                            "num_attention_heads": sm["num_attention_heads"],
                            "num_key_value_heads": sm["num_key_value_heads"],
                            "head_dim": sm["head_dim"],
                            "num_experts": sm["num_experts"],
                            "num_experts_per_tok": sm["num_experts_per_tok"],
                        },
                        "image_token_id": sm["vocab_size"] - 1,
                        "video_token_id": sm["vocab_size"] - 2,
                    }
                )
            ),
            {"use_vision": True, "image_size": 8},
        ),
    ],
    ids=["qwen2_moe", "glm4_moe_lite", "glm4v", "qwen3_omni_moe"],
)
def test_get_flops_smoke_across_model_families(builder, flop_kwargs, small_model_config):
    """Representative model families should all expose positive FLOPs estimates."""
    model = builder(small_model_config)
    flops = model.get_flops(batch_size=small_model_config["batch_size"], sequence_length=8, **flop_kwargs)
    assert isinstance(flops, int)
    assert flops > 0


def test_qwen3_omni_audio_flops_not_implemented(small_model_config):
    """Audio FLOPs should fail explicitly until audio inference exists in MLX."""
    model = Qwen3OmniMoeForConditionalGeneration(
        Qwen3OmniMoeConfig(
            thinker_config={
                "audio_config": {},
                "vision_config": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_heads": 4,
                    "depth": 1,
                    "out_hidden_size": small_model_config["hidden_size"],
                    "patch_size": 4,
                    "in_channels": 3,
                },
                "text_config": {
                    "vocab_size": small_model_config["vocab_size"],
                    "hidden_size": small_model_config["hidden_size"],
                    "intermediate_size": small_model_config["intermediate_size"],
                    "num_hidden_layers": small_model_config["num_hidden_layers"],
                    "num_attention_heads": small_model_config["num_attention_heads"],
                    "num_key_value_heads": small_model_config["num_key_value_heads"],
                    "head_dim": small_model_config["head_dim"],
                    "num_experts": small_model_config["num_experts"],
                    "num_experts_per_tok": small_model_config["num_experts_per_tok"],
                },
                "image_token_id": small_model_config["vocab_size"] - 1,
                "video_token_id": small_model_config["vocab_size"] - 2,
            }
        )
    )

    with pytest.raises(NotImplementedError, match="Audio FLOPs estimation"):
        model.get_flops(batch_size=2, sequence_length=8, use_audio=True)
