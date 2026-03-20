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


"""Shared FLOPs estimators for easymlx models.

The estimators in this module are intentionally analytical. They use the
runtime configuration to approximate the floating-point work performed by a
single forward pass, including architecture-specific paths such as:

- dense decoder-only transformers
- sliding/chunked attention
- hybrid Qwen3-Next linear attention
- MoE feed-forward blocks
- GLM-4 MoE Lite MLA attention
- common vision towers used by the multimodal models in this repository
"""

from __future__ import annotations

import math
import typing as tp

from .base_config import EasyMLXBaseConfig


def estimate_module_forward_flops(
    module: tp.Any,
    *,
    batch_size: int,
    sequence_length: int,
    use_vision: bool = False,
    use_audio: bool = False,
    include_lm_head: bool | None = None,
    image_size: int | tuple[int, int] | None = None,
    vision_batch_size: int | None = None,
    vision_height: int | None = None,
    vision_width: int | None = None,
    vision_frames: int | None = None,
    vision_sequence_length: int | None = None,
    audio_sequence_length: int | None = None,
) -> int:
    """Estimate FLOPs for a single model forward pass.

    Args:
        module: Instantiated easymlx model/module.
        batch_size: Text batch size.
        sequence_length: Text sequence length seen by the language model.
        use_vision: Whether the forward path includes the vision encoder.
        use_audio: Whether the forward path includes the audio encoder.
        include_lm_head: Override whether to include the output vocabulary
            projection. ``None`` infers the value from the module shape.
        image_size: Convenience override for square or ``(height, width)``
            image inputs.
        vision_batch_size: Number of vision items processed by the vision
            tower. Defaults to ``batch_size``.
        vision_height: Vision input height when ``use_vision=True``.
        vision_width: Vision input width when ``use_vision=True``.
        vision_frames: Number of frames/temporal patches per vision input.
        vision_sequence_length: Explicit number of raw vision tokens before
            model-specific merging/projectors. Use this when the preprocessor
            already determines the exact patch count.
        audio_sequence_length: Placeholder for future audio support.

    Returns:
        Integer FLOPs for one forward pass.
    """
    batch_size = _require_positive_int("batch_size", batch_size)
    sequence_length = _require_positive_int("sequence_length", sequence_length)
    if vision_batch_size is None:
        vision_batch_size = batch_size
    else:
        vision_batch_size = _require_positive_int("vision_batch_size", vision_batch_size)

    if vision_frames is None:
        vision_frames = 1
    else:
        vision_frames = _require_positive_int("vision_frames", vision_frames)

    if vision_sequence_length is not None:
        vision_sequence_length = _require_positive_int("vision_sequence_length", vision_sequence_length)
    if audio_sequence_length is not None:
        audio_sequence_length = _require_positive_int("audio_sequence_length", audio_sequence_length)

    if image_size is not None:
        if isinstance(image_size, tuple):
            if len(image_size) != 2:
                raise ValueError("image_size must be an int or a (height, width) tuple.")
            vision_height, vision_width = image_size
        else:
            vision_height = image_size
            vision_width = image_size

    root_config, text_config, vision_config, audio_config = _resolve_model_configs(module.config)
    total = _estimate_text_model_flops(text_config, batch_size=batch_size, sequence_length=sequence_length)

    if include_lm_head is None:
        include_lm_head = _should_include_lm_head(module)
    if include_lm_head:
        total += _estimate_lm_head_flops(text_config, batch_size=batch_size, sequence_length=sequence_length)

    if use_vision:
        if vision_config is None:
            raise ValueError(f"{module.__class__.__name__} does not expose a vision configuration.")
        total += _estimate_vision_path_flops(
            module,
            root_config=root_config,
            vision_config=vision_config,
            text_config=text_config,
            vision_batch_size=vision_batch_size,
            vision_height=vision_height,
            vision_width=vision_width,
            vision_frames=vision_frames,
            vision_sequence_length=vision_sequence_length,
        )

    if use_audio:
        if audio_config is None:
            raise ValueError(f"{module.__class__.__name__} does not expose an audio configuration.")
        raise NotImplementedError(
            "Audio FLOPs estimation is not implemented because audio inference is not supported in the MLX runtime yet."
        )

    del audio_sequence_length, audio_config
    return int(total)


def _resolve_model_configs(
    config: EasyMLXBaseConfig,
) -> tuple[EasyMLXBaseConfig, EasyMLXBaseConfig, EasyMLXBaseConfig | None, EasyMLXBaseConfig | None]:
    """Resolve the text/vision/audio configs exposed by a model config."""
    _call_finalize(config)

    if callable(getattr(config, "to_thinker_config", None)):
        thinker_config = config.to_thinker_config()
        _call_finalize(thinker_config)
        text_config = _get_text_config(thinker_config)
        vision_config = getattr(thinker_config, "vision_config", None)
        audio_config = getattr(thinker_config, "audio_config", None)
        return thinker_config, text_config, vision_config, audio_config

    if callable(getattr(config, "to_model_config", None)):
        model_config = config.to_model_config()
        _call_finalize(model_config)
        text_config = _get_text_config(model_config)
        vision_config = getattr(model_config, "vision_config", None)
        return model_config, text_config, vision_config, None

    text_config = _get_text_config(config)
    vision_config = getattr(config, "vision_config", None)
    audio_config = getattr(config, "audio_config", None)
    return config, text_config, vision_config, audio_config


def _get_text_config(config: EasyMLXBaseConfig) -> EasyMLXBaseConfig:
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        _call_finalize(text_config)
        return text_config
    return config


def _estimate_text_model_flops(
    config: EasyMLXBaseConfig,
    *,
    batch_size: int,
    sequence_length: int,
) -> int:
    tokens = batch_size * sequence_length

    if _is_mla_text_config(config):
        return _estimate_mla_text_flops(config, batch_size=batch_size, sequence_length=sequence_length, tokens=tokens)
    if _is_qwen3_next_text_config(config):
        return _estimate_qwen3_next_text_flops(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tokens=tokens,
        )
    return _estimate_standard_text_flops(config, batch_size=batch_size, sequence_length=sequence_length, tokens=tokens)


def _estimate_standard_text_flops(
    config: EasyMLXBaseConfig,
    *,
    batch_size: int,
    sequence_length: int,
    tokens: int,
) -> int:
    total = 0
    num_layers = _require_positive_int("num_hidden_layers", config.num_hidden_layers)
    hidden_size = _get_hidden_size(config)
    num_heads = _require_positive_int("num_attention_heads", config.num_attention_heads)
    num_kv_heads = _get_num_kv_heads(config, num_heads)
    head_dim = _get_head_dim(config, hidden_size=hidden_size, num_heads=num_heads)
    q_proj_dim = num_heads * head_dim
    kv_proj_dim = num_kv_heads * head_dim

    for layer_idx in range(num_layers):
        key_length = _attention_window_for_layer(config, layer_idx, sequence_length)
        total += _attention_flops(
            tokens=tokens,
            batch_size=batch_size,
            query_length=sequence_length,
            key_length=key_length,
            hidden_size=hidden_size,
            num_query_heads=num_heads,
            query_proj_dim=q_proj_dim,
            key_proj_dim=kv_proj_dim,
            value_proj_dim=kv_proj_dim,
            output_proj_dim=q_proj_dim,
            qk_head_dim=head_dim,
            value_head_dim=head_dim,
        )
        if _is_sparse_mlp_layer(config, layer_idx):
            total += _estimate_sparse_mlp_flops(config, tokens=tokens, hidden_size=hidden_size)
        else:
            total += _gated_mlp_flops(tokens, hidden_size, _dense_intermediate_size(config))

    return total


def _estimate_qwen3_next_text_flops(
    config: EasyMLXBaseConfig,
    *,
    batch_size: int,
    sequence_length: int,
    tokens: int,
) -> int:
    total = 0
    hidden_size = _get_hidden_size(config)
    num_layers = _require_positive_int("num_hidden_layers", config.num_hidden_layers)

    num_heads = _require_positive_int("num_attention_heads", config.num_attention_heads)
    num_kv_heads = _get_num_kv_heads(config, num_heads)
    head_dim = _get_head_dim(config, hidden_size=hidden_size, num_heads=num_heads)

    linear_num_key_heads = _require_positive_int("linear_num_key_heads", config.linear_num_key_heads)
    linear_num_value_heads = _require_positive_int("linear_num_value_heads", config.linear_num_value_heads)
    linear_key_head_dim = _require_positive_int("linear_key_head_dim", config.linear_key_head_dim)
    linear_value_head_dim = _require_positive_int("linear_value_head_dim", config.linear_value_head_dim)
    linear_conv_kernel_dim = _require_positive_int("linear_conv_kernel_dim", config.linear_conv_kernel_dim)

    full_q_proj_dim = 2 * num_heads * head_dim
    full_kv_proj_dim = num_kv_heads * head_dim

    key_dim = linear_num_key_heads * linear_key_head_dim
    value_dim = linear_num_value_heads * linear_value_head_dim
    linear_inner_dim = 2 * key_dim + value_dim

    for layer_idx in range(num_layers):
        if _is_full_attention_layer(config, layer_idx):
            total += _attention_flops(
                tokens=tokens,
                batch_size=batch_size,
                query_length=sequence_length,
                key_length=sequence_length,
                hidden_size=hidden_size,
                num_query_heads=num_heads,
                query_proj_dim=full_q_proj_dim,
                key_proj_dim=full_kv_proj_dim,
                value_proj_dim=full_kv_proj_dim,
                output_proj_dim=num_heads * head_dim,
                qk_head_dim=head_dim,
                value_head_dim=head_dim,
            )
        else:
            total += _linear_flops(tokens, hidden_size, linear_inner_dim)
            total += _linear_flops(tokens, hidden_size, value_dim)
            total += _linear_flops(tokens, hidden_size, linear_num_value_heads)
            total += _linear_flops(tokens, hidden_size, linear_num_value_heads)
            total += _linear_flops(tokens, value_dim, hidden_size)
            total += 2 * batch_size * sequence_length * linear_inner_dim * linear_conv_kernel_dim
            total += (
                batch_size * sequence_length * linear_num_value_heads * (7 * linear_key_head_dim * linear_value_head_dim)
            )

        if _is_sparse_mlp_layer(config, layer_idx):
            total += _estimate_sparse_mlp_flops(config, tokens=tokens, hidden_size=hidden_size)
        else:
            total += _gated_mlp_flops(tokens, hidden_size, _dense_intermediate_size(config))

    return total


def _estimate_mla_text_flops(
    config: EasyMLXBaseConfig,
    *,
    batch_size: int,
    sequence_length: int,
    tokens: int,
) -> int:
    total = 0
    hidden_size = _get_hidden_size(config)
    num_layers = _require_positive_int("num_hidden_layers", config.num_hidden_layers)
    num_heads = _require_positive_int("num_attention_heads", config.num_attention_heads)

    qk_rope_head_dim = _require_positive_int("qk_rope_head_dim", config.qk_rope_head_dim)
    qk_nope_head_dim = _require_positive_int("qk_nope_head_dim", config.qk_nope_head_dim)
    q_head_dim = qk_rope_head_dim + qk_nope_head_dim
    v_head_dim = _require_positive_int("v_head_dim", config.v_head_dim)
    kv_lora_rank = _require_positive_int("kv_lora_rank", config.kv_lora_rank)
    q_lora_rank = getattr(config, "q_lora_rank", None)
    if q_lora_rank is not None:
        q_lora_rank = _require_positive_int("q_lora_rank", q_lora_rank)

    for layer_idx in range(num_layers):
        key_length = _attention_window_for_layer(config, layer_idx, sequence_length)
        if q_lora_rank is None:
            total += _linear_flops(tokens, hidden_size, num_heads * q_head_dim)
        else:
            total += _linear_flops(tokens, hidden_size, q_lora_rank)
            total += _linear_flops(tokens, q_lora_rank, num_heads * q_head_dim)

        total += _linear_flops(tokens, hidden_size, kv_lora_rank + qk_rope_head_dim)
        total += _linear_flops(tokens, kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
        total += _linear_flops(tokens, num_heads * v_head_dim, hidden_size)
        total += 2 * batch_size * num_heads * sequence_length * key_length * (q_head_dim + v_head_dim)

        if _is_sparse_mlp_layer(config, layer_idx):
            total += _estimate_sparse_mlp_flops(config, tokens=tokens, hidden_size=hidden_size)
        else:
            total += _gated_mlp_flops(tokens, hidden_size, _dense_intermediate_size(config))

    return total


def _estimate_sparse_mlp_flops(config: EasyMLXBaseConfig, *, tokens: int, hidden_size: int) -> int:
    class_name = type(config).__name__

    if hasattr(config, "shared_expert_intermediate_size"):
        router_experts = _require_positive_int("num_experts", config.num_experts)
        top_k = _require_positive_int("num_experts_per_tok", config.num_experts_per_tok)
        routed = _gated_mlp_flops(
            tokens * top_k, hidden_size, _require_positive_int("moe_intermediate_size", config.moe_intermediate_size)
        )
        shared = _gated_mlp_flops(
            tokens,
            hidden_size,
            _require_positive_int("shared_expert_intermediate_size", config.shared_expert_intermediate_size),
        )
        shared_gate = _linear_flops(tokens, hidden_size, hidden_size)
        router = _linear_flops(tokens, hidden_size, router_experts)
        return router + routed + shared + shared_gate

    if class_name == "Llama4TextConfig":
        router_experts = _require_positive_int("num_local_experts", config.num_local_experts)
        top_k = _require_positive_int("num_experts_per_tok", config.num_experts_per_tok)
        router = _linear_flops(tokens, hidden_size, router_experts)
        routed = _gated_mlp_flops(
            tokens * top_k, hidden_size, _require_positive_int("intermediate_size", config.intermediate_size)
        )
        return router + routed

    if class_name == "GptOssConfig":
        router_experts = _require_positive_int("num_local_experts", config.num_local_experts)
        top_k = _require_positive_int("num_experts_per_tok", config.num_experts_per_tok)
        router = _linear_flops(tokens, hidden_size, router_experts)
        routed = _gated_mlp_flops(
            tokens * top_k, hidden_size, _require_positive_int("intermediate_size", config.intermediate_size)
        )
        return router + routed

    if class_name == "Glm4MoeConfig":
        router_experts = _require_positive_int("n_routed_experts", config.n_routed_experts)
        top_k = _require_positive_int("num_experts_per_tok", config.num_experts_per_tok)
        router = _linear_flops(tokens, hidden_size, router_experts)
        routed = _gated_mlp_flops(
            tokens * top_k,
            hidden_size,
            _require_positive_int("moe_intermediate_size", config.moe_intermediate_size),
        )
        shared = 0
        if getattr(config, "n_shared_experts", None):
            shared = _gated_mlp_flops(
                tokens,
                hidden_size,
                _require_positive_int("moe_intermediate_size", config.moe_intermediate_size)
                * _require_positive_int("n_shared_experts", config.n_shared_experts),
            )
        return router + routed + shared

    if class_name in {"Glm4VMoeTextConfig", "Glm4MoeLiteConfig"}:
        router_experts = _require_positive_int("n_routed_experts", config.n_routed_experts)
        top_k = _require_positive_int("num_experts_per_tok", config.num_experts_per_tok)
        router = _linear_flops(tokens, hidden_size, router_experts)
        routed = _gated_mlp_flops(
            tokens * top_k,
            hidden_size,
            _require_positive_int("moe_intermediate_size", config.moe_intermediate_size),
        )
        shared = 0
        if getattr(config, "n_shared_experts", None):
            shared = _gated_mlp_flops(tokens, hidden_size, _dense_intermediate_size(config))
        return router + routed + shared

    if hasattr(config, "num_experts") and hasattr(config, "moe_intermediate_size"):
        router_experts = _require_positive_int("num_experts", config.num_experts)
        top_k = _require_positive_int("num_experts_per_tok", config.num_experts_per_tok)
        router = _linear_flops(tokens, hidden_size, router_experts)
        routed = _gated_mlp_flops(
            tokens * top_k,
            hidden_size,
            _require_positive_int("moe_intermediate_size", config.moe_intermediate_size),
        )
        return router + routed

    raise ValueError(f"Unsupported sparse MLP configuration: {type(config).__name__}")


def _estimate_lm_head_flops(
    config: EasyMLXBaseConfig,
    *,
    batch_size: int,
    sequence_length: int,
) -> int:
    hidden_size = _get_hidden_size(config)
    vocab_size = _require_positive_int("vocab_size", config.vocab_size)
    return _linear_flops(batch_size * sequence_length, hidden_size, vocab_size)


def _estimate_vision_path_flops(
    module: tp.Any,
    *,
    root_config: EasyMLXBaseConfig,
    vision_config: EasyMLXBaseConfig,
    text_config: EasyMLXBaseConfig,
    vision_batch_size: int,
    vision_height: int | None,
    vision_width: int | None,
    vision_frames: int,
    vision_sequence_length: int | None,
) -> int:
    owner = _find_model_owner(module)
    vision_tower = getattr(owner, "vision_tower", None)
    if vision_tower is None:
        raise ValueError(f"{module.__class__.__name__} does not have a vision tower.")

    if vision_sequence_length is None:
        default_image_size = getattr(vision_config, "image_size", None)
        if vision_height is None:
            vision_height = default_image_size
        if vision_width is None:
            vision_width = default_image_size
        if vision_height is None or vision_width is None:
            raise ValueError(
                "Vision FLOPs require `image_size`, `vision_height`/`vision_width`, or `vision_sequence_length` "
                "when the model config does not provide a default image size."
            )
        vision_height = _require_positive_int("vision_height", vision_height)
        vision_width = _require_positive_int("vision_width", vision_width)

    if vision_frames is None:
        vision_frames = max(1, int(getattr(vision_config, "temporal_patch_size", 1)))

    class_name = type(vision_tower).__name__
    if class_name == "Qwen3_5VisionModel":
        flops, output_tokens = _estimate_qwen3_5_vision_flops(
            vision_config,
            vision_batch_size=vision_batch_size,
            vision_height=vision_height,
            vision_width=vision_width,
            vision_frames=vision_frames,
            vision_sequence_length=vision_sequence_length,
        )
    elif class_name == "VisionModel":
        flops, output_tokens = _estimate_glm4v_vision_flops(
            vision_config,
            vision_batch_size=vision_batch_size,
            vision_height=vision_height,
            vision_width=vision_width,
            vision_frames=vision_frames,
            vision_sequence_length=vision_sequence_length,
        )
    else:
        flops, output_tokens = _estimate_standard_vision_flops(
            vision_config,
            vision_batch_size=vision_batch_size,
            vision_height=vision_height,
            vision_width=vision_width,
            vision_frames=vision_frames,
            vision_sequence_length=vision_sequence_length,
            add_internal_projection=hasattr(vision_tower, "proj")
            and class_name in {"Llama4VisionModel", "Qwen3OmniMoeVisionModel"},
        )

    projector_owner = owner if hasattr(owner, "vision_proj") else module if hasattr(module, "vision_proj") else None
    if projector_owner is not None:
        flops += _projector_flops_from_attr(projector_owner.vision_proj, output_tokens)

    del root_config, text_config
    return flops


def _estimate_standard_vision_flops(
    config: EasyMLXBaseConfig,
    *,
    vision_batch_size: int,
    vision_height: int | None,
    vision_width: int | None,
    vision_frames: int,
    vision_sequence_length: int | None,
    add_internal_projection: bool,
) -> tuple[int, int]:
    dim = _vision_hidden_size(config)
    depth = _vision_depth(config)
    num_heads = _vision_num_heads(config)
    mlp_hidden = _vision_intermediate_size(config)
    in_channels = _vision_in_channels(config)
    patch_size = _require_positive_int("patch_size", config.patch_size)

    raw_tokens = _vision_sequence_length(
        vision_sequence_length=vision_sequence_length,
        vision_batch_size=vision_batch_size,
        vision_height=vision_height,
        vision_width=vision_width,
        vision_frames=vision_frames,
        patch_size=patch_size,
        temporal_patch_size=1,
    )
    seq_per_item = raw_tokens // vision_batch_size
    total = _vision_patch_embed_2d_flops(
        vision_batch_size=vision_batch_size,
        vision_frames=vision_frames,
        in_channels=in_channels,
        out_channels=dim,
        patch_size=patch_size,
        vision_height=vision_height,
        vision_width=vision_width,
    )

    head_dim = dim // num_heads
    for _ in range(depth):
        total += _linear_flops(raw_tokens, dim, 3 * dim)
        total += _linear_flops(raw_tokens, dim, dim)
        total += 4 * vision_batch_size * num_heads * seq_per_item * seq_per_item * head_dim
        total += _dense_mlp_flops(raw_tokens, dim, mlp_hidden)

    if add_internal_projection:
        output_dim = int(getattr(config, "vision_output_dim", config.out_hidden_size))
        total += _linear_flops(raw_tokens, dim, output_dim)

    return total, raw_tokens


def _estimate_qwen3_5_vision_flops(
    config: EasyMLXBaseConfig,
    *,
    vision_batch_size: int,
    vision_height: int | None,
    vision_width: int | None,
    vision_frames: int,
    vision_sequence_length: int | None,
) -> tuple[int, int]:
    dim = _vision_hidden_size(config)
    depth = _vision_depth(config)
    num_heads = _vision_num_heads(config)
    mlp_hidden = _vision_intermediate_size(config)
    in_channels = _vision_in_channels(config)
    patch_size = _require_positive_int("patch_size", config.patch_size)
    spatial_merge_size = max(1, int(getattr(config, "spatial_merge_size", 2)))
    output_dim = int(getattr(config, "out_hidden_size", dim))

    raw_tokens = _vision_sequence_length(
        vision_sequence_length=vision_sequence_length,
        vision_batch_size=vision_batch_size,
        vision_height=vision_height,
        vision_width=vision_width,
        vision_frames=vision_frames,
        patch_size=patch_size,
        temporal_patch_size=1,
    )
    seq_per_item = raw_tokens // vision_batch_size
    total = _vision_patch_embed_2d_flops(
        vision_batch_size=vision_batch_size,
        vision_frames=vision_frames,
        in_channels=in_channels,
        out_channels=dim,
        patch_size=patch_size,
        vision_height=vision_height,
        vision_width=vision_width,
    )

    head_dim = dim // num_heads
    for _ in range(depth):
        total += _linear_flops(raw_tokens, dim, 3 * dim)
        total += _linear_flops(raw_tokens, dim, dim)
        total += 4 * vision_batch_size * num_heads * seq_per_item * seq_per_item * head_dim
        total += _dense_mlp_flops(raw_tokens, dim, mlp_hidden)

    merge_tokens = raw_tokens // (spatial_merge_size * spatial_merge_size)
    merge_dim = dim * spatial_merge_size * spatial_merge_size
    total += _linear_flops(merge_tokens, merge_dim, merge_dim)
    total += _linear_flops(merge_tokens, merge_dim, output_dim)
    return total, merge_tokens


def _estimate_glm4v_vision_flops(
    config: EasyMLXBaseConfig,
    *,
    vision_batch_size: int,
    vision_height: int | None,
    vision_width: int | None,
    vision_frames: int,
    vision_sequence_length: int | None,
) -> tuple[int, int]:
    hidden_size = _vision_hidden_size(config)
    depth = _vision_depth(config)
    num_heads = _vision_num_heads(config)
    in_channels = _vision_in_channels(config)
    patch_size = _require_positive_int("patch_size", config.patch_size)
    temporal_patch_size = max(1, int(getattr(config, "temporal_patch_size", 1)))
    spatial_merge_size = max(1, int(getattr(config, "spatial_merge_size", 1)))
    out_hidden_size = _require_positive_int("out_hidden_size", config.out_hidden_size)
    merger_hidden = _require_positive_int("intermediate_size", config.intermediate_size)

    raw_tokens = _vision_sequence_length(
        vision_sequence_length=vision_sequence_length,
        vision_batch_size=vision_batch_size,
        vision_height=vision_height,
        vision_width=vision_width,
        vision_frames=vision_frames,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
    )
    seq_per_item = raw_tokens // vision_batch_size

    total = _vision_patch_embed_3d_flops(
        vision_batch_size=vision_batch_size,
        in_channels=in_channels,
        out_channels=hidden_size,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        vision_height=vision_height,
        vision_width=vision_width,
        vision_frames=vision_frames,
    )

    head_dim = hidden_size // num_heads
    for _ in range(depth):
        total += _linear_flops(raw_tokens, hidden_size, 3 * hidden_size)
        total += _linear_flops(raw_tokens, hidden_size, hidden_size)
        total += 4 * vision_batch_size * num_heads * seq_per_item * seq_per_item * head_dim
        total += _gated_mlp_flops(raw_tokens, hidden_size, out_hidden_size)

    merge_tokens = raw_tokens // (spatial_merge_size * spatial_merge_size)
    total += _conv2d_flops(
        output_tokens=merge_tokens,
        in_channels=hidden_size,
        out_channels=out_hidden_size,
        kernel_h=spatial_merge_size,
        kernel_w=spatial_merge_size,
    )
    total += _linear_flops(merge_tokens, out_hidden_size, out_hidden_size)
    total += _gated_mlp_flops(merge_tokens, out_hidden_size, merger_hidden)
    return total, merge_tokens


def _vision_patch_embed_2d_flops(
    *,
    vision_batch_size: int,
    vision_frames: int,
    in_channels: int,
    out_channels: int,
    patch_size: int,
    vision_height: int | None,
    vision_width: int | None,
) -> int:
    if vision_height is None or vision_width is None:
        raise ValueError("vision_height and vision_width are required to derive patch embedding FLOPs.")
    out_h = _ceil_div(vision_height, patch_size)
    out_w = _ceil_div(vision_width, patch_size)
    return 2 * vision_batch_size * vision_frames * out_h * out_w * in_channels * out_channels * patch_size * patch_size


def _vision_patch_embed_3d_flops(
    *,
    vision_batch_size: int,
    in_channels: int,
    out_channels: int,
    patch_size: int,
    temporal_patch_size: int,
    vision_height: int | None,
    vision_width: int | None,
    vision_frames: int,
) -> int:
    if vision_height is None or vision_width is None:
        raise ValueError("vision_height and vision_width are required to derive patch embedding FLOPs.")
    out_h = _ceil_div(vision_height, patch_size)
    out_w = _ceil_div(vision_width, patch_size)
    out_t = _ceil_div(vision_frames, temporal_patch_size)
    return (
        2
        * vision_batch_size
        * out_t
        * out_h
        * out_w
        * in_channels
        * out_channels
        * temporal_patch_size
        * patch_size
        * patch_size
    )


def _vision_sequence_length(
    *,
    vision_sequence_length: int | None,
    vision_batch_size: int,
    vision_height: int | None,
    vision_width: int | None,
    vision_frames: int,
    patch_size: int,
    temporal_patch_size: int,
) -> int:
    if vision_sequence_length is not None:
        return vision_sequence_length * vision_batch_size
    if vision_height is None or vision_width is None:
        raise ValueError("vision_height and vision_width are required when vision_sequence_length is not provided.")
    return (
        vision_batch_size
        * _ceil_div(vision_frames, temporal_patch_size)
        * _ceil_div(vision_height, patch_size)
        * _ceil_div(vision_width, patch_size)
    )


def _attention_flops(
    *,
    tokens: int,
    batch_size: int,
    query_length: int,
    key_length: int,
    hidden_size: int,
    num_query_heads: int,
    query_proj_dim: int,
    key_proj_dim: int,
    value_proj_dim: int,
    output_proj_dim: int,
    qk_head_dim: int,
    value_head_dim: int,
) -> int:
    return (
        _linear_flops(tokens, hidden_size, query_proj_dim)
        + _linear_flops(tokens, hidden_size, key_proj_dim)
        + _linear_flops(tokens, hidden_size, value_proj_dim)
        + _linear_flops(tokens, output_proj_dim, hidden_size)
        + 2 * batch_size * num_query_heads * query_length * key_length * (qk_head_dim + value_head_dim)
    )


def _projector_flops_from_attr(projector: tp.Any, tokens: int) -> int:
    weight = getattr(projector, "weight", None)
    if weight is None or len(weight.shape) != 2:
        return 0
    out_dim, in_dim = int(weight.shape[0]), int(weight.shape[1])
    return _linear_flops(tokens, in_dim, out_dim)


def _should_include_lm_head(module: tp.Any) -> bool:
    if callable(getattr(module, "get_task_head", None)) or hasattr(module, "lm_head"):
        return True

    owner = _find_model_owner(module)
    language_model = getattr(owner, "language_model", None)
    if language_model is not None and hasattr(language_model, "lm_head"):
        return True

    class_name = module.__class__.__name__
    return "ForCausalLM" in class_name or "ForConditionalGeneration" in class_name


def _find_model_owner(module: tp.Any) -> tp.Any:
    if hasattr(module, "vision_tower") or hasattr(module, "language_model"):
        return module
    for attr in ("model", "base_model"):
        child = getattr(module, attr, None)
        if child is not None and (hasattr(child, "vision_tower") or hasattr(child, "language_model")):
            return child
    return module


def _is_mla_text_config(config: EasyMLXBaseConfig) -> bool:
    return hasattr(config, "kv_lora_rank") and hasattr(config, "qk_nope_head_dim")


def _is_qwen3_next_text_config(config: EasyMLXBaseConfig) -> bool:
    return hasattr(config, "linear_num_key_heads") and hasattr(config, "linear_num_value_heads")


def _is_full_attention_layer(config: EasyMLXBaseConfig, layer_idx: int) -> bool:
    if callable(getattr(config, "is_full_attention_layer", None)):
        return bool(config.is_full_attention_layer(layer_idx))
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return True
    return layer_types[layer_idx] != "linear_attention"


def _is_sparse_mlp_layer(config: EasyMLXBaseConfig, layer_idx: int) -> bool:
    if callable(getattr(config, "is_moe_layer", None)):
        return bool(config.is_moe_layer(layer_idx))

    if hasattr(config, "mlp_layer_types"):
        return config.mlp_layer_types[layer_idx] == "sparse"

    if hasattr(config, "moe_layers") and getattr(config, "moe_layers", None) is not None:
        return layer_idx in config.moe_layers

    if hasattr(config, "first_k_dense_replace") and hasattr(config, "n_routed_experts"):
        return bool(config.n_routed_experts) and layer_idx >= int(config.first_k_dense_replace)

    if hasattr(config, "decoder_sparse_step") and hasattr(config, "num_experts"):
        mlp_only_layers = set(getattr(config, "mlp_only_layers", []) or [])
        if layer_idx in mlp_only_layers:
            return False
        return layer_idx % max(int(config.decoder_sparse_step), 1) == 0

    return False


def _attention_window_for_layer(config: EasyMLXBaseConfig, layer_idx: int, sequence_length: int) -> int:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return sequence_length

    layer_type = layer_types[layer_idx]
    if layer_type == "sliding_attention":
        sliding_window = getattr(config, "sliding_window", None)
        return min(sequence_length, int(sliding_window)) if sliding_window is not None else sequence_length
    if layer_type == "chunked_attention":
        chunk_size = getattr(config, "attention_chunk_size", None)
        return min(sequence_length, int(chunk_size)) if chunk_size is not None else sequence_length
    return sequence_length


def _dense_intermediate_size(config: EasyMLXBaseConfig) -> int:
    if hasattr(config, "intermediate_size_mlp"):
        return _require_positive_int("intermediate_size_mlp", config.intermediate_size_mlp)
    return _require_positive_int("intermediate_size", config.intermediate_size)


def _get_hidden_size(config: EasyMLXBaseConfig) -> int:
    return _require_positive_int("hidden_size", config.hidden_size)


def _get_num_kv_heads(config: EasyMLXBaseConfig, fallback: int) -> int:
    value = getattr(config, "num_key_value_heads", None)
    if value is None:
        return fallback
    return _require_positive_int("num_key_value_heads", value)


def _get_head_dim(config: EasyMLXBaseConfig, *, hidden_size: int, num_heads: int) -> int:
    if getattr(config, "head_dim", None) is not None:
        return _require_positive_int("head_dim", config.head_dim)
    if getattr(config, "kv_channels", None) is not None:
        return _require_positive_int("kv_channels", config.kv_channels)
    return hidden_size // num_heads


def _vision_hidden_size(config: EasyMLXBaseConfig) -> int:
    if hasattr(config, "embed_dim"):
        return _require_positive_int("embed_dim", config.embed_dim)
    return _require_positive_int("hidden_size", config.hidden_size)


def _vision_intermediate_size(config: EasyMLXBaseConfig) -> int:
    if hasattr(config, "hidden_size") and hasattr(config, "embed_dim"):
        return _require_positive_int("hidden_size", config.hidden_size)
    return _require_positive_int("intermediate_size", config.intermediate_size)


def _vision_depth(config: EasyMLXBaseConfig) -> int:
    if hasattr(config, "depth"):
        return _require_positive_int("depth", config.depth)
    return _require_positive_int("num_hidden_layers", config.num_hidden_layers)


def _vision_num_heads(config: EasyMLXBaseConfig) -> int:
    if hasattr(config, "num_heads"):
        return _require_positive_int("num_heads", config.num_heads)
    return _require_positive_int("num_attention_heads", config.num_attention_heads)


def _vision_in_channels(config: EasyMLXBaseConfig) -> int:
    if hasattr(config, "in_channels"):
        return _require_positive_int("in_channels", config.in_channels)
    return _require_positive_int("num_channels", config.num_channels)


def _call_finalize(config: tp.Any) -> None:
    finalize = getattr(config, "finalize", None)
    if callable(finalize):
        finalize()


def _require_positive_int(name: str, value: tp.Any) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    numerator = _require_positive_int("numerator", numerator)
    denominator = _require_positive_int("denominator", denominator)
    return math.ceil(numerator / denominator)


def _linear_flops(tokens: int, input_dim: int, output_dim: int) -> int:
    return 2 * int(tokens) * int(input_dim) * int(output_dim)


def _dense_mlp_flops(tokens: int, hidden_size: int, intermediate_size: int) -> int:
    return 4 * int(tokens) * int(hidden_size) * int(intermediate_size)


def _gated_mlp_flops(tokens: int, hidden_size: int, intermediate_size: int) -> int:
    return 6 * int(tokens) * int(hidden_size) * int(intermediate_size)


def _conv2d_flops(*, output_tokens: int, in_channels: int, out_channels: int, kernel_h: int, kernel_w: int) -> int:
    return 2 * int(output_tokens) * int(in_channels) * int(out_channels) * int(kernel_h) * int(kernel_w)
