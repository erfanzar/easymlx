# Copyright 2026 The EASYDEL / EASYMLX Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""DFlash draft-model support for eSurge speculative decoding."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from easymlx.infra.etils import QuantizationSpec
from easymlx.infra.mixins.bridge import (
    _cast_weights,
    _collect_quantized_weight_map,
    _load_weights_file,
    _no_weight_init,
    _prepare_quantization_plan,
    _quantize_weights,
    _resolve_model_path,
    _resolve_weight_files,
    _validate_quantization_kernel,
)
from easymlx.layers.rotary import get_rope

logger = logging.getLogger(__name__)


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    mask_token_id: int = 0
    rope_scaling: dict[str, Any] | None = None
    sliding_window_size: int | None = None

    @classmethod
    def from_json(cls, path: Path, *, sliding_window_size: int | None = None) -> "DFlashConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        dflash = raw.get("dflash_config") or {}
        return cls(
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            intermediate_size=int(raw["intermediate_size"]),
            vocab_size=int(raw["vocab_size"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(raw["rope_theta"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            block_size=int(raw["block_size"]),
            target_layer_ids=tuple(int(idx) for idx in dflash["target_layer_ids"]),
            num_target_layers=int(raw["num_target_layers"]),
            mask_token_id=int(dflash.get("mask_token_id", 0)),
            rope_scaling=raw.get("rope_scaling"),
            sliding_window_size=sliding_window_size,
        )


class _DFlashKVCache:
    """Minimal KV cache needed by the DFlash draft attention block."""

    def __init__(self, *, max_size: int | None = None):
        self.max_size = max_size
        self.offset = 0
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self.offset += int(keys.shape[2])
        if self.max_size is not None and int(self.keys.shape[2]) > self.max_size:
            self.keys = self.keys[:, :, -self.max_size :, :]
            self.values = self.values[:, :, -self.max_size :, :]
        return self.keys, self.values


class DFlashMLP(nn.Module):
    """SwiGLU MLP matching DFlash draft checkpoints."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, x_ctx: mx.array, rope: nn.Module, cache: _DFlashKVCache) -> mx.array:
        batch, length, _ = x.shape
        ctx_length = x_ctx.shape[1]

        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)

        queries = self.q_norm(queries.reshape(batch, length, self.n_heads, -1)).transpose(0, 2, 1, 3)
        ctx_keys = self.k_norm(ctx_keys.reshape(batch, ctx_length, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(batch, ctx_length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(batch, length, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(batch, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = rope(queries, offset=cache.offset + ctx_length)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + ctx_length)

        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)

        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.self_attn = DFlashAttention(config)
        self.mlp = DFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, x_ctx: mx.array, rope: nn.Module, cache: _DFlashKVCache) -> mx.array:
        hidden = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class DFlashDraftModel(nn.Module):
    """DFlash block-diffusion draft model.

    DFlash draft checkpoints do not contain token embeddings or an LM head.
    Those are borrowed from the target model via :meth:`bind`.
    """

    speculative_kind = "dflash"

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = get_rope(
            dims=config.head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.embed_tokens: Any | None = None
        self._target_model: Any | None = None

    @property
    def target_feature_layer_indices(self) -> tuple[int, ...]:
        """Target layer indices in EasyMLX's 1-based hidden-feature convention."""
        return tuple(layer_id + 1 for layer_id in self.config.target_layer_ids)

    def bind(self, target_model: Any) -> "DFlashDraftModel":
        get_embedding = getattr(target_model, "get_embedding", None)
        if callable(get_embedding):
            self.embed_tokens = get_embedding()
        elif hasattr(target_model, "get_input_embeddings") and callable(target_model.get_input_embeddings):
            self.embed_tokens = target_model.get_input_embeddings()
        elif hasattr(target_model, "base_model") and hasattr(target_model.base_model, "embed_tokens"):
            self.embed_tokens = target_model.base_model.embed_tokens
        elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            self.embed_tokens = target_model.model.embed_tokens
        elif hasattr(target_model, "language_model") and hasattr(target_model.language_model, "embed_tokens"):
            self.embed_tokens = target_model.language_model.embed_tokens
        else:
            raise AttributeError(f"Cannot find token embeddings in target model {type(target_model).__name__}.")

        self._target_model = target_model
        return self

    def make_cache(self) -> list[_DFlashKVCache]:
        return [_DFlashKVCache(max_size=self.config.sliding_window_size) for _ in self.layers]

    def __call__(self, inputs: mx.array, target_hidden: mx.array, cache: list[_DFlashKVCache]) -> mx.array:
        if self.embed_tokens is None:
            raise RuntimeError("DFlash draft model must be bound to a target model before use.")
        if self._target_model is None:
            raise RuntimeError("DFlash draft model has no bound target model.")

        hidden = self.embed_tokens(inputs)
        hidden_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, layer_cache in zip(self.layers, cache, strict=False):
            hidden = layer(hidden, hidden_ctx, self.rope, layer_cache)

        hidden = self.norm(hidden)
        compute_lm_logits = getattr(self._target_model, "compute_lm_logits", None)
        if callable(compute_lm_logits):
            return compute_lm_logits(hidden)

        lm_head = getattr(self._target_model, "lm_head", None)
        if callable(lm_head):
            return lm_head(hidden)
        if hasattr(self.embed_tokens, "as_linear"):
            return self.embed_tokens.as_linear(hidden)
        raise AttributeError(f"Cannot find LM head on target model {type(self._target_model).__name__}.")

    def dflash_logits(
        self,
        *,
        first_token: int,
        hidden_states: tuple[mx.array, ...] | mx.array,
        max_tokens: int,
        target_model: Any,
        cache: list[_DFlashKVCache] | None = None,
    ) -> mx.array:
        """Return draft logits for tokens after ``first_token``."""
        if max_tokens <= 0:
            return mx.zeros((0, self.config.vocab_size))
        self.bind(target_model)
        if isinstance(hidden_states, mx.array):
            target_hidden = hidden_states
        else:
            if len(hidden_states) != len(self.config.target_layer_ids):
                raise RuntimeError(
                    "DFlash target feature count mismatch: "
                    f"draft expects {len(self.config.target_layer_ids)}, got {len(hidden_states)}."
                )
            target_hidden = mx.concatenate(list(hidden_states), axis=-1)
        block = mx.array(
            [[int(first_token)] + [int(self.config.mask_token_id)] * int(max_tokens)],
            dtype=mx.int32,
        )
        logits = self(block, target_hidden, cache if cache is not None else self.make_cache())
        return logits[0, -int(max_tokens) :, :]


def _looks_like_dflash_config(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.exists():
        return False
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(raw.get("dflash_config"), dict)


def load_dflash_draft_model(
    pretrained_model_name_or_path: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    weights_name: str | None = None,
    dtype: mx.Dtype | None = None,
    device: mx.Device | None = None,
    lazy: bool = False,
    quantization: QuantizationSpec | None = None,
    sliding_window_size: int | None = None,
) -> DFlashDraftModel:
    """Load a DFlash draft checkpoint without treating it as a causal LM."""
    if sliding_window_size is not None and int(sliding_window_size) <= 0:
        raise ValueError(f"sliding_window_size must be positive or None, got {sliding_window_size}")
    if device is not None:
        mx.set_default_device(device)

    model_path = _resolve_model_path(
        pretrained_model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
    )
    if not _looks_like_dflash_config(model_path):
        raise ValueError(f"{pretrained_model_name_or_path!r} does not look like a DFlash draft checkpoint.")

    config = DFlashConfig.from_json(model_path / "config.json", sliding_window_size=sliding_window_size)
    with _no_weight_init():
        model = DFlashDraftModel(config)

    weight_quantization_map: dict[str, dict[str, int | str]] = {}
    quantization_plan = _prepare_quantization_plan(quantization)
    if quantization_plan is not None:
        logger.info("Quantizing DFlash draft model: %s", quantization_plan.summary)
        for quant_kwargs in quantization_plan.unique_quant_configs:
            _validate_quantization_kernel(quant_kwargs)
        nn.quantize(model, **quantization_plan.nn_quantize_kwargs)
        if quantization_plan.fixed_quant_kwargs is not None:
            weight_quantization_map = _collect_quantized_weight_map(model, quantization_plan.fixed_quant_kwargs)
        else:
            weight_quantization_map = dict(quantization_plan.weight_quantization_map)

    weight_files = _resolve_weight_files(model_path, weights_name=weights_name)
    if not weight_files:
        raise FileNotFoundError(f"No DFlash weights found under {model_path!s}.")

    loaded_keys: set[str] = set()
    for weight_file in weight_files:
        shard = _load_weights_file(weight_file)
        shard = _cast_weights(shard, dtype=dtype)
        if weight_quantization_map:
            shard = _quantize_weights(shard, weight_quantization_map)
        shard_items = list(shard.items())
        loaded_keys.update(k for k, _ in shard_items)
        model.load_weights(shard_items, strict=False)
        if not lazy:
            mx.eval([v for _, v in shard_items])
        del shard, shard_items

    model_keys = {key for key, _ in tree_flatten(model.parameters())}
    missing = model_keys - loaded_keys
    extra = loaded_keys - model_keys
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"Missing keys in DFlash checkpoint: {missing}")
        if extra:
            parts.append(f"Unexpected keys in DFlash checkpoint: {extra}")
        raise ValueError(". ".join(parts))

    if not lazy:
        mx.eval(model.parameters())
    model.eval()
    model.name_or_path = str(model_path)
    return model


__all__ = ("DFlashConfig", "DFlashDraftModel", "load_dflash_draft_model")
