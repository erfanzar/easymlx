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

"""Native EasyMLX implementation of the FLUX.2 klein stack."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.core.fast import scaled_dot_product_attention
from mlx.utils import tree_flatten
from transformers import AutoTokenizer

from easymlx.infra import EasyMLXBaseModule, TaskType
from easymlx.infra.factory import register_module
from easymlx.modules.qwen3 import Qwen3ForCausalLM
from easymlx.utils.hf_composite import resolve_hf_composite_repo, resolve_local_hf_path

from .flux2_klein_configuration import (
    AutoencoderKLFlux2Config,
    FlowMatchEulerDiscreteSchedulerConfig,
    Flux2KleinConfig,
    Flux2TransformerConfig,
)

try:
    _ROTARY_METAL_KERNEL = (
        mx.fast.metal_kernel(
            name="flux2_rotary_apply",
            input_names=["x", "cos", "sin", "params"],
            output_names=["out"],
            source="""
                uint idx = thread_position_in_grid.x;
                uint total = params[0];
                if (idx >= total) return;
                uint dim = params[1];
                uint heads = params[2];
                uint seq_len = params[3];
                uint d = idx % dim;
                uint tmp = idx / dim;
                uint t = (tmp / heads) % seq_len;
                uint mate = (d & 1) ? (idx - 1) : (idx + 1);
                T xv = x[idx];
                T xm = x[mate];
                T cv = cos[t * dim + d];
                T sv = sin[t * dim + d];
                out[idx] = (d & 1) ? (xv * cv + xm * sv) : (xv * cv - xm * sv);
            """,
        )
        if mx.metal.is_available()
        else None
    )
except Exception:
    _ROTARY_METAL_KERNEL = None

_ROTARY_METAL_DISABLED = _ROTARY_METAL_KERNEL is None
_ROTARY_METAL_WARMED = False


def _to_nhwc(x: mx.array, channels: int | None = None) -> mx.array:
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape={x.shape}.")
    if channels is not None and x.shape[-1] == channels:
        return x
    if channels is not None and x.shape[1] == channels:
        return x.transpose(0, 2, 3, 1)
    if x.shape[-1] <= 4 and x.shape[1] > 4:
        return x
    return x.transpose(0, 2, 3, 1)


def _to_nchw(x: mx.array) -> mx.array:
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape={x.shape}.")
    return x.transpose(0, 3, 1, 2)


def _conv_nchw(module: nn.Module, x: mx.array) -> mx.array:
    return _to_nchw(module(_to_nhwc(x)))


def _identity_array(x: mx.array) -> mx.array:
    return x


class _Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class _SiLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(x)


def _repeat_interleave_last(x: mx.array, repeats: int) -> mx.array:
    return mx.repeat(x, repeats, axis=-1)


def _get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be 1D.")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = timesteps.astype(mx.float32)[:, None] * mx.exp(exponent)[None, :]
    emb = scale * emb
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, ((0, 0), (0, 1)))
    return emb


def _get_1d_rotary_pos_embed(
    dim: int,
    pos: mx.array,
    *,
    theta: float = 10000.0,
) -> tuple[mx.array, mx.array]:
    if dim % 2 != 0:
        raise ValueError(f"Rotary dim must be even, got {dim}.")
    pos = pos.astype(mx.float32)
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    freqs = pos[..., None] * freqs[None, :]
    return _repeat_interleave_last(mx.cos(freqs), 2), _repeat_interleave_last(mx.sin(freqs), 2)


def _apply_rotary_emb(
    x: mx.array,
    rotary_emb: tuple[mx.array, mx.array],
    *,
    sequence_dim: int = 1,
) -> mx.array:
    global _ROTARY_METAL_DISABLED
    cos, sin = rotary_emb
    if sequence_dim != 1:
        raise ValueError(f"Only sequence_dim=1 is supported, got {sequence_dim}.")
    if not _ROTARY_METAL_DISABLED and _ROTARY_METAL_KERNEL is not None and x.ndim == 4:
        try:
            batch, seq_len, heads, dim = x.shape
            total = batch * seq_len * heads * dim
            params = mx.array([total, dim, heads, seq_len], dtype=mx.uint32)
            return _ROTARY_METAL_KERNEL(
                inputs=[
                    x.reshape((-1,)),
                    cos.reshape((-1,)).astype(x.dtype),
                    sin.reshape((-1,)).astype(x.dtype),
                    params,
                ],
                template=[("T", x.dtype)],
                output_shapes=[(total,)],
                output_dtypes=[x.dtype],
                grid=(total, 1, 1),
                threadgroup=(256, 1, 1),
            )[0].reshape(x.shape)
        except Exception:
            _ROTARY_METAL_DISABLED = True
    cos = cos[None, :, None, :].astype(x.dtype)
    sin = sin[None, :, None, :].astype(x.dtype)
    x_pair = x.reshape(*x.shape[:-1], -1, 2)
    x_real = x_pair[..., 0]
    x_imag = x_pair[..., 1]
    x_rot = mx.stack([-x_imag, x_real], axis=-1).reshape(x.shape)
    return x * cos + x_rot * sin


def _warmup_rotary_metal_kernel() -> None:
    global _ROTARY_METAL_DISABLED, _ROTARY_METAL_WARMED
    if _ROTARY_METAL_DISABLED or _ROTARY_METAL_WARMED or _ROTARY_METAL_KERNEL is None:
        return
    try:
        warm_x = mx.zeros((1, 2, 1, 2), dtype=mx.float16)
        warm_cos = mx.ones((2, 2), dtype=mx.float16)
        warm_sin = mx.zeros((2, 2), dtype=mx.float16)
        mx.eval(_apply_rotary_emb(warm_x, (warm_cos, warm_sin)))
        _ROTARY_METAL_WARMED = True
    except Exception:
        _ROTARY_METAL_DISABLED = True


def _cartesian_prod(*arrays: np.ndarray) -> np.ndarray:
    grids = np.meshgrid(*arrays, indexing="ij")
    return np.stack([grid.reshape(-1) for grid in grids], axis=-1)


def _ensure_numpy_image(image: Any) -> np.ndarray:
    if isinstance(image, mx.array):
        arr = np.array(image)
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for PIL image inputs.") from exc

        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape={arr.shape}.")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


@dataclass(slots=True)
class Flux2Transformer2DModelOutput:
    sample: mx.array


@dataclass(slots=True)
class DecoderOutput:
    sample: mx.array


@dataclass(slots=True)
class AutoencoderKLOutput:
    latent_dist: "DiagonalGaussianDistribution"


@dataclass(slots=True)
class FlowMatchEulerDiscreteSchedulerOutput:
    prev_sample: mx.array


@dataclass(slots=True)
class Flux2KleinPipelineOutput:
    images: Any


class Flux2SwiGLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        hidden_dim = x.shape[-1] // 2
        x1 = x[..., :hidden_dim]
        x2 = x[..., hidden_dim:]
        return nn.silu(x1) * x2


class Flux2FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_out: int | None = None,
        mult: float = 3.0,
        inner_dim: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.inner_dim = int(dim * mult) if inner_dim is None else int(inner_dim)
        self.linear_in = nn.Linear(dim, self.inner_dim * 2, bias=bias)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = nn.Linear(self.inner_dim, dim if dim_out is None else dim_out, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_out(self.act_fn(self.linear_in(x)))


class Flux2Attention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        heads: int,
        dim_head: int,
        added_kv_proj_dim: int | None = None,
        out_dim: int | None = None,
        eps: float = 1e-5,
        bias: bool = False,
        added_proj_bias: bool = True,
        out_bias: bool = True,
    ):
        super().__init__()
        self.head_dim = int(dim_head)
        self.query_dim = int(query_dim)
        self.inner_dim = out_dim or (heads * dim_head)
        self.out_dim = out_dim or query_dim
        self.heads = self.inner_dim // self.head_dim
        self.scale = self.head_dim**-0.5
        self.added_kv_proj_dim = added_kv_proj_dim

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        self.to_out = [nn.Linear(self.inner_dim, self.out_dim, bias=out_bias), _Identity()]

        if added_kv_proj_dim is not None:
            self.norm_added_q = nn.RMSNorm(self.head_dim, eps=eps)
            self.norm_added_k = nn.RMSNorm(self.head_dim, eps=eps)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=out_bias)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        encoder_hidden_states: mx.array | None = None,
        image_rotary_emb: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array | tuple[mx.array, mx.array]:
        batch, seq_len = hidden_states.shape[:2]

        query = self.to_q(hidden_states).reshape(batch, seq_len, self.heads, self.head_dim)
        key = self.to_k(hidden_states).reshape(batch, seq_len, self.heads, self.head_dim)
        value = self.to_v(hidden_states).reshape(batch, seq_len, self.heads, self.head_dim)
        query = self.norm_q(query)
        key = self.norm_k(key)

        context_len = 0
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            context_len = encoder_hidden_states.shape[1]
            encoder_query = self.add_q_proj(encoder_hidden_states).reshape(batch, context_len, self.heads, self.head_dim)
            encoder_key = self.add_k_proj(encoder_hidden_states).reshape(batch, context_len, self.heads, self.head_dim)
            encoder_value = self.add_v_proj(encoder_hidden_states).reshape(batch, context_len, self.heads, self.head_dim)
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)
            query = mx.concatenate([encoder_query, query], axis=1)
            key = mx.concatenate([encoder_key, key], axis=1)
            value = mx.concatenate([encoder_value, value], axis=1)

        if image_rotary_emb is not None:
            query = _apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = _apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        attn = scaled_dot_product_attention(
            query.transpose(0, 2, 1, 3),
            key.transpose(0, 2, 1, 3),
            value.transpose(0, 2, 1, 3),
            scale=self.scale,
            mask=None,
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, query.shape[1], self.inner_dim).astype(hidden_states.dtype)

        if context_len:
            encoder_attn = self.to_add_out(attn[:, :context_len])
            image_attn = self.to_out[0](attn[:, context_len:])
            image_attn = self.to_out[1](image_attn)
            return image_attn, encoder_attn

        attn = self.to_out[0](attn)
        return self.to_out[1](attn)


class Flux2ParallelSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        heads: int,
        dim_head: int,
        out_dim: int | None = None,
        eps: float = 1e-5,
        bias: bool = False,
        out_bias: bool = True,
        mlp_ratio: float = 4.0,
        mlp_mult_factor: int = 2,
    ):
        super().__init__()
        self.head_dim = int(dim_head)
        self.query_dim = int(query_dim)
        self.inner_dim = out_dim or (heads * dim_head)
        self.out_dim = out_dim or query_dim
        self.heads = self.inner_dim // self.head_dim
        self.scale = self.head_dim**-0.5
        self.mlp_ratio = float(mlp_ratio)
        self.mlp_hidden_dim = int(query_dim * self.mlp_ratio)
        self.mlp_mult_factor = int(mlp_mult_factor)

        self.to_qkv_mlp_proj = nn.Linear(
            self.query_dim,
            self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=bias,
        )
        self.mlp_act_fn = Flux2SwiGLU()
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.Linear(self.inner_dim + self.mlp_hidden_dim, self.out_dim, bias=out_bias)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        image_rotary_emb: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch, seq_len = hidden_states.shape[:2]
        projected = self.to_qkv_mlp_proj(hidden_states)
        qkv_dim = self.inner_dim * 3
        qkv = projected[..., :qkv_dim]
        mlp_hidden_states = projected[..., qkv_dim:]

        query = qkv[..., : self.inner_dim].reshape(batch, seq_len, self.heads, self.head_dim)
        key = qkv[..., self.inner_dim : 2 * self.inner_dim].reshape(batch, seq_len, self.heads, self.head_dim)
        value = qkv[..., 2 * self.inner_dim :].reshape(batch, seq_len, self.heads, self.head_dim)

        query = self.norm_q(query)
        key = self.norm_k(key)
        if image_rotary_emb is not None:
            query = _apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = _apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        attn = scaled_dot_product_attention(
            query.transpose(0, 2, 1, 3),
            key.transpose(0, 2, 1, 3),
            value.transpose(0, 2, 1, 3),
            scale=self.scale,
            mask=None,
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.inner_dim).astype(hidden_states.dtype)

        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)
        return self.to_out(mx.concatenate([attn, mlp_hidden_states], axis=-1))


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, *, mod_param_sets: int = 2, bias: bool = False):
        super().__init__()
        self.mod_param_sets = int(mod_param_sets)
        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)
        self.act_fn = _SiLU()

    def __call__(self, temb: mx.array) -> mx.array:
        return self.linear(self.act_fn(temb))

    @staticmethod
    def split(mod: mx.array, mod_param_sets: int) -> tuple[tuple[mx.array, mx.array, mx.array], ...]:
        if mod.ndim == 2:
            mod = mod[:, None, :]
        step = mod.shape[-1] // (3 * mod_param_sets)
        chunks = [mod[..., i * step : (i + 1) * step] for i in range(3 * mod_param_sets)]
        return tuple((chunks[3 * i], chunks[3 * i + 1], chunks[3 * i + 2]) for i in range(mod_param_sets))


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, affine=False, bias=False)
        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            eps=eps,
            bias=bias,
            out_bias=bias,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        encoder_hidden_states: mx.array | None,
        temb_mod: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array | tuple[mx.array, mx.array]:
        text_seq_len = None
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
        attn_output = self.attn(norm_hidden_states, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + mod_gate * attn_output

        if text_seq_len is None:
            return hidden_states
        return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]


class Flux2TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps, affine=False, bias=False)
        self.norm1_context = nn.LayerNorm(dim, eps=eps, affine=False, bias=False)
        self.attn = Flux2Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
        )
        self.norm2 = nn.LayerNorm(dim, eps=eps, affine=False, bias=False)
        self.ff = Flux2FeedForward(dim, dim_out=dim, mult=mlp_ratio, bias=bias)
        self.norm2_context = nn.LayerNorm(dim, eps=eps, affine=False, bias=False)
        self.ff_context = Flux2FeedForward(dim, dim_out=dim, mult=mlp_ratio, bias=bias)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        encoder_hidden_states: mx.array,
        temb_mod_img: mx.array,
        temb_mod_txt: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, mx.array]:
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(temb_mod_img, 2)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(
            temb_mod_txt, 2
        )

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        attn_output, context_attn_output = self.attn(
            norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (1 + scale_mlp) * norm_hidden_states + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_mlp) * norm_encoder_hidden_states + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(norm_encoder_hidden_states)

        return encoder_hidden_states, hidden_states


class Flux2PosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: tuple[int, ...] | list[int]):
        super().__init__()
        self.theta = int(theta)
        self.axes_dim = tuple(int(v) for v in axes_dim)

    def __call__(self, ids: mx.array) -> tuple[mx.array, mx.array]:
        cos_out: list[mx.array] = []
        sin_out: list[mx.array] = []
        pos = ids.astype(mx.float32)
        for axis_index, axis_dim in enumerate(self.axes_dim):
            cos, sin = _get_1d_rotary_pos_embed(axis_dim, pos[..., axis_index], theta=self.theta)
            cos_out.append(cos)
            sin_out.append(sin)
        return mx.concatenate(cos_out, axis=-1), mx.concatenate(sin_out, axis=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, *, sample_proj_bias: bool = True):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=sample_proj_bias)
        self.act = _SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=sample_proj_bias)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, *, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = int(num_channels)
        self.flip_sin_to_cos = bool(flip_sin_to_cos)
        self.downscale_freq_shift = float(downscale_freq_shift)
        self.scale = int(scale)

    def __call__(self, timesteps: mx.array) -> mx.array:
        return _get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        guidance_embeds: bool = True,
    ):
        super().__init__()
        self.time_proj = Timesteps(in_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels, embedding_dim, sample_proj_bias=bias)
        self.guidance_embedder = (
            TimestepEmbedding(in_channels, embedding_dim, sample_proj_bias=bias) if guidance_embeds else None
        )

    def __call__(self, timestep: mx.array, guidance: mx.array | None) -> mx.array:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.astype(timestep.dtype))
        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.astype(guidance.dtype))
            return timesteps_emb + guidance_emb
        return timesteps_emb


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        *,
        elementwise_affine: bool = False,
        eps: float = 1e-5,
        bias: bool = False,
    ):
        super().__init__()
        self.silu = _SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, affine=elementwise_affine, bias=bias)

    def __call__(self, x: mx.array, conditioning_embedding: mx.array) -> mx.array:
        emb = self.linear(self.silu(conditioning_embedding).astype(x.dtype))
        scale, shift = mx.split(emb, 2, axis=-1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]


@register_module(task_type=TaskType.BASE_MODULE, config=Flux2TransformerConfig, model_type="flux2_transformer")
class Flux2Transformer2DModel(EasyMLXBaseModule):
    """FLUX.2 image transformer."""

    config_class = Flux2TransformerConfig

    def __init__(self, config: Flux2TransformerConfig):
        super().__init__(config)
        self.out_channels = config.out_channels or config.in_channels
        self.inner_dim = config.num_attention_heads * config.attention_head_dim
        self.pos_embed = Flux2PosEmbed(theta=config.rope_theta, axes_dim=config.axes_dims_rope)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=config.timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=config.guidance_embeds,
        )
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)
        self.x_embedder = nn.Linear(config.in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(config.joint_attention_dim, self.inner_dim, bias=False)
        self.transformer_blocks = [
            Flux2TransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                mlp_ratio=config.mlp_ratio,
                eps=config.eps,
                bias=False,
            )
            for _ in range(config.num_layers)
        ]
        self.single_transformer_blocks = [
            Flux2SingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                mlp_ratio=config.mlp_ratio,
                eps=config.eps,
                bias=False,
            )
            for _ in range(config.num_single_layers)
        ]
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=config.eps,
            bias=False,
        )
        self.proj_out = nn.Linear(
            self.inner_dim,
            config.patch_size * config.patch_size * self.out_channels,
            bias=False,
        )

    def _run_blocks(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        double_stream_mod_img: mx.array,
        double_stream_mod_txt: mx.array,
        single_stream_mod: mx.array,
        cos_rope: mx.array,
        sin_rope: mx.array,
    ) -> mx.array:
        concat_rotary_emb = (cos_rope, sin_rope)
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
            )
        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
            )
        return hidden_states

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        img_ids: mx.array,
        txt_ids: mx.array,
        guidance: mx.array | None = None,
        return_dict: bool = True,
        image_rotary_emb: tuple[mx.array, mx.array] | None = None,
        _blocks_fn: Any = None,
        _encoder_embedded: bool = False,
        **_unused: Any,
    ) -> Flux2Transformer2DModelOutput | tuple[mx.array]:
        num_txt_tokens = encoder_hidden_states.shape[1]

        timestep = timestep.astype(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.astype(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)
        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        hidden_states = self.x_embedder(hidden_states)
        if not _encoder_embedded:
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if image_rotary_emb is None:
            if img_ids.ndim == 3:
                img_ids = img_ids[0]
            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            img_rope = self.pos_embed(img_ids)
            txt_rope = self.pos_embed(txt_ids)
            cos_rope = mx.concatenate([txt_rope[0], img_rope[0]], axis=0)
            sin_rope = mx.concatenate([txt_rope[1], img_rope[1]], axis=0)
        else:
            cos_rope, sin_rope = image_rotary_emb

        run_blocks = _blocks_fn if _blocks_fn is not None else self._run_blocks
        hidden_states = run_blocks(
            hidden_states,
            encoder_hidden_states,
            double_stream_mod_img,
            double_stream_mod_txt,
            single_stream_mod,
            cos_rope,
            sin_rope,
        )

        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Flux2Transformer2DModelOutput(sample=output)


class Flux2ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        temb_channels: int | None = None,
        groups: int = 32,
        groups_out: int | None = None,
        eps: float = 1e-6,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        groups_out = groups if groups_out is None else groups_out
        self.output_scale_factor = float(output_scale_factor)
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = None if temb_channels is None else nn.Linear(temb_channels, out_channels, bias=True)
        self.norm2 = nn.GroupNorm(groups_out, out_channels, eps=eps, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def __call__(self, input_tensor: mx.array, *, temb: mx.array | None = None) -> mx.array:
        hidden_states = nn.silu(self.norm1(input_tensor))
        hidden_states = self.conv1(hidden_states)

        if temb is not None and self.time_emb_proj is not None:
            hidden_states = hidden_states + self.time_emb_proj(nn.silu(temb))[:, None, None, :]

        hidden_states = nn.silu(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        out = input_tensor + hidden_states
        if self.output_scale_factor != 1.0:
            out = out / self.output_scale_factor
        return out


class Downsample2D(nn.Module):
    def __init__(self, channels: int, *, out_channels: int | None = None, padding: int = 1):
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels or channels)
        self.padding = int(padding)
        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, stride=2, padding=padding, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self.padding == 0:
            hidden_states = mx.pad(hidden_states, ((0, 0), (0, 1), (0, 1), (0, 0)))
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(self, channels: int, *, out_channels: int | None = None):
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels or channels)
        self.interp = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.conv(self.interp(hidden_states))


class Flux2SpatialAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        eps: float = 1e-6,
        norm_num_groups: int = 32,
        rescale_output_factor: float = 1.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.heads = 1
        self.dim_head = self.channels
        self.scale = self.dim_head**-0.5
        self.rescale_output_factor = float(rescale_output_factor)
        self.group_norm = nn.GroupNorm(norm_num_groups, self.channels, eps=eps, pytorch_compatible=True)
        self.to_q = nn.Linear(self.channels, self.channels, bias=True)
        self.to_k = nn.Linear(self.channels, self.channels, bias=True)
        self.to_v = nn.Linear(self.channels, self.channels, bias=True)
        self.to_out = [nn.Linear(self.channels, self.channels, bias=True), _Identity()]

    def __call__(self, hidden_states: mx.array, *, temb: mx.array | None = None) -> mx.array:
        del temb
        residual = hidden_states
        batch, height, width, channels = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.reshape(batch, height * width, channels)

        query = self.to_q(hidden_states).reshape(batch, height * width, self.heads, self.dim_head)
        key = self.to_k(hidden_states).reshape(batch, height * width, self.heads, self.dim_head)
        value = self.to_v(hidden_states).reshape(batch, height * width, self.heads, self.dim_head)

        attn = scaled_dot_product_attention(
            query.astype(mx.float32).transpose(0, 2, 1, 3),
            key.astype(mx.float32).transpose(0, 2, 1, 3),
            value.astype(mx.float32).transpose(0, 2, 1, 3),
            scale=self.scale,
            mask=None,
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, height * width, channels).astype(hidden_states.dtype)
        attn = self.to_out[0](attn)
        attn = self.to_out[1](attn).reshape(batch, height, width, channels)
        out = attn + residual
        if self.rescale_output_factor != 1.0:
            out = out / self.rescale_output_factor
        return out


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        temb_channels: int | None,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        add_attention: bool = True,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        self.resnets = [
            Flux2ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
                eps=resnet_eps,
                output_scale_factor=output_scale_factor,
            ),
            Flux2ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
                eps=resnet_eps,
                output_scale_factor=output_scale_factor,
            ),
        ]
        self.attentions = (
            [Flux2SpatialAttention(in_channels, eps=resnet_eps, norm_num_groups=resnet_groups)]
            if add_attention
            else [None]
        )

    def __call__(self, hidden_states: mx.array, temb: mx.array | None = None) -> mx.array:
        hidden_states = self.resnets[0](hidden_states, temb=temb)
        attn = self.attentions[0]
        if attn is not None:
            hidden_states = attn(hidden_states, temb=temb)
        return self.resnets[1](hidden_states, temb=temb)


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.resnets = [
            Flux2ResnetBlock2D(
                in_channels=in_channels if index == 0 else out_channels,
                out_channels=out_channels,
                temb_channels=None,
                groups=resnet_groups,
                eps=resnet_eps,
            )
            for index in range(num_layers)
        ]
        self.downsamplers = (
            [Downsample2D(out_channels, out_channels=out_channels, padding=0)] if add_downsample else None
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        add_upsample: bool = True,
        temb_channels: int | None = None,
    ):
        super().__init__()
        self.resnets = [
            Flux2ResnetBlock2D(
                in_channels=in_channels if index == 0 else out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
                eps=resnet_eps,
            )
            for index in range(num_layers)
        ]
        self.upsamplers = [Upsample2D(out_channels, out_channels=out_channels)] if add_upsample else None

    def __call__(self, hidden_states: mx.array, temb: mx.array | None = None) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class DiagonalGaussianDistribution:
    def __init__(self, parameters: mx.array, deterministic: bool = False):
        self.parameters = parameters
        channels = parameters.shape[1] // 2
        self.mean = parameters[:, :channels]
        self.logvar = mx.clip(parameters[:, channels:], -30.0, 20.0)
        self.deterministic = bool(deterministic)
        self.std = mx.exp(0.5 * self.logvar)
        self.var = mx.exp(self.logvar)
        if self.deterministic:
            self.std = mx.zeros_like(self.mean)
            self.var = mx.zeros_like(self.mean)

    def sample(self) -> mx.array:
        noise = mx.random.normal(shape=self.mean.shape).astype(self.mean.dtype)
        return self.mean + self.std * noise

    def mode(self) -> mx.array:
        return self.mean


class Flux2Encoder(nn.Module):
    def __init__(self, config: AutoencoderKLFlux2Config):
        super().__init__()
        self.conv_in = nn.Conv2d(config.in_channels, config.block_out_channels[0], kernel_size=3, stride=1, padding=1)
        self.down_blocks = []
        output_channel = config.block_out_channels[0]
        for index, _block_type in enumerate(config.down_block_types):
            input_channel = output_channel
            output_channel = config.block_out_channels[index]
            is_final_block = index == len(config.block_out_channels) - 1
            self.down_blocks.append(
                DownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=config.layers_per_block,
                    resnet_eps=1e-6,
                    resnet_groups=config.norm_num_groups,
                    add_downsample=not is_final_block,
                )
            )
        self.mid_block = UNetMidBlock2D(
            in_channels=config.block_out_channels[-1],
            temb_channels=None,
            resnet_eps=1e-6,
            resnet_groups=config.norm_num_groups,
            add_attention=config.mid_block_add_attention,
            output_scale_factor=1.0,
        )
        self.conv_norm_out = nn.GroupNorm(
            config.norm_num_groups,
            config.block_out_channels[-1],
            eps=1e-6,
            pytorch_compatible=True,
        )
        self.conv_act = _SiLU()
        self.conv_out = nn.Conv2d(
            config.block_out_channels[-1],
            2 * config.latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(self, sample: mx.array) -> mx.array:
        hidden_states = _to_nhwc(sample, channels=sample.shape[1] if sample.shape[1] <= 4 else None)
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        hidden_states = self.mid_block(hidden_states)
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return _to_nchw(hidden_states)


class Flux2Decoder(nn.Module):
    def __init__(self, config: AutoencoderKLFlux2Config):
        super().__init__()
        block_out_channels = config.decoder_block_out_channels or config.block_out_channels
        self.conv_in = nn.Conv2d(config.latent_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            resnet_eps=1e-6,
            resnet_groups=config.norm_num_groups,
            add_attention=config.mid_block_add_attention,
            output_scale_factor=1.0,
        )
        reversed_channels = list(reversed(block_out_channels))
        self.up_blocks = []
        output_channel = reversed_channels[0]
        for index, _block_type in enumerate(config.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_channels[index]
            is_final_block = index == len(block_out_channels) - 1
            self.up_blocks.append(
                UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    num_layers=config.layers_per_block + 1,
                    resnet_eps=1e-6,
                    resnet_groups=config.norm_num_groups,
                    add_upsample=not is_final_block,
                    temb_channels=None,
                )
            )
        self.conv_norm_out = nn.GroupNorm(
            config.norm_num_groups,
            block_out_channels[0],
            eps=1e-6,
            pytorch_compatible=True,
        )
        self.conv_act = _SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], config.out_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, sample: mx.array, latent_embeds: mx.array | None = None) -> mx.array:
        del latent_embeds
        hidden_states = _to_nhwc(sample, channels=sample.shape[1] if sample.shape[1] > 4 else None)
        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.mid_block(hidden_states, None)
        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states, None)
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return _to_nchw(hidden_states)


@register_module(task_type=TaskType.BASE_MODULE, config=AutoencoderKLFlux2Config, model_type="autoencoder_kl_flux2")
class AutoencoderKLFlux2(EasyMLXBaseModule):
    """FLUX.2 VAE."""

    config_class = AutoencoderKLFlux2Config

    def __init__(self, config: AutoencoderKLFlux2Config):
        super().__init__(config)
        self.encoder = Flux2Encoder(config)
        self.decoder = Flux2Decoder(config)
        self.quant_conv = (
            nn.Conv2d(2 * config.latent_channels, 2 * config.latent_channels, kernel_size=1)
            if config.use_quant_conv
            else None
        )
        self.post_quant_conv = (
            nn.Conv2d(config.latent_channels, config.latent_channels, kernel_size=1)
            if config.use_post_quant_conv
            else None
        )
        self.bn = nn.BatchNorm(
            math.prod(config.patch_size) * config.latent_channels,
            eps=config.batch_norm_eps,
            momentum=config.batch_norm_momentum,
            affine=False,
            track_running_stats=True,
        )
        self.use_slicing = False
        self.use_tiling = False
        self.tile_sample_min_size = config.sample_size
        self.tile_latent_min_size = int(config.sample_size / (2 ** (len(config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

    def _encode(self, x: mx.array) -> mx.array:
        hidden_states = self.encoder(x)
        if self.quant_conv is not None:
            hidden_states = _conv_nchw(self.quant_conv, hidden_states)
        return hidden_states

    def encode(
        self, x: mx.array, *, return_dict: bool = True
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        posterior = DiagonalGaussianDistribution(self._encode(x))
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: mx.array, *, return_dict: bool = True) -> DecoderOutput | tuple[mx.array]:
        if self.post_quant_conv is not None:
            z = _conv_nchw(self.post_quant_conv, z)
        decoded = self.decoder(z)
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def decode(self, z: mx.array, *, return_dict: bool = True) -> DecoderOutput | tuple[mx.array]:
        if not return_dict:
            return self._decode(z, return_dict=False)
        return self._decode(z, return_dict=True)

    def __call__(
        self,
        sample: mx.array,
        *,
        sample_posterior: bool = False,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple[mx.array]:
        posterior = self.encode(sample).latent_dist
        latents = posterior.sample() if sample_posterior else posterior.mode()
        decoded = self.decode(latents, return_dict=False)[0]
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        expected_shapes = {key: value.shape for key, value in tree_flatten(self.parameters())}
        sanitized: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.endswith("num_batches_tracked"):
                continue
            if key.endswith(".weight") and value.ndim == 4:
                expected_shape = expected_shapes.get(key)
                transposed = value.transpose(0, 2, 3, 1)
                if expected_shape == value.shape:
                    sanitized[key] = value
                elif expected_shape == transposed.shape:
                    sanitized[key] = transposed
                else:
                    sanitized[key] = value
                continue
            sanitized[key] = value
        return sanitized


class FlowMatchEulerDiscreteScheduler:
    """MLX port of the FLUX.2 flow-match Euler scheduler."""

    config_class = FlowMatchEulerDiscreteSchedulerConfig
    order = 1

    def __init__(self, config: FlowMatchEulerDiscreteSchedulerConfig):
        self.config = config
        timesteps = np.linspace(1, config.num_train_timesteps, config.num_train_timesteps, dtype=np.float32)[::-1].copy()
        sigmas = timesteps / config.num_train_timesteps
        if not config.use_dynamic_shifting:
            sigmas = config.shift * sigmas / (1 + (config.shift - 1) * sigmas)
        self.timesteps = mx.array(sigmas * config.num_train_timesteps, dtype=mx.float32)
        self.sigmas = mx.array(sigmas, dtype=mx.float32)
        self.sigma_min = float(sigmas[-1])
        self.sigma_max = float(sigmas[0])
        self._shift = float(config.shift)
        self._step_index: int | None = None
        self._begin_index: int | None = None
        self.num_inference_steps: int | None = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> "FlowMatchEulerDiscreteScheduler":
        root = resolve_local_hf_path(pretrained_model_name_or_path)
        if root.exists():
            base = root / subfolder if subfolder else root
            config_path = base / "scheduler_config.json"
            if not config_path.exists():
                config_path = base / "config.json"
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            from huggingface_hub import hf_hub_download

            filename = f"{subfolder}/scheduler_config.json" if subfolder else "scheduler_config.json"
            try:
                config_path = hf_hub_download(
                    repo_id=str(pretrained_model_name_or_path),
                    filename=filename,
                    revision=revision,
                    local_files_only=local_files_only,
                    repo_type="model",
                )
            except Exception:
                filename = f"{subfolder}/config.json" if subfolder else "config.json"
                config_path = hf_hub_download(
                    repo_id=str(pretrained_model_name_or_path),
                    filename=filename,
                    revision=revision,
                    local_files_only=local_files_only,
                    repo_type="model",
                )
            payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return cls(cls.config_class(**payload))

    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "scheduler_config.json").write_text(self.config.to_json_string(use_diff=False), encoding="utf-8")

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = int(begin_index)

    def set_shift(self, shift: float) -> None:
        self._shift = float(shift)

    def _sigma_to_t(self, sigma: float | np.ndarray) -> float | np.ndarray:
        return sigma * self.config.num_train_timesteps

    def _time_shift_exponential(self, mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
        return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)

    def _time_shift_linear(self, mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
        return mu / (mu + (1.0 / t - 1.0) ** sigma)

    def time_shift(self, mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
        if self.config.time_shift_type == "linear":
            return self._time_shift_linear(mu, sigma, t)
        return self._time_shift_exponential(mu, sigma, t)

    def stretch_shift_to_terminal(self, t: np.ndarray) -> np.ndarray:
        one_minus_z = 1.0 - t
        scale_factor = one_minus_z[-1] / (1.0 - float(self.config.shift_terminal))
        return 1.0 - (one_minus_z / scale_factor)

    def _convert_to_karras(self, in_sigmas: np.ndarray, num_inference_steps: int) -> np.ndarray:
        sigma_min = float(in_sigmas[-1])
        sigma_max = float(in_sigmas[0])
        rho = 7.0
        ramp = np.linspace(0.0, 1.0, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    def _convert_to_exponential(self, in_sigmas: np.ndarray, num_inference_steps: int) -> np.ndarray:
        sigma_min = float(in_sigmas[-1])
        sigma_max = float(in_sigmas[0])
        return np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        *,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
    ) -> None:
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be provided when dynamic shifting is enabled.")

        if sigmas is not None and timesteps is not None:
            raise ValueError("Only one of `sigmas` or `timesteps` can be set.")

        if num_inference_steps is None:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps or [])
        self.num_inference_steps = int(num_inference_steps)

        if timesteps is not None:
            timestep_values = np.array(timesteps, dtype=np.float32)
        else:
            timestep_values = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

        if sigmas is None:
            sigma_values = timestep_values / self.config.num_train_timesteps
        else:
            sigma_values = np.array(sigmas, dtype=np.float32)
            timestep_values = sigma_values * self.config.num_train_timesteps

        if self.config.use_dynamic_shifting:
            sigma_values = self.time_shift(float(mu), 1.0, sigma_values)
        else:
            sigma_values = self.shift * sigma_values / (1 + (self.shift - 1) * sigma_values)

        if self.config.shift_terminal is not None:
            sigma_values = self.stretch_shift_to_terminal(sigma_values)
        if self.config.use_karras_sigmas:
            sigma_values = self._convert_to_karras(sigma_values, num_inference_steps)
        elif self.config.use_exponential_sigmas:
            sigma_values = self._convert_to_exponential(sigma_values, num_inference_steps)

        sigmas_arr = mx.array(sigma_values, dtype=mx.float32)
        timesteps_arr = mx.array(
            timestep_values if timesteps is not None else sigma_values * self.config.num_train_timesteps,
            dtype=mx.float32,
        )

        if self.config.invert_sigmas:
            sigmas_arr = 1.0 - sigmas_arr
            timesteps_arr = sigmas_arr * self.config.num_train_timesteps
            sigmas_arr = mx.concatenate([sigmas_arr, mx.ones((1,), dtype=mx.float32)], axis=0)
        else:
            sigmas_arr = mx.concatenate([sigmas_arr, mx.zeros((1,), dtype=mx.float32)], axis=0)

        self.timesteps = timesteps_arr
        self.sigmas = sigmas_arr
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep: float | mx.array, schedule_timesteps: mx.array | None = None) -> int:
        schedule = self.timesteps if schedule_timesteps is None else schedule_timesteps
        target = float(np.array(timestep).reshape(-1)[0])
        matches = np.nonzero(np.isclose(np.array(schedule), target))[0]
        if len(matches) == 0:
            raise ValueError(f"Timestep {target} is not in the scheduler schedule.")
        return int(matches[1] if len(matches) > 1 else matches[0])

    def _init_step_index(self, timestep: float | mx.array) -> None:
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_noise(self, sample: mx.array, timestep: mx.array, noise: mx.array | None = None) -> mx.array:
        noise = mx.zeros_like(sample) if noise is None else noise
        indices = [self.index_for_timestep(t) for t in np.array(timestep).reshape(-1)]
        sigma = self.sigmas[mx.array(indices, dtype=mx.int32)]
        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]
        return sigma * noise + (1.0 - sigma) * sample

    def step(
        self,
        model_output: mx.array,
        timestep: float | mx.array,
        sample: mx.array,
        *,
        return_dict: bool = True,
    ) -> FlowMatchEulerDiscreteSchedulerOutput | tuple[mx.array]:
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        dt = sigma_next - sigma

        if self.config.stochastic_sampling:
            x0 = sample - sigma * model_output
            noise = mx.random.normal(shape=sample.shape).astype(sample.dtype)
            prev_sample = (1.0 - sigma_next) * x0 + sigma_next * noise
        else:
            prev_sample = sample + dt * model_output

        self._step_index += 1
        prev_sample = prev_sample.astype(model_output.dtype)
        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self) -> int:
        return self.config.num_train_timesteps


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


@register_module(task_type=TaskType.BASE_MODULE, config=Flux2KleinConfig, model_type="flux2_klein")
class Flux2KleinPipeline(EasyMLXBaseModule):
    """Composite native EasyMLX FLUX.2 klein pipeline."""

    config_class = Flux2KleinConfig

    def __init__(
        self,
        config: Flux2KleinConfig,
        *,
        scheduler: FlowMatchEulerDiscreteScheduler | None = None,
        vae: AutoencoderKLFlux2 | None = None,
        text_encoder: Qwen3ForCausalLM | None = None,
        tokenizer: Any | None = None,
        transformer: Flux2Transformer2DModel | None = None,
    ):
        super().__init__(config)
        self.scheduler = scheduler or FlowMatchEulerDiscreteScheduler(config.scheduler_config)
        self.vae = vae or AutoencoderKLFlux2(config.vae_config)
        self.text_encoder = text_encoder or Qwen3ForCausalLM(config.text_config)
        self.tokenizer = tokenizer
        self.transformer = transformer or Flux2Transformer2DModel(config.transformer_config)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        dtype: mx.Dtype | None = None,
        device: mx.Device | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        weights_name: str | None = None,
        strict: bool = True,
        lazy: bool = False,
        auto_convert_hf: bool | None = None,
        converted_cache_dir: str | Path | None = None,
        force_conversion: bool = False,
        copy_support_files: bool = True,
        quantization: Any = None,
    ) -> "Flux2KleinPipeline":
        composite_config = Flux2KleinConfig.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
        )
        resolved = resolve_hf_composite_repo(
            pretrained_model_name_or_path,
            task_type=TaskType.CAUSAL_LM,
            revision=revision,
            local_files_only=local_files_only,
        )
        tokenizer_subfolder = resolved.tokenizer_subfolder or "tokenizer"
        raw_source = Path(pretrained_model_name_or_path).expanduser()
        if raw_source.exists():
            tokenizer_source_ref: str | Path = resolve_local_hf_path(pretrained_model_name_or_path)
        elif local_files_only:
            tokenizer_source_ref = resolve_local_hf_path(pretrained_model_name_or_path)
        else:
            tokenizer_source_ref = pretrained_model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source_ref,
            subfolder=tokenizer_subfolder,
            trust_remote_code=True,
            revision=revision,
            local_files_only=local_files_only,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=resolved.model_subfolder or "text_encoder",
            dtype=dtype,
            device=device,
            revision=revision,
            local_files_only=local_files_only,
            weights_name=weights_name,
            strict=strict,
            lazy=lazy,
            auto_convert_hf=auto_convert_hf,
            converted_cache_dir=converted_cache_dir,
            force_conversion=force_conversion,
            copy_support_files=copy_support_files,
            quantization=quantization,
        )
        transformer = Flux2Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            dtype=dtype,
            device=device,
            revision=revision,
            local_files_only=local_files_only,
            weights_name=weights_name,
            strict=strict,
            lazy=lazy,
            auto_convert_hf=auto_convert_hf,
            converted_cache_dir=converted_cache_dir,
            force_conversion=force_conversion,
            copy_support_files=copy_support_files,
            quantization=quantization,
        )
        vae = AutoencoderKLFlux2.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            dtype=dtype,
            device=device,
            revision=revision,
            local_files_only=local_files_only,
            weights_name=weights_name,
            strict=strict,
            lazy=lazy,
            auto_convert_hf=auto_convert_hf,
            converted_cache_dir=converted_cache_dir,
            force_conversion=force_conversion,
            copy_support_files=copy_support_files,
            quantization=None,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
            revision=revision,
            local_files_only=local_files_only,
        )
        instance = cls(
            composite_config,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        _warmup_rotary_metal_kernel()
        return instance

    def save_pretrained(self, save_directory: str | Path, *, weights_name: str = "model.safetensors") -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        if self.tokenizer is None or not hasattr(self.tokenizer, "save_pretrained"):
            raise ValueError("Flux2KleinPipeline.save_pretrained() requires a tokenizer with save_pretrained().")

        self.text_encoder.save_pretrained(path / "text_encoder", weights_name=weights_name)
        self.transformer.save_pretrained(path / "transformer", weights_name=weights_name)
        self.vae.save_pretrained(path / "vae", weights_name=weights_name)
        self.scheduler.save_pretrained(path / "scheduler")
        self.tokenizer.save_pretrained(str(path / "tokenizer"))

        model_index = {
            "_class_name": type(self).__name__,
            "is_distilled": bool(getattr(self.config, "is_distilled", True)),
            "scheduler": ["diffusers", type(self.scheduler).__name__],
            "text_encoder": ["transformers", type(self.text_encoder).__name__],
            "tokenizer": ["transformers", type(self.tokenizer).__name__],
            "transformer": ["diffusers", type(self.transformer).__name__],
            "vae": ["diffusers", type(self.vae).__name__],
        }
        (path / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _prepare_text_ids(prompt_embeds: mx.array) -> mx.array:
        batch_size, seq_len, _ = prompt_embeds.shape
        coords = _cartesian_prod(
            np.arange(1, dtype=np.int32),
            np.arange(1, dtype=np.int32),
            np.arange(1, dtype=np.int32),
            np.arange(seq_len, dtype=np.int32),
        )
        return mx.array(np.repeat(coords[None, :, :], batch_size, axis=0), dtype=mx.int32)

    @staticmethod
    def _prepare_latent_ids(latents: mx.array) -> mx.array:
        batch_size, _, height, width = latents.shape
        coords = _cartesian_prod(
            np.arange(1, dtype=np.int32),
            np.arange(height, dtype=np.int32),
            np.arange(width, dtype=np.int32),
            np.arange(1, dtype=np.int32),
        )
        return mx.array(np.repeat(coords[None, :, :], batch_size, axis=0), dtype=mx.int32)

    @staticmethod
    def _prepare_image_ids(image_latents: list[mx.array], *, scale: int = 10) -> mx.array:
        all_coords = []
        for index, image_latent in enumerate(image_latents):
            _, _, height, width = image_latent.shape
            coords = _cartesian_prod(
                np.array([scale + scale * index], dtype=np.int32),
                np.arange(height, dtype=np.int32),
                np.arange(width, dtype=np.int32),
                np.arange(1, dtype=np.int32),
            )
            all_coords.append(coords)
        return mx.array(np.concatenate(all_coords, axis=0)[None, :, :], dtype=mx.int32)

    @staticmethod
    def _patchify_latents(latents: mx.array) -> mx.array:
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.transpose(0, 1, 3, 5, 2, 4)
        return latents.reshape(batch_size, channels * 4, height // 2, width // 2)

    @staticmethod
    def _unpatchify_latents(latents: mx.array) -> mx.array:
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
        latents = latents.transpose(0, 1, 4, 2, 5, 3)
        return latents.reshape(batch_size, channels // 4, height * 2, width * 2)

    @staticmethod
    def _pack_latents(latents: mx.array) -> mx.array:
        batch_size, channels, height, width = latents.shape
        return latents.reshape(batch_size, channels, height * width).transpose(0, 2, 1)

    @staticmethod
    def _unpack_latents_with_ids(
        packed: mx.array,
        x_ids: mx.array,
        height: int,
        width: int,
    ) -> mx.array:
        outputs: list[mx.array] = []
        for data, pos in zip(packed, x_ids, strict=False):
            channels = data.shape[-1]
            flat_ids = (pos[:, 1].astype(mx.int32) * width + pos[:, 2].astype(mx.int32)).reshape(-1, 1)
            out = mx.zeros((height * width, channels), dtype=data.dtype)
            out = mx.put_along_axis(out, flat_ids, data, axis=0)
            outputs.append(out.reshape(height, width, channels).transpose(2, 0, 1))
        return mx.stack(outputs, axis=0)

    def _tokenize_prompts(self, prompt: str | list[str], *, max_sequence_length: int) -> tuple[mx.array, mx.array]:
        if self.tokenizer is None:
            raise ValueError(
                "Flux2KleinPipeline requires a tokenizer for prompt encoding. "
                "Load it with `from_pretrained()` or pass `prompt_embeds` directly."
            )
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        formatted: list[str] = []
        for text in prompts:
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": text}]
                try:
                    formatted.append(
                        self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )
                except TypeError:
                    formatted.append(
                        self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    )
            else:
                formatted.append(text)

        inputs = self.tokenizer(
            formatted,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        return mx.array(inputs["input_ids"], dtype=mx.int32), mx.array(inputs["attention_mask"], dtype=mx.int32)

    def _forward_qwen3_hidden_states(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        max_layer: int | None = None,
    ) -> list[mx.array]:
        model = self.text_encoder.base_model
        hidden_states = model.embed_tokens(input_ids)
        batch_size, seq_len = hidden_states.shape[:2]
        from easymlx.layers.attention import build_attention_mask

        mask = build_attention_mask(attention_mask, batch_size=batch_size, seq_len=seq_len)
        limit = len(model.layers) if max_layer is None else min(max_layer + 1, len(model.layers))
        if not hasattr(self, "_compiled_text_enc_fn") or self._compiled_text_enc_limit != limit:
            _layers = [model.layers[i] for i in range(limit)]
            _state = [[layer.state for layer in _layers]]
            _selected_idxs = getattr(self, "_text_selected_idxs", None)

            def _enc(hs, msk, *selected_placeholder):
                outs = [hs]
                for layer in _layers:
                    hs = layer(hs, mask=msk)
                    outs.append(hs)
                return outs

            self._compiled_text_enc_fn = mx.compile(_enc, inputs=_state)
            self._compiled_text_enc_limit = limit

        outputs = self._compiled_text_enc_fn(hidden_states, mask)
        return outputs

    def encode_prompt(
        self,
        prompt: str | list[str] | None = None,
        *,
        prompt_embeds: mx.array | None = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ) -> tuple[mx.array, mx.array]:
        if prompt_embeds is None:
            prompt = "" if prompt is None else prompt
            input_ids, attention_mask = self._tokenize_prompts(prompt, max_sequence_length=max_sequence_length)
            hidden_states = self._forward_qwen3_hidden_states(
                input_ids,
                attention_mask,
                max_layer=max(text_encoder_out_layers),
            )
            selected = mx.stack([hidden_states[index] for index in text_encoder_out_layers], axis=1)
            batch_size, num_channels, seq_len, hidden_dim = selected.shape
            prompt_embeds = selected.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        batch_size, seq_len, dim = prompt_embeds.shape
        prompt_embeds = mx.repeat(prompt_embeds, num_images_per_prompt, axis=0).reshape(
            batch_size * num_images_per_prompt,
            seq_len,
            dim,
        )
        text_ids = self._prepare_text_ids(prompt_embeds)
        return prompt_embeds, text_ids

    def _encode_vae_image(self, image: mx.array) -> mx.array:
        latents = self.vae.encode(image).latent_dist.mode()
        latents = self._patchify_latents(latents)
        bn_mean = self.vae.bn.running_mean.reshape(1, -1, 1, 1).astype(latents.dtype)
        bn_std = mx.sqrt(
            self.vae.bn.running_var.reshape(1, -1, 1, 1).astype(latents.dtype) + self.vae.config.batch_norm_eps
        )
        return (latents - bn_mean) / bn_std

    def prepare_latents(
        self,
        *,
        batch_size: int,
        num_latents_channels: int,
        height: int,
        width: int,
        dtype: mx.Dtype,
        latents: mx.array | None = None,
        seed: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if seed is not None:
            mx.random.seed(seed)
        if latents is None:
            latents = mx.random.normal(shape=shape).astype(dtype)
        else:
            latents = latents.astype(dtype)
        latent_ids = self._prepare_latent_ids(latents)
        return self._pack_latents(latents), latent_ids

    def prepare_image_latents(self, images: list[mx.array], *, batch_size: int) -> tuple[mx.array, mx.array]:
        image_latents = [self._encode_vae_image(image) for image in images]
        image_latent_ids = self._prepare_image_ids(image_latents)
        packed = [self._pack_latents(latent).squeeze(0) for latent in image_latents]
        image_latents = mx.concatenate(packed, axis=0)[None, :, :]
        image_latents = mx.repeat(image_latents, batch_size, axis=0)
        image_latent_ids = mx.repeat(image_latent_ids, batch_size, axis=0)
        return image_latents, image_latent_ids

    def preprocess_images(
        self, image: Any | list[Any] | None, *, height: int | None, width: int | None
    ) -> tuple[list[mx.array] | None, int | None, int | None]:
        if image is None:
            return None, height, width
        images = image if isinstance(image, list) else [image]
        processed: list[mx.array] = []
        for item in images:
            arr = _ensure_numpy_image(item)
            target_height = height or arr.shape[0]
            target_width = width or arr.shape[1]
            multiple = self.vae_scale_factor * 2
            target_height = max(multiple, (target_height // multiple) * multiple)
            target_width = max(multiple, (target_width // multiple) * multiple)
            if arr.shape[0] != target_height or arr.shape[1] != target_width:
                try:
                    from PIL import Image
                except ImportError as exc:
                    raise ImportError("Pillow is required for resizing images.") from exc
                arr = np.array(Image.fromarray(arr).resize((target_width, target_height), Image.Resampling.LANCZOS))
            tensor = mx.array(arr.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None, ...]
            processed.append(tensor)
            height = target_height
            width = target_width
        return processed, height, width

    def postprocess_images(self, image: mx.array, *, output_type: str = "pil") -> Any:
        if output_type == "latent":
            return image
        if output_type == "mx":
            return image
        array = np.clip((np.array(image).transpose(0, 2, 3, 1) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        if output_type == "numpy":
            return array
        if output_type != "pil":
            raise ValueError(f"Unsupported output_type={output_type!r}.")
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for `output_type='pil'`.") from exc
        return [Image.fromarray(frame) for frame in array]

    def __call__(
        self,
        *,
        prompt: str | list[str] | None = None,
        image: Any | list[Any] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: mx.array | None = None,
        negative_prompt_embeds: mx.array | None = None,
        latents: mx.array | None = None,
        seed: int | None = None,
        output_type: str = "pil",
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
        return_dict: bool = True,
    ) -> Flux2KleinPipelineOutput | tuple[Any]:
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1

        prompt_embeds, text_ids = self.encode_prompt(
            prompt,
            prompt_embeds=prompt_embeds,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if guidance_scale > 1.0 and not self.config.is_distilled:
            negative_prompt = "" if negative_prompt is None else negative_prompt
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )
        else:
            negative_text_ids = None

        condition_images, height, width = self.preprocess_images(image, height=height, width=width)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            latents=latents,
            seed=seed,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(condition_images, batch_size=batch_size)

        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float32)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas.tolist(), mu=mu)
        timesteps = self.scheduler.timesteps
        self.scheduler.set_begin_index(0)

        _t_dtype = self.transformer.config.mlx_dtype
        if not hasattr(self, "_compiled_blocks_fn"):
            _transformer = self.transformer
            _state = [_transformer.state]

            def _blocks(hs, ehs, mod_i, mod_t, mod_s, cos_r, sin_r):
                return _transformer._run_blocks(hs, ehs, mod_i, mod_t, mod_s, cos_r, sin_r)

            self._compiled_blocks_fn = mx.compile(_blocks, inputs=_state)
        _blocks_fn = self._compiled_blocks_fn

        _img_ids = latent_ids[0] if latent_ids.ndim == 3 else latent_ids
        _txt_ids = text_ids[0] if text_ids.ndim == 3 else text_ids
        if image_latents is not None and image_latent_ids is not None:
            _full_img_ids = mx.concatenate(
                [_img_ids, image_latent_ids[0] if image_latent_ids.ndim == 3 else image_latent_ids],
                axis=0,
            )
        else:
            _full_img_ids = _img_ids
        _img_rope = self.transformer.pos_embed(_full_img_ids)
        _txt_rope = self.transformer.pos_embed(_txt_ids)
        _cat_rope = (
            mx.concatenate([_txt_rope[0], _img_rope[0]], axis=0),
            mx.concatenate([_txt_rope[1], _img_rope[1]], axis=0),
        )
        _prompt_embeds_enc = self.transformer.context_embedder(prompt_embeds)
        mx.eval(_cat_rope, _prompt_embeds_enc)

        for _step_idx, timestep_value in enumerate(timesteps):
            timestep = mx.full((latents.shape[0],), float(timestep_value), dtype=latents.dtype)

            latent_model_input = latents if latents.dtype == _t_dtype else latents.astype(_t_dtype)
            latent_image_ids = latent_ids
            if image_latents is not None and image_latent_ids is not None:
                latent_model_input = mx.concatenate([latents, image_latents.astype(latents.dtype)], axis=1)
                latent_image_ids = mx.concatenate([latent_ids, image_latent_ids], axis=1)

            noise_pred = self.transformer(
                latent_model_input,
                timestep=timestep / 1000.0,
                guidance=None,
                encoder_hidden_states=_prompt_embeds_enc,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                image_rotary_emb=_cat_rope,
                _blocks_fn=_blocks_fn,
                _encoder_embedded=True,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.shape[1]]

            if guidance_scale > 1.0 and negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise_pred = self.transformer(
                    latent_model_input,
                    timestep=timestep / 1000.0,
                    guidance=None,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                negative_noise_pred = negative_noise_pred[:, : latents.shape[1]]
                noise_pred = negative_noise_pred + guidance_scale * (noise_pred - negative_noise_pred)

            latents = self.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]

        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        latents = self._unpack_latents_with_ids(latents, latent_ids, latent_height // 2, latent_width // 2)
        if not hasattr(self, "_cached_bn"):
            _bm = self.vae.bn.running_mean.reshape(1, -1, 1, 1).astype(latents.dtype)
            _bs = mx.sqrt(
                self.vae.bn.running_var.reshape(1, -1, 1, 1).astype(latents.dtype) + self.vae.config.batch_norm_eps
            )
            mx.eval(_bm, _bs)
            self._cached_bn = (_bm, _bs)
        bn_mean, bn_std = self._cached_bn
        latents = latents * bn_std + bn_mean
        latents = self._unpatchify_latents(latents)

        if output_type == "latent":
            images = latents
        else:
            if not hasattr(self, "_compiled_vae_decode_fn"):
                _vae = self.vae
                _state = [_vae.state]

                def _vae_dec(z):
                    return _vae.decode(z, return_dict=False)[0]

                self._compiled_vae_decode_fn = mx.compile(_vae_dec, inputs=_state, outputs=_state)
            images = self._compiled_vae_decode_fn(latents)
            images = self.postprocess_images(images, output_type=output_type)

        if not return_dict:
            return (images,)
        return Flux2KleinPipelineOutput(images=images)

    def generate(self, prompt: str | list[str], **kwargs: Any) -> Flux2KleinPipelineOutput | tuple[Any]:
        return self(prompt=prompt, **kwargs)


__all__ = (
    "AutoencoderKLFlux2",
    "AutoencoderKLOutput",
    "DecoderOutput",
    "DiagonalGaussianDistribution",
    "FlowMatchEulerDiscreteScheduler",
    "FlowMatchEulerDiscreteSchedulerOutput",
    "Flux2KleinPipeline",
    "Flux2KleinPipelineOutput",
    "Flux2Transformer2DModel",
    "Flux2Transformer2DModelOutput",
    "compute_empirical_mu",
)
