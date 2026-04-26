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

"""Text generation helpers for easymlx (serving-only).

Implements a small, MLX-native subset of HuggingFace generation features:
- greedy decoding
- sampling (temperature, top-k, top-p)
- ``init_operations_cache`` for allocating the right cache type

Models are expected to be callable and return either:
- an object with a ``.logits`` attribute, or
- raw logits as an ``mx.array`` of shape ``[batch, seq_len, vocab]``.
"""

from __future__ import annotations

import copy
import typing as tp

import mlx.core as mx
from transformers.generation.configuration_utils import GenerationConfig

from ..modeling_outputs import GreedySearchOutput, SampleOutput


def _as_logits(output: tp.Any) -> mx.array:
    """Extract logits from a model output.

    Args:
        output: Either a raw ``mx.array`` of logits, or an object with
            a ``.logits`` attribute (e.g. :class:`CausalLMOutput`).

    Returns:
        The logits as an ``mx.array``.

    Raises:
        TypeError: If the output is neither an ``mx.array`` nor has a
            ``.logits`` attribute of type ``mx.array``.
    """
    if isinstance(output, mx.array):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, mx.array):
        return logits
    raise TypeError("Model output must be an mx.array or have a `.logits: mx.array` attribute")


def _apply_temperature(logits: mx.array, temperature: float) -> mx.array:
    """Scale logits by a temperature value.

    Args:
        logits: Raw logits of shape ``[batch, vocab_size]``.
        temperature: Positive float controlling randomness. Values
            below 1.0 sharpen the distribution; values above 1.0
            flatten it.

    Returns:
        Temperature-scaled logits.

    Raises:
        ValueError: If ``temperature`` is not positive.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if temperature == 1.0:
        return logits
    return logits / float(temperature)


def _apply_top_k(logits: mx.array, top_k: int) -> mx.array:
    """Mask logits to keep only the top-k highest values.

    All positions outside the top-k are set to the dtype's minimum
    representable value.

    Args:
        logits: Logits of shape ``[batch, vocab_size]``.
        top_k: Number of top entries to keep. If ``<= 0`` or
            ``>= vocab_size``, the logits are returned unchanged.

    Returns:
        Filtered logits with non-top-k positions masked.
    """
    if top_k <= 0:
        return logits
    vocab = logits.shape[-1]
    if top_k >= vocab:
        return logits
    kth = mx.sort(logits, axis=-1)[..., -top_k]
    very_neg = mx.finfo(logits.dtype).min
    return mx.where(logits < kth[..., None], very_neg, logits)


def _apply_top_p(logits: mx.array, top_p: float) -> mx.array:
    """Apply nucleus (top-p) filtering to logits.

    Keeps the smallest set of tokens whose cumulative probability
    exceeds *top_p*, masking all others.

    Args:
        logits: Logits of shape ``[batch, vocab_size]``.
        top_p: Cumulative probability threshold in ``(0, 1]``.
            ``1.0`` disables filtering.

    Returns:
        Filtered logits.

    Raises:
        ValueError: If ``top_p`` is not in ``(0, 1]``.
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if top_p == 1.0:
        return logits

    sorted_idx = mx.argsort(logits, axis=-1)[:, ::-1]
    sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cum_probs = mx.cumsum(sorted_probs, axis=-1)

    keep_sorted = cum_probs <= float(top_p)
    keep_sorted = mx.put_along_axis(
        keep_sorted,
        mx.zeros((keep_sorted.shape[0], 1), dtype=mx.int32),
        mx.ones((keep_sorted.shape[0], 1), dtype=mx.bool_),
        axis=-1,
    )

    keep = mx.put_along_axis(mx.zeros_like(keep_sorted), sorted_idx, keep_sorted, axis=-1)
    very_neg = mx.finfo(logits.dtype).min
    return mx.where(keep, logits, very_neg)


class EasyGenerationMixin:
    """Mixin providing text generation capabilities for easymlx models.

    Adds :meth:`init_operations_cache` for cache allocation and
    :meth:`generate` for autoregressive text generation with greedy
    or sampling strategies.

    Attributes:
        config: The model configuration (expected to be set by the
            host class).
    """

    config: tp.Any

    def init_operations_cache(
        self,
        batch_size: int,
        max_length: int,
        *,
        page_size: int = 16,
        dtype: mx.Dtype = mx.float16,
        cache_type: str | None = None,
    ):
        """Allocate the recommended cache for this model.

        Uses :meth:`get_operations_cache_info` (from
        :class:`OperationCacheMixin`) to determine the right cache
        family and then creates it.

        Args:
            batch_size: Number of sequences in the batch.
            max_length: Maximum total sequence length (prompt +
                generation).
            page_size: Block size for paged attention caches.
            dtype: The MLX dtype for cache tensors.
            cache_type: Force a specific cache type (``"transformer"``,
                ``"paged"``, or ``"hybrid"``). If ``None``, the
                recommended type is used.

        Returns:
            For ``"transformer"`` -- a ``TransformerCache``.
            For ``"paged"`` -- a ``PageCache`` (one view per layer).
            For ``"hybrid"`` -- a ``HybridCache``.
        """
        from easymlx.caching import (
            HybridCache,
            HybridCacheConfig,
            PageCache,
            PageCacheConfig,
            PageCacheView,
            TransformerCache,
            TransformerCacheConfig,
        )

        cache_info = self.get_operations_cache_info()
        recommended = cache_type or cache_info.get_recommended_cache_type()

        num_heads, head_dim, num_kv_heads = self._resolve_head_geometry()
        config = getattr(self, "config", None)
        num_hidden_layers: int = getattr(config, "num_hidden_layers", len(cache_info.layers) or 1)
        sliding_window: int | None = getattr(config, "sliding_window", None)

        if recommended == "transformer":
            cache_cfg = TransformerCacheConfig(
                batch_size=batch_size,
                num_hidden_layers=num_hidden_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                max_sequence_length=max_length,
                num_key_value_heads=num_kv_heads,
                sliding_window=sliding_window,
                dtype=dtype,
            )
            return TransformerCache.init_cache(cache_cfg)

        if recommended == "paged":
            model_cache_dtype = getattr(config, "cache_dtype", None)
            model_cache_bits = getattr(config, "cache_bits", None)
            ragged_cfg = PageCacheConfig(
                num_hidden_layers=num_hidden_layers,
                num_kv_heads=num_kv_heads or num_heads,
                head_dim=head_dim,
                page_size=page_size,
                max_num_reqs=batch_size,
                max_model_length=max_length,
                dtype=dtype,
                cache_dtype=model_cache_dtype,
                cache_bits=model_cache_bits,
            )
            ragged_cache = PageCache.init_cache(ragged_cfg)

            for layer_idx in range(num_hidden_layers):
                if layer_idx < len(cache_info.layers):
                    layer_info = cache_info.layers[layer_idx]
                    op_name = layer_info.operation_name
                    if op_name in ("linear_attention", "gated_delta_rule", "gdr"):
                        layer_nkv = ragged_cfg.num_kv_heads
                        layer_hd = ragged_cfg.head_dim
                        text_cfg = getattr(config, "text_config", config)
                        layer_nkv = getattr(text_cfg, "linear_num_value_heads", layer_nkv)
                        layer_hd = getattr(text_cfg, "linear_value_head_dim", layer_hd)
                        ragged_cache[layer_idx] = PageCacheView.init(
                            num_seqs=batch_size,
                            max_seq_len=max_length,
                            num_kv_heads=layer_nkv,
                            head_dim=layer_hd,
                            block_size=page_size,
                            dtype=dtype,
                            cache_dtype=model_cache_dtype,
                            cache_bits=model_cache_bits,
                        )
            return ragged_cache

        layer_types_raw = getattr(config, "layer_types", None)
        if layer_types_raw is not None:
            layer_types = tuple(str(lt) for lt in layer_types_raw)
        else:
            layer_types = tuple(
                "full_attention" if layer.is_attention_layer else "linear_attention" for layer in cache_info.layers
            )

        hybrid_cfg = HybridCacheConfig(
            batch_size=batch_size,
            num_hidden_layers=num_hidden_layers,
            layer_types=layer_types,
            num_heads=num_heads,
            head_dim=head_dim,
            num_key_value_heads=num_kv_heads,
            max_sequence_length=max_length,
            sliding_window=sliding_window,
            conv_dim=getattr(config, "conv_dimension", 0),
            conv_kernel_size=getattr(config, "conv_kernel", 4),
            recurrent_state_shape=tuple(getattr(config, "recurrent_state_shape", ())),
            dtype=dtype,
        )
        return HybridCache.init_cache(hybrid_cfg)

    def _resolve_head_geometry(self) -> tuple[int, int, int | None]:
        """Return ``(num_heads, head_dim, num_kv_heads)`` from the model.

        Tries the config first, then walks the model's layer list to find
        the first attention sub-module and reads its shapes.

        Returns:
            A tuple of ``(num_heads, head_dim, num_kv_heads)``.
            ``num_kv_heads`` may be ``None`` when not explicitly
            configured (i.e. MHA without GQA).
        """
        config = getattr(self, "config", None)
        if config is not None:
            num_heads = getattr(config, "num_attention_heads", None)
            head_dim = getattr(config, "head_dim", None)
            num_kv_heads = getattr(config, "num_key_value_heads", None)
            if num_heads is not None and head_dim is not None:
                return int(num_heads), int(head_dim), int(num_kv_heads) if num_kv_heads else None

        for attr_name in ("layers", "h", "blocks"):
            layer_list = getattr(self, attr_name, None)
            if not isinstance(layer_list, list) or not layer_list:
                continue
            first_layer = layer_list[0]
            for sub_name in ("self_attn", "attn", "attention"):
                attn_mod = getattr(first_layer, sub_name, None)
                if attn_mod is None:
                    continue
                nh = getattr(attn_mod, "num_heads", None) or getattr(attn_mod, "n_head", None)
                hd = getattr(attn_mod, "head_dim", None) or getattr(attn_mod, "d_head", None)
                nkv = getattr(attn_mod, "num_key_value_heads", None) or getattr(attn_mod, "n_kv_head", None)
                if nh is not None and hd is not None:
                    return int(nh), int(hd), int(nkv) if nkv else None

        hidden = getattr(config, "hidden_size", 4096) if config else 4096
        nh = getattr(config, "num_attention_heads", 32) if config else 32
        return int(nh), int(hidden) // int(nh), None

    def generate(
        self,
        input_ids: mx.ArrayLike,
        *,
        generation_config: GenerationConfig | None = None,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        **model_kwargs,
    ) -> mx.array | GreedySearchOutput | SampleOutput:
        """Generate token sequences autoregressively.

        Supports both greedy decoding and sampling with temperature,
        top-k, and top-p filtering.

        Args:
            input_ids: Input token IDs of shape ``[batch, seq_len]``
                or ``[seq_len]``.
            generation_config: A HuggingFace ``GenerationConfig``
                object. If ``None``, a default one is constructed from
                the model config.
            max_new_tokens: Maximum number of new tokens to generate.
                Overrides the value in *generation_config*.
            do_sample: Whether to use sampling. ``False`` uses greedy
                decoding.
            temperature: Sampling temperature. Only used when
                ``do_sample=True``.
            top_k: Top-k filtering parameter.
            top_p: Nucleus sampling probability threshold.
            eos_token_id: End-of-sequence token ID for early stopping.
            pad_token_id: Padding token ID for finished sequences.
            return_dict_in_generate: If ``True``, return a
                :class:`GreedySearchOutput` or :class:`SampleOutput`
                instead of a raw array.
            output_scores: If ``True``, include per-step logits in the
                output.
            **model_kwargs: Additional keyword arguments passed to the
                model's ``__call__`` method.

        Returns:
            If ``return_dict_in_generate`` is ``False`` (default),
            returns an ``mx.array`` of shape
            ``[batch, prompt_len + generated_len]``. Otherwise returns
            a :class:`GreedySearchOutput` or :class:`SampleOutput`.

        Raises:
            ValueError: If ``max_new_tokens`` is not positive.
        """
        if generation_config is None:
            generation_config = getattr(self, "generation_config", None)
        if generation_config is None:
            model_config = getattr(self, "config", None)
            generation_config = (
                GenerationConfig.from_model_config(model_config) if model_config is not None else GenerationConfig()
            )
        generation_config = copy.deepcopy(generation_config)

        if max_new_tokens is not None:
            generation_config.max_new_tokens = int(max_new_tokens)
        if do_sample is not None:
            generation_config.do_sample = bool(do_sample)
        if temperature is not None:
            generation_config.temperature = float(temperature)
        if top_k is not None:
            generation_config.top_k = int(top_k)
        if top_p is not None:
            generation_config.top_p = float(top_p)
        if eos_token_id is not None:
            generation_config.eos_token_id = int(eos_token_id)
        if pad_token_id is not None:
            generation_config.pad_token_id = int(pad_token_id)

        max_new_tokens = int(generation_config.max_new_tokens or 0)
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0 (set via generation_config or argument)")

        do_sample = bool(getattr(generation_config, "do_sample", False))
        temperature = float(getattr(generation_config, "temperature", 1.0) or 1.0)
        top_k = int(getattr(generation_config, "top_k", 0) or 0)
        top_p = float(getattr(generation_config, "top_p", 1.0) or 1.0)

        eos_token_id = generation_config.eos_token_id if eos_token_id is None else eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        eos_token_id = int(eos_token_id) if eos_token_id is not None else None

        pad_token_id = generation_config.pad_token_id if pad_token_id is None else pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id
        pad_token_id = int(pad_token_id) if pad_token_id is not None else None

        sequences = mx.array(input_ids, dtype=mx.int32)
        if sequences.ndim == 1:
            sequences = sequences[None, :]
        batch = sequences.shape[0]
        finished = mx.zeros((batch,), dtype=mx.bool_)

        scores: list[mx.array] | None = [] if output_scores else None

        for _ in range(max_new_tokens):
            logits = _as_logits(self(sequences, **model_kwargs))
            step_logits = logits[:, -1, :]

            finished_before = finished
            if do_sample:
                step_logits = _apply_temperature(step_logits, temperature)
                if top_k:
                    step_logits = _apply_top_k(step_logits, top_k)
                if top_p < 1.0:
                    step_logits = _apply_top_p(step_logits, top_p)
                next_token = mx.random.categorical(step_logits).astype(mx.int32)
            else:
                next_token = mx.argmax(step_logits, axis=-1).astype(mx.int32)

            if scores is not None:
                scores.append(step_logits)

            if eos_token_id is not None:
                eos_hit = next_token == eos_token_id
                finished = finished | eos_hit

            if pad_token_id is not None and eos_token_id is not None:
                next_token = mx.where(finished_before, mx.array(pad_token_id, dtype=mx.int32), next_token)

            sequences = mx.concatenate([sequences, next_token[:, None]], axis=-1)

            if eos_token_id is not None and bool(mx.all(finished).item()):
                break

        if not return_dict_in_generate:
            return sequences

        scores_out = tuple(scores) if scores is not None else None
        if do_sample:
            return SampleOutput(sequences=sequences, scores=scores_out)
        return GreedySearchOutput(sequences=sequences, scores=scores_out)
