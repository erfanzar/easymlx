#!/usr/bin/env python3
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

"""Benchmark DFlash MLX scenarios for Qwen3.5-style models.

This script intentionally lives outside the package: it depends on the
upstream DFlash MLX implementation and `mlx-lm`, so it is a local
optimization harness rather than a library API.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load
from mlx_lm.generate import stream_generate as mlx_stream_generate
from mlx_lm.sample_utils import make_sampler

DEFAULT_PROMPT = (
    "Write a long technical manual section about speculative decoding and "
    "production serving. Do not conclude early; keep expanding with concrete "
    "implementation details, benchmarks, failure modes, memory behavior, "
    "scheduling, cache rollback, and optimization notes."
)


def _import_dflash(dflash_repo: str | None):
    if dflash_repo:
        repo = Path(dflash_repo).expanduser().resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
    return importlib.import_module("dflash.model_mlx")


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def _parse_csv_optional_ints(value: str) -> list[int | None]:
    out: list[int | None] = []
    for part in value.split(","):
        part = part.strip().lower()
        if not part:
            continue
        out.append(None if part in {"none", "null", "0"} else int(part))
    return out


def _format_optional_int(value: int | None) -> str:
    return "none" if value is None else str(value)


def _safe_quant_predicate(mode: str, group_size: int):
    def predicate(path: str, module: nn.Module) -> bool:
        del path
        weight = getattr(module, "weight", None)
        return hasattr(module, "to_quantized") and weight is not None and weight.shape[-1] % group_size == 0

    return predicate


def _quant_defaults(mode: str) -> tuple[int, int]:
    if mode == "mxfp4":
        return (32, 4)
    if mode == "nvfp4":
        return (16, 4)
    if mode == "mxfp8":
        return (32, 8)
    if mode == "affine4":
        return (64, 4)
    if mode == "affine8":
        return (64, 8)
    raise ValueError(f"Unsupported draft quantization mode: {mode}")


def _quantize_draft(draft: nn.Module, mode: str) -> dict[str, Any]:
    if mode == "none":
        return {"mode": "none"}
    group_size, bits = _quant_defaults(mode)
    q_mode = "affine" if mode.startswith("affine") else mode
    start = time.perf_counter()
    nn.quantize(
        draft,
        group_size=group_size,
        bits=bits,
        mode=q_mode,
        class_predicate=_safe_quant_predicate(q_mode, group_size),
    )
    mx.eval(draft.parameters())
    return {
        "mode": mode,
        "mlx_mode": q_mode,
        "group_size": group_size,
        "bits": bits,
        "seconds": time.perf_counter() - start,
    }


def _build_prompt(tokenizer: Any, prompt: str, *, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _drain_baseline(model: nn.Module, tokenizer: Any, prompt: str, *, max_tokens: int, sampler: Any) -> dict[str, Any]:
    last = None
    start = time.perf_counter()
    for response in mlx_stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
        last = response
    elapsed = time.perf_counter() - start
    if last is None:
        return {
            "kind": "baseline",
            "tokens": 0,
            "tps": 0.0,
            "elapsed": elapsed,
            "prompt_tps": 0.0,
            "finish_reason": None,
        }
    return {
        "kind": "baseline",
        "tokens": int(last.generation_tokens),
        "tps": float(last.generation_tps),
        "elapsed": elapsed,
        "prompt_tps": float(last.prompt_tps),
        "finish_reason": last.finish_reason,
        "peak_gb": float(mx.get_peak_memory() / 1e9),
    }


def _drain_dflash(
    stream_generate: Any,
    model: nn.Module,
    draft: nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    block_size: int,
    max_tokens: int,
    sampler: Any,
    disable_clear_cache: bool,
) -> dict[str, Any]:
    original_clear_cache = mx.clear_cache
    if disable_clear_cache:
        mx.clear_cache = lambda: None  # type: ignore[assignment]
    last = None
    accepted_sum = 0
    chunks = 0
    start = time.perf_counter()
    try:
        for response in stream_generate(
            model,
            draft,
            tokenizer,
            prompt,
            block_size=block_size,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            last = response
            accepted_sum += int(response.accepted)
            chunks += 1
    finally:
        if disable_clear_cache:
            mx.clear_cache = original_clear_cache  # type: ignore[assignment]

    elapsed = time.perf_counter() - start
    if last is None:
        return {
            "kind": "dflash",
            "tokens": 0,
            "tps": 0.0,
            "elapsed": elapsed,
            "prompt_tps": 0.0,
            "finish_reason": None,
            "accepted_sum": accepted_sum,
            "chunks": chunks,
            "avg_accept": 0.0,
        }
    return {
        "kind": "dflash",
        "tokens": int(last.generation_tokens),
        "tps": float(last.generation_tps),
        "elapsed": elapsed,
        "prompt_tps": float(last.prompt_tps),
        "finish_reason": last.finish_reason,
        "accepted_sum": accepted_sum,
        "chunks": chunks,
        "avg_accept": accepted_sum / max(chunks, 1),
        "peak_gb": float(mx.get_peak_memory() / 1e9),
    }


def _print_json(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def _iter_draft_scenarios(
    draft_quant_modes: Iterable[str],
    sliding_windows: Iterable[int | None],
) -> Iterable[tuple[str, int | None]]:
    for quant_mode in draft_quant_modes:
        for sliding_window in sliding_windows:
            yield quant_mode, sliding_window


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DFlash MLX scenarios.")
    parser.add_argument("--target", required=True, help="Target MLX-LM model path or HF id.")
    parser.add_argument("--draft", required=True, help="DFlash draft HF id.")
    parser.add_argument("--dflash-repo", default="/tmp/dflash-read", help="Path to cloned z-lab/dflash repo.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--block-sizes", default="4,8,16")
    parser.add_argument("--draft-quant-modes", default="none,mxfp4")
    parser.add_argument("--sliding-windows", default="none,4096")
    parser.add_argument(
        "--clear-cache-modes",
        default="enabled,disabled",
        help="Comma-separated: enabled,disabled",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    dflash_mlx = _import_dflash(args.dflash_repo)
    sampler = make_sampler(temp=args.temperature)
    block_sizes = _parse_csv_ints(args.block_sizes)
    draft_quant_modes = [part.strip().lower() for part in args.draft_quant_modes.split(",") if part.strip()]
    sliding_windows = _parse_csv_optional_ints(args.sliding_windows)
    clear_cache_modes = [
        part.strip().lower() in {"disabled", "disable", "off", "none"}
        for part in args.clear_cache_modes.split(",")
        if part.strip()
    ]

    load_start = time.perf_counter()
    model, tokenizer = mlx_load(args.target)
    mx.eval(model.parameters())
    prompt = _build_prompt(tokenizer, args.prompt, enable_thinking=args.enable_thinking)
    _print_json(
        {
            "event": "target_loaded",
            "target": args.target,
            "seconds": time.perf_counter() - load_start,
            "prompt_chars": len(prompt),
            "peak_gb": float(mx.get_peak_memory() / 1e9),
        }
    )

    if not args.skip_baseline:
        _drain_baseline(model, tokenizer, prompt, max_tokens=args.warmup_tokens, sampler=sampler)
        for i in range(args.repeat):
            result = _drain_baseline(model, tokenizer, prompt, max_tokens=args.max_tokens, sampler=sampler)
            result.update({"event": "result", "repeat": i})
            _print_json(result)
        mx.clear_cache()

    for quant_mode, sliding_window in _iter_draft_scenarios(
        draft_quant_modes,
        sliding_windows,
    ):
        draft_load_start = time.perf_counter()
        draft = dflash_mlx.load_draft(args.draft, sliding_window_size=sliding_window)
        mx.eval(draft.parameters())
        quant_info = _quantize_draft(draft, quant_mode)
        draft.bind(model)
        _print_json(
            {
                "event": "draft_loaded",
                "draft": args.draft,
                "draft_quant": quant_info,
                "sliding_window": _format_optional_int(sliding_window),
                "seconds": time.perf_counter() - draft_load_start,
                "peak_gb": float(mx.get_peak_memory() / 1e9),
            }
        )

        for disable_clear_cache in clear_cache_modes:
            for block_size in block_sizes:
                _drain_dflash(
                    dflash_mlx.stream_generate,
                    model,
                    draft,
                    tokenizer,
                    prompt,
                    block_size=block_size,
                    max_tokens=args.warmup_tokens,
                    sampler=sampler,
                    disable_clear_cache=disable_clear_cache,
                )
                for i in range(args.repeat):
                    result = _drain_dflash(
                        dflash_mlx.stream_generate,
                        model,
                        draft,
                        tokenizer,
                        prompt,
                        block_size=block_size,
                        max_tokens=args.max_tokens,
                        sampler=sampler,
                        disable_clear_cache=disable_clear_cache,
                    )
                    result.update(
                        {
                            "event": "result",
                            "repeat": i,
                            "block_size": block_size,
                            "draft_quant": quant_mode,
                            "sliding_window": _format_optional_int(sliding_window),
                            "clear_cache": "disabled" if disable_clear_cache else "enabled",
                        }
                    )
                    _print_json(result)
                mx.clear_cache()

        del draft
        mx.clear_cache()


if __name__ == "__main__":
    main()
