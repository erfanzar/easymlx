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

"""Stage-level profiler for the upstream DFlash MLX generation loop."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

PROMPT = (
    "Write a long technical manual section about speculative decoding and "
    "production serving. Do not conclude early; keep expanding with concrete "
    "implementation details, benchmarks, failure modes, memory behavior, "
    "scheduling, cache rollback, and optimization notes."
)


def _import_dflash(repo: str | None):
    if repo:
        resolved = str(Path(repo).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    return importlib.import_module("dflash.model_mlx")


def _quant_defaults(mode: str) -> tuple[int, int, str]:
    if mode == "none":
        return (0, 0, mode)
    if mode == "mxfp4":
        return (32, 4, mode)
    if mode == "nvfp4":
        return (16, 4, mode)
    if mode == "mxfp8":
        return (32, 8, mode)
    if mode == "affine4":
        return (64, 4, "affine")
    raise ValueError(f"Unsupported quant mode {mode!r}")


def _quantize(model: nn.Module, mode: str) -> None:
    if mode == "none":
        return
    group_size, bits, mlx_mode = _quant_defaults(mode)

    def predicate(path: str, module: nn.Module) -> bool:
        del path
        weight = getattr(module, "weight", None)
        return hasattr(module, "to_quantized") and weight is not None and weight.shape[-1] % group_size == 0

    nn.quantize(model, group_size=group_size, bits=bits, mode=mlx_mode, class_predicate=predicate)
    mx.eval(model.parameters())


def _record(totals: dict[str, float], counts: dict[str, int], name: str, start: float) -> None:
    totals[name] += time.perf_counter() - start
    counts[name] += 1


def _tokenize_prompt(tokenizer: Any, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def profile(args: argparse.Namespace) -> dict[str, Any]:
    dflash = _import_dflash(args.dflash_repo)
    sampler = make_sampler(temp=args.temperature)
    model, tokenizer = mlx_load(args.target)
    draft = dflash.load_draft(args.draft, sliding_window_size=args.sliding_window)
    _quantize(draft, args.draft_quant)
    draft.bind(model)
    mx.eval(model.parameters(), draft.parameters())

    dflash._patch_model(model, draft.config.target_layer_ids)
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    prompt = _tokenize_prompt(tokenizer, args.enable_thinking)
    prompt_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=tokenizer.bos_token is None))
    tokens = prompt_ids.tolist()
    mask_id = int(draft.config.mask_token_id)
    target_cache = make_prompt_cache(model)
    draft_cache = make_prompt_cache(draft)
    target_can_trim = can_trim_prompt_cache(target_cache)
    capture = dflash._GDNStateCapture() if not target_can_trim else None
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    accepted_lengths: list[int] = []
    n = 0

    try:
        start = time.perf_counter()
        with mx.stream(generation_stream):
            logits = model(prompt_ids[None], target_cache)
            hidden = mx.concatenate(model._hidden_states, axis=-1)
        mx.eval(logits, hidden)
        _record(totals, counts, "prefill_target", start)

        start = time.perf_counter()
        token = sampler(logits[:, -1:])[0, 0].item()
        tokens.append(token)
        n = 1
        _record(totals, counts, "first_sample", start)

        while n < args.max_tokens:
            bs = min(args.block_size, args.max_tokens - n + 1)
            if bs <= 1:
                break

            start = time.perf_counter()
            with mx.stream(generation_stream):
                block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
                draft_logits = draft(block, hidden, draft_cache)
                if (
                    draft.config.sliding_window_size is None
                    and (trim_n := draft_cache[0].offset - (prompt_ids.size + n - 1)) > 0
                ):
                    trim_prompt_cache(draft_cache, trim_n)
                draft_tokens = sampler(draft_logits[:, 1 - bs :])
            mx.eval(draft_tokens)
            _record(totals, counts, "draft", start)

            if capture is not None:
                capture.clear()
            start = time.perf_counter()
            with mx.stream(generation_stream):
                verify_input = mx.concatenate([mx.array([[tokens[-1]]]), draft_tokens], axis=1)
                logits = model(verify_input, target_cache)
                hidden = mx.concatenate(model._hidden_states, axis=-1)
                target_tokens = sampler(logits)
            mx.eval(target_tokens, hidden)
            _record(totals, counts, "verify_target", start)

            start = time.perf_counter()
            d_list = draft_tokens[0].tolist()
            t_list = target_tokens[0].tolist()
            accepted = next((i for i in range(len(d_list)) if d_list[i] != t_list[i]), len(d_list))
            new_tokens = [*d_list[:accepted], t_list[accepted]]
            new_tokens = new_tokens[: args.max_tokens - n]
            tokens.extend(new_tokens)
            n += len(new_tokens)
            trim = bs - accepted - 1
            if trim > 0:
                if target_can_trim:
                    trim_prompt_cache(target_cache, trim)
                elif capture is not None:
                    capture.rollback(target_cache, accepted, trim)
            hidden = hidden[:, : accepted + 1, :]
            accepted_lengths.append(accepted + 1)
            _record(totals, counts, "accept_rollback", start)
    finally:
        if capture is not None:
            capture.close()

    total_decode = totals["draft"] + totals["verify_target"] + totals["accept_rollback"]
    return {
        "target": args.target,
        "draft": args.draft,
        "draft_quant": args.draft_quant,
        "block_size": args.block_size,
        "sliding_window": args.sliding_window,
        "tokens": n,
        "decode_seconds_profiled": total_decode,
        "profiled_tps": n / total_decode if total_decode > 0 else 0.0,
        "avg_accept": sum(accepted_lengths) / max(len(accepted_lengths), 1),
        "chunks": len(accepted_lengths),
        "peak_gb": float(mx.get_peak_memory() / 1e9),
        "totals": dict(totals),
        "counts": dict(counts),
        "per_chunk_ms": {
            key: (1000.0 * totals[key] / max(counts[key], 1)) for key in ("draft", "verify_target", "accept_rollback")
        },
        "share": {
            key: (totals[key] / total_decode if total_decode > 0 else 0.0)
            for key in ("draft", "verify_target", "accept_rollback")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DFlash MLX stages.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--dflash-repo", default="/tmp/dflash-read")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--draft-quant", default="affine4")
    parser.add_argument("--sliding-window", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()
    print(json.dumps(profile(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
