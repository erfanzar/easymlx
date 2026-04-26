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

"""Benchmark effective TFLOP/s for easymlx causal-LM generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
import typing as tp
from pathlib import Path

from mlx import core as mx

python_src = Path(__file__).resolve().parents[1] / "lib" / "python"
if str(python_src) not in sys.path:
    sys.path.insert(0, str(python_src))

import easymlx as ex  # noqa: E402
from easymlx.infra.flops import (  # noqa: E402
    _attention_flops,
    _attention_window_for_layer,
    _dense_intermediate_size,
    _estimate_lm_head_flops,
    _estimate_sparse_mlp_flops,
    _estimate_text_model_flops,
    _gated_mlp_flops,
    _get_head_dim,
    _get_hidden_size,
    _get_num_kv_heads,
    _is_full_attention_layer,
    _is_mla_text_config,
    _is_qwen3_next_text_config,
    _is_sparse_mlp_layer,
    _linear_flops,
    _require_positive_int,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local easymlx checkpoint directory or repo id.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer repo id or local tokenizer directory.")
    parser.add_argument("--prompt", required=True, help="User prompt to send through chat().")
    parser.add_argument("--max-tokens", type=int, default=128, help="Generated tokens for the measured run.")
    parser.add_argument("--warmup-tokens", type=int, default=24, help="Warmup generation length before measuring.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling cutoff.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling cutoff.")
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use stochastic sampling. Disable for greedy/speculative decode benchmarks.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for MLX sampling.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
        help="Model compute dtype.",
    )
    parser.add_argument(
        "--cache-dtype",
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
        help="KV cache dtype.",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional easymlx quantization mode, for example 'mxfp4'.",
    )
    parser.add_argument(
        "--attn-mechanism",
        default="unified",
        choices=("paged", "unified"),
        help="Attention/cache path to benchmark.",
    )
    parser.add_argument("--page-size", type=int, default=128, help="Paged KV page size.")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum modeled context length.")
    parser.add_argument("--max-num-seqs", type=int, default=1, help="Maximum concurrent running sequences.")
    parser.add_argument(
        "--memory-utilization",
        type=float,
        default=0.45,
        help="Fraction of available memory the scheduler may reserve.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to pass enable_thinking into the chat template.",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the streaming chat API. Disable to exercise eSurge blocking/speculative paths.",
    )
    parser.add_argument(
        "--runner-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose eSurge runner logs.",
    )
    parser.add_argument(
        "--compile-runner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable eSurge runner compile/warmup.",
    )
    parser.add_argument("--speculative-model", default=None, help="Optional draft/EAGLE3 model for eSurge speculation.")
    parser.add_argument("--num-speculative-tokens", type=int, default=0, help="Draft tokens proposed per round.")
    parser.add_argument(
        "--speculative-method",
        default="draft",
        choices=("draft", "dflash", "eagle3"),
        help="Speculative decode method.",
    )
    parser.add_argument(
        "--speculative-quantization",
        default=None,
        help="Optional quantization mode for a string-loaded speculative model.",
    )
    return parser.parse_args()


def _mx_dtype(name: str) -> mx.Dtype:
    return {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }[name]


def _sampling_params(args: argparse.Namespace) -> ex.SamplingParams:
    return ex.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=bool(args.do_sample),
    )


def _warmup_sampling_params(args: argparse.Namespace) -> ex.SamplingParams:
    return ex.SamplingParams(
        max_tokens=args.warmup_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=bool(args.do_sample),
    )


def _run_chat(
    engine: ex.eSurge,
    *,
    prompt: str,
    sampling_params: ex.SamplingParams,
    enable_thinking: bool,
    stream: bool,
) -> tuple[tp.Any, float, float | None, float]:
    messages = [{"role": "user", "content": prompt}]
    start = time.perf_counter()
    first_token_at = None
    final_output = None
    if not stream:
        final_output = engine.chat(
            messages=messages,
            sampling_params=sampling_params,
            stream=False,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        end = time.perf_counter()
        return final_output, start, None, end

    for output in engine.chat(
        messages=messages,
        sampling_params=sampling_params,
        stream=True,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    ):
        final_output = output
        if first_token_at is None and output.num_generated_tokens > 0:
            first_token_at = time.perf_counter()
    end = time.perf_counter()
    if final_output is None:
        raise RuntimeError("engine.chat() produced no output.")
    return final_output, start, first_token_at, end


def _dense_or_sparse_mlp_flops(config: tp.Any, *, tokens: int, hidden_size: int, layer_idx: int) -> int:
    if _is_sparse_mlp_layer(config, layer_idx):
        return _estimate_sparse_mlp_flops(config, tokens=tokens, hidden_size=hidden_size)
    return _gated_mlp_flops(tokens, hidden_size, _dense_intermediate_size(config))


def _estimate_standard_cached_decode_step_flops(config: tp.Any, *, context_length: int) -> int:
    hidden_size = _get_hidden_size(config)
    num_layers = _require_positive_int("num_hidden_layers", config.num_hidden_layers)
    num_heads = _require_positive_int("num_attention_heads", config.num_attention_heads)
    num_kv_heads = _get_num_kv_heads(config, num_heads)
    head_dim = _get_head_dim(config, hidden_size=hidden_size, num_heads=num_heads)
    q_proj_dim = num_heads * head_dim
    kv_proj_dim = num_kv_heads * head_dim

    total = 0
    for layer_idx in range(num_layers):
        key_length = _attention_window_for_layer(config, layer_idx, context_length)
        total += _attention_flops(
            tokens=1,
            batch_size=1,
            query_length=1,
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
        total += _dense_or_sparse_mlp_flops(config, tokens=1, hidden_size=hidden_size, layer_idx=layer_idx)

    total += _estimate_lm_head_flops(config, batch_size=1, sequence_length=1)
    return int(total)


def _estimate_qwen3_next_cached_decode_step_flops(config: tp.Any, *, context_length: int) -> int:
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

    total = 0
    for layer_idx in range(num_layers):
        if _is_full_attention_layer(config, layer_idx):
            total += _attention_flops(
                tokens=1,
                batch_size=1,
                query_length=1,
                key_length=context_length,
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
            total += _linear_flops(1, hidden_size, linear_inner_dim)
            total += _linear_flops(1, hidden_size, value_dim)
            total += _linear_flops(1, hidden_size, linear_num_value_heads)
            total += _linear_flops(1, hidden_size, linear_num_value_heads)
            total += _linear_flops(1, value_dim, hidden_size)
            total += 2 * linear_inner_dim * linear_conv_kernel_dim
            total += linear_num_value_heads * (7 * linear_key_head_dim * linear_value_head_dim)

        total += _dense_or_sparse_mlp_flops(config, tokens=1, hidden_size=hidden_size, layer_idx=layer_idx)

    total += _estimate_lm_head_flops(config, batch_size=1, sequence_length=1)
    return int(total)


def _estimate_mla_cached_decode_step_flops(config: tp.Any, *, context_length: int) -> int:
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

    total = 0
    for layer_idx in range(num_layers):
        key_length = _attention_window_for_layer(config, layer_idx, context_length)
        if q_lora_rank is None:
            total += _linear_flops(1, hidden_size, num_heads * q_head_dim)
        else:
            total += _linear_flops(1, hidden_size, q_lora_rank)
            total += _linear_flops(1, q_lora_rank, num_heads * q_head_dim)

        total += _linear_flops(1, hidden_size, kv_lora_rank + qk_rope_head_dim)
        total += _linear_flops(1, kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
        total += _linear_flops(1, num_heads * v_head_dim, hidden_size)
        total += 2 * num_heads * key_length * (q_head_dim + v_head_dim)
        total += _dense_or_sparse_mlp_flops(config, tokens=1, hidden_size=hidden_size, layer_idx=layer_idx)

    total += _estimate_lm_head_flops(config, batch_size=1, sequence_length=1)
    return int(total)


def _estimate_cached_decode_step_flops(config: tp.Any, *, context_length: int) -> int:
    if _is_mla_text_config(config):
        return _estimate_mla_cached_decode_step_flops(config, context_length=context_length)
    if _is_qwen3_next_text_config(config):
        return _estimate_qwen3_next_cached_decode_step_flops(config, context_length=context_length)
    return _estimate_standard_cached_decode_step_flops(config, context_length=context_length)


def _prefill_flops(config: tp.Any, *, prompt_length: int, paged_runtime: bool) -> int:
    lm_head_tokens = 1 if paged_runtime else prompt_length
    return _estimate_text_model_flops(config, batch_size=1, sequence_length=prompt_length) + _estimate_lm_head_flops(
        config,
        batch_size=1,
        sequence_length=lm_head_tokens,
    )


def _build_result(
    *,
    text_config: tp.Any,
    output: tp.Any,
    started_at: float,
    first_token_at: float | None,
    ended_at: float,
    attn_mechanism: str,
) -> dict[str, tp.Any]:
    prompt_tokens = len(output.prompt_token_ids)
    generated_tokens = int(output.num_generated_tokens)
    total_wall_s = ended_at - started_at
    ttft_s = (first_token_at - started_at) if first_token_at is not None else None
    post_first_wall_s = (ended_at - first_token_at) if first_token_at is not None else None

    prefill_flops = _prefill_flops(
        text_config,
        prompt_length=prompt_tokens,
        paged_runtime=attn_mechanism == "paged",
    )
    decode_total_flops = 0
    for generated_idx in range(1, generated_tokens):
        decode_total_flops += _estimate_cached_decode_step_flops(
            text_config,
            context_length=prompt_tokens + generated_idx,
        )
    total_flops = int(prefill_flops + decode_total_flops)

    full_attention_layers = None
    if _is_qwen3_next_text_config(text_config):
        full_attention_layers = sum(
            1 for layer_idx in range(text_config.num_hidden_layers) if _is_full_attention_layer(text_config, layer_idx)
        )

    return {
        "model_type": type(text_config).__name__,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "engine_tokens_per_second": float(output.tokens_per_second),
        "time_to_first_token_s": ttft_s,
        "total_wall_s": total_wall_s,
        "post_first_wall_s": post_first_wall_s,
        "prefill_flops": int(prefill_flops),
        "decode_total_flops": int(decode_total_flops),
        "total_flops": int(total_flops),
        "effective_end_to_end_tflops": total_flops / total_wall_s / 1e12,
        "effective_post_first_tflops": (
            decode_total_flops / post_first_wall_s / 1e12 if post_first_wall_s and generated_tokens > 1 else None
        ),
        "avg_total_gflops_per_generated_token": total_flops / max(generated_tokens, 1) / 1e9,
        "avg_decode_gflops_per_post_first_token": (
            decode_total_flops / max(generated_tokens - 1, 1) / 1e9 if generated_tokens > 1 else None
        ),
        "hidden_size": text_config.hidden_size,
        "intermediate_size": getattr(text_config, "intermediate_size", None),
        "num_hidden_layers": text_config.num_hidden_layers,
        "num_attention_heads": text_config.num_attention_heads,
        "num_key_value_heads": getattr(text_config, "num_key_value_heads", None),
        "full_attention_layers": full_attention_layers,
        "metrics": output.metrics,
    }


def main() -> None:
    args = _parse_args()
    mx.random.seed(args.seed)

    quantization = None
    if args.quantization is not None:
        quantization = ex.QuantizationConfig(mode=args.quantization)

    speculative_model_kwargs: dict[str, tp.Any] = {}
    if args.speculative_quantization is not None:
        speculative_model_kwargs["quantization"] = ex.QuantizationConfig(mode=args.speculative_quantization)

    model = ex.AutoEasyMLXModelForCausalLM.from_pretrained(
        args.model,
        device=mx.gpu,
        dtype=_mx_dtype(args.dtype),
        quantization=quantization,
        config={
            "attn_mechanism": args.attn_mechanism,
            "cache_dtype": args.cache_dtype,
        },
    )
    engine = ex.eSurge(
        model,
        tokenizer=args.tokenizer,
        max_model_len=args.max_model_len,
        page_size=args.page_size,
        runner_verbose=args.runner_verbose,
        max_num_seqs=args.max_num_seqs,
        memory_utilization=args.memory_utilization,
        compile_runner=bool(args.compile_runner),
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        speculative_method=args.speculative_method,
        speculative_model_kwargs=speculative_model_kwargs,
    )
    text_config = model.config.text_config if hasattr(model.config, "text_config") else model.config

    _run_chat(
        engine,
        prompt=args.prompt,
        sampling_params=_warmup_sampling_params(args),
        enable_thinking=args.enable_thinking,
        stream=args.stream,
    )
    output, started_at, first_token_at, ended_at = _run_chat(
        engine,
        prompt=args.prompt,
        sampling_params=_sampling_params(args),
        enable_thinking=args.enable_thinking,
        stream=args.stream,
    )

    result = _build_result(
        text_config=text_config,
        output=output,
        started_at=started_at,
        first_token_at=first_token_at,
        ended_at=ended_at,
        attn_mechanism=args.attn_mechanism,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
