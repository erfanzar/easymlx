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
import numpy as np

from easymlx.caching import PagedKVCache, build_query_start_loc
from easymlx.operations.kernels.unified_attention import paged_attention


def _naive_paged_attention(
    queries: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    block_tables: np.ndarray,
    kv_lens: np.ndarray,
    query_start_loc: np.ndarray,
    *,
    softmax_scale: float,
    sliding_window: int,
) -> np.ndarray:
    _total_tokens, num_q_heads, head_dim = queries.shape
    num_blocks, block_size, num_kv_heads, _ = key_cache.shape
    num_seqs = block_tables.shape[0]
    num_queries_per_kv = num_q_heads // num_kv_heads

    out = np.zeros_like(queries, dtype=np.float32)
    for seq_idx in range(num_seqs):
        q_start = int(query_start_loc[seq_idx])
        q_end = int(query_start_loc[seq_idx + 1])
        q_len = q_end - q_start
        if q_len <= 0:
            continue
        seq_len = int(kv_lens[seq_idx])
        if seq_len <= 0:
            continue

        k_seq = np.zeros((seq_len, num_kv_heads, head_dim), dtype=np.float32)
        v_seq = np.zeros((seq_len, num_kv_heads, head_dim), dtype=np.float32)
        for pos in range(seq_len):
            block = int(block_tables[seq_idx, pos // block_size])
            if block < 0 or block >= num_blocks:
                continue
            offset = pos % block_size
            k_seq[pos] = key_cache[block, offset]
            v_seq[pos] = value_cache[block, offset]

        for local_idx in range(q_len):
            token_idx = q_start + local_idx
            q_vec = queries[token_idx].astype(np.float32) * softmax_scale
            context_len = seq_len - q_len
            max_k = min(seq_len - 1, context_len + local_idx)
            if max_k < 0:
                continue
            start_k = 0
            if sliding_window > 0:
                start_k = max(0, max_k - sliding_window + 1)

            k_slice = k_seq[start_k : max_k + 1]
            v_slice = v_seq[start_k : max_k + 1]
            if k_slice.size == 0:
                continue

            k_full = np.repeat(k_slice, repeats=num_queries_per_kv, axis=1)
            v_full = np.repeat(v_slice, repeats=num_queries_per_kv, axis=1)

            scores = np.einsum("hd,khd->hk", q_vec, k_full, optimize=True)
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights = weights / np.sum(weights, axis=-1, keepdims=True)
            out[token_idx] = np.einsum("hk,khd->hd", weights, v_full, optimize=True)
    return out.astype(queries.dtype)


def test_paged_attention_reference_matches():
    rng = np.random.default_rng(0)
    num_seqs = 2
    num_q_heads = 2
    num_kv_heads = 1
    head_dim = 4
    block_size = 4

    seq_lens = [5, 3]
    query_lens = [2, 1]
    cache = PagedKVCache.allocate(
        num_seqs=num_seqs,
        max_seq_len=max(seq_lens),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=mx.float32,
    )

    for seq_idx, seq_len in enumerate(seq_lens):
        k = mx.array(rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float32))
        v = mx.array(rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float32))
        cache.append(seq_idx, k, v)

    q_list = []
    for q_len in query_lens:
        q_list.append(rng.standard_normal((q_len, num_q_heads, head_dim)).astype(np.float32))
    queries = np.concatenate(q_list, axis=0)

    key_cache = cache.key_cache
    value_cache = cache.value_cache
    block_tables = cache.block_tables
    kv_lens = cache.kv_lens
    query_start_loc = build_query_start_loc(query_lens)

    scale = 1.0 / np.sqrt(head_dim)
    expected = _naive_paged_attention(
        queries,
        np.asarray(key_cache),
        np.asarray(value_cache),
        np.asarray(block_tables),
        np.asarray(kv_lens),
        np.asarray(query_start_loc),
        softmax_scale=scale,
        sliding_window=0,
    )

    actual = paged_attention(
        mx.array(queries),
        key_cache,
        value_cache,
        block_tables,
        kv_lens,
        query_start_loc,
        softmax_scale=scale,
        sliding_window=0,
        use_metal=False,
    )

    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-5, rtol=1e-5)
