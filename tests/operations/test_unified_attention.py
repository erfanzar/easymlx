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

from easymlx.caching import PageCacheView, PageMetadata, build_query_start_loc
from easymlx.layers.rotary import get_rope
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
    cache = PageCacheView.allocate(
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


def test_paged_attention_metal_matches_reference(monkeypatch):
    monkeypatch.setenv("EASYMLX_UNIFIED_ATTENTION_ALLOW_FALLBACK", "0")

    rng = np.random.default_rng(1)
    num_seqs = 2
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 8
    block_size = 4

    seq_lens = [6, 4]
    query_lens = [2, 1]
    cache = PageCacheView.allocate(
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

    queries = np.concatenate(
        [
            rng.standard_normal((query_lens[0], num_q_heads, head_dim)).astype(np.float32),
            rng.standard_normal((query_lens[1], num_q_heads, head_dim)).astype(np.float32),
        ],
        axis=0,
    )
    query_start_loc = build_query_start_loc(query_lens)
    scale = 1.0 / np.sqrt(head_dim)

    expected = _naive_paged_attention(
        queries,
        np.asarray(cache.key_cache),
        np.asarray(cache.value_cache),
        np.asarray(cache.block_tables),
        np.asarray(cache.kv_lens),
        np.asarray(query_start_loc),
        softmax_scale=scale,
        sliding_window=0,
    )

    actual = paged_attention(
        mx.array(queries),
        cache.key_cache,
        cache.value_cache,
        cache.block_tables,
        cache.kv_lens,
        query_start_loc,
        softmax_scale=scale,
        sliding_window=0,
        use_metal=True,
        threadgroup_size=128,
    )

    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-5, rtol=1e-5)


def test_paged_attention_metal_decode_fast_path_matches_reference(monkeypatch):
    monkeypatch.setenv("EASYMLX_UNIFIED_ATTENTION_ALLOW_FALLBACK", "0")

    rng = np.random.default_rng(11)
    num_seqs = 3
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 8
    block_size = 4

    seq_lens = [5, 7, 3]
    query_lens = [1, 1, 1]
    cache = PageCacheView.allocate(
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

    queries = rng.standard_normal((num_seqs, num_q_heads, head_dim)).astype(np.float32)
    query_start_loc = build_query_start_loc(query_lens)
    scale = 1.0 / np.sqrt(head_dim)

    expected = _naive_paged_attention(
        queries,
        np.asarray(cache.key_cache),
        np.asarray(cache.value_cache),
        np.asarray(cache.block_tables),
        np.asarray(cache.kv_lens),
        np.asarray(query_start_loc),
        softmax_scale=scale,
        sliding_window=0,
    )

    actual = paged_attention(
        mx.array(queries),
        cache.key_cache,
        cache.value_cache,
        cache.block_tables,
        cache.kv_lens,
        query_start_loc,
        softmax_scale=scale,
        sliding_window=0,
        use_metal=True,
        threadgroup_size=128,
    )

    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-5, rtol=1e-5)


def test_page_cache_concatenate_to_cache_updates_multiple_slots():
    rng = np.random.default_rng(2)
    cache = PageCacheView.allocate(
        num_seqs=3,
        max_seq_len=8,
        num_kv_heads=1,
        head_dim=4,
        block_size=4,
        dtype=mx.float32,
    )

    initial_lens = [1, 0, 2]
    for seq_idx, seq_len in enumerate(initial_lens):
        if seq_len == 0:
            continue
        k = mx.array(rng.standard_normal((seq_len, 1, 4)).astype(np.float32))
        v = mx.array(rng.standard_normal((seq_len, 1, 4)).astype(np.float32))
        cache.append(seq_idx, k, v)

    initial_key_cache = np.asarray(cache.key_cache).copy()
    initial_value_cache = np.asarray(cache.value_cache).copy()
    initial_page_key_cache = np.asarray(cache.page_key_cache).copy()
    initial_page_value_cache = np.asarray(cache.page_value_cache).copy()
    initial_kv_lens = np.asarray(cache.kv_lens).copy()
    block_tables = np.asarray(cache.block_tables)
    page_vec_size = int(cache.page_vec_size)

    query_lens = [2, 0, 3]
    query_start_loc = build_query_start_loc(query_lens)
    queries = mx.array(rng.standard_normal((sum(query_lens), 2, 4)).astype(np.float32))
    keys = mx.array(rng.standard_normal((sum(query_lens), 1, 4)).astype(np.float32))
    values = mx.array(rng.standard_normal((sum(query_lens), 1, 4)).astype(np.float32))

    expected_key_cache = initial_key_cache.copy()
    expected_value_cache = initial_value_cache.copy()
    expected_page_key_cache = initial_page_key_cache.copy()
    expected_page_value_cache = initial_page_value_cache.copy()
    expected_kv_lens = initial_kv_lens.copy()
    qsl = np.asarray(query_start_loc)

    for seq_idx, slot in enumerate([0, 1, 2]):
        q_start = int(qsl[seq_idx])
        q_end = int(qsl[seq_idx + 1])
        q_len = q_end - q_start
        if q_len <= 0:
            continue
        offset = int(initial_kv_lens[slot])
        for local_idx in range(q_len):
            pos = offset + local_idx
            block = int(block_tables[slot, pos // cache.block_size])
            in_block = pos % cache.block_size
            key_token = np.asarray(keys)[q_start + local_idx]
            value_token = np.asarray(values)[q_start + local_idx]
            expected_key_cache[block, in_block] = key_token
            expected_value_cache[block, in_block] = value_token
            expected_page_key_cache[block, :, :, in_block, :] = key_token.reshape(
                key_token.shape[0],
                key_token.shape[1] // page_vec_size,
                page_vec_size,
            )
            expected_page_value_cache[block, :, :, in_block] = value_token
        expected_kv_lens[slot] = offset + q_len

    prepared_queries, key_cache, value_cache, metadata = cache.concatenate_to_cache(
        queries,
        keys,
        values,
        cache_metadata=PageMetadata(query_start_loc=query_start_loc),
        slot_ids=[0, 1, 2],
    )

    np.testing.assert_allclose(np.asarray(prepared_queries), np.asarray(queries), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(key_cache), expected_key_cache, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(value_cache), expected_value_cache, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(cache.page_key_cache), expected_page_key_cache, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(cache.page_value_cache), expected_page_value_cache, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(metadata.kv_lens), expected_kv_lens)
    np.testing.assert_array_equal(np.asarray(metadata.block_tables), block_tables[[0, 1, 2]])


def test_page_cache_single_token_decode_fast_path_applies_rope_and_updates_cache():
    rng = np.random.default_rng(7)
    cache = PageCacheView.allocate(
        num_seqs=3,
        max_seq_len=8,
        num_kv_heads=1,
        head_dim=4,
        block_size=4,
        dtype=mx.float32,
    )

    initial_lens = [2, 1, 3]
    for seq_idx, seq_len in enumerate(initial_lens):
        k = mx.array(rng.standard_normal((seq_len, 1, 4)).astype(np.float32))
        v = mx.array(rng.standard_normal((seq_len, 1, 4)).astype(np.float32))
        cache.append(seq_idx, k, v)

    queries = mx.array(rng.standard_normal((3, 2, 4)).astype(np.float32))
    keys = mx.array(rng.standard_normal((3, 1, 4)).astype(np.float32))
    values = mx.array(rng.standard_normal((3, 1, 4)).astype(np.float32))
    rope = get_rope(dims=4, base=10000.0, traditional=False)

    expected_queries = []
    expected_keys = []
    for seq_idx, offset in enumerate(initial_lens):
        q_chunk = queries[seq_idx : seq_idx + 1][None].transpose(0, 2, 1, 3)
        k_chunk = keys[seq_idx : seq_idx + 1][None].transpose(0, 2, 1, 3)
        expected_queries.append(rope(q_chunk, offset=offset).transpose(0, 2, 1, 3)[0, 0])
        expected_keys.append(rope(k_chunk, offset=offset).transpose(0, 2, 1, 3)[0, 0])
    expected_queries = np.stack([np.asarray(q) for q in expected_queries], axis=0)
    expected_keys = np.stack([np.asarray(k) for k in expected_keys], axis=0)

    prepared_queries, key_cache, value_cache, metadata = cache.concatenate_to_cache(
        queries,
        keys,
        values,
        cache_metadata=PageMetadata(query_start_loc=build_query_start_loc([1, 1, 1])),
        slot_ids=[0, 1, 2],
        query_lens=[1, 1, 1],
        rope=rope,
    )

    np.testing.assert_allclose(np.asarray(prepared_queries), expected_queries, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(metadata.kv_lens), np.asarray(initial_lens, dtype=np.int32) + 1)

    key_cache_np = np.asarray(key_cache)
    value_cache_np = np.asarray(value_cache)
    page_key_cache_np = np.asarray(cache.page_key_cache)
    page_value_cache_np = np.asarray(cache.page_value_cache)
    block_tables_np = np.asarray(cache.block_tables)
    page_vec_size = int(cache.page_vec_size)
    for seq_idx, offset in enumerate(initial_lens):
        block = int(block_tables_np[seq_idx, offset // cache.block_size])
        in_block = offset % cache.block_size
        np.testing.assert_allclose(key_cache_np[block, in_block], expected_keys[seq_idx], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(value_cache_np[block, in_block], np.asarray(values)[seq_idx], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            page_key_cache_np[block, :, :, in_block, :].reshape(1, 4),
            expected_keys[seq_idx].reshape(1, 4 // page_vec_size, page_vec_size).reshape(1, 4),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            page_value_cache_np[block, :, :, in_block],
            np.asarray(values)[seq_idx],
            atol=1e-5,
            rtol=1e-5,
        )
