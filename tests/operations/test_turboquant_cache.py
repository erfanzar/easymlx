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

import math

import mlx.core as mx
import numpy as np
import pytest
from easymlx.caching import PageCacheView, PageMetadata, build_query_start_loc
from easymlx.caching.paged import turboquant as turboquant_module
from easymlx.caching.paged.turboquant import get_turboquant_params
from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.layers.attention import scaled_dot_product_attention


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(a_flat, b_flat) / denom)


def test_base_config_accepts_turboquant_cache():
    config = EasyMLXBaseConfig(cache_dtype="turboquant", cache_bits=4)
    assert config.is_turboquant_cache is True
    assert config.cache_mlx_dtype == mx.uint32


def test_turboquant_page_cache_uses_packed_storage():
    head_dim = 64
    bits = 3
    block_size = 8
    num_seqs = 2
    num_kv_heads = 3
    max_seq_len = 24

    cache = PageCacheView.allocate(
        num_seqs=num_seqs,
        max_seq_len=max_seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        cache_dtype="turboquant",
        cache_bits=bits,
    )

    params = get_turboquant_params(head_dim, bits)
    assert getattr(cache, "cache_dtype_is_turboquant", False) is True
    assert cache.key_cache.dtype == mx.uint32
    assert cache.value_cache.dtype == mx.uint32
    assert cache.key_cache.shape[-1] == (head_dim * params.mse_bits + 31) // 32
    assert cache.value_cache.shape[-1] == (head_dim * bits + 31) // 32
    assert cache.key_qjl_signs.shape[-1] == (head_dim + 31) // 32

    turbo_bits = (
        np.asarray(cache.key_cache).size * 32
        + np.asarray(cache.value_cache).size * 32
        + np.asarray(cache.key_qjl_signs).size * 32
        + (
            np.asarray(cache.key_norms).size
            + np.asarray(cache.key_residual_norms).size
            + np.asarray(cache.value_norms).size
        )
        * 16
    )
    fp16_bits = num_seqs * math.ceil(max_seq_len / block_size) * block_size * num_kv_heads * head_dim * 2 * 16
    assert turbo_bits < fp16_bits


def test_turboquant_paged_attention_tracks_dense_cache():
    rng = np.random.default_rng(23)
    num_seqs = 2
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    block_size = 8
    seq_lens = [18, 13]
    query_lens = [2, 1]

    dense_cache = PageCacheView.allocate(
        num_seqs=num_seqs,
        max_seq_len=max(seq_lens),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=mx.float32,
    )
    turbo_cache = PageCacheView.allocate(
        num_seqs=num_seqs,
        max_seq_len=max(seq_lens),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=mx.float32,
        cache_dtype="turboquant",
        cache_bits=4,
    )

    for seq_idx, seq_len in enumerate(seq_lens):
        keys = mx.array(rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float32))
        values = mx.array(rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float32))
        dense_cache.append(seq_idx, keys, values)
        turbo_cache.append(seq_idx, keys, values)

    queries = mx.array(
        np.concatenate(
            [
                rng.standard_normal((query_lens[0], num_q_heads, head_dim)).astype(np.float32),
                rng.standard_normal((query_lens[1], num_q_heads, head_dim)).astype(np.float32),
            ],
            axis=0,
        )
    )
    query_start_loc = build_query_start_loc(query_lens)
    scale = 1.0 / math.sqrt(head_dim)

    dense_metadata = PageMetadata(
        query_start_loc=query_start_loc,
        block_tables=dense_cache.block_tables,
        kv_lens=dense_cache.kv_lens,
        block_size=dense_cache.block_size,
    )
    turbo_metadata = PageMetadata(
        query_start_loc=query_start_loc,
        block_tables=turbo_cache.block_tables,
        kv_lens=turbo_cache.kv_lens,
        block_size=turbo_cache.block_size,
    )

    expected = scaled_dot_product_attention(
        queries,
        dense_cache.key_cache,
        dense_cache.value_cache,
        cache_metadata=dense_metadata,
        cache_view=dense_cache,
        scale=scale,
        use_metal=False,
    )
    actual = scaled_dot_product_attention(
        queries,
        turbo_cache.key_cache,
        turbo_cache.value_cache,
        cache_metadata=turbo_metadata,
        cache_view=turbo_cache,
        scale=scale,
        use_metal=False,
    )

    expected_np = np.asarray(expected)
    actual_np = np.asarray(actual)
    assert actual_np.shape == expected_np.shape
    assert np.isfinite(actual_np).all()
    assert _cosine_similarity(actual_np, expected_np) > 0.97


def test_turboquant_runtime_append_and_attention_avoid_numpy_materialization(monkeypatch: pytest.MonkeyPatch):
    rng = np.random.default_rng(11)
    bits = 3
    head_dim = 32
    num_q_heads = 4
    num_kv_heads = 2

    cache = PageCacheView.allocate(
        num_seqs=1,
        max_seq_len=8,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=4,
        dtype=mx.float32,
        cache_dtype="turboquant",
        cache_bits=bits,
    )
    queries = mx.array(rng.standard_normal((2, num_q_heads, head_dim)).astype(np.float32))
    keys = mx.array(rng.standard_normal((2, num_kv_heads, head_dim)).astype(np.float32))
    values = mx.array(rng.standard_normal((2, num_kv_heads, head_dim)).astype(np.float32))
    metadata = PageMetadata(
        query_start_loc=build_query_start_loc([2]),
        block_tables=cache.block_tables,
        kv_lens=cache.kv_lens,
        block_size=cache.block_size,
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("TurboQuant runtime should stay on MLX arrays.")

    monkeypatch.setattr(turboquant_module.np, "array", _fail)
    monkeypatch.setattr(turboquant_module.np, "asarray", _fail)

    cache.append(0, keys, values)
    metadata = PageMetadata(
        query_start_loc=build_query_start_loc([2]),
        block_tables=cache.block_tables,
        kv_lens=cache.kv_lens,
        block_size=cache.block_size,
    )
    output = scaled_dot_product_attention(
        queries,
        cache.key_cache,
        cache.value_cache,
        cache_metadata=metadata,
        cache_view=cache,
        scale=1.0 / math.sqrt(head_dim),
        use_metal=False,
    )

    assert output.shape == queries.shape


def test_turboquant_single_token_decode_fast_path_matches_generic_path():
    rng = np.random.default_rng(19)
    num_seqs = 2
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 64
    block_size = 8
    seq_lens = [18, 13]
    query_lens = [1, 1]

    cache = PageCacheView.allocate(
        num_seqs=num_seqs,
        max_seq_len=max(seq_lens),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=mx.float32,
        cache_dtype="turboquant",
        cache_bits=3,
    )

    for seq_idx, seq_len in enumerate(seq_lens):
        keys = mx.array(rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float32))
        values = mx.array(rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float32))
        cache.append(seq_idx, keys, values)

    queries = mx.array(
        np.concatenate(
            [
                rng.standard_normal((query_lens[0], num_q_heads, head_dim)).astype(np.float32),
                rng.standard_normal((query_lens[1], num_q_heads, head_dim)).astype(np.float32),
            ],
            axis=0,
        )
    )
    query_start_loc = build_query_start_loc(query_lens)
    scale = 1.0 / math.sqrt(head_dim)

    slow_metadata = PageMetadata(
        query_start_loc=query_start_loc,
        block_tables=cache.block_tables,
        kv_lens=cache.kv_lens,
        block_size=cache.block_size,
        is_single_token_decode=False,
    )
    fast_metadata = PageMetadata(
        query_start_loc=query_start_loc,
        block_tables=cache.block_tables,
        kv_lens=cache.kv_lens,
        block_size=cache.block_size,
        is_single_token_decode=True,
    )

    expected = scaled_dot_product_attention(
        queries,
        cache.key_cache,
        cache.value_cache,
        cache_metadata=slow_metadata,
        cache_view=cache,
        scale=scale,
        use_metal=False,
    )
    actual = scaled_dot_product_attention(
        queries,
        cache.key_cache,
        cache.value_cache,
        cache_metadata=fast_metadata,
        cache_view=cache,
        scale=scale,
        use_metal=False,
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=5e-3, rtol=2e-3)


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_turboquant_pack_bits_roundtrip(bits: int) -> None:
    rng = np.random.default_rng(100 + bits)
    values = mx.array(rng.integers(0, 1 << bits, size=(3, 2, 37), dtype=np.uint32))

    packed = turboquant_module._pack_bits(values, bits)
    unpacked = turboquant_module._unpack_bits(packed, bits, values.shape[-1])

    np.testing.assert_array_equal(np.asarray(unpacked), np.asarray(values))


def test_turboquant_pack_sign_bits_roundtrip() -> None:
    rng = np.random.default_rng(211)
    signs = mx.array(rng.integers(0, 2, size=(2, 3, 65), dtype=np.uint32)).astype(mx.bool_)

    packed = turboquant_module._pack_sign_bits(signs)
    unpacked = turboquant_module._unpack_sign_bits(packed, signs.shape[-1])

    expected = np.where(np.asarray(signs), 1.0, -1.0).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(unpacked), expected)
