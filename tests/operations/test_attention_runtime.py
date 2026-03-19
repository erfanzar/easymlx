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

import importlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from easymlx.caching import PageCacheView, PageMetadata, build_query_start_loc
from easymlx.layers.attention import FlexibleAttentionModule, scaled_dot_product_attention
from easymlx.operations import OperationRegistry, page_attention, paged_attention

_METAL_KERNEL_DIR = Path(__file__).resolve().parents[2] / "easymlx" / "operations" / "kernels" / "metal" / "page_attention"
_HAS_PAGE_ATTENTION_METAL = _METAL_KERNEL_DIR.is_dir()


def test_operation_registry_exposes_attention_impls():
    available = set(OperationRegistry.available_impls())
    assert "scaled_dot_product_attention" in available
    assert "sdpa" in available
    assert "vanilla_attention" in available
    assert "vanilla" in available
    assert "unified_attention" in available
    assert "page_attention" in available
    assert "paged_attention" in available


def test_exported_paged_attention_symbols_point_to_page_attention_module():
    assert page_attention.__module__ == "easymlx.operations.kernels.page_attention"
    assert paged_attention.__module__ == "easymlx.operations.kernels.page_attention"


@pytest.mark.skipif(not _HAS_PAGE_ATTENTION_METAL, reason="page_attention Metal kernels not vendored")
def test_page_attention_vendors_mistralrs_metal_kernel_files():
    kernel_dir = Path(__file__).resolve().parents[2] / "easymlx" / "operations" / "kernels" / "metal" / "page_attention"
    expected = {
        "copy_blocks.metal",
        "float8.metal",
        "gather_kv_cache.metal",
        "kv_scale_update.metal",
        "pagedattention.metal",
        "reshape_and_cache.metal",
        "runtime_header.metal",
        "utils.metal",
    }
    assert expected.issubset({path.name for path in kernel_dir.iterdir()})


@pytest.mark.skipif(not _HAS_PAGE_ATTENTION_METAL, reason="page_attention Metal kernels not vendored")
def test_flexible_attention_paged_routes_to_page_attention(monkeypatch):
    calls: dict[str, int] = {"page": 0, "unified": 0}
    page_attention_module = importlib.import_module("easymlx.operations.kernels.page_attention")

    def fake_page(
        queries,
        key_cache,
        value_cache,
        block_tables,
        kv_lens,
        query_start_loc,
        **kwargs,
    ):
        del key_cache, value_cache, block_tables, kv_lens, query_start_loc, kwargs
        calls["page"] += 1
        return mx.zeros_like(queries)

    def fake_unified(*args, **kwargs):
        del args, kwargs
        calls["unified"] += 1
        raise AssertionError("unified attention should not be used for page_attention execution")

    monkeypatch.setattr(page_attention_module, "paged_attention", fake_page)
    monkeypatch.setattr("easymlx.operations.kernels.unified_attention.paged_attention", fake_unified)

    cache = PageCacheView.allocate(
        num_seqs=1,
        max_seq_len=8,
        num_kv_heads=1,
        head_dim=4,
        block_size=4,
        dtype=mx.float32,
    )
    q = mx.zeros((2, 2, 4), dtype=mx.float32)
    k = mx.zeros((2, 1, 4), dtype=mx.float32)
    v = mx.zeros((2, 1, 4), dtype=mx.float32)
    _, key_cache, value_cache, metadata = cache.concatenate_to_cache(
        q,
        k,
        v,
        cache_metadata=PageMetadata(query_start_loc=build_query_start_loc([2])),
        slot_ids=[0],
    )

    out = scaled_dot_product_attention(
        q,
        key_cache,
        value_cache,
        cache_metadata=metadata,
        cache_view=cache,
        scale=0.5,
        mask=None,
        paged_attention_mechanism="page_attention",
    )
    assert out.shape == q.shape
    assert calls == {"page": 1, "unified": 0}


@pytest.mark.skipif(not _HAS_PAGE_ATTENTION_METAL, reason="page_attention Metal kernels not vendored")
def test_page_attention_uses_packed_cache_only_for_single_token_decode(monkeypatch):
    page_attention_module = importlib.import_module("easymlx.operations.kernels.page_attention")
    calls: list[tuple[int, bool | None]] = []

    def fake_page(
        queries,
        key_cache,
        value_cache,
        block_tables,
        kv_lens,
        query_start_loc,
        *,
        single_token_decode=None,
        **kwargs,
    ):
        del value_cache, block_tables, kv_lens, query_start_loc, kwargs
        calls.append((key_cache.ndim, single_token_decode))
        return mx.zeros_like(queries)

    monkeypatch.setattr(page_attention_module, "paged_attention", fake_page)

    cache = PageCacheView.allocate(
        num_seqs=1,
        max_seq_len=8,
        num_kv_heads=1,
        head_dim=4,
        block_size=4,
        dtype=mx.float32,
    )
    op = page_attention_module.PageAttention()

    q_prefill = mx.zeros((2, 2, 4), dtype=mx.float32)
    k_prefill = mx.zeros((2, 1, 4), dtype=mx.float32)
    v_prefill = mx.zeros((2, 1, 4), dtype=mx.float32)
    _, key_cache, value_cache, prefill_metadata = cache.concatenate_to_cache(
        q_prefill,
        k_prefill,
        v_prefill,
        cache_metadata=PageMetadata(query_start_loc=build_query_start_loc([2])),
        slot_ids=[0],
        query_lens=[2],
    )
    out = op.forward_native(
        query=q_prefill,
        key=key_cache,
        value=value_cache,
        cache_metadata=prefill_metadata,
        cache_view=cache,
        scale=0.5,
    )
    assert out.attention_outputs.shape == q_prefill.shape

    q_decode = mx.zeros((1, 2, 4), dtype=mx.float32)
    k_decode = mx.zeros((1, 1, 4), dtype=mx.float32)
    v_decode = mx.zeros((1, 1, 4), dtype=mx.float32)
    _, key_cache, value_cache, decode_metadata = cache.concatenate_to_cache(
        q_decode,
        k_decode,
        v_decode,
        cache_metadata=PageMetadata(
            query_start_loc=build_query_start_loc([1]),
            is_single_token_decode=True,
        ),
        slot_ids=[0],
        query_lens=[1],
    )
    out = op.forward_native(
        query=q_decode,
        key=key_cache,
        value=value_cache,
        cache_metadata=decode_metadata,
        cache_view=cache,
        scale=0.5,
    )
    assert out.attention_outputs.shape == q_decode.shape
    assert calls == [(4, False), (5, True)]


@pytest.mark.skip(reason="page_attention Metal kernel compilation not yet validated")
def test_page_attention_packed_decode_partitioned_matches_reference(monkeypatch):
    monkeypatch.setenv("EASYMLX_PAGE_ATTENTION_PARTITION_SIZE", "512")
    monkeypatch.setenv("EASYMLX_PAGE_ATTENTION_V2_MIN_CONTEXT", "1024")

    page_attention_module = importlib.import_module("easymlx.operations.kernels.page_attention")

    rng = np.random.default_rng(29)
    num_seqs = 2
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 256
    block_size = 64
    seq_lens = [1089, 1537]
    query_lens = [1, 1]

    cache = PageCacheView.allocate(
        num_seqs=num_seqs,
        max_seq_len=max(seq_lens),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=mx.float16,
    )
    for seq_idx, seq_len in enumerate(seq_lens):
        k = mx.array(rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float16))
        v = mx.array(rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float16))
        cache.append(seq_idx, k, v)

    queries = mx.array(rng.standard_normal((num_seqs, num_q_heads, head_dim)).astype(np.float16))
    query_start_loc = build_query_start_loc(query_lens)
    scale = 1.0 / np.sqrt(head_dim)

    expected = page_attention_module._paged_attention_reference_packed(
        queries,
        cache.page_key_cache,
        cache.page_value_cache,
        cache.block_tables,
        cache.kv_lens,
        query_start_loc,
        softmax_scale=scale,
        sliding_window=0,
    )
    actual = page_attention_module.paged_attention(
        queries,
        cache.page_key_cache,
        cache.page_value_cache,
        cache.block_tables,
        cache.kv_lens,
        query_start_loc,
        softmax_scale=scale,
        sliding_window=0,
        use_metal=True,
        single_token_decode=True,
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=3e-2, rtol=3e-2)


def test_page_attention_public_api_raises_when_metal_is_disabled():
    page_attention_module = importlib.import_module("easymlx.operations.kernels.page_attention")

    queries = mx.zeros((1, 2, 4), dtype=mx.float32)
    key_cache = mx.zeros((1, 4, 1, 4), dtype=mx.float32)
    value_cache = mx.zeros((1, 4, 1, 4), dtype=mx.float32)
    block_tables = mx.array([[0]], dtype=mx.int32)
    kv_lens = mx.array([1], dtype=mx.int32)
    query_start_loc = build_query_start_loc([1])

    with pytest.raises(NotImplementedError, match="Metal-only"):
        page_attention_module.paged_attention(
            queries,
            key_cache,
            value_cache,
            block_tables,
            kv_lens,
            query_start_loc,
            use_metal=False,
        )


def test_flexible_attention_dense_routes_to_sdpa(monkeypatch):
    calls: dict[str, int] = {"sdpa": 0, "paged": 0}

    def fake_sdpa(query, key, value, *, scale=None, mask=None, sinks=None):
        del key, value, scale, mask, sinks
        calls["sdpa"] += 1
        return mx.zeros_like(query)

    def fake_paged(*args, **kwargs):
        del args, kwargs
        calls["paged"] += 1
        raise AssertionError("paged attention should not be used for dense execution")

    monkeypatch.setattr(
        "easymlx.operations.kernels.scaled_dot_product_attention.mx.fast.scaled_dot_product_attention",
        fake_sdpa,
    )
    monkeypatch.setattr(
        "easymlx.operations.kernels.unified_attention.paged_attention",
        fake_paged,
    )

    q = mx.zeros((1, 2, 3, 4), dtype=mx.float32)
    k = mx.zeros((1, 1, 3, 4), dtype=mx.float32)
    v = mx.zeros((1, 1, 3, 4), dtype=mx.float32)

    out = scaled_dot_product_attention(q, k, v, scale=0.5, mask="causal")
    assert out.shape == q.shape
    assert calls == {"sdpa": 1, "paged": 0}


def test_flexible_attention_paged_routes_to_unified(monkeypatch):
    calls: dict[str, int] = {"sdpa": 0, "paged": 0}

    def fake_sdpa(*args, **kwargs):
        del args, kwargs
        calls["sdpa"] += 1
        raise AssertionError("sdpa should not be used for paged execution")

    def fake_paged(
        queries,
        key_cache,
        value_cache,
        block_tables,
        kv_lens,
        query_start_loc,
        *,
        softmax_scale=None,
        sliding_window=None,
        use_metal=None,
        single_token_decode=None,
        threadgroup_size=256,
    ):
        del key_cache, value_cache, block_tables, kv_lens, query_start_loc
        del softmax_scale, sliding_window, use_metal, single_token_decode, threadgroup_size
        calls["paged"] += 1
        return mx.zeros_like(queries)

    monkeypatch.setattr(
        "easymlx.operations.kernels.scaled_dot_product_attention.mx.fast.scaled_dot_product_attention",
        fake_sdpa,
    )
    monkeypatch.setattr(
        "easymlx.operations.kernels.unified_attention.paged_attention",
        fake_paged,
    )

    cache = PageCacheView.allocate(
        num_seqs=1,
        max_seq_len=8,
        num_kv_heads=1,
        head_dim=4,
        block_size=4,
        dtype=mx.float32,
    )
    q = mx.zeros((2, 2, 4), dtype=mx.float32)
    k = mx.zeros((2, 1, 4), dtype=mx.float32)
    v = mx.zeros((2, 1, 4), dtype=mx.float32)
    _, key_cache, value_cache, metadata = cache.concatenate_to_cache(
        q,
        k,
        v,
        cache_metadata=PageMetadata(query_start_loc=build_query_start_loc([2])),
        slot_ids=[0],
    )

    out = scaled_dot_product_attention(
        q,
        key_cache,
        value_cache,
        cache_metadata=metadata,
        cache_view=cache,
        scale=0.5,
        mask=None,
        paged_attention_mechanism="unified_attention",
    )
    assert out.shape == q.shape
    assert calls == {"sdpa": 0, "paged": 1}


def test_flexible_attention_module_validates_paged_inputs():
    module = FlexibleAttentionModule()
    q = mx.zeros((1, 2, 3, 4), dtype=mx.float32)
    k = mx.zeros((1, 1, 3, 4), dtype=mx.float32)
    v = mx.zeros((1, 1, 3, 4), dtype=mx.float32)

    try:
        module(q, k, v, cache_metadata=PageMetadata(query_start_loc=[0, 3]), scale=0.5)
    except ValueError as exc:
        assert "cache_metadata and cache_view" in str(exc)
    else:
        raise AssertionError("FlexibleAttentionModule should require both cache_metadata and cache_view")
