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

from easymlx.caching import PageCache, PagedKVCache, PageMetadata, build_query_start_loc
from easymlx.layers.attention import FlexibleAttentionModule, scaled_dot_product_attention
from easymlx.operations import OperationRegistry


def test_operation_registry_exposes_attention_impls():
    available = set(OperationRegistry.available_impls())
    assert "scaled_dot_product_attention" in available
    assert "sdpa" in available
    assert "vanilla_attention" in available
    assert "vanilla" in available
    assert "unified_attention" in available
    assert "page_attention" in available
    assert "paged_attention" in available


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
        threadgroup_size=256,
    ):
        del key_cache, value_cache, block_tables, kv_lens, query_start_loc
        del softmax_scale, sliding_window, use_metal, threadgroup_size
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

    cache = PagedKVCache.allocate(
        num_seqs=1,
        max_seq_len=8,
        num_kv_heads=1,
        head_dim=4,
        block_size=4,
        dtype=mx.float32,
    )
    cache_view = PageCache(cache, [0])
    q = mx.zeros((2, 2, 4), dtype=mx.float32)
    k = mx.zeros((2, 1, 4), dtype=mx.float32)
    v = mx.zeros((2, 1, 4), dtype=mx.float32)
    _, key_cache, value_cache, metadata = cache_view.concatenate_to_cache(
        q,
        k,
        v,
        cache_metadata=PageMetadata(query_start_loc=build_query_start_loc([2])),
    )

    out = scaled_dot_product_attention(
        q,
        key_cache,
        value_cache,
        cache_metadata=metadata,
        cache_view=cache_view,
        scale=0.5,
        mask=None,
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
