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
import pytest

from easymlx.caching import PageCache, PagedKVCache, PageMetadata, build_query_start_loc
from easymlx.layers.attention import AttentionPerformer, prepare_paged_attention_inputs
from easymlx.modules.llama.llama_configuration import LlamaConfig
from easymlx.modules.llama.modeling_llama import LlamaAttention
from easymlx.modules.qwen2.modeling_qwen2 import Qwen2Attention
from easymlx.modules.qwen2.qwen2_configuration import Qwen2Config
from easymlx.operations.kernels.unified_attention import paged_attention


def _clone_cache(cache: PagedKVCache) -> PagedKVCache:
    return PagedKVCache(
        key_cache=mx.array(cache.key_cache),
        value_cache=mx.array(cache.value_cache),
        block_tables=mx.array(cache.block_tables),
        kv_lens=mx.array(cache.kv_lens),
        block_size=cache.block_size,
        num_kv_heads=cache.num_kv_heads,
        head_dim=cache.head_dim,
    )


def test_attention_performer_paged_matches_manual_cache_update():
    rng = np.random.default_rng(0)
    num_q_heads = 2
    num_kv_heads = 1
    head_dim = 4
    performer = AttentionPerformer(scale=1.0 / np.sqrt(head_dim))

    cache = PagedKVCache.allocate(
        num_seqs=2,
        max_seq_len=8,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=4,
        dtype=mx.float32,
    )
    initial_k0 = mx.array(rng.standard_normal((2, num_kv_heads, head_dim)).astype(np.float32))
    initial_v0 = mx.array(rng.standard_normal((2, num_kv_heads, head_dim)).astype(np.float32))
    initial_k1 = mx.array(rng.standard_normal((1, num_kv_heads, head_dim)).astype(np.float32))
    initial_v1 = mx.array(rng.standard_normal((1, num_kv_heads, head_dim)).astype(np.float32))
    cache.append(0, initial_k0, initial_v0)
    cache.append(1, initial_k1, initial_v1)

    manual_cache = _clone_cache(cache)
    query_lens = [2, 1]
    query_start_loc = build_query_start_loc(query_lens)
    queries = mx.array(rng.standard_normal((sum(query_lens), num_q_heads, head_dim)).astype(np.float32))
    keys = mx.array(rng.standard_normal((sum(query_lens), num_kv_heads, head_dim)).astype(np.float32))
    values = mx.array(rng.standard_normal((sum(query_lens), num_kv_heads, head_dim)).astype(np.float32))

    manual_cache.append(0, keys[:2], values[:2])
    manual_cache.append(1, keys[2:], values[2:])

    active_block_tables = manual_cache.block_tables[mx.array([0, 1])]
    active_kv_lens = manual_cache.kv_lens[mx.array([0, 1])]
    expected = paged_attention(
        queries,
        manual_cache.key_cache,
        manual_cache.value_cache,
        active_block_tables,
        active_kv_lens,
        mx.array(query_start_loc, dtype=mx.int32) if not isinstance(query_start_loc, mx.array) else query_start_loc,
        softmax_scale=1.0 / np.sqrt(head_dim),
        sliding_window=0,
        use_metal=False,
    )

    # Use unified __call__ with PageCache + PageMetadata
    cache_view = PageCache(cache, [0, 1])
    cache_metadata = PageMetadata(query_start_loc=query_start_loc)
    actual = performer(
        queries,
        keys,
        values,
        cache_view=cache_view,
        cache_metadata=cache_metadata,
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(cache.kv_lens), np.asarray(manual_cache.kv_lens))


def test_prepare_paged_attention_inputs_returns_page_metadata():
    cache = PagedKVCache.allocate(
        num_seqs=1,
        max_seq_len=4,
        num_kv_heads=1,
        head_dim=4,
        block_size=4,
        dtype=mx.float32,
    )
    prepared = prepare_paged_attention_inputs(
        mx.zeros((2, 2, 4), dtype=mx.float32),
        mx.zeros((2, 1, 4), dtype=mx.float32),
        mx.zeros((2, 1, 4), dtype=mx.float32),
        kv_cache=cache,
        query_start_loc=build_query_start_loc([2]),
        slot_ids=[0],
    )

    assert isinstance(prepared.metadata, PageMetadata)
    assert prepared.metadata.block_size == 4


@pytest.mark.parametrize(
    ("factory", "hidden_size", "num_heads", "num_kv_heads"),
    [
        (
            lambda: LlamaAttention(
                LlamaConfig(
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    max_position_embeddings=16,
                )
            ),
            8,
            2,
            1,
        ),
        (
            lambda: Qwen2Attention(
                Qwen2Config(
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    max_position_embeddings=16,
                    layer_types=["full_attention"],
                )
            ),
            8,
            2,
            1,
        ),
    ],
)
def test_attention_modules_delegate_all_to_performer(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
):
    """Model attention modules must delegate everything to the performer via __call__.

    The model attention should only: project Q/K/V → call performer → output project.
    No branching, no layout transforms, no RoPE — the performer handles it all.
    """
    attention = factory()
    head_dim = hidden_size // num_heads

    # -- Dense path --
    calls: dict[str, object] = {}

    def fake_performer(queries, keys, values, **kwargs):
        calls["queries_shape"] = tuple(queries.shape)
        calls["keys_shape"] = tuple(keys.shape)
        calls["mask"] = kwargs.get("mask")
        calls["cache_view"] = kwargs.get("cache_view")
        calls["cache_metadata"] = kwargs.get("cache_metadata")
        calls["rope"] = kwargs.get("rope")
        # Return same shape as queries (performer contract)
        return mx.zeros_like(queries)

    monkeypatch.setattr(attention, "attention_performer", type("FakePerformer", (), {"__call__": lambda self, *a, **kw: fake_performer(*a, **kw)})())
    dense_out = attention(mx.zeros((1, 3, hidden_size), dtype=mx.float32), mask="causal")

    # Model passes BTHD [B, L, H, D] to performer
    assert calls["queries_shape"] == (1, 3, num_heads, head_dim)
    assert calls["keys_shape"] == (1, 3, num_kv_heads, head_dim)
    assert calls["mask"] == "causal"
    assert calls["cache_view"] is None
    assert calls["cache_metadata"] is None
    assert calls["rope"] is not None  # rope should always be passed
    assert dense_out.shape == (1, 3, hidden_size)

    # -- Paged path (3D input, single sequence) --
    calls.clear()
    paged_cache = PagedKVCache.allocate(
        num_seqs=2,
        max_seq_len=8,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=4,
        dtype=mx.float32,
    )
    cache_view = PageCache(paged_cache, [0])
    cache_metadata = PageMetadata(query_start_loc=build_query_start_loc([2]))
    cache_out = attention(
        mx.zeros((1, 2, hidden_size), dtype=mx.float32),
        cache_view=cache_view,
        cache_metadata=cache_metadata,
    )

    # Model passes BTHD [1, 2, H, D] to performer — performer flattens internally
    assert calls["queries_shape"] == (1, 2, num_heads, head_dim)
    assert calls["keys_shape"] == (1, 2, num_kv_heads, head_dim)
    assert isinstance(calls["cache_view"], PageCache)
    assert isinstance(calls["cache_metadata"], PageMetadata)
    assert cache_out.shape == (1, 2, hidden_size)

    # -- Paged path (multi-sequence, 3D input) --
    calls.clear()
    paged_cache2 = PagedKVCache.allocate(
        num_seqs=2,
        max_seq_len=8,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=4,
        dtype=mx.float32,
    )
    paged_view = PageCache(paged_cache2, [0, 1])
    paged_metadata = PageMetadata(query_start_loc=build_query_start_loc([2, 1]))
    paged_out = attention(
        mx.zeros((1, 3, hidden_size), dtype=mx.float32),
        cache_view=paged_view,
        cache_metadata=paged_metadata,
    )
    assert calls["queries_shape"] == (1, 3, num_heads, head_dim)
    assert isinstance(calls["cache_view"], PageCache)
    assert list(calls["cache_view"].slot_ids) == [0, 1]
    assert paged_out.shape == (1, 3, hidden_size)
