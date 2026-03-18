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

"""Focused tests for the refactored eSurge runner stack."""

from __future__ import annotations

import numpy as np

from easymlx.inference.esurge.runners import (
    ExecutionManager,
    ExecutionRequest,
    ModelRunner,
    ScheduledSequence,
    SequenceBuffer,
)
from easymlx.inference.esurge.sampling_params import SamplingParams


class DummyPagedModel:
    """Deterministic paged model used for runner tests."""

    def __init__(self, vocab_size: int = 8):
        self.vocab_size = vocab_size

    def __call__(self, input_ids=None, *, positions=None, slot_ids=None, **_kwargs):
        del positions
        logits = np.full((len(input_ids), self.vocab_size), -50.0, dtype=np.float32)
        for index, token_ids in enumerate(input_ids):
            target = (int(sum(token_ids)) + int(slot_ids[index])) % self.vocab_size
            logits[index, target] = 10.0
        return {"logits": logits}


def test_sequence_buffer_row_ops_and_page_table() -> None:
    buffer = SequenceBuffer(max_num_rows=6)
    row0 = buffer.begin_sequence("req-0", [1, 2], page_ids=[10, 11], row_index=0)
    row2 = buffer.begin_sequence("req-2", [3], page_ids=[20], row_index=2)
    row1 = buffer.begin_sequence("req-1", [4], page_ids=[30, 31])
    assert (row0, row1, row2) == (0, 1, 2)

    removed = buffer.remove_sequence("req-1")
    assert removed.request_id == "req-1"

    moved = buffer.compact_holes(0, 3)
    assert moved == [(2, 1)]
    assert buffer.get_row_index("req-2") == 1

    buffer.swap_rows(0, 1)
    assert buffer.get_row_index("req-0") == 1
    assert buffer.get_row_index("req-2") == 0

    buffer.move_row(1, 3)
    assert buffer.get_row_index("req-0") == 3

    page_table = buffer.page_table()
    assert page_table.shape == (4, 2)
    assert page_table[0, 0] == 20
    assert page_table[3, 0] == 10


def test_model_runner_normalizes_step_updates_and_mutates_buffer() -> None:
    model = DummyPagedModel(vocab_size=7)
    buffer = SequenceBuffer()
    buffer.begin_sequence("req-a", [10], row_index=0)
    buffer.begin_sequence("req-b", [20], row_index=1)
    runner = ModelRunner(model, sequence_buffer=buffer, seed=0)

    request = ExecutionRequest(
        step_id=42,
        mode="mixed",
        sequences=[
            ScheduledSequence(request_id="req-a", row_index=0, token_ids=[10], num_computed_tokens=1),
            ScheduledSequence(request_id="req-b", row_index=1, token_ids=[20], num_computed_tokens=1),
        ],
        sampling_by_request={
            "req-a": SamplingParams(max_tokens=1, do_sample=False),
            "req-b": SamplingParams(max_tokens=1, do_sample=False),
        },
    )

    result = runner.run(request)
    assert result.step_id == 42
    assert result.req_ids == ["req-a", "req-b"]
    assert result.metadata["mode"] == "mixed"
    assert result.sampled_token_ids == [[3], [0]]
    assert buffer.get_row("req-a").output_token_ids == [3]
    assert buffer.get_row("req-b").output_token_ids == [0]


def test_execution_manager_async_collect() -> None:
    model = DummyPagedModel(vocab_size=5)
    runner = ModelRunner(model, seed=0)
    manager = ExecutionManager(runner)
    request = ExecutionRequest(
        step_id=7,
        sequences=[ScheduledSequence(request_id="req-x", row_index=0, token_ids=[2], num_computed_tokens=0)],
        sampling_by_request={"req-x": SamplingParams(max_tokens=1, do_sample=False)},
    )

    pending = manager.execute_async(request)
    result = manager.collect(pending, timeout=1.0)
    assert result.step_id == 7
    assert result.req_ids == ["req-x"]
    assert result.sampled_token_ids == [[2]]
    manager.close()
