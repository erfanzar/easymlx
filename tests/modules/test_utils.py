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

"""Minimal test helpers for easymlx model smoke tests."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class TestResult:
    __test__ = False
    success: bool
    error_message: str | None = None


def _get_vocab_size(config: tp.Any) -> int:
    if hasattr(config, "vocab_size"):
        return int(config.vocab_size)
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        if isinstance(text_cfg, dict):
            return int(text_cfg["vocab_size"])
        return int(text_cfg.vocab_size)
    thinker_cfg = getattr(config, "thinker_config", None)
    if isinstance(thinker_cfg, dict):
        return int(thinker_cfg["text_config"]["vocab_size"])
    raise ValueError("Unable to resolve vocab_size from config.")


def _get_image_token_id(config: tp.Any) -> int:
    if hasattr(config, "image_token_id"):
        return int(config.image_token_id)
    if hasattr(config, "image_token_index"):
        return int(config.image_token_index)
    thinker_cfg = getattr(config, "thinker_config", None)
    if isinstance(thinker_cfg, dict):
        return int(thinker_cfg["image_token_id"])
    raise ValueError("Unable to resolve image_token_id from config.")


def _make_input_ids(batch_size: int, seq_len: int, vocab_size: int) -> mx.array:
    base = mx.arange(batch_size * seq_len, dtype=mx.int32) % int(vocab_size)
    return base.reshape(batch_size, seq_len)


class CausalLMTester:
    def run(
        self,
        *,
        model_cls: type,
        config: tp.Any,
        small_model_config: dict[str, tp.Any],
        forward_kwargs: dict[str, tp.Any] | None = None,
        module_name: str | None = None,
        task: str | None = None,
    ) -> TestResult:
        del module_name, task
        try:
            model = model_cls(config)
            batch_size = int(small_model_config["batch_size"])
            seq_len = int(small_model_config["sequence_length"])
            vocab_size = _get_vocab_size(config)

            input_ids = _make_input_ids(batch_size, seq_len, vocab_size)
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
            kwargs = forward_kwargs or {}
            output = model(
                input_ids,
                attention_mask=attention_mask,
                return_dict=False,
                **kwargs,
            )
            logits = output if isinstance(output, mx.array) else output.logits
            expected = (batch_size, seq_len, vocab_size)
            if logits.shape != expected:
                raise AssertionError(f"Unexpected logits shape {logits.shape}, expected {expected}.")
            return TestResult(success=True)
        except Exception as exc:  # pragma: no cover - surfaced in test failures
            return TestResult(success=False, error_message=str(exc))

    def test_generation(
        self,
        *,
        model_cls: type,
        config: tp.Any,
        small_model_config: dict[str, tp.Any],
        max_new_tokens: int = 4,
        module_name: str | None = None,
        task: str | None = None,
    ) -> TestResult:
        del module_name, task
        try:
            model = model_cls(config)
            batch_size = int(small_model_config["batch_size"])
            seq_len = int(small_model_config["sequence_length"])
            vocab_size = _get_vocab_size(config)

            input_ids = _make_input_ids(batch_size, seq_len, vocab_size)
            generated = model.generate(input_ids, max_new_tokens=max_new_tokens)
            if generated.shape[0] != batch_size:
                raise AssertionError("Batch size mismatch in generated output.")
            if generated.shape[1] < seq_len:
                raise AssertionError("Generated sequence is shorter than input sequence.")
            return TestResult(success=True)
        except Exception as exc:  # pragma: no cover - surfaced in test failures
            return TestResult(success=False, error_message=str(exc))


class VisionLanguageTester:
    def run(
        self,
        *,
        model_cls: type,
        config: tp.Any,
        small_model_config: dict[str, tp.Any],
        vlm_config: dict[str, tp.Any],
        module_name: str | None = None,
        task: str | None = None,
    ) -> TestResult:
        del module_name, task
        try:
            model = model_cls(config)
            batch_size = int(small_model_config["batch_size"])
            vocab_size = _get_vocab_size(config)
            num_image_tokens = int(vlm_config["num_image_tokens"])
            seq_len = max(int(small_model_config["sequence_length"]), num_image_tokens + 1)
            image_token_id = int(vlm_config.get("image_token_id", _get_image_token_id(config)))

            if image_token_id >= vocab_size:
                raise AssertionError("image_token_id must be within vocab_size.")

            prefix = _make_input_ids(batch_size, 1, vocab_size)
            image_tokens = mx.full((batch_size, num_image_tokens), image_token_id, dtype=mx.int32)
            suffix_len = seq_len - (1 + num_image_tokens)
            if suffix_len > 0:
                suffix = _make_input_ids(batch_size, suffix_len, vocab_size)
                input_ids = mx.concatenate([prefix, image_tokens, suffix], axis=1)
            else:
                input_ids = mx.concatenate([prefix, image_tokens], axis=1)

            pixel_values_shape = vlm_config["pixel_values_shape"]
            pixel_values = mx.zeros(pixel_values_shape, dtype=mx.float32)
            attention_mask = mx.ones((batch_size, input_ids.shape[1]), dtype=mx.int32)

            output = model(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                return_dict=False,
            )
            logits = output if isinstance(output, mx.array) else output.logits
            expected = (batch_size, input_ids.shape[1], vocab_size)
            if logits.shape != expected:
                raise AssertionError(f"Unexpected logits shape {logits.shape}, expected {expected}.")
            return TestResult(success=True)
        except Exception as exc:  # pragma: no cover - surfaced in test failures
            return TestResult(success=False, error_message=str(exc))


class EasyMLXOnlyTester(CausalLMTester):
    """Alias for EasyDeLOnlyTester-style tests."""


__all__ = ("CausalLMTester", "EasyMLXOnlyTester", "TestResult", "VisionLanguageTester")
