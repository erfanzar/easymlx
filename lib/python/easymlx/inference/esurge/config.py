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

"""Minimal eSurge configuration objects for easymlx.

Provides dataclass-based configuration for the scheduler, KV cache
management, speculative decoding, and the unified engine config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class SchedulerConfig:
    """Configuration for request scheduling.

    Controls how the scheduler batches and prioritizes inference requests
    including maximum concurrency, token budgets, and chunked-prefill settings.

    Attributes:
        max_num_seqs: Maximum number of sequences that can be in-flight
            simultaneously.
        max_num_batched_tokens: Optional cap on the total number of tokens
            in a single batch. Clamped to ``max_model_len`` if larger.
        max_model_len: Maximum supported model context length.
        policy: Scheduling policy. ``"fcfs"`` for first-come-first-served,
            ``"priority"`` for priority-based ordering.
        long_prefill_token_threshold: Optional threshold above which a
            prefill is considered "long" and may be chunked.
        chunked_prefill_enabled: Whether to enable chunked prefill for
            long prompts.
        token_safety_margin: Optional token budget safety margin to reserve.
        max_num_seq_buckets: Optional tuple of bucket sizes for adaptive
            batching.
        async_scheduling: Whether the engine may schedule the next step
            while a previously submitted runner step is still pending.
    """

    max_num_seqs: int = 1
    max_num_batched_tokens: int | None = None
    max_model_len: int = 4096
    policy: Literal["priority", "fcfs"] = "fcfs"
    long_prefill_token_threshold: int | None = None
    chunked_prefill_enabled: bool = False
    token_safety_margin: int | None = None
    max_num_seq_buckets: tuple[int, ...] | None = None
    async_scheduling: bool = True

    def __post_init__(self) -> None:
        """Validate scheduler configuration parameters.

        Raises:
            ValueError: If any parameter violates its constraints (e.g.,
                non-positive values where positive values are required).
        """
        if self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {self.max_num_seqs}")
        if self.max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {self.max_model_len}")
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens <= 0:
            raise ValueError(f"max_num_batched_tokens must be positive, got {self.max_num_batched_tokens}")
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens > self.max_model_len:
            self.max_num_batched_tokens = self.max_model_len
        if self.long_prefill_token_threshold is not None and self.long_prefill_token_threshold < 0:
            raise ValueError(
                f"long_prefill_token_threshold must be non-negative, got {self.long_prefill_token_threshold}"
            )
        if self.token_safety_margin is not None and self.token_safety_margin < 0:
            raise ValueError(f"token_safety_margin must be non-negative, got {self.token_safety_margin}")
        if self.max_num_seq_buckets is not None:
            if not self.max_num_seq_buckets:
                raise ValueError("max_num_seq_buckets cannot be empty")
            if any(bucket <= 0 for bucket in self.max_num_seq_buckets):
                raise ValueError(f"All bucket sizes must be positive, got {self.max_num_seq_buckets}")


@dataclass
class CacheConfig:
    """Configuration for KV cache management.

    Attributes:
        num_pages: Total number of pages to allocate. ``None`` for automatic
            sizing based on available memory.
        page_size: Number of tokens stored per page.
        enable_prefix_caching: Whether to enable prefix-aware caching for
            shared prompt prefixes.
    """

    num_pages: int | None = None
    page_size: int = 16
    enable_prefix_caching: bool = False

    def __post_init__(self) -> None:
        """Validate cache configuration parameters.

        Raises:
            ValueError: If *page_size* is not positive or *num_pages* is
                specified but not positive.
        """
        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}")
        if self.num_pages is not None and self.num_pages <= 0:
            raise ValueError(f"num_pages must be positive when specified, got {self.num_pages}")


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.

    Attributes:
        num_speculative_tokens: Number of tokens to speculatively decode
            per step. ``0`` disables speculative decoding.
        speculative_model: Path, identifier, or already-loaded draft model
            used in speculative decoding. ``None`` disables speculative
            decoding.
        speculative_method: Speculative decoding method. ``"draft"`` uses
            a smaller draft language model. ``"dflash"`` uses a DFlash
            block-diffusion drafter. ``"eagle3"`` uses an EAGLE3 proposer
            fed by target-model hidden-state features.
        eagle3_feature_layer_indices: Optional target hidden-state layer
            indices to expose to EAGLE3. If omitted, the model chooses
            lower, middle, and final-layer features.
    """

    num_speculative_tokens: int = 0
    speculative_model: str | Any | None = None
    speculative_method: Literal["draft", "dflash", "eagle3"] = "draft"
    eagle3_feature_layer_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.num_speculative_tokens = int(self.num_speculative_tokens or 0)
        if self.num_speculative_tokens < 0:
            raise ValueError(f"num_speculative_tokens must be non-negative, got {self.num_speculative_tokens}")
        if self.speculative_method not in {"draft", "dflash", "eagle3"}:
            raise ValueError(f"Unsupported speculative_method: {self.speculative_method!r}")
        if self.eagle3_feature_layer_indices is not None:
            self.eagle3_feature_layer_indices = tuple(int(idx) for idx in self.eagle3_feature_layer_indices)

    def enabled(self) -> bool:
        """Determine whether speculative decoding is enabled.

        Returns:
            ``True`` if both ``num_speculative_tokens > 0`` and a draft
            or EAGLE3 model is configured.
        """
        return self.num_speculative_tokens > 0 and self.speculative_model is not None

    def use_eagle(self) -> bool:
        """Backward-compatible alias for older EasyDeL-style checks."""
        return self.enabled()

    def use_eagle3(self) -> bool:
        """Return whether the configured speculative method is EAGLE3."""
        return self.enabled() and self.speculative_method == "eagle3"


@dataclass
class Config:
    """Unified eSurge engine configuration.

    Aggregates scheduler, cache, and speculative decoding configurations
    into a single top-level object.

    Attributes:
        scheduler_config: Scheduling and batching configuration.
        cache_config: KV cache management configuration.
        speculative_config: Optional speculative decoding configuration.
            ``None`` disables speculative decoding.
    """

    scheduler_config: SchedulerConfig
    cache_config: CacheConfig
    speculative_config: SpeculativeConfig | None = None
