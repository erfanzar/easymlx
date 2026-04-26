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

"""Token-budget helpers for schedulers.

Provides a simple two-tier budgeting system: :class:`TokenBudget` is a
mutable budget for a single scheduling step, and
:class:`TokenBudgetManager` is a reusable factory that stamps out fresh
budgets with a configurable safety margin.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TokenBudget:
    """Mutable token budget for one schedule step.

    Tracks a fixed total capacity, a reserved portion (safety margin),
    and the number of tokens consumed so far.

    Attributes:
        total: Total token capacity for the step.
        reserved: Tokens reserved as a safety margin.
        used: Tokens consumed so far.
    """

    total: int
    reserved: int = 0
    used: int = 0

    @property
    def available(self) -> int:
        """Return the number of tokens still available for allocation.

        Returns:
            ``total - reserved - used``, clamped to >= 0.
        """
        return max(int(self.total) - int(self.reserved) - int(self.used), 0)

    def reset(self, *, total: int | None = None, reserved: int | None = None) -> None:
        """Reset the budget, optionally changing total or reserved values.

        Args:
            total: New total capacity.  ``None`` keeps the current value.
            reserved: New reserved amount.  ``None`` keeps the current
                value.
        """
        if total is not None:
            self.total = int(total)
        if reserved is not None:
            self.reserved = int(reserved)
        self.used = 0

    def can_fit(self, token_count: int) -> bool:
        """Check whether *token_count* tokens can fit in the remaining budget.

        Args:
            token_count: Number of tokens to check.

        Returns:
            ``True`` if the budget has enough room.
        """
        return self.available >= max(int(token_count), 0)

    def consume(self, token_count: int) -> int:
        """Consume up to *token_count* tokens from the budget.

        If the requested amount exceeds the available budget, only the
        available portion is consumed.

        Args:
            token_count: Number of tokens to consume.

        Returns:
            The actual number of tokens consumed (may be less than
            requested).
        """
        wanted = max(int(token_count), 0)
        granted = min(wanted, self.available)
        self.used += granted
        return granted


@dataclass(slots=True)
class TokenBudgetManager:
    """Reusable budget planner with safety margin support.

    Creates fresh :class:`TokenBudget` instances for each scheduling
    step, applying a configurable safety margin.

    Attributes:
        max_batch_tokens: Maximum tokens per scheduling step.
        safety_margin_tokens: Tokens to reserve as a safety margin.
    """

    max_batch_tokens: int
    safety_margin_tokens: int = 0

    def make_step_budget(self) -> TokenBudget:
        """Create a new :class:`TokenBudget` for a scheduling step.

        Returns:
            A fresh budget with the configured total and reserved
            amounts.
        """
        return TokenBudget(total=max(int(self.max_batch_tokens), 0), reserved=max(int(self.safety_margin_tokens), 0))
