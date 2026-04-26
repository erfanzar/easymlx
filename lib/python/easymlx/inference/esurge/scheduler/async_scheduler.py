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

"""Async-aware scheduler wrapper.

Extends :class:`~easymlx.inference.esurge.scheduler.scheduler.Scheduler`
with output-placeholder tracking for asynchronous decode steps.  When
tokens are scheduled for decode, a placeholder count is incremented on
the request; when model output arrives, the placeholders are consumed.
"""

from __future__ import annotations

from ..request import EngineRequest
from .output import SchedulerStepOutput
from .scheduler import Scheduler


class AsyncScheduler(Scheduler):
    """Scheduler variant that tracks async decode placeholders.

    Overrides :meth:`schedule` to increment
    ``num_output_placeholders`` on decode-phase requests, and
    :meth:`update_from_model_output` to decrement them when sampled
    tokens arrive.
    """

    def schedule(
        self,
        *,
        blocked_request_ids: set[str] | frozenset[str] | None = None,
        allow_preemption: bool = True,
    ) -> SchedulerStepOutput:
        """Produce a scheduling decision and update placeholder counts.

        Calls the parent :meth:`Scheduler.schedule` and then increments
        ``num_output_placeholders`` for each decode-phase scheduled
        request.

        Returns:
            The :class:`SchedulerStepOutput` for this step.
        """
        output = super().schedule(
            blocked_request_ids=blocked_request_ids,
            allow_preemption=allow_preemption,
        )
        for scheduled in output.scheduled:
            if scheduled.is_prefill:
                continue
            request = self.get_request(scheduled.request_id)
            if request is None:
                continue
            request.num_output_placeholders += scheduled.num_tokens
        return output

    async def step(self) -> SchedulerStepOutput:
        """Async entry point that delegates to :meth:`schedule`.

        Returns:
            The :class:`SchedulerStepOutput` for this step.
        """
        return self.schedule()

    def update_from_model_output(
        self,
        step_output: SchedulerStepOutput,
        *,
        sampled_token_ids: dict[str, list["int"]] | None = None,
        stop_reasons: dict[str, str | None] | None = None,
        failed_requests: dict[str, str] | None = None,
    ) -> list[EngineRequest]:
        """Apply model output and consume decode placeholders.

        Decrements ``num_output_placeholders`` by the number of sampled
        tokens for each request before delegating to the parent
        implementation.

        Args:
            step_output: The scheduling decision that produced this
                output.
            sampled_token_ids: Mapping of request IDs to sampled token
                ID lists.
            stop_reasons: Mapping of request IDs to stop-reason strings.
            failed_requests: Mapping of request IDs to failure
                descriptions.

        Returns:
            List of :class:`EngineRequest` objects that finished during
            this update.
        """
        sampled_token_ids = sampled_token_ids or {}
        for request_id, token_ids in sampled_token_ids.items():
            request = self.get_request(request_id)
            if request is None:
                continue
            request.num_output_placeholders = max(request.num_output_placeholders - len(token_ids), 0)
        return super().update_from_model_output(
            step_output,
            sampled_token_ids=sampled_token_ids,
            stop_reasons=stop_reasons,
            failed_requests=failed_requests,
        )
