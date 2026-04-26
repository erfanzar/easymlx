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

"""Scheduler protocols.

Defines the :class:`SchedulerProtocol` structural typing interface that
all scheduler implementations must satisfy for use by the eSurge engine
loop.
"""

from __future__ import annotations

from typing import Protocol

from ..request import EngineRequest
from .output import SchedulerStepOutput


class SchedulerProtocol(Protocol):
    """Protocol for schedulers used by eSurge runners.

    Any class that implements :meth:`add_request`, :meth:`cancel_request`,
    :meth:`schedule`, and :meth:`update_from_model_output` with the
    signatures below is compatible with this protocol.
    """

    def add_request(self, request: EngineRequest) -> None:
        """Add a new request to the scheduler's waiting queue.

        Args:
            request: The :class:`EngineRequest` to enqueue.
        """
        ...

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request by ID if it is still pending or running.

        Args:
            request_id: The unique request identifier.

        Returns:
            ``True`` if the request was found and canceled.
        """
        ...

    def schedule(self) -> SchedulerStepOutput:
        """Produce a scheduling decision for the next engine step.

        Returns:
            A :class:`SchedulerStepOutput` describing which requests
            to process and with how many tokens.
        """
        ...

    def update_from_model_output(
        self,
        step_output: SchedulerStepOutput,
        *,
        sampled_token_ids: dict[str, list["int"]] | None = None,
        stop_reasons: dict[str, str | None] | None = None,
        failed_requests: dict[str, str] | None = None,
    ) -> list[EngineRequest]:
        """Apply model output to the scheduler state and finalize requests.

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
        ...
