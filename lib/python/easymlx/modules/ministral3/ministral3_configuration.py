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

"""Ministral3 configuration for EasyMLX inference."""

from __future__ import annotations

from typing import Any

from easymlx.infra.factory import register_config

from ..llama import LlamaConfig


@register_config("ministral3")
class Ministral3Config(LlamaConfig):
    """Configuration for the Ministral3 text model.

    Ministral3 extends the Llama architecture with a sliding/full attention
    pattern controlled per layer via ``layer_types``. It also supports
    Llama-4-style attention scaling (``llama_4_scaling_beta``) and extracts
    RoPE parameters from a ``rope_parameters`` dict for compatibility with
    upstream Mistral checkpoints.

    Inherits all attributes from ``LlamaConfig``.

    Attributes:
        model_type: The model type identifier (``"ministral3"``).
        rope_parameters: Raw RoPE parameter dict from the upstream config.
        llama_4_scaling_beta: Beta factor for Llama-4 attention scaling.
            When non-zero, queries are scaled by
            ``1 + beta * log(1 + floor(pos / original_max_position_embeddings))``.
        original_max_position_embeddings: Original maximum position before
            any RoPE scaling was applied.

    Example:
        >>> config = Ministral3Config(
        ...     hidden_size=4096,
        ...     layer_types=["full_attention", "sliding_attention"] * 16,
        ... )
        >>> config.model_type
        'ministral3'
    """

    model_type = "ministral3"

    def __init__(
        self,
        *,
        rope_parameters: dict[str, Any] | None = None,
        llama_4_scaling_beta: float | None = None,
        original_max_position_embeddings: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Ministral3 configuration.

        Args:
            rope_parameters: Dict of RoPE parameters from the upstream
                config. May contain ``rope_theta``, ``rope_type``,
                ``llama_4_scaling_beta``, and
                ``original_max_position_embeddings``.
            llama_4_scaling_beta: Explicit Llama-4 scaling beta override.
                Falls back to ``rope_parameters["llama_4_scaling_beta"]``
                when ``None``.
            original_max_position_embeddings: Explicit original max position
                override. Falls back to ``rope_parameters`` value when ``None``.
            **kwargs: All remaining arguments forwarded to ``LlamaConfig``.
        """
        rope_parameters = dict(rope_parameters or {})
        rope_scaling = kwargs.pop("rope_scaling", None)

        if "rope_theta" in rope_parameters and "rope_theta" not in kwargs:
            kwargs["rope_theta"] = rope_parameters["rope_theta"]
        if rope_scaling is None and rope_parameters:
            rope_scaling = {
                key: value for key, value in rope_parameters.items() if key not in {"rope_theta", "llama_4_scaling_beta"}
            }
        if rope_scaling is not None:
            kwargs["rope_scaling"] = rope_scaling

        super().__init__(**kwargs)

        self.rope_parameters = rope_parameters
        self.llama_4_scaling_beta = float(
            rope_parameters.get("llama_4_scaling_beta", 0.0) if llama_4_scaling_beta is None else llama_4_scaling_beta
        )
        self.original_max_position_embeddings = int(
            rope_parameters.get("original_max_position_embeddings", self.max_position_embeddings)
            if original_max_position_embeddings is None
            else original_max_position_embeddings
        )


__all__ = ("Ministral3Config",)
