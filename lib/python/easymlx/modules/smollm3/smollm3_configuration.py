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

"""SmolLM-3 configuration for serving and inference."""

from __future__ import annotations

from easymlx.infra.factory import register_config

from ..llama import LlamaConfig


@register_config("smollm3")
class SmolLM3Config(LlamaConfig):
    """Configuration for the SmolLM-3 transformer model.

    SmolLM-3 extends the Llama architecture by selectively disabling
    Rotary Positional Embeddings (RoPE) on certain layers. The layer
    pattern is controlled by ``no_rope_layer_interval`` or an explicit
    ``no_rope_layers`` mask. Layers where RoPE is disabled use a no-op
    positional embedding (``SmolLM3NoPE``).

    Inherits all attributes from ``LlamaConfig``.

    Attributes:
        model_type: The model type identifier (``"smollm3"``).
        no_rope_layer_interval: Interval for disabling RoPE. Every
            ``no_rope_layer_interval``-th layer (1-indexed) keeps RoPE;
            all others disable it.
        no_rope_layers: Explicit per-layer mask. ``1`` means the layer
            uses RoPE, ``0`` means it does not. When ``None``, the mask
            is derived from ``no_rope_layer_interval``.

    Example:
        >>> config = SmolLM3Config(num_hidden_layers=8, no_rope_layer_interval=4)
        >>> config.no_rope_layers
        [1, 1, 1, 0, 1, 1, 1, 0]
    """

    model_type = "smollm3"

    def __init__(
        self,
        *,
        no_rope_layer_interval: int = 4,
        no_rope_layers: list[int] | None = None,
        **kwargs,
    ):
        """Initialize a SmolLM-3 configuration.

        Args:
            no_rope_layer_interval: Interval for the automatic RoPE
                disable pattern. Must be a positive integer.
            no_rope_layers: Explicit per-layer 0/1 mask indicating
                which layers use RoPE (``1``) and which skip it (``0``).
                Must match ``num_hidden_layers`` in length if provided.
            **kwargs: All remaining arguments are forwarded to ``LlamaConfig``.

        Raises:
            ValueError: If ``no_rope_layer_interval`` is not positive.
            ValueError: If ``no_rope_layers`` length does not match
                ``num_hidden_layers``.
        """
        super().__init__(**kwargs)

        self.no_rope_layer_interval = int(no_rope_layer_interval)
        if self.no_rope_layer_interval <= 0:
            raise ValueError(f"no_rope_layer_interval must be positive, got {self.no_rope_layer_interval}")

        if no_rope_layers is None:
            self.no_rope_layers = [
                int((idx + 1) % self.no_rope_layer_interval != 0) for idx in range(self.num_hidden_layers)
            ]
        else:
            if len(no_rope_layers) != self.num_hidden_layers:
                raise ValueError(
                    "no_rope_layers length must match num_hidden_layers, "
                    f"got {len(no_rope_layers)} and {self.num_hidden_layers}"
                )
            self.no_rope_layers = [int(bool(flag)) for flag in no_rope_layers]


Smollm3Config = SmolLM3Config

__all__ = ("SmolLM3Config", "Smollm3Config")
