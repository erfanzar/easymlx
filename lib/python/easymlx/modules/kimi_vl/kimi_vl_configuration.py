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

"""Kimi VL configuration for EasyMLX.

Kimi VL is a vision-language model whose text backbone is DeepSeek V3.
This config wraps the DeepSeek V3 text config under ``text_config``
and registers as ``kimi_vl``.  Only the text backbone is exposed as a
CAUSAL_LM; the vision tower is stripped in ``sanitize``.
"""

from __future__ import annotations

import typing as tp

from easymlx.infra.factory import register_config

from ..deepseek_v3 import DeepseekV3Config


@register_config("kimi_vl")
class KimiVLConfig(DeepseekV3Config):
    """Configuration for the Kimi VL text backbone.

    Upstream Kimi VL wraps a DeepSeek V3 text model.  We inherit all
    DeepSeek V3 parameters and add ``model_type = "kimi_vl"`` so the
    registry can resolve it.
    """

    model_type = "kimi_vl"

    def __init__(
        self,
        *,
        text_config: dict[str, tp.Any] | None = None,
        **kwargs,
    ):
        """Initialize KimiVLConfig.

        Args:
            text_config: Optional dict of text backbone parameters. If
                provided, these are merged into ``kwargs`` before passing
                to the DeepSeek V3 config constructor.
            **kwargs: All other keyword arguments are forwarded to
                ``DeepseekV3Config.__init__``.
        """
        if text_config is not None:
            kwargs.update(text_config)
        super().__init__(**kwargs)


__all__ = ("KimiVLConfig",)
