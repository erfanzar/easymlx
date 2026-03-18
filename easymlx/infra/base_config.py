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

"""Base configuration for easymlx models (serving-only).

This module intentionally implements a small subset of EasyDeL's config ideas:
- HuggingFace ``PretrainedConfig`` compatibility (load/save ``config.json``)
- Serving-oriented knobs (attention selection, dtype, generation defaults)

Typical usage::

    config = EasyMLXBaseConfig(attn_mechanism="sdpa", dtype="bfloat16")
    print(config.mlx_dtype)  # mx.bfloat16
"""

from __future__ import annotations

import mlx.core as mx
from transformers.configuration_utils import PretrainedConfig

from .errors import EasyMLXConfigError
from .etils import DEFAULT_ATTENTION_MECHANISM, AttentionMechanism


class EasyMLXBaseConfig(PretrainedConfig):
    """Base configuration class for all easymlx models.

    Extends HuggingFace's ``PretrainedConfig`` with MLX-specific serving
    knobs such as attention mechanism selection and dtype mapping.

    Attributes:
        model_type: Identifier string for the model type. Defaults to
            ``"easymlx"``.
        attn_mechanism: The attention implementation to use. One of
            ``"auto"``, ``"vanilla"``, ``"sdpa"``, or ``"paged"``.
        dtype: String representation of the desired MLX dtype (e.g.
            ``"float16"``, ``"bfloat16"``, ``"float32"``).

    Example::

        config = EasyMLXBaseConfig(attn_mechanism="sdpa", dtype="float16")
        assert config.mlx_dtype == mx.float16
    """

    model_type = "easymlx"

    def __init__(
        self,
        *,
        attn_mechanism: AttentionMechanism = DEFAULT_ATTENTION_MECHANISM,
        dtype: str = "float16",
        **kwargs,
    ):
        """Initialize the base configuration.

        Args:
            attn_mechanism: Attention mechanism to use. Must be one of
                ``"auto"``, ``"vanilla"``, ``"sdpa"``, or ``"paged"``.
                Defaults to the value of ``DEFAULT_ATTENTION_MECHANISM``.
            dtype: String name of the desired MLX floating-point dtype.
                Supported values: ``"float16"``/``"fp16"``,
                ``"bfloat16"``/``"bf16"``, ``"float32"``/``"fp32"``.
                Defaults to ``"float16"``.
            **kwargs: Additional keyword arguments forwarded to
                ``PretrainedConfig.__init__``.

        Raises:
            EasyMLXConfigError: If ``attn_mechanism`` is not a supported
                value.
        """
        super().__init__(**kwargs)
        self.attn_mechanism = attn_mechanism
        self.dtype = dtype
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values.

        Raises:
            EasyMLXConfigError: If ``attn_mechanism`` is not one of the
                supported values (``"auto"``, ``"vanilla"``, ``"sdpa"``,
                ``"paged"``).
        """
        if self.attn_mechanism not in ("auto", "vanilla", "sdpa", "paged"):
            raise EasyMLXConfigError(f"Unsupported attn_mechanism={self.attn_mechanism!r}")

    @property
    def mlx_dtype(self) -> mx.Dtype:
        """Convert the string dtype to an ``mx.Dtype`` object.

        Returns:
            The corresponding MLX dtype for this configuration's
            ``dtype`` string.

        Raises:
            EasyMLXConfigError: If the configured dtype string is not
                recognized.
        """
        dtype = str(getattr(self, "dtype", "float16")).lower()
        mapping = {
            "float16": mx.float16,
            "fp16": mx.float16,
            "bfloat16": mx.bfloat16,
            "bf16": mx.bfloat16,
            "float32": mx.float32,
            "fp32": mx.float32,
        }
        try:
            return mapping[dtype]
        except KeyError as exc:  # pragma: no cover
            raise EasyMLXConfigError(f"Unsupported dtype={dtype!r}; supported: {sorted(mapping)}") from exc
