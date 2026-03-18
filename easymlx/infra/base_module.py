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

"""Base module for easymlx models (serving-only).

Provides the foundational ``EasyMLXBaseModule`` class that all easymlx model
implementations should inherit from. It combines MLX's ``nn.Module`` with
bridge (I/O), generation, and operation-cache mixins.
"""

from __future__ import annotations

import typing as tp

import mlx.nn as nn

from .base_config import EasyMLXBaseConfig
from .mixins.bridge import EasyBridgeMixin
from .mixins.generation import EasyGenerationMixin
from .mixins.operation_cache import OperationCacheMixin


class EasyMLXBaseModule(nn.Module, EasyBridgeMixin, EasyGenerationMixin, OperationCacheMixin):
    """Abstract base class for all easymlx neural-network modules.

    Combines MLX's native ``nn.Module`` with the following mixins:

    - :class:`EasyBridgeMixin` -- model saving/loading and HuggingFace
      checkpoint conversion.
    - :class:`EasyGenerationMixin` -- text generation (greedy / sampling).
    - :class:`OperationCacheMixin` -- cache requirement discovery and
      allocation.

    Subclasses **must** override ``__call__`` to implement the forward pass.

    Attributes:
        config_class: The configuration class associated with this module.
            Defaults to :class:`EasyMLXBaseConfig`.
        config: The active configuration instance.
        _model_task: Optional string identifying the model task (e.g.
            ``"causal-language-model"``). Set by the registry decorator.
        _model_type: Optional string identifying the model type (e.g.
            ``"llama"``). Set by the registry decorator.
    """

    config_class = EasyMLXBaseConfig
    config: EasyMLXBaseConfig

    _model_task: str | None = None
    _model_type: str | None = None

    def __init__(self, config: EasyMLXBaseConfig, **_unused: tp.Any):
        """Initialize the base module with a configuration.

        Args:
            config: The model configuration object.
            **_unused: Ignored keyword arguments for forward-compatibility
                with subclass constructors.
        """
        super().__init__()
        self.config = config

    def __call__(self, *args: tp.Any, **kwargs: tp.Any):  # pragma: no cover
        """Execute the forward pass.

        Subclasses must override this method to define the model's
        forward computation.

        Args:
            *args: Positional arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.

        Raises:
            NotImplementedError: Always, unless overridden by a
                subclass.
        """
        raise NotImplementedError("Subclasses must implement `__call__`.")
