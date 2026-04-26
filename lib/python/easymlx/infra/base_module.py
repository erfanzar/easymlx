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
from .factory import TaskType
from .flops import estimate_module_forward_flops
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

    def __call__(self, *args: tp.Any, **kwargs: tp.Any):
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

    def get_flops(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        use_vision: bool = False,
        use_audio: bool = False,
        include_lm_head: bool | None = None,
        image_size: int | tuple[int, int] | None = None,
        vision_batch_size: int | None = None,
        vision_height: int | None = None,
        vision_width: int | None = None,
        vision_frames: int | None = None,
        vision_sequence_length: int | None = None,
        audio_sequence_length: int | None = None,
    ) -> int:
        """Estimate FLOPs for a single forward pass.

        Args:
            batch_size: Text batch size.
            sequence_length: Text sequence length processed by the model.
            use_vision: Whether to include the vision encoder path.
            use_audio: Whether to include the audio path.
            include_lm_head: Whether to include the output vocabulary
                projection. ``None`` infers the value from the module.
            image_size: Convenience alias for square or ``(height, width)``
                vision inputs.
            vision_batch_size: Number of images/videos processed by the
                vision tower. Defaults to ``batch_size``.
            vision_height: Vision input height.
            vision_width: Vision input width.
            vision_frames: Number of frames per vision item.
            vision_sequence_length: Exact raw vision token count per item,
                before model-specific merging/projectors.
            audio_sequence_length: Reserved for future audio support.

        Returns:
            Integer FLOPs for one forward pass.
        """
        supported_tasks = {
            TaskType.CAUSAL_LM.value,
            TaskType.BASE_MODULE.value,
        }
        task_type = getattr(self, "_model_task", None)
        if isinstance(task_type, str) and task_type not in supported_tasks:
            raise NotImplementedError(
                f"FLOPs estimation is not supported for task type {task_type!r}; "
                "supported task types are base modules and causal LMs."
            )
        return estimate_module_forward_flops(
            self,
            batch_size=batch_size,
            sequence_length=sequence_length,
            use_vision=use_vision,
            use_audio=use_audio,
            include_lm_head=include_lm_head,
            image_size=image_size,
            vision_batch_size=vision_batch_size,
            vision_height=vision_height,
            vision_width=vision_width,
            vision_frames=vision_frames,
            vision_sequence_length=vision_sequence_length,
            audio_sequence_length=audio_sequence_length,
        )
