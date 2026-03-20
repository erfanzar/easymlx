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

"""Auto model classes for easymlx on MLX.

This module provides ``Auto`` model classes that automatically resolve the
correct module class from the model registry based on the model type found
in a pretrained checkpoint's configuration. Each ``Auto`` class is bound to
a specific ``TaskType`` and provides ``from_pretrained`` and ``from_config``
class methods.
"""

from __future__ import annotations

import os
from typing import Literal, TypedDict, Unpack

import mlx.core as mx

from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.infra.base_module import EasyMLXBaseModule
from easymlx.infra.etils import QuantizationConfig, QuantizationMode
from easymlx.infra.factory import TaskType, registry


class ConfigOverrides(TypedDict, total=False):
    """Typed overrides applied on top of the loaded model config.

    Attributes:
        attn_mechanism: Attention mechanism to use. Overrides the value
            loaded from the pretrained config.
        cache_dtype: Data type for KV cache storage. ``"auto"`` uses the
            model dtype, ``"fp8"`` uses FP8 E4M3 quantization (uint8
            storage) for ~50% KV cache memory reduction.
    """

    attn_mechanism: Literal["auto", "vanilla", "sdpa", "paged", "unified"]
    cache_dtype: Literal["auto", "float16", "bfloat16", "float32", "fp8"]


class FromPretrainedKwargs(TypedDict, total=False):
    """Keyword arguments accepted by ``from_pretrained`` methods.

    Attributes:
        config: Config overrides applied on top of the pretrained config.
            Pass a :class:`ConfigOverrides` dict to selectively update
            fields (e.g. ``config={"attn_mechanism": "sdpa"}``).
        dtype: Target data type for model parameters (e.g., ``mx.float16``).
        device: Target device for model parameters.
        revision: Git revision (branch, tag, commit) for HuggingFace Hub.
        local_files_only: If True, only use local files and do not download.
        weights_name: Custom weight file name (e.g., ``"model.safetensors"``).
        strict: If True, raise on missing/unexpected keys during loading.
        lazy: If True, lazily load weights (useful for large models).
        auto_convert_hf: If True, auto-convert HuggingFace weights to MLX
            format. If None, decided automatically.
        converted_cache_dir: Directory to cache converted weights.
        force_conversion: If True, re-convert even if cached weights exist.
        copy_support_files: If True, copy tokenizer and other support files.
        quantization: Quantization config. Can be a mode string
            (``"affine"``, ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``) or a
            :class:`QuantizationConfig` dict with ``mode``, ``bits``,
            ``group_size``.
    """

    config: ConfigOverrides | None
    dtype: mx.Dtype | None
    device: mx.Device | None
    revision: str | None
    local_files_only: bool
    weights_name: str | None
    strict: bool
    lazy: bool
    auto_convert_hf: bool | None
    converted_cache_dir: str | os.PathLike | None
    force_conversion: bool
    copy_support_files: bool
    quantization: QuantizationConfig | QuantizationMode | None


class BaseAutoEasyModel:
    """Base class for all Auto EasyMLX model classes.

    Provides from_pretrained() and from_config() for loading models
    by resolving model_type from config and looking up the registry.
    """

    model_task: TaskType

    @classmethod
    def from_config(
        cls,
        config: EasyMLXBaseConfig,
    ) -> EasyMLXBaseModule:
        """Instantiate a model from a configuration object.

        Args:
            config: Model configuration with a ``model_type`` attribute.

        Returns:
            The instantiated EasyMLX model module.

        Raises:
            KeyError: If no module is registered for the given model type
                and task.
        """
        registration = registry.get_module_registration(cls.model_task, config.model_type)
        return registration.module(config=config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Unpack[FromPretrainedKwargs],
    ) -> EasyMLXBaseModule:
        """Load a model from a pretrained checkpoint.

        Resolves model_type from config, looks up the module in the
        registry by task_type, and delegates to the module's
        from_pretrained().

        Args:
            pretrained_model_name_or_path: HF model ID or local path.
            **kwargs: See :class:`FromPretrainedKwargs` for all options.

        Returns:
            The loaded EasyMLX model.
        """
        from transformers import AutoConfig

        config_overrides: ConfigOverrides | None = kwargs.pop("config", None)

        hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        model_type: str = hf_config.model_type

        registration = registry.get_module_registration(cls.model_task, model_type)
        module_class = registration.module
        # quantization is passed through to bridge's from_pretrained so
        # it can quantize per-shard (much lower peak memory).
        model = module_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if config_overrides:
            config_obj = model.config
            for key, value in config_overrides.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                elif hasattr(config_obj, "text_config") and hasattr(config_obj.text_config, key):
                    setattr(config_obj.text_config, key, value)

        return model


class AutoEasyMLXModelForCausalLM(BaseAutoEasyModel):
    """Auto model loader for causal language modeling tasks."""

    model_task: TaskType = TaskType.CAUSAL_LM


class AutoEasyMLXModelForSequenceClassification(BaseAutoEasyModel):
    """Auto model loader for sequence classification tasks."""

    model_task: TaskType = TaskType.SEQUENCE_CLASSIFICATION


class AutoEasyMLXModelForTokenClassification(BaseAutoEasyModel):
    """Auto model loader for token classification tasks."""

    model_task: TaskType = TaskType.TOKEN_CLASSIFICATION


class AutoEasyMLXModelForQuestionAnswering(BaseAutoEasyModel):
    """Auto model loader for question answering tasks."""

    model_task: TaskType = TaskType.QUESTION_ANSWERING


class AutoEasyMLXModelForImageTextToText(BaseAutoEasyModel):
    """Auto model loader for vision-language (image-text-to-text) tasks."""

    model_task: TaskType = TaskType.IMAGE_TEXT_TO_TEXT


class AutoEasyMLXModelForEmbedding(BaseAutoEasyModel):
    """Auto model loader for embedding / retrieval tasks."""

    model_task: TaskType = TaskType.EMBEDDING


class AutoEasyMLXModelForImageClassification(BaseAutoEasyModel):
    """Auto model loader for image classification tasks."""

    model_task: TaskType = TaskType.IMAGE_CLASSIFICATION


class AutoEasyMLXModel(BaseAutoEasyModel):
    """Auto model loader for base (backbone) models."""

    model_task: TaskType = TaskType.BASE_MODULE
