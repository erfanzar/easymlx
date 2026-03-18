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
from easymlx.infra.factory import TaskType, registry

QuantizationMode = Literal["affine", "mxfp4", "mxfp8", "nvfp4"]


class QuantizationConfig(TypedDict, total=False):
    """Quantization configuration for model loading.

    Attributes:
        mode: Quantization mode. Supported modes:

            - ``"affine"`` — Standard affine quantization (default). Requires
              ``bits`` and ``group_size``.
            - ``"mxfp4"`` — Microscaling FP4 format.
            - ``"mxfp8"`` — Microscaling FP8 format.
            - ``"nvfp4"`` — NVIDIA FP4 format.
        bits: Number of bits per weight element. Only used for ``"affine"``
            mode. Common values: ``4`` (default), ``8``, ``2``.
        group_size: Number of elements sharing a scale/bias. Only used for
            ``"affine"`` mode. Default ``64``.

    Example:
        >>> QuantizationConfig(mode="affine", bits=4, group_size=64)
        >>> QuantizationConfig(mode="mxfp4")
        >>> QuantizationConfig(mode="nvfp4")
    """

    mode: QuantizationMode
    bits: int
    group_size: int


def _apply_quantization(
    model: EasyMLXBaseModule, quantization: QuantizationConfig | QuantizationMode | None
) -> EasyMLXBaseModule:
    """Apply quantization to a model in-place.

    Args:
        model: The model to quantize.
        quantization: Quantization specification. Can be:

            - ``None`` — No quantization.
            - :class:`QuantizationMode` — Shorthand mode literal
              (``"affine"``, ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``).
              Uses default bits/group_size for the mode.
            - :class:`QuantizationConfig` dict with ``mode``, ``bits``,
              ``group_size`` keys.

    Returns:
        The quantized model (modified in-place).

    Raises:
        ValueError: If the quantization mode is not recognized.
    """
    import mlx.nn as nn

    if quantization is None or quantization == "":
        return model

    if isinstance(quantization, str):
        quantization = QuantizationConfig(mode=quantization)

    mode = quantization.get("mode", "affine")
    valid_modes = ("affine", "mxfp4", "mxfp8", "nvfp4")
    if mode not in valid_modes:
        raise ValueError(f"Unknown quantization mode '{mode}'. Supported: {valid_modes}")

    _MODE_DEFAULTS: dict[str, dict[str, int]] = {
        "affine": {"bits": 4, "group_size": 64},
        "mxfp4": {"bits": 4, "group_size": 32},
        "mxfp8": {"bits": 8, "group_size": 32},
        "nvfp4": {"bits": 4, "group_size": 16},
    }

    defaults = _MODE_DEFAULTS[mode]
    kwargs: dict = {
        "mode": mode,
        "bits": quantization.get("bits", defaults["bits"]),
        "group_size": quantization.get("group_size", defaults["group_size"]),
    }

    import logging

    logger = logging.getLogger("easymlx.quantization")

    logger.info(
        "Quantizing model: mode=%s, bits=%d, group_size=%d",
        kwargs["mode"],
        kwargs["bits"],
        kwargs["group_size"],
    )

    try:
        gs = kwargs["group_size"]
        bits = kwargs["bits"]
        dummy_w = mx.ones((gs, gs))
        quant_result = mx.quantize(dummy_w, gs, bits, mode=kwargs["mode"])
        scales = quant_result[1]
        biases = quant_result[2] if len(quant_result) == 3 else None
        dummy_x = mx.ones((1, gs))
        out = mx.quantized_matmul(
            dummy_x,
            quant_result[0],
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=gs,
            bits=bits,
            mode=kwargs["mode"],
        )
        mx.eval(out)
    except RuntimeError as exc:
        if "Unable to load kernel" in str(exc):
            device_info = mx.metal.device_info()
            device_name = device_info.get("device_name", "unknown")
            raise RuntimeError(
                f"Quantization mode '{mode}' is not supported on {device_name} "
                f"(arch={device_info.get('architecture', '?')}). "
                f"The required Metal kernel is not available in MLX {mx.__version__}. "
                f"Try mode='affine' (bits=4, group_size=64) which works on all Apple Silicon, "
                f"or upgrade MLX: pip install -U mlx"
            ) from exc
        raise

    nn.quantize(model, **kwargs)
    logger.info("Quantization complete")
    return model


class FromPretrainedKwargs(TypedDict, total=False):
    """Keyword arguments accepted by ``from_pretrained`` methods.

    Attributes:
        config: Pre-loaded config, config dict, or path. If None, loaded
            automatically from the checkpoint.
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

    config: EasyMLXBaseConfig | dict | str | os.PathLike | None
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

        quantization = kwargs.pop("quantization", None)

        hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        model_type: str = hf_config.model_type

        registration = registry.get_module_registration(cls.model_task, model_type)
        module_class = registration.module
        model = module_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if quantization is not None:
            model = _apply_quantization(model, quantization)

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
