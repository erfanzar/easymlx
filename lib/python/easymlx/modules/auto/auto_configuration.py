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

"""Auto configuration helpers for easymlx models.

This module provides utilities for automatically resolving model configurations
from pretrained checkpoints. It includes task normalization, module lookup by
model type, task inference from HuggingFace config files, and the
``AutoEasyMLXConfig`` factory class.
"""

from __future__ import annotations

import json
import logging
import typing as tp

from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.infra.base_module import EasyMLXBaseModule
from easymlx.infra.factory import TaskType, registry
from easymlx.utils.hf_composite import resolve_hf_composite_repo, resolve_local_hf_path

logger = logging.getLogger(__name__)


TASK_ALIASES: dict[str, TaskType] = {
    "causal_lm": TaskType.CAUSAL_LM,
    "lm": TaskType.CAUSAL_LM,
    "seq2seq": TaskType.SEQUENCE_TO_SEQUENCE,
    "sequence_to_sequence": TaskType.SEQUENCE_TO_SEQUENCE,
    "image_text_to_text": TaskType.IMAGE_TEXT_TO_TEXT,
    "embedding": TaskType.EMBEDDING,
    "base": TaskType.BASE_MODULE,
    "sequence_classification": TaskType.SEQUENCE_CLASSIFICATION,
    "token_classification": TaskType.TOKEN_CLASSIFICATION,
    "question_answering": TaskType.QUESTION_ANSWERING,
    "image_classification": TaskType.IMAGE_CLASSIFICATION,
}


def normalize_task(t: TaskType | str | None) -> TaskType | None:
    """Normalize a task specification to a TaskType enum.

    Args:
        t: TaskType, string alias, or None.

    Returns:
        Normalized TaskType or None if not recognized.
    """
    if t is None:
        return None
    if isinstance(t, TaskType):
        return t
    return TASK_ALIASES.get(str(t).strip().lower().replace("-", "_"))


def get_modules_by_type(
    model_type: str,
    task_type: TaskType,
) -> tuple[type[EasyMLXBaseConfig], type[EasyMLXBaseModule] | tp.Any]:
    """Retrieve config and module classes for a given model type and task.

    Args:
        model_type: HuggingFace model type string (e.g. "llama", "qwen2").
        task_type: Task type to look up.

    Returns:
        Tuple of (config_class, module_class).
    """
    registration = registry.get_module_registration(
        task_type=task_type,
        model_type=model_type,
    )
    return (registration.config, registration.module)


def infer_task_from_hf_config(model_name_or_path: str) -> TaskType | None:
    """Infer task type from a HuggingFace model config.

    Fetches config.json and determines the task from model architecture.

    Args:
        model_name_or_path: HuggingFace model ID or local path.

    Returns:
        Inferred TaskType, or None (caller should fallback to CAUSAL_LM).
    """
    try:
        from pathlib import Path

        base_resolved_repo = resolve_hf_composite_repo(model_name_or_path, task_type=TaskType.BASE_MODULE)
        if base_resolved_repo.composite_model_type is not None:
            return TaskType.BASE_MODULE

        resolved_repo = resolve_hf_composite_repo(model_name_or_path, task_type=TaskType.CAUSAL_LM)
        config_subpath = (
            Path(resolved_repo.model_subfolder) / "config.json" if resolved_repo.model_subfolder else Path("config.json")
        )
        local_path = resolve_local_hf_path(model_name_or_path)
        if local_path.is_dir():
            config_file = local_path / config_subpath
            if config_file.exists():
                config = json.loads(config_file.read_text())
            else:
                logger.warning("No config.json found in %s. Falling back to CAUSAL_LM.", model_name_or_path)
                return None
        else:
            try:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(
                    repo_id=model_name_or_path,
                    filename=str(config_subpath),
                    repo_type="model",
                )
                config = json.loads(Path(config_path).read_text())
            except Exception:
                logger.warning("Failed to fetch config for %s. Falling back to CAUSAL_LM.", model_name_or_path)
                return None

        architectures = config.get("architectures", [])
        model_type = config.get("model_type", "").lower()

        if not architectures:
            return None

        arch = architectures[0]
        if "ForCausalLM" in arch:
            return TaskType.CAUSAL_LM
        elif "ForConditionalGeneration" in arch:
            return TaskType.IMAGE_TEXT_TO_TEXT
        elif "ForSequenceClassification" in arch:
            return TaskType.SEQUENCE_CLASSIFICATION
        elif "ForTokenClassification" in arch:
            return TaskType.TOKEN_CLASSIFICATION
        elif "ForQuestionAnswering" in arch:
            return TaskType.QUESTION_ANSWERING
        elif "ForImageClassification" in arch:
            return TaskType.IMAGE_CLASSIFICATION
        elif "ForEmbedding" in arch or "ForSentenceEmbedding" in arch:
            return TaskType.EMBEDDING

        if "vision" in model_type or "clip" in model_type:
            return TaskType.BASE_VISION

        logger.warning("Could not map architecture '%s' to a TaskType. Falling back to CAUSAL_LM.", arch)
        return None

    except Exception as e:
        logger.warning("Error inferring task for %s: %s. Falling back to CAUSAL_LM.", model_name_or_path, e)
        return None


class AutoEasyMLXConfig:
    """Factory for loading easymlx model configurations.

    Resolves the correct config class based on model type and task, then
    loads from a pretrained HuggingFace-compatible config.json.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        task_type: TaskType | str | None = None,
        **kwargs,
    ) -> EasyMLXBaseConfig:
        """Load a model config from a pretrained checkpoint.

        Args:
            pretrained_model_name_or_path: HF model ID or local path.
            task_type: Override task type. If None, inferred from config.
            **kwargs: Extra attributes set on the config.

        Returns:
            EasyMLXBaseConfig instance.
        """
        from transformers import AutoConfig

        resolved_task = normalize_task(task_type)
        if resolved_task is None:
            resolved_task = infer_task_from_hf_config(pretrained_model_name_or_path)
        if resolved_task is None:
            resolved_task = TaskType.CAUSAL_LM

        subfolder: str | None = kwargs.pop("subfolder", None)
        resolved_repo = None
        if subfolder is None:
            resolved_repo = resolve_hf_composite_repo(
                pretrained_model_name_or_path,
                task_type=resolved_task,
            )

        if (
            resolved_task == TaskType.BASE_MODULE
            and subfolder is None
            and resolved_repo is not None
            and resolved_repo.composite_model_type is not None
        ):
            config_class, _ = get_modules_by_type(resolved_repo.composite_model_type, resolved_task)
            config = config_class.from_pretrained(
                pretrained_model_name_or_path,
                revision=kwargs.get("revision"),
                local_files_only=bool(kwargs.get("local_files_only", False)),
            )

            for k, v in kwargs.items():
                setattr(config, k, v)

            return config

        if subfolder is None and resolved_repo is not None:
            subfolder = resolved_repo.model_subfolder

        config_kwargs = {"trust_remote_code": True}
        if subfolder is not None:
            config_kwargs["subfolder"] = subfolder
        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            **config_kwargs,
        )
        model_type: str = hf_config.model_type

        config_class, _ = get_modules_by_type(model_type, resolved_task)
        config = config_class.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            revision=kwargs.get("revision"),
            local_files_only=bool(kwargs.get("local_files_only", False)),
        )

        for k, v in kwargs.items():
            setattr(config, k, v)

        return config
