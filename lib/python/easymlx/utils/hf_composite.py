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

"""Helpers for Hugging Face composite repositories.

Composite repos such as diffusers pipelines often expose a
``model_index.json`` at the root and place transformer-compatible
components in subfolders like ``text_encoder/`` and ``tokenizer/``.
EasyMLX is primarily a transformer loader, so these helpers resolve the
component subfolder that should be used for a given task.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from easymlx.infra.factory import TaskType

_MODEL_COMPONENT_CANDIDATES: dict[TaskType, tuple[str, ...]] = {
    TaskType.CAUSAL_LM: ("text_encoder", "text_encoder_2", "language_model", "llm", "model"),
    TaskType.IMAGE_TEXT_TO_TEXT: ("language_model", "text_encoder", "llm", "model"),
    TaskType.EMBEDDING: ("text_encoder", "text_encoder_2", "model"),
    TaskType.SEQUENCE_CLASSIFICATION: ("text_encoder", "language_model", "model"),
    TaskType.TOKEN_CLASSIFICATION: ("text_encoder", "language_model", "model"),
    TaskType.QUESTION_ANSWERING: ("text_encoder", "language_model", "model"),
    TaskType.BASE_MODULE: ("model", "text_encoder", "language_model", "llm"),
}

_TASK_ALIASES: dict[str, TaskType] = {
    "causal_lm": TaskType.CAUSAL_LM,
    "lm": TaskType.CAUSAL_LM,
    "image_text_to_text": TaskType.IMAGE_TEXT_TO_TEXT,
    "embedding": TaskType.EMBEDDING,
    "sequence_classification": TaskType.SEQUENCE_CLASSIFICATION,
    "token_classification": TaskType.TOKEN_CLASSIFICATION,
    "question_answering": TaskType.QUESTION_ANSWERING,
    "base": TaskType.BASE_MODULE,
    "base_module": TaskType.BASE_MODULE,
}

_TOKENIZER_COMPONENT_CANDIDATES = ("tokenizer", "tokenizer_2")
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.model",
)
_COMPOSITE_PIPELINE_TYPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "Flux2KleinPipeline": (
        "flux2_klein",
        (
            "text_encoder/config.json",
            "transformer/config.json",
            "vae/config.json",
            "scheduler/scheduler_config.json",
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class ResolvedHFCompositeRepo:
    """Resolved component selection for a repo or local directory."""

    source: str | os.PathLike[str]
    model_subfolder: str | None = None
    tokenizer_subfolder: str | None = None
    composite_class_name: str | None = None
    composite_model_type: str | None = None


def _hf_cache_root() -> Path:
    with suppress(Exception):
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE).expanduser()
    return Path("~/.cache/huggingface/hub").expanduser()


def _repo_id_cache_root(source: str | os.PathLike[str]) -> Path | None:
    text = str(source).strip()
    if not text or text.startswith(".") or text.startswith("~") or text.startswith("/"):
        return None
    if text.count("/") > 1:
        return None

    cache_name = text.replace("/", "--")
    return _hf_cache_root() / f"models--{cache_name}"


def resolve_local_hf_path(source: str | os.PathLike[str]) -> Path:
    """Resolve a local source path, including ``~`` and HF cache roots.

    Hugging Face's on-disk cache uses a repo root like
    ``models--org--name/`` with real files under ``snapshots/<revision>/``.
    This helper expands ``~`` and, when given that cache root, resolves it
    to the current snapshot directory.
    """
    path = Path(source).expanduser()
    if not path.exists():
        cache_root = _repo_id_cache_root(source)
        if cache_root is not None and cache_root.exists():
            path = cache_root
    if not path.exists() or not path.is_dir():
        return path

    snapshots_dir = path / "snapshots"
    if not snapshots_dir.is_dir():
        return path

    for marker in ("config.json", "model_index.json"):
        if (path / marker).exists():
            return path

    ref_main = path / "refs" / "main"
    if ref_main.exists():
        revision = ref_main.read_text(encoding="utf-8").strip()
        if revision:
            snapshot_path = snapshots_dir / revision
            if snapshot_path.is_dir():
                return snapshot_path

    snapshot_dirs = sorted(
        (candidate for candidate in snapshots_dir.iterdir() if candidate.is_dir()), key=lambda p: p.name
    )
    if snapshot_dirs:
        return snapshot_dirs[-1]

    return path


def _normalize_task(task_type: TaskType | str | None) -> TaskType | None:
    if task_type is None:
        return None
    if isinstance(task_type, TaskType):
        return task_type
    with suppress(ValueError):
        return TaskType(str(task_type))
    return _TASK_ALIASES.get(str(task_type).strip().lower().replace("-", "_"))


def _component_candidates(task_type: TaskType | str | None) -> tuple[str, ...]:
    normalized = _normalize_task(task_type)
    if normalized is None:
        return ()
    return _MODEL_COMPONENT_CANDIDATES.get(normalized, ())


def _download_repo_file(
    repo_id: str,
    filename: str,
    *,
    revision: str | None,
    local_files_only: bool,
) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None

    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_files_only=local_files_only,
                repo_type="model",
            )
        )
    except Exception:
        return None


def _has_top_level_config(
    source: str | os.PathLike[str],
    *,
    revision: str | None,
    local_files_only: bool,
) -> bool:
    local_path = resolve_local_hf_path(source)
    if local_path.exists():
        return (local_path / "config.json").exists()
    return (
        _download_repo_file(str(source), "config.json", revision=revision, local_files_only=local_files_only) is not None
    )


def _load_model_index(
    source: str | os.PathLike[str],
    *,
    revision: str | None,
    local_files_only: bool,
) -> dict[str, Any] | None:
    local_path = resolve_local_hf_path(source)
    if local_path.exists():
        path = local_path / "model_index.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    downloaded = _download_repo_file(
        str(source),
        "model_index.json",
        revision=revision,
        local_files_only=local_files_only,
    )
    if downloaded is None:
        return None
    return json.loads(downloaded.read_text(encoding="utf-8"))


def _is_transformers_component(entry: Any) -> bool:
    return isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[0] == "transformers"


def _component_has_file(
    source: str | os.PathLike[str],
    subfolder: str,
    filename: str,
    *,
    revision: str | None,
    local_files_only: bool,
) -> bool:
    local_path = resolve_local_hf_path(source)
    if local_path.exists():
        return (local_path / subfolder / filename).exists()
    return (
        _download_repo_file(
            str(source),
            f"{subfolder}/{filename}",
            revision=revision,
            local_files_only=local_files_only,
        )
        is not None
    )


def _path_exists(
    source: str | os.PathLike[str],
    filename: str,
    *,
    revision: str | None,
    local_files_only: bool,
) -> bool:
    local_path = resolve_local_hf_path(source)
    if local_path.exists():
        return (local_path / filename).exists()
    return (
        _download_repo_file(
            str(source),
            filename,
            revision=revision,
            local_files_only=local_files_only,
        )
        is not None
    )


def _resolve_composite_model_type(
    source: str | os.PathLike[str],
    model_index: dict[str, Any],
    *,
    revision: str | None,
    local_files_only: bool,
) -> tuple[str | None, str | None]:
    class_name = model_index.get("_class_name")
    if not isinstance(class_name, str):
        return (None, None)

    registration = _COMPOSITE_PIPELINE_TYPES.get(class_name)
    if registration is None:
        return (class_name, None)

    model_type, required_files = registration
    if not all(
        _path_exists(
            source,
            filename,
            revision=revision,
            local_files_only=local_files_only,
        )
        for filename in required_files
    ):
        return (class_name, None)

    return (class_name, model_type)


def resolve_hf_composite_repo(
    source: str | os.PathLike[str],
    *,
    task_type: TaskType | str | None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> ResolvedHFCompositeRepo:
    """Resolve a composite repo component for the requested task."""

    if _has_top_level_config(source, revision=revision, local_files_only=local_files_only):
        return ResolvedHFCompositeRepo(source=source)

    model_index = _load_model_index(source, revision=revision, local_files_only=local_files_only)
    if not isinstance(model_index, dict):
        return ResolvedHFCompositeRepo(source=source)

    composite_class_name, composite_model_type = _resolve_composite_model_type(
        source,
        model_index,
        revision=revision,
        local_files_only=local_files_only,
    )
    model_subfolder: str | None = None
    for candidate in _component_candidates(task_type):
        entry = model_index.get(candidate)
        if not _is_transformers_component(entry):
            continue
        if _component_has_file(
            source,
            candidate,
            "config.json",
            revision=revision,
            local_files_only=local_files_only,
        ):
            model_subfolder = candidate
            break

    tokenizer_subfolder: str | None = None
    for candidate in _TOKENIZER_COMPONENT_CANDIDATES:
        entry = model_index.get(candidate)
        if not _is_transformers_component(entry):
            continue
        if any(
            _component_has_file(
                source,
                candidate,
                filename,
                revision=revision,
                local_files_only=local_files_only,
            )
            for filename in _TOKENIZER_FILES
        ):
            tokenizer_subfolder = candidate
            break

    return ResolvedHFCompositeRepo(
        source=source,
        model_subfolder=model_subfolder,
        tokenizer_subfolder=tokenizer_subfolder,
        composite_class_name=composite_class_name,
        composite_model_type=composite_model_type,
    )


__all__ = ("ResolvedHFCompositeRepo", "resolve_hf_composite_repo")
