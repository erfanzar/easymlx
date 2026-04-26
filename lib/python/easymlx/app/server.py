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

"""Local EasyMLX application server.

The app server layers a model manager and a browser UI on top of the
existing OpenAI-compatible eSurge API server. Inference clients continue
to use the standard ``/v1`` endpoints while the UI uses ``/app/api`` for
model lifecycle operations.
"""

from __future__ import annotations

import importlib
import inspect
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

import easymlx.modules  # noqa: F401  (registers @register_module model classes)
from easymlx import __version__
from easymlx.inference.esurge import eSurge
from easymlx.inference.esurge.server.api_server import eSurgeApiServer

STATIC_DIR = Path(__file__).with_name("static")


def _signature_default(name: str) -> Any:
    parameter = inspect.signature(eSurge.__init__).parameters.get(name)
    if parameter is None or parameter.default is inspect._empty:
        return None
    value = parameter.default
    if isinstance(value, tuple):
        return list(value)
    return value


def _schema_field(
    name: str,
    label: str,
    field_type: str,
    category: str,
    *,
    choices: list[str] | None = None,
    hint: str | None = None,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    step: int | float | None = None,
    placeholder: str | None = None,
    default: Any = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "type": field_type,
        "category": category,
        "default": _signature_default(name) if default is None else default,
        "choices": choices or [],
        "hint": hint,
        "min": minimum,
        "max": maximum,
        "step": step,
        "placeholder": placeholder,
    }


ENGINE_CONFIG_SCHEMA: list[dict[str, Any]] = [
    _schema_field("dtype", "Runtime dtype", "select", "Runtime", choices=["", "float16", "bfloat16", "float32"]),
    _schema_field("max_model_len", "Context length", "int", "Runtime", minimum=1, step=128, default=4096),
    _schema_field("max_num_seqs", "Max sequences", "int", "Runtime", minimum=1, default=1),
    _schema_field("max_num_batched_tokens", "Max batched tokens", "int", "Runtime", minimum=1, default=4096),
    _schema_field(
        "hbm_utilization", "HBM utilization", "float", "Runtime", minimum=0.05, maximum=1.0, step=0.01, default=0.45
    ),
    _schema_field("reserve_tokens", "Reserve tokens", "int", "Runtime", minimum=1),
    _schema_field("seed", "Seed", "int", "Runtime", minimum=0),
    _schema_field("min_input_pad", "Min input pad", "int", "Runtime", minimum=1),
    _schema_field("min_token_pad", "Min token pad", "int", "Runtime", minimum=1),
    _schema_field("page_size", "Page size", "int", "Cache", minimum=1, default=128),
    _schema_field("enable_prefix_caching", "Prefix caching", "bool", "Cache"),
    _schema_field("long_prefill_token_threshold", "Long prefill threshold", "int", "Scheduler", minimum=0),
    _schema_field("max_num_seq_buckets", "Sequence buckets", "int-list", "Scheduler", placeholder="1,2,4,8"),
    _schema_field("async_scheduling", "Async scheduling", "bool", "Scheduler"),
    _schema_field("overlap_execution", "Overlap execution", "bool", "Scheduler"),
    _schema_field(
        "speculative_model",
        "Draft model",
        "text",
        "Speculative",
        placeholder="z-lab/Qwen3.5-9B-DFlash or /path/to/dflash-draft",
        hint=("DFlash drafter for Qwen/Qwen3.5-9B. eSurge passes it as the configured speculative draft/EAGLE3 model."),
    ),
    _schema_field(
        "num_speculative_tokens",
        "Draft tokens",
        "int",
        "Speculative",
        minimum=0,
        default=0,
        hint="Set > 0 to enable speculative decoding.",
    ),
    _schema_field(
        "speculative_method",
        "Speculative method",
        "select",
        "Speculative",
        choices=["dflash", "draft", "eagle3"],
        default="dflash",
    ),
    _schema_field(
        "speculative_model_kwargs",
        "Draft model kwargs",
        "json",
        "Speculative",
        placeholder='{"quantization":"mxfp4"}',
        hint="JSON forwarded to the draft model loader.",
    ),
    _schema_field(
        "eagle3_feature_layer_indices",
        "EAGLE3 layers",
        "int-list",
        "Speculative",
        placeholder="8,16,32",
    ),
    _schema_field("compile_runner", "Compile runner", "bool", "Compiler"),
    _schema_field("use_aot_forward", "AOT forward", "bool", "Compiler"),
    _schema_field("bind_graphstate_for_aot", "Bind graphstate", "bool", "Compiler"),
    _schema_field("runner_verbose", "Runner verbose", "bool", "Compiler"),
    _schema_field("silent_mode", "Silent mode", "bool", "Compiler"),
    _schema_field("auto_truncate_prompt", "Auto truncate prompt", "bool", "Context"),
    _schema_field("auto_cap_new_tokens", "Auto cap new tokens", "bool", "Context"),
    _schema_field("strict_context", "Strict context", "bool", "Context"),
    _schema_field("truncate_mode", "Truncate mode", "select", "Context", choices=["left", "right", "middle"]),
    _schema_field("prefer_preserve_prompt", "Preserve prompt", "bool", "Context"),
    _schema_field("decode_truncated_prompt", "Decode truncated prompt", "bool", "Context"),
    _schema_field("extra_eos_token_ids", "Extra EOS token IDs", "int-list", "Context", placeholder="151643,151645"),
    _schema_field("extra_stops", "Extra stop strings", "string-list", "Context", placeholder="</tool>,<|end|>"),
    _schema_field("tool_parser", "Tool parser", "text", "Parsers", placeholder="auto, llama3_json, hermes, qwen3xml"),
    _schema_field("reasoning_parser", "Reasoning parser", "text", "Parsers", placeholder="auto, qwen3, deepseek_r1"),
    _schema_field("ignore_stop_strings_in_reasoning", "Ignore reasoning stops", "bool", "Parsers"),
    _schema_field("esurge_name", "Engine name", "text", "Identity"),
    _schema_field("enable_window_aware_runtime_cap", "Window-aware cap", "bool", "Advanced"),
    _schema_field("destroy_pages_on_pause", "Destroy pages on pause", "bool", "Advanced"),
    _schema_field("detokenizer_max_states", "Detokenizer states", "int", "Advanced", minimum=1),
    _schema_field("tokenizer_endpoint", "Tokenizer endpoint", "text", "Advanced"),
    _schema_field("detokenizer_endpoint", "Detokenizer endpoint", "text", "Advanced"),
    _schema_field("worker_startup_timeout", "Worker startup timeout", "float", "Advanced", minimum=0.1, step=0.1),
    _schema_field("max_request_outputs", "Max request outputs", "int", "Advanced", minimum=1),
    _schema_field("idle_reset_seconds", "Idle reset seconds", "float", "Advanced", minimum=0),
    _schema_field("idle_reset_min_interval", "Idle reset interval", "float", "Advanced", minimum=0),
    _schema_field("resolution_buckets", "Resolution buckets", "pair-list", "Advanced", placeholder="224x224,448x448"),
    _schema_field("vision_cache_capacity_mb", "Vision cache MB", "int", "Advanced", minimum=0),
]


SAMPLING_SCHEMA: list[dict[str, Any]] = [
    {"name": "max_tokens", "label": "Max tokens", "type": "int", "default": 4096, "min": 1, "step": 1},
    {"name": "temperature", "label": "Temperature", "type": "float", "default": 0.7, "min": 0, "max": 2, "step": 0.05},
    {"name": "top_p", "label": "Top P", "type": "float", "default": 0.95, "min": 0, "max": 1, "step": 0.01},
    {"name": "top_k", "label": "Top K", "type": "int", "default": 0, "min": 0, "step": 1},
    {"name": "presence_penalty", "label": "Presence penalty", "type": "float", "default": 0, "step": 0.05},
    {"name": "repetition_penalty", "label": "Repetition penalty", "type": "float", "default": 1, "step": 0.05},
]


class ModelLoadRequest(BaseModel):
    """Request body for loading an eSurge model."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., min_length=1)
    served_name: str | None = None
    tokenizer: str | None = None
    revision: str | None = None
    local_files_only: bool = False
    converted_cache_dir: str | None = None
    force_conversion: bool = False
    model_class: str | None = None
    replace: bool = False
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    engine_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("served_name", "tokenizer", "revision", "converted_cache_dir", "model_class", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class ModelState(BaseModel):
    """Serializable model lifecycle state."""

    served_name: str
    model_id: str
    status: Literal["loading", "ready", "error"]
    error: str | None = None
    created_at: float
    loaded_at: float | None = None
    load_seconds: float | None = None
    tokenizer: str | None = None
    tokenizer_source: str | None = None
    has_chat_template: bool | None = None
    revision: str | None = None
    local_files_only: bool = False
    converted_cache_dir: str | None = None
    force_conversion: bool = False
    model_class: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    engine_kwargs: dict[str, Any] = Field(default_factory=dict)


@dataclass
class _RuntimeRecord:
    served_name: str
    model_id: str
    status: Literal["loading", "ready", "error"] = "loading"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    loaded_at: float | None = None
    tokenizer: str | None = None
    tokenizer_source: str | None = None
    has_chat_template: bool | None = None
    revision: str | None = None
    local_files_only: bool = False
    converted_cache_dir: str | None = None
    force_conversion: bool = False
    model_class: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    load_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_state(self) -> ModelState:
        load_seconds = None
        if self.loaded_at is not None:
            load_seconds = round(max(self.loaded_at - self.created_at, 0.0), 3)
        elif self.status == "loading":
            load_seconds = round(max(time.time() - self.created_at, 0.0), 3)
        return ModelState(
            served_name=self.served_name,
            model_id=self.model_id,
            status=self.status,
            error=self.error,
            created_at=self.created_at,
            loaded_at=self.loaded_at,
            load_seconds=load_seconds,
            tokenizer=self.tokenizer,
            tokenizer_source=self.tokenizer_source,
            has_chat_template=self.has_chat_template,
            revision=self.revision,
            local_files_only=self.local_files_only,
            converted_cache_dir=self.converted_cache_dir,
            force_conversion=self.force_conversion,
            model_class=self.model_class,
            model_kwargs=self.model_kwargs,
            engine_kwargs=self.engine_kwargs,
        )


class EasyMLXAppServer:
    """Browser app + model lifecycle controller for EasyMLX serving."""

    def __init__(
        self,
        *,
        title: str = "EasyMLX App",
        require_api_key: bool = False,
        admin_api_key: str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, _RuntimeRecord] = {}
        self.api_server = eSurgeApiServer(
            {},
            title=title,
            require_api_key=require_api_key,
            admin_api_key=admin_api_key,
        )
        self.app = self.api_server.app
        self._register_routes()

    def _register_routes(self) -> None:
        self.app.get("/", include_in_schema=False)(self._index)
        self.app.get("/app", include_in_schema=False)(self._index)
        self.app.get("/app/api/info")(self._info)
        self.app.get("/app/api/config-schema")(self._config_schema)
        self.app.get("/app/api/models")(self._list_models)
        self.app.post("/app/api/models/load", status_code=202)(self._load_model)
        self.app.post("/app/api/models/{served_name}/unload")(self._unload_model)
        self.app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="easymlx-app-assets")

    def _index(self) -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    def _info(self) -> dict[str, Any]:
        return {
            "name": "easymlx-app",
            "easymlx_version": __version__,
            "openai_base_url": "/v1",
            "models_endpoint": "/v1/models",
            "chat_endpoint": "/v1/chat/completions",
        }

    def _config_schema(self) -> dict[str, Any]:
        return {
            "engine": ENGINE_CONFIG_SCHEMA,
            "sampling": SAMPLING_SCHEMA,
            "model": {
                "quantization_modes": ["", "affine", "mxfp4", "mxfp8", "nvfp4"],
                "dtype_choices": ["", "float16", "bfloat16", "float32"],
            },
        }

    def _list_models(self) -> dict[str, Any]:
        with self._lock:
            states = [record.to_state().model_dump() for record in self._records.values()]
        states.sort(key=lambda item: item["created_at"], reverse=True)
        return {"data": states}

    def _load_model(self, payload: ModelLoadRequest) -> dict[str, Any]:
        served_name = payload.served_name or self._default_served_name(payload.model_id)
        existing_engine: Any | None = None
        record = _RuntimeRecord(
            served_name=served_name,
            model_id=payload.model_id,
            tokenizer=payload.tokenizer,
            revision=payload.revision,
            local_files_only=payload.local_files_only,
            converted_cache_dir=payload.converted_cache_dir,
            force_conversion=payload.force_conversion,
            model_class=payload.model_class,
            model_kwargs=dict(payload.model_kwargs),
            engine_kwargs=dict(payload.engine_kwargs),
        )
        with self._lock:
            if served_name in self._records and not payload.replace:
                raise HTTPException(
                    status_code=409,
                    detail=f"Model {served_name!r} already exists. Set replace=true to reload it.",
                )
            if payload.replace:
                existing_engine = self.api_server.engines.pop(served_name, None)
                self._records.pop(served_name, None)
            self._records[served_name] = record

        self._close_engine(existing_engine)
        thread = threading.Thread(
            target=self._load_model_in_background,
            args=(record, payload),
            name=f"easymlx-load-{served_name}",
            daemon=True,
        )
        thread.start()
        return {"model": record.to_state().model_dump()}

    def _unload_model(self, served_name: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.pop(served_name, None)
            engine = self.api_server.engines.pop(served_name, None)
        if record is None and engine is None:
            raise HTTPException(status_code=404, detail=f"Model {served_name!r} not found")
        self._close_engine(engine)
        return {"status": "unloaded", "model": served_name}

    def _load_model_in_background(self, record: _RuntimeRecord, payload: ModelLoadRequest) -> None:
        engine: Any | None = None
        try:
            model_kwargs = self._coerce_model_kwargs(payload.model_kwargs)
            engine_kwargs = self._coerce_engine_kwargs(payload.engine_kwargs)
            model_class = self._load_symbol(payload.model_class) if payload.model_class else None
            engine = eSurge.from_pretrained(
                payload.model_id,
                model_class=model_class,
                tokenizer=payload.tokenizer,
                revision=payload.revision,
                local_files_only=payload.local_files_only,
                converted_cache_dir=payload.converted_cache_dir,
                force_conversion=payload.force_conversion,
                model_kwargs=model_kwargs,
                **engine_kwargs,
            )
        except Exception as exc:
            with self._lock:
                current = self._records.get(record.served_name)
                if current is not None and current.load_id == record.load_id:
                    current.status = "error"
                    current.error = f"{type(exc).__name__}: {exc}"
            return

        should_close = False
        with self._lock:
            current = self._records.get(record.served_name)
            if current is None or current.load_id != record.load_id:
                should_close = True
            else:
                self.api_server.engines[record.served_name] = engine
                current.status = "ready"
                current.error = None
                current.loaded_at = time.time()
                tokenizer = getattr(engine, "tokenizer", None)
                current.tokenizer_source = (
                    getattr(tokenizer, "name_or_path", None) or payload.tokenizer or payload.model_id
                )
                current.has_chat_template = bool(getattr(tokenizer, "chat_template", None))
        if should_close:
            self._close_engine(engine)

    @staticmethod
    def _default_served_name(model_id: str) -> str:
        name = model_id.rstrip("/").split("/")[-1] or "model"
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-._")
        return name.lower() or "model"

    @staticmethod
    def _load_symbol(symbol_path: str) -> Any:
        if ":" in symbol_path:
            module_name, _, attr_name = symbol_path.partition(":")
        else:
            module_name, _, attr_name = symbol_path.rpartition(".")
        if not module_name or not attr_name:
            raise ValueError("model_class must be a dotted path like 'pkg.module:ClassName'")
        module = importlib.import_module(module_name)
        value: Any = module
        for part in attr_name.split("."):
            value = getattr(value, part)
        return value

    @classmethod
    def _coerce_model_kwargs(cls, values: dict[str, Any]) -> dict[str, Any]:
        cleaned = cls._drop_empty_values(values)
        if "dtype" in cleaned:
            cleaned["dtype"] = cls._coerce_mx_dtype(cleaned["dtype"])
        if "device" in cleaned:
            cleaned["device"] = cls._coerce_mx_device(cleaned["device"])
        return cleaned

    @classmethod
    def _coerce_engine_kwargs(cls, values: dict[str, Any]) -> dict[str, Any]:
        cleaned = cls._drop_empty_values(values)
        if "dtype" in cleaned:
            cleaned["dtype"] = cls._coerce_mx_dtype(cleaned["dtype"])
        if isinstance(cleaned.get("speculative_model_kwargs"), dict):
            cleaned["speculative_model_kwargs"] = cls._coerce_model_kwargs(cleaned["speculative_model_kwargs"])
        for key in ("max_num_seq_buckets",):
            if isinstance(cleaned.get(key), list):
                cleaned[key] = tuple(cleaned[key])
        if isinstance(cleaned.get("resolution_buckets"), list):
            cleaned["resolution_buckets"] = [
                tuple(item) if isinstance(item, list) else item for item in cleaned["resolution_buckets"]
            ]
        return cleaned

    @classmethod
    def _drop_empty_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in values.items():
            if value is None or value == "":
                continue
            if isinstance(value, dict):
                nested = cls._drop_empty_values(value)
                if nested:
                    cleaned[key] = nested
            else:
                cleaned[key] = value
        return cleaned

    @staticmethod
    def _coerce_mx_dtype(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.lower().replace("mx.", "").replace("mlx.core.", "").replace("torch.", "")
        try:
            import mlx.core as mx
        except Exception:
            return value
        return {
            "float16": mx.float16,
            "fp16": mx.float16,
            "half": mx.float16,
            "bfloat16": mx.bfloat16,
            "bf16": mx.bfloat16,
            "float32": mx.float32,
            "fp32": mx.float32,
            "float": mx.float32,
        }.get(normalized, value)

    @staticmethod
    def _coerce_mx_device(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        if not normalized or normalized == "default":
            return None
        try:
            import mlx.core as mx
        except Exception:
            return value
        match = re.fullmatch(r"(cpu|gpu)(?::(\d+))?", normalized)
        if match is None:
            return value
        device_type = mx.cpu if match.group(1) == "cpu" else mx.gpu
        index = int(match.group(2) or 0)
        return mx.Device(device_type, index)

    @staticmethod
    def _close_engine(engine: Any | None) -> None:
        if engine is None:
            return
        close = getattr(engine, "close", None)
        if callable(close):
            close()


def create_app_server(
    *,
    title: str = "EasyMLX App",
    require_api_key: bool = False,
    admin_api_key: str | None = None,
) -> EasyMLXAppServer:
    """Create an EasyMLX app server instance."""

    return EasyMLXAppServer(
        title=title,
        require_api_key=require_api_key,
        admin_api_key=admin_api_key,
    )
