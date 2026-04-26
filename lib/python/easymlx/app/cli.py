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

"""Command line entry point for the EasyMLX local app."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import uvicorn

from .server import ModelLoadRequest, create_app_server


def _json_object(value: str | None, *, option: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{option} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{option} must be a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="easymlx app",
        description="Run the EasyMLX local app and OpenAI-compatible server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", default=8000, type=int, help="HTTP port to bind.")
    parser.add_argument("--title", default="EasyMLX App", help="FastAPI application title.")
    parser.add_argument("--model", help="Optional model ID or local path to load at startup.")
    parser.add_argument("--served-name", help="Name exposed through /v1/models and request payloads.")
    parser.add_argument("--tokenizer", help="Tokenizer ID/path. Defaults to the model path.")
    parser.add_argument("--revision", help="Hub revision for model and tokenizer loading.")
    parser.add_argument("--local-files-only", action="store_true", help="Disable Hugging Face network access.")
    parser.add_argument("--converted-cache-dir", help="Directory for converted EasyMLX checkpoints.")
    parser.add_argument("--force-conversion", action="store_true", help="Re-convert source weights even if cached.")
    parser.add_argument("--model-class", help="Optional dotted model class path, e.g. pkg.module:ModelClass.")
    parser.add_argument("--model-kwargs-json", help="JSON object forwarded as model_kwargs.")
    parser.add_argument("--engine-kwargs-json", help="JSON object forwarded to eSurge.")
    parser.add_argument("--require-api-key", action="store_true", help="Require API keys on inference endpoints.")
    parser.add_argument("--admin-api-key", help="Bootstrap admin API key for /v1/admin/keys.")
    parser.add_argument("--access-log", action="store_true", help="Print one log line for every HTTP request.")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Start the Bun/Vite frontend dev server alongside the Python backend.",
    )
    parser.add_argument("--frontend-host", default="127.0.0.1", help="Frontend dev server host.")
    parser.add_argument("--frontend-port", default=5173, type=int, help="Frontend dev server port.")
    parser.add_argument("--frontend-dir", help="Path to the Bun/Vite easymlx-app directory.")
    parser.add_argument("--bun-bin", default="bun", help="Bun executable to use for the frontend dev server.")
    parser.add_argument(
        "--skip-frontend-install",
        action="store_true",
        help="Do not run `bun install` before starting the frontend dev server.",
    )
    return parser


def _find_frontend_dir(explicit: str | None = None) -> Path:
    if explicit:
        frontend_dir = Path(explicit).expanduser().resolve()
        if (frontend_dir / "package.json").is_file():
            return frontend_dir
        raise FileNotFoundError(f"Frontend package.json not found under {frontend_dir}")

    env_dir = os.environ.get("EASYMLX_FRONTEND_DIR")
    if env_dir:
        return _find_frontend_dir(env_dir)

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidates = (
            parent / "lib" / "typescript" / "easymlx-app",
            parent / "typescript" / "easymlx-app",
        )
        for candidate in candidates:
            if (candidate / "package.json").is_file():
                return candidate
    raise FileNotFoundError("Could not find lib/typescript/easymlx-app. Pass --frontend-dir.")


def _proxy_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def _start_frontend(args: argparse.Namespace) -> subprocess.Popen[str]:
    frontend_dir = _find_frontend_dir(args.frontend_dir)
    bun_bin = shutil.which(args.bun_bin) if not Path(args.bun_bin).is_absolute() else args.bun_bin
    if bun_bin is None:
        raise FileNotFoundError("Bun was not found. Install Bun or pass --bun-bin.")

    if not args.skip_frontend_install and not (frontend_dir / "node_modules").is_dir():
        subprocess.run([bun_bin, "install"], cwd=frontend_dir, check=True)

    env = os.environ.copy()
    env["EASYMLX_API_TARGET"] = f"http://{_proxy_host(args.host)}:{args.port}"
    command = [
        bun_bin,
        "run",
        "vite",
        "--host",
        args.frontend_host,
        "--port",
        str(args.frontend_port),
        "--strictPort",
    ]
    return subprocess.Popen(command, cwd=frontend_dir, env=env, text=True)


def _stop_frontend(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    app_server = create_app_server(
        title=args.title,
        require_api_key=args.require_api_key,
        admin_api_key=args.admin_api_key,
    )
    if args.model:
        app_server._load_model(
            ModelLoadRequest(
                model_id=args.model,
                served_name=args.served_name,
                tokenizer=args.tokenizer,
                revision=args.revision,
                local_files_only=args.local_files_only,
                converted_cache_dir=args.converted_cache_dir,
                force_conversion=args.force_conversion,
                model_class=args.model_class,
                replace=True,
                model_kwargs=_json_object(args.model_kwargs_json, option="--model-kwargs-json"),
                engine_kwargs=_json_object(args.engine_kwargs_json, option="--engine-kwargs-json"),
            )
        )

    frontend_process: subprocess.Popen[str] | None = None
    try:
        if args.dev:
            frontend_process = _start_frontend(args)
            print(f"EasyMLX backend:  http://{_proxy_host(args.host)}:{args.port}", flush=True)
            print(f"EasyMLX frontend: http://{args.frontend_host}:{args.frontend_port}", flush=True)
        uvicorn.run(app_server.app, host=args.host, port=args.port, access_log=args.access_log)
    finally:
        _stop_frontend(frontend_process)


if __name__ == "__main__":
    main()
