#!/usr/bin/env bash
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
#
# EasyMLX one-shot installer.
#
# Installs uv (Python package manager), bun (frontend tooling), the EasyMLX
# Python project, and the Electron/Vite frontend. After this finishes, the
# `easymlx` and `easymlx-app` CLIs are available on PATH.
#
# Usage:
#   bash scripts/install.sh                # full install
#   bash scripts/install.sh --no-frontend  # skip bun + frontend deps
#   bash scripts/install.sh --no-global    # only set up project venv (no PATH CLI)
#   bash scripts/install.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FRONTEND_DIR="${PROJECT_ROOT}/lib/typescript/easymlx-app"
PYTHON_VERSION="3.13"

INSTALL_FRONTEND=1
INSTALL_GLOBAL=1
ASSUME_YES=0

usage() {
    cat <<'EOF'
EasyMLX installer.

Usage: bash scripts/install.sh [options]

Options:
  --no-frontend   Skip Bun + Electron frontend install.
  --no-global     Skip `uv tool install` (no `easymlx` on PATH; use `uv run easymlx`).
  -y, --yes       Do not prompt; accept defaults.
  -h, --help      Show this help and exit.

What it does:
  1. Installs uv (https://astral.sh) if missing.
  2. Installs bun (https://bun.com) if missing (unless --no-frontend).
  3. Provisions Python 3.13 via uv if needed.
  4. Runs `uv sync` to install the Python project and lock file.
  5. Runs `bun install` inside lib/typescript/easymlx-app (unless --no-frontend).
  6. Runs `uv tool install --editable .` to expose `easymlx` on PATH (unless --no-global).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-frontend) INSTALL_FRONTEND=0; shift ;;
        --no-global)   INSTALL_GLOBAL=0; shift ;;
        -y|--yes)      ASSUME_YES=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

log()   { printf '[install] %s\n' "$*"; }
warn()  { printf '[install] WARN: %s\n' "$*" >&2; }
fatal() { printf '[install] ERROR: %s\n' "$*" >&2; exit 1; }

confirm() {
    local prompt="$1"
    if [[ ${ASSUME_YES} -eq 1 ]]; then return 0; fi
    read -r -p "[install] ${prompt} [y/N] " reply
    [[ "${reply}" == "y" || "${reply}" == "Y" ]]
}

require_macos_arm64() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"
    if [[ "${os}" != "Darwin" ]]; then
        warn "EasyMLX targets macOS (MLX is Apple Silicon only). Detected: ${os}"
        confirm "Continue anyway?" || exit 1
    elif [[ "${arch}" != "arm64" ]]; then
        warn "MLX requires Apple Silicon (arm64). Detected: ${arch}"
        confirm "Continue anyway?" || exit 1
    fi
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        log "uv found: $(uv --version)"
        return
    fi
    log "Installing uv from https://astral.sh/uv/install.sh ..."
    if command -v brew >/dev/null 2>&1; then
        brew install uv
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    if ! command -v uv >/dev/null 2>&1; then
        local cargo_bin="${HOME}/.cargo/bin"
        local local_bin="${HOME}/.local/bin"
        for candidate in "${cargo_bin}" "${local_bin}"; do
            if [[ -x "${candidate}/uv" ]]; then
                export PATH="${candidate}:${PATH}"
                break
            fi
        done
    fi
    command -v uv >/dev/null 2>&1 || fatal "uv install failed; ensure ~/.local/bin or ~/.cargo/bin is on PATH and re-run."
    log "uv installed: $(uv --version)"
}

ensure_bun() {
    if command -v bun >/dev/null 2>&1; then
        log "bun found: $(bun --version)"
        return
    fi
    log "Installing bun from https://bun.com/install ..."
    if command -v brew >/dev/null 2>&1; then
        brew install oven-sh/bun/bun
    else
        curl -fsSL https://bun.com/install | bash
    fi
    if ! command -v bun >/dev/null 2>&1; then
        local bun_bin="${HOME}/.bun/bin"
        if [[ -x "${bun_bin}/bun" ]]; then
            export PATH="${bun_bin}:${PATH}"
        fi
    fi
    command -v bun >/dev/null 2>&1 || fatal "bun install failed; add ~/.bun/bin to PATH and re-run."
    log "bun installed: $(bun --version)"
}

ensure_python() {
    if uv python find "${PYTHON_VERSION}" >/dev/null 2>&1; then
        log "Python ${PYTHON_VERSION} available via uv."
        return
    fi
    log "Installing Python ${PYTHON_VERSION} via uv ..."
    uv python install "${PYTHON_VERSION}"
}

install_python_project() {
    log "Syncing Python dependencies (uv sync) ..."
    cd "${PROJECT_ROOT}"
    uv sync --python "${PYTHON_VERSION}"
}

install_frontend() {
    if [[ ! -d "${FRONTEND_DIR}" ]]; then
        warn "Frontend directory not found at ${FRONTEND_DIR}; skipping."
        return
    fi
    log "Installing frontend dependencies (bun install) ..."
    cd "${FRONTEND_DIR}"
    bun install
}

install_global_cli() {
    log "Installing easymlx CLI globally (uv tool install --editable .) ..."
    cd "${PROJECT_ROOT}"
    uv tool install --editable . --python "${PYTHON_VERSION}" --force
    if ! command -v easymlx >/dev/null 2>&1; then
        warn "easymlx CLI is installed but not on PATH."
        warn "Run: uv tool update-shell    (then reopen your terminal)"
    fi
}

print_summary() {
    cat <<EOF

[install] Done.

Project root: ${PROJECT_ROOT}
Python venv:  ${PROJECT_ROOT}/.venv

Quick start:
  easymlx --help
  easymlx app --dev          # backend + Vite frontend on :5173
  easymlx-app --help

If 'easymlx' is not yet on PATH, run one of:
  uv run easymlx --help
  source .venv/bin/activate && easymlx --help
  uv tool update-shell       # one-time, then reopen terminal
EOF
}

main() {
    log "EasyMLX installer starting in ${PROJECT_ROOT}"
    require_macos_arm64
    ensure_uv
    if [[ ${INSTALL_FRONTEND} -eq 1 ]]; then
        ensure_bun
    fi
    ensure_python
    install_python_project
    if [[ ${INSTALL_FRONTEND} -eq 1 ]]; then
        install_frontend
    fi
    if [[ ${INSTALL_GLOBAL} -eq 1 ]]; then
        install_global_cli
    fi
    print_summary
}

main "$@"
