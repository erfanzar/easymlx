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
INSTALL_APP=1
ASSUME_YES=0
APP_NAME="EasyMLX"
APP_BUNDLE_ID="dev.easymlx.app"
APP_INSTALL_DIR="${HOME}/Applications"

usage() {
    cat <<'EOF'
EasyMLX installer.

Usage: bash scripts/install.sh [options]

Options:
  --no-frontend   Skip Bun + Electron frontend install (also implies --no-app).
  --no-app        Skip building the desktop app bundle and installing the .app shim.
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
  7. Builds the desktop bundle (`bun run build && bun run desktop:build`) and
     installs ${HOME}/Applications/EasyMLX.app (macOS only; unless --no-app).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-frontend) INSTALL_FRONTEND=0; INSTALL_APP=0; shift ;;
        --no-app)      INSTALL_APP=0; shift ;;
        --no-global)   INSTALL_GLOBAL=0; shift ;;
        -y|--yes)      ASSUME_YES=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" && ${INSTALL_APP} -eq 1 ]]; then
    INSTALL_APP=0
fi

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

build_desktop_app() {
    log "Building desktop app bundle (Vite + Electron main) ..."
    cd "${FRONTEND_DIR}"
    bun run build
    bun run desktop:build
}

write_app_icon() {
    local app_path="$1"
    local png_src="${PROJECT_ROOT}/images/easymlx-logo.png"
    local resources_dir="${app_path}/Contents/Resources"
    mkdir -p "${resources_dir}"
    if [[ ! -f "${png_src}" ]]; then
        return
    fi
    if ! command -v sips >/dev/null 2>&1 || ! command -v iconutil >/dev/null 2>&1; then
        cp "${png_src}" "${resources_dir}/icon.png"
        return
    fi
    local tmp iconset
    tmp="$(mktemp -d)"
    iconset="${tmp}/icon.iconset"
    mkdir -p "${iconset}"
    local size doubled
    for size in 16 32 64 128 256 512; do
        doubled=$((size * 2))
        sips -z "${size}" "${size}" "${png_src}" --out "${iconset}/icon_${size}x${size}.png" >/dev/null 2>&1 || true
        sips -z "${doubled}" "${doubled}" "${png_src}" --out "${iconset}/icon_${size}x${size}@2x.png" >/dev/null 2>&1 || true
    done
    if iconutil -c icns "${iconset}" -o "${resources_dir}/icon.icns" 2>/dev/null; then
        :
    else
        cp "${png_src}" "${resources_dir}/icon.png"
    fi
    rm -rf "${tmp}"
}

install_macos_app_shim() {
    local app_path="${APP_INSTALL_DIR}/${APP_NAME}.app"
    local bun_bin
    bun_bin="$(command -v bun)" || fatal "bun not found on PATH; cannot create .app shim."

    log "Installing ${app_path} ..."
    mkdir -p "${APP_INSTALL_DIR}"
    rm -rf "${app_path}"
    mkdir -p "${app_path}/Contents/MacOS"

    cat > "${app_path}/Contents/MacOS/${APP_NAME}" <<EOF
#!/usr/bin/env bash
# Auto-generated by easymlx install.sh; launches the EasyMLX desktop app.
export PATH="${HOME}/.bun/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:\${PATH}"
cd "${FRONTEND_DIR}"
exec "${bun_bin}" scripts/launch-electron.ts "\$@"
EOF
    chmod +x "${app_path}/Contents/MacOS/${APP_NAME}"

    cat > "${app_path}/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${APP_BUNDLE_ID}</string>
    <key>CFBundleVersion</key>
    <string>0.0.1</string>
    <key>CFBundleShortVersionString</key>
    <string>0.0.1</string>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>CFBundleIconFile</key>
    <string>icon</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
EOF

    write_app_icon "${app_path}"
    /usr/bin/touch "${app_path}" 2>/dev/null || true
    log "${app_path} installed. Launch via Spotlight, Launchpad, or 'open -a ${APP_NAME}'."
}

install_desktop_app() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        warn "Desktop .app shim is macOS-only; skipping."
        return
    fi
    build_desktop_app
    install_macos_app_shim
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
EOF
    if [[ ${INSTALL_APP} -eq 1 ]]; then
        cat <<EOF

Desktop app:
  open -a ${APP_NAME}        # or launch ${APP_NAME}.app from Spotlight/Launchpad
  Installed at: ${APP_INSTALL_DIR}/${APP_NAME}.app
EOF
    fi
    cat <<EOF

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
    if [[ ${INSTALL_APP} -eq 1 ]]; then
        install_desktop_app
    fi
    print_summary
}

main "$@"
