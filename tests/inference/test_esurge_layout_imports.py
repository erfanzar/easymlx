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

"""Import-contract tests for the eSurge package layout migration."""

from __future__ import annotations

import importlib
from importlib.util import find_spec

import easymlx.inference as inference_pkg
import easymlx.inference.esurge as esurge_pkg


def test_esurge_legacy_public_imports_are_stable():
    assert esurge_pkg.eSurge is not None
    assert esurge_pkg.Config is not None
    assert esurge_pkg.SamplingParams is not None
    assert esurge_pkg.RequestOutput is not None
    assert esurge_pkg.MetricsCollector is not None
    assert esurge_pkg.eSurgeApiServer is not None
    assert inference_pkg.eSurge is not None
    assert inference_pkg.eSurgeApiServer is not None
    assert inference_pkg.SamplingParams is not None


def _assert_module_import(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name


def test_esurge_new_layout_packages_import_if_present():
    if find_spec("easymlx.inference.esurge.core") is None:
        return

    for module_name in (
        "easymlx.inference.esurge.distributed",
        "easymlx.inference.esurge.core",
        "easymlx.inference.esurge.mixins",
        "easymlx.inference.esurge.runners",
        "easymlx.inference.esurge.scheduler",
        "easymlx.inference.esurge.multimodal",
    ):
        _assert_module_import(module_name)

    for module_name in (
        "easymlx.inference.esurge.core.interface",
        "easymlx.inference.esurge.esurge_engine",
        "easymlx.inference.esurge.server.api_keys",
        "easymlx.inference.esurge.server.auth_endpoints",
    ):
        if find_spec(module_name) is not None:
            _assert_module_import(module_name)


def test_esurge_symbol_aliases_between_legacy_and_mirrored_layout():
    legacy_engine = importlib.import_module("easymlx.inference.esurge.engine")
    if find_spec("easymlx.inference.esurge.esurge_engine") is None:
        return

    mirrored_engine = importlib.import_module("easymlx.inference.esurge.esurge_engine")
    assert legacy_engine.eSurge is mirrored_engine.eSurge
    assert mirrored_engine.eSurge.__name__ == "eSurge"
