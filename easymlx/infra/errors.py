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

"""easymlx infra exceptions.

This package is intentionally inference/serving focused. Training-specific errors
and concepts should live elsewhere.

Exception hierarchy::

    EasyMLXError
    +-- EasyMLXConfigError
    +-- EasyMLXRuntimeError
    +-- EasyMLXNotImplementedFeatureError
"""


class EasyMLXError(Exception):
    """Base class for all easymlx errors.

    All custom exceptions raised by the easymlx library inherit from
    this class, making it possible to catch any easymlx-specific error
    with a single ``except EasyMLXError`` clause.
    """


class EasyMLXConfigError(EasyMLXError):
    """Raised when a configuration is invalid or inconsistent.

    This typically occurs when an unsupported attention mechanism or
    dtype string is specified in :class:`EasyMLXBaseConfig`.
    """


class EasyMLXRuntimeError(EasyMLXError):
    """Raised when an unexpected runtime condition prevents execution.

    Examples include missing optional dependencies (e.g.
    ``huggingface_hub``, ``safetensors``, ``torch``) or I/O failures
    during model loading.
    """


class EasyMLXNotImplementedFeatureError(EasyMLXError):
    """Raised when a requested feature is not implemented in easymlx.

    Use this instead of the built-in ``NotImplementedError`` to keep
    all easymlx errors under the :class:`EasyMLXError` hierarchy.
    """
