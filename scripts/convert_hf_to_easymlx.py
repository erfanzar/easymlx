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

"""Convert a raw Hugging Face checkpoint into an easymlx checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoConfig

import easymlx.modules  # noqa: F401
from easymlx.infra.factory import TaskType, registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Hugging Face repo id or local raw checkpoint directory.")
    parser.add_argument("--out", required=True, help="Target directory for converted easymlx weights.")
    parser.add_argument(
        "--task",
        default="causal_lm",
        choices=("causal_lm", "base"),
        help="Which easymlx task head to convert.",
    )
    parser.add_argument("--revision", default=None, help="Optional HF revision.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local cached Hugging Face files.",
    )
    parser.add_argument(
        "--no-copy-support-files",
        action="store_true",
        help="Do not copy tokenizer/generation support files into the converted directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    hf_config = AutoConfig.from_pretrained(
        args.source,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )

    task_type = TaskType.CAUSAL_LM if args.task == "causal_lm" else TaskType.BASE_MODULE
    registration = registry.get_module_registration(task_type, hf_config.model_type)
    save_path = registration.module.convert_hf_checkpoint(
        args.source,
        save_directory=Path(args.out),
        revision=args.revision,
        local_files_only=args.local_files_only,
        copy_support_files=not args.no_copy_support_files,
    )
    print(save_path)


if __name__ == "__main__":
    main()
