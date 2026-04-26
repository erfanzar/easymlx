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

"""Top-level EasyMLX command line interface."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "app":
        from easymlx.app.cli import main as app_main

        app_main(args[1:])
        return

    parser = argparse.ArgumentParser(prog="easymlx")
    parser.add_argument("command", nargs="?", choices=["app"], help="Command to run.")
    namespace = parser.parse_args(args)
    if namespace.command is None:
        parser.print_help()


if __name__ == "__main__":
    main()
