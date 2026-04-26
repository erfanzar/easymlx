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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

project = "EasyMLX"

copyright = "2026, Erfan Zare Chavoshi"

author = "Erfan Zare Chavoshi"

release = "0.0.1"



extensions = [

    "myst_parser",

    "sphinx.ext.autodoc",

    "sphinx.ext.napoleon",

    "sphinx.ext.intersphinx",

    "sphinx.ext.viewcode",

    "sphinx_design",

    "sphinx_copybutton",

]



templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]





source_suffix = {

    ".rst": "restructuredtext",

    ".md": "markdown",

}





html_theme = "sphinx_book_theme"

html_static_path = ["_static"]

html_css_files = ["style.css"]

html_theme_options = {

    "repository_url": "https://github.com/erfanzar/easymlx",

    "use_repository_button": True,

    "use_issues_button": True,

    "use_edit_page_button": False,

    "show_toc_level": 2,

}

html_title = "EasyMLX"





myst_enable_extensions = [

    "colon_fence",

    "deflist",

    "fieldlist",

    "tasklist",

]

myst_heading_anchors = 3





intersphinx_mapping = {

    "python": ("https://docs.python.org/3", None),

    "numpy": ("https://numpy.org/doc/stable/", None),

}





autodoc_member_order = "bysource"

autodoc_typehints = "description"





copybutton_prompt_text = r">>> |\.\.\. |\$ "

copybutton_prompt_is_regexp = True
