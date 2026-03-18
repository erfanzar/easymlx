# Configuration file for the Sphinx documentation builder.

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

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Theme
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

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
