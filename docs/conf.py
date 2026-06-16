from __future__ import annotations

from importlib.metadata import metadata

project = "DownClim"
meta = metadata(project.lower())
release = meta.get("version")
copyright = "MIT - CIRAD"
author = "Thomas Arsouze"
desc = meta["Summary"]
version = ".".join(release.split(".")[:3])

extensions = [
    "sphinx.ext.autodoc",  # support for automatic inclusion of docstring
    "sphinx.ext.autosummary",  # generates autodoc summaries
    "sphinx.ext.doctest",  # inclusion and testing of doctest code snippets
    "sphinx.ext.intersphinx",  # support for linking to other projects
    "sphinx.ext.mathjax",  # support for math equations
    "sphinx.ext.ifconfig",  # support for conditional content
    "sphinx.ext.viewcode",  # support for links to source code
    "sphinx.ext.coverage",  # includes doc coverage stats in the documentation
    "sphinx.ext.todo",  # support for todo items
    "sphinx.ext.napoleon",  # support for numpy and google style docstrings
    "sphinx_favicon",  # support for favicon
    "sphinx_copybutton",  # support for copybutton in code blocks
    # "myst_parser",   # for parsing .md files
    "nbsphinx",  # nbgallery directive and notebook processing
    "myst_nb",  # myst with .md and notebooks (used for hiding cells)
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]


language = "en"
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

autosummary_generate = True

# Additional configuration for better cross-referencing
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# MyST configuration
myst_heading_anchors = 3
myst_url_schemes = ("http", "https", "mailto")

html_theme = "furo"
html_baseurl = "https://downclim.readthedocs.io/en/latest/"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

suppress_warnings = [
    "myst.xref_missing",
    "myst.header",
    "docutils",
]

always_document_param_types = True

# Notebook execution — uncomment when notebooks can run without failures
# nb_execution_mode = "auto"
# nb_execution_timeout = 600
# nb_execution_allow_errors = True
# nb_execution_excludepatterns = [
#     "examples/get_available_simulation.ipynb",
# ]


nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. note::

        | This page was generated from `{{ docname }}`__.
        | Interactive online version: :raw-html:`<a href="https://mybinder.org/v2/gh/DownClim/DownClim/main?urlpath=lab/tree/doc/{{ docname }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>`

        __ https://github.com/DownClim/DownClim/blob/main/doc/{{ docname }}
"""
