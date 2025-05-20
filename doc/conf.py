from __future__ import annotations

from importlib.metadata import metadata

project = "DownClim"
meta = metadata(project.lower())
release = meta.get("version")
copyright = 'MIT - CIRAD'
author = "Thomas Arsouze"
desc = meta['Summary']
version = ".".join(release.split('.')[:3])

extensions = [
    'sphinx.ext.autodoc',  # support for automatic inclusion of docstring
    'sphinx.ext.autosummary',  # generates autodoc summaries
    'sphinx.ext.doctest',  # inclusion and testing of doctest code snippets
    'sphinx.ext.intersphinx', # support for linking to other projects
    'sphinx.ext.mathjax',  # support for math equations
    'sphinx.ext.ifconfig',  # support for conditional content
    'sphinx.ext.viewcode',  # support for links to source code
    'sphinx.ext.coverage',  # includes doc coverage stats in the documentation
    'sphinx.ext.todo',      # support for todo items
    'sphinx.ext.napoleon',  # support for numpy and google style docstrings
    "sphinx_favicon",      # support for favicon
    "sphinx_copybutton",      # support for copybutton in code blocks
    "nbsphinx",     # for integrating jupyter notebooks
    "myst_parser"   # for parsing .md files
    ]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
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
pygments_style = 'sphinx'

autosummary_generate = True

html_theme = "furo"
html_baseurl = "https://DownClim.github.io/DownClim/"

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True


nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. note::

        | This page was generated from `{{ docname }}`__.
        | Interactive online version: :raw-html:`<a href="https://mybinder.org/v2/gh/geopandas/geopandas/main?urlpath=lab/tree/doc/source/{{ docname }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>`

        __ https://github.com/geopandas/geopandas/blob/main/doc/source/{{ docname }}
"""
