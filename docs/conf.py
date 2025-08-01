# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from __future__ import annotations

import os
import sys
from pprint import pformat

import multipac_testbench
import sphinx
from sphinx.util import inspect

sys.path.append(os.path.abspath("./_ext"))

project = "MULTIPAC test bench"
author = "Adrien Plaçais"
copyright = "2025, " + author

version = multipac_testbench.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_extensions",
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinx_tabs.tabs",
    "sphinxcontrib.bibtex",
]

# -- autodoc -------------------------------------------------------------
autodoc_default_options = {
    "ignore-module-all": True,
    # Ref of inherited methods, when not specifically redefined
    # for example: :meth:`.ForwardPower.where_is_growing`:
    "inherited-members": True,
    "members": True,
    "member-order": "bysource",
    "private-members": True,
    "special-members": "__init__, __post_init__, __str__",  # Document those special members
    "undoc-members": True,  # Document members without doc
    "show_inheritance": True,
}
autodoc_mock_imports = []

napoleon_use_rtype = True
typehints_document_rtype = True
typehints_use_rtype = True

# sphinx-autodoc-typehints
always_document_param_types = True
always_use_bars_union = True

# -----------------------------------------------------------------------------
typehints_defaults = "comma"
add_module_names = False
default_role = "literal"
todo_include_todos = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "experimental",
    "**/.pytest_cache/*",
]
bibtex_bibfiles = ["references.bib"]

# -- Check that there is no broken link --------------------------------------
nitpicky = True
nitpick_ignore = [
    # Not recognized by Sphinx, don't know if this is normal
    ("py:class", "optional"),
    ("py:class", "T"),
    ("py:class", "np.float64"),
    ("py:class", "numpy.float64"),
    ("py:class", "numpy.int32"),
    ("py:obj", "numpy._typing._array_like._ScalarType_co"),
    # Temporary fix, see https://github.com/sphinx-doc/sphinx/issues/13178
    ("py:class", "pathlib._local.Path"),
    ("py:class", "Path"),
]

# Link to other libraries
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_sidebars = {
    "**": [
        "versions.html",
    ],
}


# -- Constants display fix ---------------------------------------------------
# https://stackoverflow.com/a/65195854
def object_description(obj: object) -> str:
    """Format the given object for a clearer printing."""
    return pformat(obj, indent=4)


inspect.object_description = object_description

# -- Bug fixes ---------------------------------------------------------------
# Fix following warning:
# <unknown>:1: WARNING: py:class reference target not found: pathlib._local.Path [ref.class]
# Note that a patch is provided by Sphinx 8.2, but nbsphinx 0.9.7 requires
# sphinx<8.2
# Associated issue:
# https://github.com/sphinx-doc/sphinx/issues/13178
if sys.version_info[:2] >= (3, 13) and sphinx.version_info[:2] < (8, 2):  # type: ignore
    import pathlib

    from sphinx.util.typing import _INVALID_BUILTIN_CLASSES

    _INVALID_BUILTIN_CLASSES[pathlib.Path] = "pathlib.Path"  # type: ignore
    nitpick_ignore.append(("py:class", "pathlib._local.Path"))
