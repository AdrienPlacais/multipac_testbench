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

# -- autodoc ---------------------------------------------------
autodoc_default_options = {
    "ignore-module-all": True,
    # Ref of inherited methods, when not specifically redefined
    # for example: :meth:`.ForwardPower.where_is_growing`:
    "inherited-members": True,
    "show_inheritance": True,
    "member-order": "bysource",  # Keep original members order
    "members": True,
    "private-members": True,  # Document _private members
    "special-members": "__init__, __post_init__, __str__",  # Document those special members
    "undoc-members": True,  # Document members without doc
}

# -- Check that there is no broken link --------------------------------------
nitpicky = True
nitpick_ignore = [
    # Not recognized by Sphinx, don't know if this is normal
    ("py:class", "optional"),
    ("py:class", "T"),
    ("py:class", "np.float64"),
    ("py:class", "numpy.float64"),
    # Temporary fix, see https://github.com/sphinx-doc/sphinx/issues/13178
    ("py:class", "pathlib._local.Path"),
    ("py:class", "ins.Instrument"),
]

# Link to other libraries
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Parameters for sphinx-autodoc-typehints and napoleon
always_document_param_types = True
always_use_bar_union = True
typehints_defaults = "comma"
typehints_use_rtype = True
autodoc_typehints = "description"
napoleon_use_rtype = False

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
