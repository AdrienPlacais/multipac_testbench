# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath(
    "/home/placais/Documents/Simulation/python/multipac_testbench/"))

project = 'MULTIPAC test bench'
copyright = '2023, A. Plaçais'
author = 'A. Plaçais'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx.ext.todo",
    "nbsphinx",
]

add_module_names = False
default_role = 'literal'
todo_include_todos = True

templates_path = ['_templates']
exclude_patterns = ['_build',
                    'Thumbs.db',
                    '.DS_Store',
                    'experimental',
                    ]
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "display_version": True,
}
html_static_path = ['_static']


# -- Options for LaTeX output ------------------------------------------------
# https://stackoverflow.com/questions/28454217/how-to-avoid-the-too-deeply-nested-error-when-creating-pdfs-with-sphinx
latex_elements = {
    'preamble': r'\usepackage{enumitem}\setlistdepth{99}'
}
