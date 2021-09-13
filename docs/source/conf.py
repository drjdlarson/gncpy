# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import sphinx_theme
sys.path.append(os.path.abspath('../../gncpy'))


# -- Project information -----------------------------------------------------

project = 'GNCPy'
copyright = '2020, Laboratory for Autonomy, GNC, and Estimation Research (LAGER)'
author = 'Laboratory for Autonomy, GNC, and Estimation Research (LAGER)'

# The full version, including alpha/beta/rc tags
release = '0.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = ['_templates']

# Configuration for autodoc/summary
# note see https://stackoverflow.com/questions/2701998/sphinx-autodoc-is-not-automatic-enough
# for help/details
autosummary_generate = True
autodoc_member_order = 'groupwise'
add_module_names = False

# Todo configuration
todo_include_todos = True
todo_link_only = True

# bibtex config
bibtex_bibfiles = ['refs.bib']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'classic'
# html_theme = 'alabaster'
html_theme = "stanford_theme"
html_theme_path = [sphinx_theme.get_html_theme_path('stanford-theme')]
html_theme_options = {
    'display_version': True,
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
