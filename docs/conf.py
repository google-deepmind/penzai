# Copyright 2024 The Penzai Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration file for the Sphinx documentation builder."""

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
import inspect
import logging
import os
import sys
import typing

import penzai.core
import penzai.data_effects
import penzai.nn
import penzai.treescope
import penzai.pz

import penzai.example_models.simple_mlp
import penzai.example_models.gemma

import penzai.toolshed.jit_wrapper
import penzai.toolshed.lora
import penzai.toolshed.interleave_intermediates
import penzai.toolshed.sharding_util
import penzai.toolshed.check_layers_by_tracing
import penzai.toolshed.isolate_submodel
import penzai.toolshed.basic_training
import penzai.toolshed.annotate_shapes
import penzai.toolshed.model_rewiring
import penzai.toolshed.token_visualization
import penzai.toolshed.unflaxify
import penzai.toolshed.patch_ipdb

from sphinx.util import logging as sphinx_logging

typing.get_type_hints = lambda obj, *unused: obj.__annotations__
sys.path.insert(0, os.path.abspath('../'))
sys.path.append(os.path.abspath('ext'))

from sphinxcontrib import katex


class IgnoreBangWarningFilter(logging.Filter):

  def filter(self, record):
    msg = record.getMessage()
    if msg.startswith('Lexing literal_block') and msg.endswith(
        'as "python" resulted in an error at token: \'!\'. Retrying in relaxed'
        ' mode.'
    ):
      return False
    else:
      return True


sphinx_logging.getLogger('sphinx.highlighting').logger.addFilter(
    IgnoreBangWarningFilter()
)

# -- Project information -----------------------------------------------------

project = 'Penzai'
copyright = '2024, The Penzai Authors'  # pylint: disable=redefined-builtin
author = 'the Penzai Authors'

# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinxcontrib.katex',
    'myst_nb',
    'sphinxcontrib.collections',
    'sphinx_contributors',
    'pz_alias_rewrite',
    'nb_output_cell_to_iframe',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

default_role = 'py:obj'

# -- Options for napoleon -----------------------------------------------------

napoleon_google_docstring = True
napoleon_use_rtype = True
napoleon_attr_annotations = True
napoleon_use_ivar = True

# typehints_use_signature = True
# typehints_use_signature_return = True

# -- Options for autodoc / autosummary -----------------------------------------

autoclass_content = 'class'
autodoc_class_signature = 'separated'
autodoc_default_options = {
    'exclude-members': (
        '__repr__, __str__, __weakref__, __hash__, __delattr__, __eq__,'
        ' __setattr__, __subclasshook__'
    ),
}
autodoc_inherit_docstrings = False

autosummary_generate = ['_autogen_root.rst']

# -- Options for sphinx-collections

collections_target = ''
collections = {
    'notebooks': {
        'driver': 'copy_folder',
        'source': '../notebooks/',
        'target': 'notebooks/',
        'ignore': [],
    }
}

# -- Options for HTML output -------------------------------------------------

html_title = 'penzai'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_book_theme'

html_theme_options = {
    'repository_url': 'https://github.com/google-deepmind/penzai',
    'use_repository_button': True,
    'use_issues_button': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['custom.js']
# html_favicon = '_static/favicon.ico'

# -- Options for myst -------------------------------------------------------
myst_enable_extensions = [
    'amsmath',
    'dollarmath',
]
# nb_execution_mode can be overridden on a notebook level via .ipynb metadata
nb_execution_mode = 'off'
nb_execution_allow_errors = False
# nb_execution_excludepatterns = ['notebooks/*']

# -- Options for katex ------------------------------------------------------

# See: https://sphinxcontrib-katex.readthedocs.io/en/0.4.1/macros.html
latex_macros = r"""
    \def \d              #1{\operatorname{#1}}
"""

# Translate LaTeX macros to KaTeX and add to options for HTML builder
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = 'macros: {' + katex_macros + '}'

# Add LaTeX macros for LATEX builder
latex_elements = {'preamble': latex_macros}

# -- Source code links -------------------------------------------------------


def linkcode_resolve(domain, info):
  """Resolve a GitHub URL corresponding to Python object."""
  if domain != 'py':
    return None

  try:
    mod = sys.modules[info['module']]
  except ImportError:
    return None

  obj = mod
  try:
    for attr in info['fullname'].split('.'):
      obj = getattr(obj, attr)
  except AttributeError:
    return None
  else:
    obj = inspect.unwrap(obj)

  try:
    filename = inspect.getsourcefile(obj)
  except TypeError:
    return None

  try:
    source, lineno = inspect.getsourcelines(obj)
  except OSError:
    return None

  return 'https://github.com/deepmind/penzai/tree/master/penzai/%s#L%d#L%d' % (
      os.path.relpath(filename, start=os.path.dirname(penzai.__file__)),
      lineno,
      lineno + len(source) - 1,
  )


# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'flax': ('https://flax.readthedocs.io/en/latest/', None),
    'optax': ('https://optax.readthedocs.io/en/latest/', None),
    'equinox': ('https://docs.kidger.site/equinox/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3/', None),
}

source_suffix = ['.rst', '.md', '.ipynb']
