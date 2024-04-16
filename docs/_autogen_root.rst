..
  This file is not actually referenced in the docs, but it is the entry point
  that generates automatic summaries for Penzai's modules. `index.rst` points
  directly at the autosummary files generated while processing this one.
  We also reference other orphaned modules here so that Sphinx doesn't give
  warnings about them.

:orphan:

.. autosummary::
  :toctree: _autosummary
  :template: pzmodule_full.rst
  :recursive:

  penzai.core
  penzai.nn
  penzai.data_effects
  penzai.example_models
  penzai.treescope
  penzai.toolshed

.. toctree::
  :hidden:

  notebooks/induction_heads_2B
  _include/_glue_figures
