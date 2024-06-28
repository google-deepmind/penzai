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
  penzai.toolshed

  penzai.treescope.arrayviz
  penzai.treescope.autovisualize
  penzai.treescope.copypaste_fallback
  penzai.treescope.default_renderer
  penzai.treescope.figures
  penzai.treescope.repr_lib
  penzai.treescope.treescope_ipython

  penzai.experimental.v2.core
  penzai.experimental.v2.nn
  penzai.experimental.v2.models
  penzai.experimental.v2.toolshed

.. toctree::
  :hidden:

  notebooks/induction_heads_2B
  notebooks/v2_induction_heads_2B
  _include/_glue_figures
