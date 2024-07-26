``pz.ts``: Treescope alias namespace
====================================

.. module:: penzai.pz.ts
.. currentmodule:: penzai

Treescope, Penzai's interactive pretty-printer, has moved to an independent
package `treescope`. See the
`Treescope documentation <https://treescope.readthedocs.io/en/stable/>`_
for more information.

The alias namespace ``pz.ts`` contains shorthand aliases for commonly-used
Treescope functions. However, we recommend that new users use `treescope`
directly.

Using Treescope in IPython Notebooks
------------------------------------

.. autosummary::

  pz.ts.basic_interactive_setup
  pz.ts.register_as_default
  pz.ts.register_autovisualize_magic
  pz.ts.register_context_manager_magic


Showing Objects Explicitly
--------------------------

.. autosummary::
  pz.ts.render_array
  pz.ts.render_array_sharding
  pz.ts.integer_digitbox
  pz.ts.text_on_color
  pz.ts.display
  pz.show


Styling Displayed Objects
-------------------------

.. autosummary::

  pz.ts.inline
  pz.ts.indented
  pz.ts.with_font_size
  pz.ts.with_color
  pz.ts.bolded
  pz.ts.styled


Configuring Treescope
---------------------

.. autosummary::

  pz.ts.active_renderer
  pz.ts.active_expansion_strategy
  pz.ts.using_expansion_strategy
  pz.ts.active_autovisualizer
  pz.ts.default_diverging_colormap
  pz.ts.default_sequential_colormap

Building Autovisualizers
------------------------

.. autosummary::

  pz.ts.ArrayAutovisualizer
  pz.ts.Autovisualizer
  pz.ts.ChildAutovisualizer
  pz.ts.IPythonVisualization
  pz.ts.vocab_autovisualizer
  pz.ts.default_magic_autovisualizer


Rendering to Strings
--------------------

.. autosummary::

  pz.ts.render_to_text
  pz.ts.render_to_html

