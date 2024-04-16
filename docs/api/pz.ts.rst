``pz.ts``: Treescope alias namespace
====================================

.. module:: penzai.pz.ts
.. currentmodule:: penzai


Using Treescope in IPython Notebooks
------------------------------------

.. autosummary::

  pz.ts.register_as_default
  pz.ts.register_autovisualize_magic
  pz.ts.register_context_manager_magic


Showing Objects Explicitly
--------------------------

.. autosummary::
  pz.ts.render_array
  pz.ts.render_array_sharding
  pz.ts.render_sharded_shape
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
  pz.ts.CustomTreescopeVisualization
  pz.ts.vocab_autovisualizer
  pz.ts.default_magic_autovisualizer


Rendering to Strings
--------------------

.. autosummary::

  pz.ts.render_to_text
  pz.ts.render_to_html


Utility Types
-------------

.. autosummary::

  pz.ts.NotRoundtrippable
