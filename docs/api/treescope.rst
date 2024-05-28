``penzai.treescope``: Penzai's interactive pretty-printer
=========================================================

.. module:: penzai.treescope

You can configure treescope as the default IPython pretty-printer using ::

  pz.ts.basic_interactive_setup()

or, for more control: ::

  pz.ts.register_as_default()
  pz.ts.register_autovisualize_magic()
  pz.enable_interactive_context()
  pz.ts.active_autovisualizer.set_interactive(pz.ts.ArrayAutovisualizer())

You can also pretty-print individual values using `pz.show` or `pz.ts.display`.
See also the :doc:`pz.ts </api/pz.ts>` module for convenient access to the most
commonly-used treescope functions and classes.

Extending and Customizing Treescope
-----------------------------------

Adding Treescope Support for New Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To teach Treescope how to render your custom type, you can define the special
method ``__penzai_repr__`` with the following signature: ::

  NodePath = <implementation detail>
  Rendering = <implementation detail>

  class MyCustomType:
    ...

    def __penzai_repr__(
      self,
      path: NodePath,
      subtree_renderer: Callable[[Any, NodePath], Rendering],
    ) -> Rendering | type(NotImplemented):
      ...

The types Treescope uses to represent node paths and renderings should be
considered internal implementation details and may change in the future. To
ensure your implementation of ``__penzai_repr__`` is compatible with future
changes, you can use the following high-level API to build renderings:

.. autosummary::
  repr_lib.render_object_constructor
  repr_lib.render_dictionary_wrapper

This API will be expanded in the future. If you would like to customize the
rendering of your type beyond what is currently possible with `repr_lib`, please
open an issue!

Customizing Automatic Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, Treescope will automatically visualize NDArrays using its own
array visualizer. However, you can hook into this system to insert arbitrary
figure objects into the leaves of your tree, replacing arbitrary objects.

To do this, you can define an autovisualization function: ::

  def autovisualizer_fn(
      value: Any,
      path: NodePath | None,
  ) -> pz.ts.IPythonVisualization | pz.ts.ChildAutovisualizer | None:
    ...

For instance, to render certain objects with Plotly, you could define ::

  def my_plotly_autovisualizer(
      value: Any,
      path: NodePath | None,
  ):
    if isinstance(value, (np.ndarray, jax.Array)):
      return pz.ts.IPythonVisualization(
          px.histogram(
              value.flatten(),
              width=400, height=200
          ).update_layout(
              margin=dict(l=20, r=20, t=20, b=20)
          )
      )

You can then enable your autovisualizer in a scope: ::

  with pz.ts.active_autovisualizer.set_scoped(
      my_plotly_autovisualizer
  ):
      IPython.display.display(some_object_with_arrays)

Alternatively, you can also pass custom visualizers to the ``%%autovisualize``
magic at the top of an IPython cell to let it handle the set_scoped boilerplate
for you: ::

  %%autovisualize my_plotly_autovisualizer
  some_object_with_arrays

You can also disable automatic visualization entirely using
``%%autovisualize None``.

Advanced: Custom Handlers and Intermediate Representation (Unstable)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Advanced users who wish to fully customize how Treescope builds renderings can
construct their own renderer objects and handlers, or directly construct
renderings using Treescope's intermediate representation. Renderer objects and
the expected types of handlers are defined in ``penzai.treescope.renderer``, and
the intermediate representation is currently defined in
``penzai.treescope.foldable_representation``.

.. warning::
  The Treescope intermediate representation and handler system will be changing
  in the future. We recommend only using this API for experimentation, and not
  using it in library code. Code that uses the internals of Penzai's
  intermediate representation may break at any time, and should pin a specific
  version of Penzai to avoid future changes.


Other Treescope Methods
------------------------------------

The following methods should usually be accessed using the
:doc:`pz.ts </api/pz.ts>` alias module instead of accessing them directly from
``treescope``.

Using Treescope in IPython Notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

  treescope_ipython.register_as_default
  treescope_ipython.register_autovisualize_magic
  treescope_ipython.register_context_manager_magic


Showing Objects Explicitly
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  treescope_ipython.display
  treescope_ipython.show
  arrayviz.arrayviz.render_array
  arrayviz.arrayviz.render_array_sharding
  arrayviz.arrayviz.render_sharded_shape
  arrayviz.arrayviz.integer_digitbox
  arrayviz.arrayviz.text_on_color


Styling Displayed Objects
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

  figures.inline
  figures.indented
  figures.with_font_size
  figures.with_color
  figures.bolded
  figures.styled


Configuring Treescope
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

  default_renderer.active_renderer
  default_renderer.active_expansion_strategy
  default_renderer.using_expansion_strategy
  autovisualize.active_autovisualizer
  arrayviz.arrayviz.default_diverging_colormap
  arrayviz.arrayviz.default_sequential_colormap

Building Autovisualizers
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

  autovisualize.Autovisualizer
  autovisualize.ChildAutovisualizer
  autovisualize.IPythonVisualization
  autovisualize.CustomTreescopeVisualization
  treescope_ipython.default_magic_autovisualizer
  arrayviz.array_autovisualizer.ArrayAutovisualizer


Rendering to Strings
^^^^^^^^^^^^^^^^^^^^

.. autosummary::

  default_renderer.render_to_text
  default_renderer.render_to_html


Utility Types
^^^^^^^^^^^^^

.. autosummary::

  copypaste_fallback.NotRoundtrippable
