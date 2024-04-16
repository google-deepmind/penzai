``pz``: Penzai's alias namespace
================================

.. module:: penzai.pz
.. currentmodule:: penzai


.. toctree::
  :hidden:

  pz.ts
  pz.nn
  pz.de


Structs and Layers
------------------

Most objects in Penzai models are subclasses of ``pz.Struct`` and
decorated with ``pz.pytree_dataclass``, which makes them into frozen Python
dataclasses that are also JAX PyTrees. Most models and layers are
subclasses of ``pz.Layer``, which means they can be called with a single
argument.

.. autosummary::

  pz.pytree_dataclass
  pz.Struct
  pz.Layer
  pz.LayerLike


PyTree Manipulation
-------------------

Penzai provides a number of utilities to make targeted modifications to PyTrees.
Since Penzai models are PyTrees, you can use them to insert new layers into
models, or modify the configuration of existing layers.

.. autosummary::
  pz.select
  pz.Selection
  pz.combine
  pz.NotInThisPartition
  pz.pretty_keystr


Named Axes
----------

``pz.nx`` is an alias for :obj:`penzai.core.named_axes`, which contains
Penzai's named axis system. Some commonly-used attributes on ``pz.nx``:

.. autosummary::
  pz.nx.NamedArray
  pz.nx.nmap
  pz.nx.wrap

See :obj:`penzai.core.named_axes` for documentation of all of the methods and
classes accessible through the ``pz.nx`` alias.

To simplify slicing named axes, Penzai also provides a helper object:

.. autosummary::
  pz.slice


Visualization
-------------

:obj:`pz.ts` is an alias namespace for Penzai's interactive pretty printer
Treescope. Some commonly-used attributes on :obj:`pz.ts`:

.. autosummary::
  pz.ts.register_as_default
  pz.ts.register_autovisualize_magic
  pz.ts.render_array

See the documentation for :obj:`pz.ts` to view all of the methods and
classes accessible through this alias namespace.

Penzai also provides a utility for quickly showing a value with Treescope in an
IPython notebook, using syntax similar to ordinary ``print``:

.. autosummary::
  pz.show


Neural Networks
---------------

:obj:`pz.nn` is an alias namespace for Penzai's declarative neural network
system, which uses a combinator-based design to expose all of your model's
operations as nodes in your model PyTree. :obj:`pz.nn` re-exports layers from
submodules of :obj:`penzai.nn` in a single convenient namespace.

See the documentation for :obj:`pz.nn` to view all of the
methods and classes accessible through this alias namespace.

Data Effects
------------

:obj:`pz.de` is an alias namespace for Penzai's "data effect" system, which
represents side inputs, side outputs, randomness, and mutable state as typed
attributes inside your model PyTree, and allows handling them using functional
effect handlers. :obj:`pz.de` re-exports effect requests and effect handlers
from submodules of :obj:`penzai.data_effects` in a single convenient namespace.

See the documentation for :obj:`pz.de` to view all of the
methods and classes accessible through this alias namespace.


Shape-Checking
--------------

``pz.chk`` is an alias for :obj:`penzai.core.shapecheck`, which contains
utilities for checking the shapes of PyTrees of positional and named arrays.
Some commonly-used attributes on ``pz.chk``:

.. autosummary::
  pz.chk.ArraySpec
  pz.chk.var
  pz.chk.vars_for_axes

See :obj:`penzai.core.shapecheck` for documentation of all of the methods and
classes accessible through the `pz.chk` alias.

The following decorators can be used to enable runtime shape-checking on
:obj:`penzai.core.layer.Layer` (``pz.Layer``) subclasses:

.. autosummary::
  pz.checked_layer_call
  pz.unchecked_layer_call


Context Management
------------------

.. autosummary::

  pz.disable_interactive_context
  pz.enable_interactive_context
  pz.ContextualValue

Dataclass and Struct Utilities
------------------------------

.. autosummary::
  pz.dataclass_from_attributes
  pz.init_takes_fields
  pz.is_pytree_dataclass_type
  pz.is_pytree_node_field
  pz.StructStaticMetadata
  pz.PyTreeDataclassSafetyError

Context and State Management
----------------------------

.. autosummary::

  pz.disable_interactive_context
  pz.enable_interactive_context
  pz.ContextualValue
  pz.RandomStream

Formatting Utilities
--------------------

.. autosummary::
  pz.oklch_color
  pz.color_from_string
