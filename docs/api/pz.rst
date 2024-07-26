``pz``: Penzai's alias namespace
================================

.. module:: penzai.pz
.. currentmodule:: penzai


.. toctree::
  :hidden:

  pz.nn
  pz.ts


Structs
-------

Most objects in Penzai models are subclasses of ``pz.Struct`` and
decorated with ``pz.pytree_dataclass``, which makes them into frozen Python
dataclasses that are also JAX PyTrees.

.. autosummary::

  pz.pytree_dataclass
  pz.Struct


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


Parameters and State Variables
------------------------------

Penzai handles mutable state by embedding stateful parameters and variables into
JAX pytrees. It provides a number of utilities to manipulate these stateful
components and support passing them across JAX transformation boundaries.

.. autosummary::

  pz.Parameter
  pz.ParameterValue
  pz.ParameterSlot
  pz.StateVariable
  pz.StateVariableValue
  pz.StateVariableSlot
  pz.unbind_variables
  pz.bind_variables
  pz.freeze_variables
  pz.variable_jit
  pz.unbind_params
  pz.freeze_params
  pz.unbind_state_vars
  pz.freeze_state_vars


.. autosummary::

  pz.VariableConflictError
  pz.UnboundVariableError
  pz.VariableLabel
  pz.AbstractVariable
  pz.AbstractVariableValue
  pz.AbstractVariableSlot
  pz.AutoStateVarLabel
  pz.ScopedStateVarLabel
  pz.scoped_auto_state_var_labels
  pz.RandomStream


Neural Networks
---------------

:obj:`pz.nn` is an alias namespace for Penzai's declarative neural network
system, which uses a combinator-based design to expose all of your model's
operations as nodes in your model PyTree. :obj:`pz.nn` re-exports layers from
submodules of :obj:`penzai.nn` in a single convenient namespace.

See the documentation for :obj:`pz.nn` to view all of the
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


Dataclass and Struct Utilities
------------------------------

.. autosummary::
  pz.is_pytree_dataclass_type
  pz.is_pytree_node_field
  pz.StructStaticMetadata
  pz.PyTreeDataclassSafetyError


Rendering and Global Configuration Management
---------------------------------------------

These utilities are available in the ``pz`` namespace for backwards
compatibility. However, they have been moved to the separate Treescope
pretty-printing package. See the
`Treescope documentation <https://treescope.readthedocs.io/en/stable/>`_
for more information.

.. autosummary::

  pz.ts
  pz.show
  pz.ContextualValue
  pz.oklch_color
  pz.color_from_string
  pz.dataclass_from_attributes
  pz.init_takes_fields
