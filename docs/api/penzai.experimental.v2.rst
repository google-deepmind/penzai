Simplified V2 API (``penzai.experimental.v2``)
==============================================

.. module:: penzai.experimental.v2


.. toctree::
  :hidden:

  ../_autosummary/penzai.experimental.v2.core
  ../_autosummary/penzai.experimental.v2.nn
  ../_autosummary/penzai.experimental.v2.models
  ../_autosummary/penzai.experimental.v2.toolshed


``penzai.experimental.v2`` is a redesign of Penzai's neural network system, which is intended to simplify the user experience and remove boilerplate in the original design. Eventually, this will be moved out of the "experimental" prefix and replace the original neural network components.

You can read about the V2 design in the guide :doc:`"How to Think in Penzai (v2 API)"</notebooks/v2_how_to_think_in_penzai>`, or :doc:`this document comparing the two APIs</guides/v2_differences>`.

To use the V2 API, we suggest importing the `pz` alias namespace from `penzai.experimental.v2.pz`: ::

  from penzai.experimental.v2 import pz

The rest of this page lists the main components used in the V2 API.


Specific to the V2 Neural Network API
-------------------------------------

Parameters and State Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The V2 API introduces stateful parameters and state variables, which simplify working with shared parameters and interventions with side effects.

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


Layers and Parameter Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  pz.nn.Layer
  pz.nn.ParameterLike
  pz.nn.derive_param_key
  pz.nn.make_parameter
  pz.nn.assert_no_parameter_slots


Basic Combinators
^^^^^^^^^^^^^^^^^

.. autosummary::

  pz.nn.Sequential
  pz.nn.NamedGroup
  pz.nn.CheckedSequential
  pz.nn.Residual
  pz.nn.BranchAndAddTogether
  pz.nn.BranchAndMultiplyTogether
  pz.nn.inline_anonymous_sequentials
  pz.nn.inline_groups
  pz.nn.is_anonymous_sequential
  pz.nn.is_sequential_or_named

Basic Operations
^^^^^^^^^^^^^^^^

.. autosummary::

  pz.nn.Elementwise
  pz.nn.Softmax
  pz.nn.CheckStructure
  pz.nn.Identity
  pz.nn.CastToDType


Linear and Affine Layers
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

  pz.nn.Linear
  pz.nn.RenameAxes
  pz.nn.AddBias
  pz.nn.Affine
  pz.nn.ConstantRescale
  pz.nn.NamedEinsum
  pz.nn.LinearInPlace
  pz.nn.LinearOperatorWeightInitializer
  pz.nn.contract
  pz.nn.variance_scaling_initializer
  pz.nn.xavier_normal_initializer
  pz.nn.xavier_uniform_initializer
  pz.nn.constant_initializer
  pz.nn.zero_initializer


Standardization
^^^^^^^^^^^^^^^

.. autosummary::

  pz.nn.LayerNorm
  pz.nn.Standardize
  pz.nn.RMSLayerNorm
  pz.nn.RMSStandardize


Dropout
^^^^^^^

.. autosummary::

  pz.nn.StochasticDropout
  pz.nn.DisabledDropout
  pz.nn.maybe_dropout


Language Modeling
^^^^^^^^^^^^^^^^^

.. autosummary::
  pz.nn.Attention
  pz.nn.KVCachingAttention
  pz.nn.ApplyExplicitAttentionMask
  pz.nn.ApplyCausalAttentionMask
  pz.nn.ApplyCausalSlidingWindowAttentionMask
  pz.nn.EmbeddingTable
  pz.nn.EmbeddingLookup
  pz.nn.EmbeddingDecode
  pz.nn.ApplyRoPE


Layer Stacks
^^^^^^^^^^^^

.. autosummary::
  pz.nn.LayerStack
  pz.nn.LayerStackVarBehavior
  pz.nn.layerstack_axes_from_keypath
  pz.nn.LayerStackGetAttrKey


Core utilities, shared with the V1 API
--------------------------------------

Structs and Layers
^^^^^^^^^^^^^^^^^^

Most objects in Penzai models are subclasses of ``pz.Struct`` and
decorated with ``pz.pytree_dataclass``, which makes them into frozen Python
dataclasses that are also JAX PyTrees.

.. autosummary::

  pz.pytree_dataclass
  pz.Struct


PyTree Manipulation
^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^

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
^^^^^^^^^^^^^

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


Shape-Checking
^^^^^^^^^^^^^^

``pz.chk`` is an alias for :obj:`penzai.core.shapecheck`, which contains
utilities for checking the shapes of PyTrees of positional and named arrays.
Some commonly-used attributes on ``pz.chk``:

.. autosummary::
  pz.chk.ArraySpec
  pz.chk.var
  pz.chk.vars_for_axes

See :obj:`penzai.core.shapecheck` for documentation of all of the methods and
classes accessible through the `pz.chk` alias.


Context Management
^^^^^^^^^^^^^^^^^^

.. autosummary::

  pz.disable_interactive_context
  pz.enable_interactive_context
  pz.ContextualValue

Dataclass and Struct Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  pz.dataclass_from_attributes
  pz.init_takes_fields
  pz.is_pytree_dataclass_type
  pz.is_pytree_node_field
  pz.StructStaticMetadata
  pz.PyTreeDataclassSafetyError
