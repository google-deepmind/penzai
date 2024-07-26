penzai.deprecated.v1 (Original V1 API)
==============================================

.. module:: penzai.deprecated.v1
.. currentmodule:: penzai.deprecated

.. toctree::
  :hidden:

  ../_autosummary/penzai.deprecated.v1.core
  ../_autosummary/penzai.deprecated.v1.nn
  ../_autosummary/penzai.deprecated.v1.example_models
  ../_autosummary/penzai.deprecated.v1.toolshed


``penzai.deprecated.v1`` is Penzai's original neural network system, which
has been replaced by the new "V2" design. It is available here for compatibility
with existing users of the V1 API. For a summary of the differences between the
V1 and V2 APIs, and a guide to migrating to the V2 API, see the
:doc:`"Changes in the V2 API" </guides/v2_differences>` overview document.

To use the V1 API, we suggest importing the `pz` alias namespace from
`penzai.deprecated.v1.pz`: ::

  from penzai.deprecated.v1 import pz

The rest of this page lists the main components used in the V1 API.


Specific to the V1 Neural Network API
-------------------------------------


V1 Layers
^^^^^^^^^

Most models and layers in the V1 API are subclasses of ``pz.Layer``, which means
they can be called with a single argument.

.. autosummary::

  v1.pz.Layer
  v1.pz.LayerLike

The following decorators can be used to enable runtime shape-checking on
:obj:`penzai.deprecated.v1.core.layer.Layer` (``pz.Layer``) subclasses:

.. autosummary::
  v1.pz.checked_layer_call
  v1.pz.unchecked_layer_call

V1 Parameters
^^^^^^^^^^^^^

In the V1 API, Parameters are always ordinary PyTree nodes. Parameter sharing
is indicated by extra metadata and PyTree transformations inside the model.

.. autosummary::

  v1.pz.nn.Parameter
  v1.pz.nn.ParameterLike
  v1.pz.nn.UninitializedParameter
  v1.pz.nn.add_parameter_prefix
  v1.pz.nn.initialize_parameters
  v1.pz.nn.FrozenParameter
  v1.pz.nn.mark_shareable
  v1.pz.nn.ShareableUninitializedParameter
  v1.pz.nn.attach_shared_parameters
  v1.pz.nn.SharedParameterLookup
  v1.pz.nn.SharedParamTag
  v1.pz.nn.SupportsParameterRenaming
  v1.pz.nn.check_no_duplicated_parameters
  v1.pz.nn.UninitializedParameterError


Basic Combinators
^^^^^^^^^^^^^^^^^

.. autosummary::

  v1.pz.nn.Sequential
  v1.pz.nn.NamedGroup
  v1.pz.nn.CheckedSequential
  v1.pz.nn.Residual
  v1.pz.nn.BranchAndAddTogether
  v1.pz.nn.BranchAndMultiplyTogether
  v1.pz.nn.inline_anonymous_sequentials
  v1.pz.nn.inline_groups
  v1.pz.nn.is_anonymous_sequential
  v1.pz.nn.is_sequential_or_named

Basic Operations
^^^^^^^^^^^^^^^^

.. autosummary::

  v1.pz.nn.Elementwise
  v1.pz.nn.Softmax
  v1.pz.nn.CheckStructure
  v1.pz.nn.Identity
  v1.pz.nn.CastToDType


Linear and Affine Layers
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

  v1.pz.nn.Linear
  v1.pz.nn.RenameAxes
  v1.pz.nn.AddBias
  v1.pz.nn.Affine
  v1.pz.nn.ConstantRescale
  v1.pz.nn.NamedEinsum
  v1.pz.nn.LinearInPlace
  v1.pz.nn.LinearOperatorWeightInitializer
  v1.pz.nn.BiasInitializer
  v1.pz.nn.contract
  v1.pz.nn.variance_scaling_initializer
  v1.pz.nn.xavier_normal_initializer
  v1.pz.nn.xavier_uniform_initializer
  v1.pz.nn.constant_initializer
  v1.pz.nn.zero_initializer


Standardization
^^^^^^^^^^^^^^^

.. autosummary::

  v1.pz.nn.LayerNorm
  v1.pz.nn.Standardize
  v1.pz.nn.RMSLayerNorm
  v1.pz.nn.RMSStandardize


Dropout
^^^^^^^

.. autosummary::

  v1.pz.nn.StochasticDropout
  v1.pz.nn.DisabledDropout
  v1.pz.nn.maybe_dropout


Language Modeling
^^^^^^^^^^^^^^^^^

.. autosummary::
  v1.pz.nn.Attention
  v1.pz.nn.KVCachingAttention
  v1.pz.nn.ApplyAttentionMask
  v1.pz.nn.EmbeddingTable
  v1.pz.nn.EmbeddingLookup
  v1.pz.nn.EmbeddingDecode
  v1.pz.nn.ApplyRoPE

Base Effect Types
^^^^^^^^^^^^^^^^^

.. autosummary::

  v1.pz.de.EffectHandler
  v1.pz.de.EffectRequest
  v1.pz.de.EffectRuntimeImpl
  v1.pz.de.HandledEffectRef
  v1.pz.de.HandlerId


Side Inputs
^^^^^^^^^^^

.. autosummary::

  v1.pz.de.SideInputEffect
  v1.pz.de.SideInputRequest
  v1.pz.de.HandledSideInputRef
  v1.pz.de.WithSideInputsFromInputTuple
  v1.pz.de.WithConstantSideInputs
  v1.pz.de.HoistedTag
  v1.pz.de.hoist_constant_side_inputs


Side Outputs
^^^^^^^^^^^^

.. autosummary::

  v1.pz.de.CollectingSideOutputs
  v1.pz.de.HandledSideOutputRef
  v1.pz.de.SideOutputEffect
  v1.pz.de.SideOutputRequest
  v1.pz.de.SideOutputValue
  v1.pz.de.TellIntermediate


Randomness
^^^^^^^^^^

.. autosummary::

  v1.pz.de.RandomEffect
  v1.pz.de.RandomRequest
  v1.pz.de.TaggedRandomRequest
  v1.pz.de.HandledRandomRef
  v1.pz.de.WithRandomKeyFromArg
  v1.pz.de.WithStatefulRandomKey
  v1.pz.de.WithFrozenRandomState


Local State
^^^^^^^^^^^

.. autosummary::

  v1.pz.de.LocalStateEffect
  v1.pz.de.InitialLocalStateRequest
  v1.pz.de.FrozenLocalStateRequest
  v1.pz.de.SharedLocalStateRequest
  v1.pz.de.HandledLocalStateRef
  v1.pz.de.WithFunctionalLocalState
  v1.pz.de.handle_local_states
  v1.pz.de.freeze_local_states
  v1.pz.de.hoist_shared_state_requests
  v1.pz.de.embed_shared_state_requests


Effect Utilities
^^^^^^^^^^^^^^^^

.. autosummary::

  v1.pz.de.all_handler_ids
  v1.pz.de.free_effect_types
  v1.pz.de.get_effect_color
  v1.pz.de.infer_or_check_handler_id
  v1.pz.de.register_effect_color
  v1.pz.de.UnhandledEffectError



Core utilities, shared with the V2 API
--------------------------------------

Structs and Layers
^^^^^^^^^^^^^^^^^^

Most objects in Penzai models are subclasses of ``pz.Struct`` and
decorated with ``pz.pytree_dataclass``, which makes them into frozen Python
dataclasses that are also JAX PyTrees.

.. autosummary::

  v1.pz.pytree_dataclass
  v1.pz.Struct


PyTree Manipulation
^^^^^^^^^^^^^^^^^^^

Penzai provides a number of utilities to make targeted modifications to PyTrees.
Since Penzai models are PyTrees, you can use them to insert new layers into
models, or modify the configuration of existing layers.

.. autosummary::
  v1.pz.select
  v1.pz.Selection
  v1.pz.combine
  v1.pz.NotInThisPartition
  v1.pz.pretty_keystr


Named Axes
^^^^^^^^^^

``pz.nx`` is an alias for :obj:`penzai.core.named_axes`, which contains
Penzai's named axis system. Some commonly-used attributes on ``pz.nx``:

.. autosummary::
  v1.pz.nx.NamedArray
  v1.pz.nx.nmap
  v1.pz.nx.wrap

See :obj:`penzai.core.named_axes` for documentation of all of the methods and
classes accessible through the ``pz.nx`` alias.

To simplify slicing named axes, Penzai also provides a helper object:

.. autosummary::
  v1.pz.slice


Shape-Checking
^^^^^^^^^^^^^^

``pz.chk`` is an alias for :obj:`penzai.core.shapecheck`, which contains
utilities for checking the shapes of PyTrees of positional and named arrays.
Some commonly-used attributes on ``pz.chk``:

.. autosummary::
  v1.pz.chk.ArraySpec
  v1.pz.chk.var
  v1.pz.chk.vars_for_axes

See :obj:`penzai.core.shapecheck` for documentation of all of the methods and
classes accessible through the `pz.chk` alias.


Dataclass and Struct Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  v1.pz.is_pytree_dataclass_type
  v1.pz.is_pytree_node_field
  v1.pz.StructStaticMetadata
  v1.pz.PyTreeDataclassSafetyError


Rendering and Global Configuration Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These utilities are available in the ``pz`` namespace for backwards
compatibility. However, they have been moved to the separate Treescope
pretty-printing package. See the
`Treescope documentation <https://treescope.readthedocs.io/en/stable/>`_
for more information.

.. autosummary::

  v1.pz.ts
  v1.pz.show
  v1.pz.ContextualValue
  v1.pz.oklch_color
  v1.pz.color_from_string
  v1.pz.dataclass_from_attributes
  v1.pz.init_takes_fields