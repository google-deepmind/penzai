``pz.nn``: Neural network alias namespace
=========================================

.. module:: penzai.pz.nn
.. currentmodule:: penzai


Layers and Parameter Utilities
------------------------------

.. autosummary::
  pz.nn.Layer
  pz.nn.ParameterLike
  pz.nn.derive_param_key
  pz.nn.make_parameter
  pz.nn.assert_no_parameter_slots


Basic Combinators
-----------------

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
----------------

.. autosummary::

  pz.nn.Elementwise
  pz.nn.Softmax
  pz.nn.CheckStructure
  pz.nn.Identity
  pz.nn.CastToDType


Linear and Affine Layers
------------------------

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
---------------

.. autosummary::

  pz.nn.LayerNorm
  pz.nn.Standardize
  pz.nn.RMSLayerNorm
  pz.nn.RMSStandardize


Dropout
-------

.. autosummary::

  pz.nn.StochasticDropout
  pz.nn.DisabledDropout
  pz.nn.maybe_dropout


Language Modeling
-----------------

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
------------

.. autosummary::
  pz.nn.LayerStack
  pz.nn.LayerStackVarBehavior
  pz.nn.layerstack_axes_from_keypath
  pz.nn.LayerStackGetAttrKey
