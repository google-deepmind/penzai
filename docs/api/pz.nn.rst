``pz.nn``: Neural network alias namespace
=========================================

.. module:: penzai.pz.nn
.. currentmodule:: penzai


Parameters
----------

.. autosummary::

  pz.nn.Parameter
  pz.nn.ParameterLike
  pz.nn.UninitializedParameter
  pz.nn.add_parameter_prefix
  pz.nn.initialize_parameters
  pz.nn.FrozenParameter
  pz.nn.mark_shareable
  pz.nn.ShareableUninitializedParameter
  pz.nn.attach_shared_parameters
  pz.nn.SharedParameterLookup
  pz.nn.SharedParamTag
  pz.nn.SupportsParameterRenaming
  pz.nn.check_no_duplicated_parameters
  pz.nn.UninitializedParameterError


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
  pz.nn.BiasInitializer
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
  pz.nn.ApplyAttentionMask
  pz.nn.EmbeddingTable
  pz.nn.EmbeddingLookup
  pz.nn.EmbeddingDecode
  pz.nn.ApplyRoPE
