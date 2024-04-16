``pz.de``: Data effects alias namespace
=======================================

.. module:: penzai.pz.de
.. currentmodule:: penzai

Base Effect Types
-----------------

.. autosummary::

  pz.de.EffectHandler
  pz.de.EffectRequest
  pz.de.EffectRuntimeImpl
  pz.de.HandledEffectRef
  pz.de.HandlerId


Side Inputs
-----------

.. autosummary::

  pz.de.SideInputEffect
  pz.de.SideInputRequest
  pz.de.HandledSideInputRef
  pz.de.WithSideInputsFromInputTuple
  pz.de.WithConstantSideInputs
  pz.de.HoistedTag
  pz.de.hoist_constant_side_inputs


Side Outputs
------------

.. autosummary::

  pz.de.CollectingSideOutputs
  pz.de.HandledSideOutputRef
  pz.de.SideOutputEffect
  pz.de.SideOutputRequest
  pz.de.SideOutputValue
  pz.de.TellIntermediate


Randomness
----------

.. autosummary::

  pz.de.RandomEffect
  pz.de.RandomRequest
  pz.de.TaggedRandomRequest
  pz.de.HandledRandomRef
  pz.de.WithRandomKeyFromArg
  pz.de.WithStatefulRandomKey
  pz.de.WithFrozenRandomState


Local State
-----------

.. autosummary::

  pz.de.LocalStateEffect
  pz.de.InitialLocalStateRequest
  pz.de.FrozenLocalStateRequest
  pz.de.SharedLocalStateRequest
  pz.de.HandledLocalStateRef
  pz.de.WithFunctionalLocalState
  pz.de.handle_local_states
  pz.de.freeze_local_states
  pz.de.hoist_shared_state_requests
  pz.de.embed_shared_state_requests


Effect Utilities
----------------

.. autosummary::

  pz.de.all_handler_ids
  pz.de.free_effect_types
  pz.de.get_effect_color
  pz.de.infer_or_check_handler_id
  pz.de.register_effect_color
  pz.de.UnhandledEffectError
