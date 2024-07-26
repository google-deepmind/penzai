# Copyright 2024 The Penzai Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Basic training logic for training simple models.

This module provides a barebones implementation of training logic that supports
training Penzai models. This can be used to train simple models that do not
require more sophisticated training loops. It also serves as a starting point
for more complex training scripts.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

import jax
import optax
from penzai import pz

ModelPyTree = Any
AuxOutPyTree = Any
LossStatePyTree = Any
OptimizerStatePyTree = Any
PRNGKeyArray = jax.Array


class LossFunction(Protocol):
  """Signature for loss functions expected by the common training step."""

  def __call__(
      self,
      *,
      model: ModelPyTree,
      state: LossStatePyTree,
      rng: PRNGKeyArray,
      **kwargs,
  ) -> tuple[jax.Array, LossStatePyTree, AuxOutPyTree]:
    """Signature for a loss function.

    Args:
      model: The structure with parameters, usually a neural network model.
      state: Arbitrary state managed by the loss function. Can be None.
      rng: A JAX PRNGKey, may be ignored.
      **kwargs: Arguments passed to the train step, usually inputs to the model
        or labels.

    Returns:
      A tuple ``(loss, new_state, aux_outputs)`` for this example. ``loss``
      should be a scalar. ``new_state`` should match the structure of ``state``.
      ``aux_outputs`` can be an arbitrary PyTree.
    """


@pz.pytree_dataclass
class InternalTrainerState(pz.Struct):
  """Internal state for the trainer.

  Attributes:
    step: Current step of training.
    opt_state: An optimizer state for the learnable parts of `model`.
    loss_fn_state: Arbitrary state managed by the loss function. For instance,
      if your model has local state, you can functionalize it using
      `pz.de.handle_local_states` and store its state dict here.
  """

  step: int | jax.Array
  opt_state: OptimizerStatePyTree
  loss_fn_state: LossStatePyTree


def _stateful_trainer_step(
    root_rng: PRNGKeyArray,
    state: InternalTrainerState,
    stateless_model: ModelPyTree,
    trainable_params: tuple[pz.ParameterValue, ...],
    frozen_params: tuple[pz.ParameterValue, ...],
    state_vars: tuple[pz.StateVariableValue, ...],
    loss_fn: LossFunction,
    optimizer_def: optax.GradientTransformation,
    kwargs: dict[str, Any],
) -> tuple[
    AuxOutPyTree,
    tuple[pz.ParameterValue, ...],
    tuple[pz.StateVariableValue, ...],
    InternalTrainerState,
]:
  """Implementation of the training step for StatefulTrainer."""
  step_rng = jax.random.fold_in(root_rng, state.step)

  def compute_loss_and_updates(
      trainable_params: tuple[pz.ParameterValue, ...],
  ) -> tuple[
      jax.Array,
      tuple[tuple[pz.ParameterValue, ...], LossStatePyTree, AuxOutPyTree],
  ]:
    """Computes loss, aux out, and state updates for a single example."""
    # Rebuild the model using the input parameters, which we are
    # differentiating with respect to, and the current states. Parameters are
    # kept frozen, since they will be updated by the optimizer, not the model.
    state_vars_mut = tuple(v.unfreeze_as_copy() for v in state_vars)
    model = pz.bind_variables(
        stateless_model,
        state_vars_mut + trainable_params + frozen_params,
    )
    loss, new_loss_fn_state, aux_outputs = loss_fn(
        model=model, state=state.loss_fn_state, rng=step_rng, **kwargs
    )
    return loss, (
        pz.freeze_variables(state_vars_mut),
        new_loss_fn_state,
        aux_outputs,
    )

  # Take gradients.
  grad_fn = jax.grad(compute_loss_and_updates, has_aux=True)
  grads, (new_state_vars, new_loss_fn_state, aux_outputs) = grad_fn(
      trainable_params
  )

  # Update parameters.
  updates, new_opt_state = optimizer_def.update(
      grads, state.opt_state, trainable_params
  )
  new_params = optax.apply_updates(trainable_params, updates)

  return (
      aux_outputs,
      new_params,
      new_state_vars,
      InternalTrainerState(
          step=state.step + 1,
          opt_state=new_opt_state,
          loss_fn_state=new_loss_fn_state,
      ),
  )


@pz.pytree_dataclass
class StatefulTrainer(pz.Struct):
  """A trainer object that updates its state in place.

  StatefulTrainer manages its own state as well as the state of the model using
  pz.StateVariable objects.

  Attributes:
    root_rng: Base random number generator; used in combination with the
      training step index to derive per-step random numbers.
    model: The model being optimized. Usually will contain Variables for
      parameters and possibly other state.
    state: The internal state of the trainer, wrapped in a Variable to allow
      in-place updates.
    optimizer_def: An optax optimizer.
    loss_fn: A user-specified loss function.
    step_fn: The function that performs a single step of training. Usually
      constructed for you by the `build()` class method.
  """

  root_rng: PRNGKeyArray
  model: ModelPyTree
  state: pz.StateVariable[InternalTrainerState]
  optimizer_def: optax.GradientTransformation = dataclasses.field(
      metadata={"pytree_node": False}
  )
  loss_fn: LossFunction = dataclasses.field(metadata={"pytree_node": False})
  step_fn: Any = dataclasses.field(metadata={"pytree_node": False})

  @classmethod
  def build(
      cls,
      root_rng: PRNGKeyArray,
      model: ModelPyTree,
      optimizer_def: optax.GradientTransformation,
      loss_fn: LossFunction,
      initial_loss_fn_state: LossStatePyTree = None,
      jit: bool = True,
      donate_states: bool = False,
  ) -> StatefulTrainer:
    _, params = pz.unbind_params(model)
    initial_opt_state = optimizer_def.init(pz.freeze_params(params))
    if jit:
      if donate_states:
        step_fn = jax.jit(
            _stateful_trainer_step,
            donate_argnames=["state", "trainable_params", "state_vars"],
            static_argnames=["loss_fn", "optimizer_def"],
        )
      else:
        step_fn = jax.jit(
            _stateful_trainer_step, static_argnames=["loss_fn", "optimizer_def"]
        )
    else:
      step_fn = _stateful_trainer_step
    return cls(
        root_rng=root_rng,
        model=model,
        state=pz.StateVariable(
            InternalTrainerState(
                step=0,
                opt_state=initial_opt_state,
                loss_fn_state=initial_loss_fn_state,
            ),
            label="StatefulTrainer.state",
        ),
        optimizer_def=optimizer_def,
        loss_fn=loss_fn,
        step_fn=step_fn,
    )

  def step(self, **kwargs) -> AuxOutPyTree:
    """Runs one step of training."""
    stateless_model, variables = pz.unbind_variables(self.model)
    trainable_params = []
    frozen_params = []
    state_vars = []
    for var in variables:
      if isinstance(var, pz.Parameter):
        if var.metadata.get("trainable", True):
          trainable_params.append(var)
        else:
          frozen_params.append(var)
      elif isinstance(var, pz.StateVariable):
        state_vars.append(var)
      else:
        raise ValueError(f"Unexpected variable type: {var}")

    aux_out, new_params, new_state_vars, new_internal_state = self.step_fn(
        root_rng=self.root_rng,
        state=self.state.value,
        stateless_model=stateless_model,
        trainable_params=pz.freeze_variables(tuple(trainable_params)),
        frozen_params=pz.freeze_variables(tuple(frozen_params)),
        state_vars=pz.freeze_variables(tuple(state_vars)),
        loss_fn=self.loss_fn,
        optimizer_def=self.optimizer_def,
        kwargs=kwargs,
    )

    for var, new_var in zip(trainable_params, new_params):
      var.update(new_var)
    for var, new_var in zip(state_vars, new_state_vars):
      var.update(new_var)
    self.state.value = new_internal_state

    return aux_out
