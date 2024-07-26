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
from penzai.deprecated.v1 import pz

ModelPyTree = Any
AuxOutPyTree = Any
LossStatePyTree = Any
OptimizerStatePyTree = Any
PRNGKeyArray = jax.Array


@pz.pytree_dataclass
class TrainState(pz.Struct):
  """Collection of state for ease of training.

  The parameters and nonlearnable parts of the model are kept separate
  internally in order to avoid overhead of PyTree traversal and to simplify
  checkpointing the parameters. You can access the full model by accessing the
  ``.model`` property.

  Attributes:
    step: Current step of training.
    root_rng: Base random number generator; used in combination with step to
      derive per-step random numbers.
    params: Values for the parameters of the model being optimized.
    model_without_params: The nonlearnable parts of the model being optimized.
      Should contain `Parameter` instances but without values.
    opt_state: An optimizer state for the learnable parts of `model`.
    loss_fn_state: Arbitrary state managed by the loss function. For instance,
      if your model has local state, you can functionalize it using
      `pz.de.handle_local_states` and store its state dict here.
    optimizer_def: An optax optimizer.
  """

  step: int
  root_rng: PRNGKeyArray
  params: dict[str, Any]
  model_without_params: ModelPyTree
  opt_state: OptimizerStatePyTree
  loss_fn_state: LossStatePyTree
  optimizer_def: optax.GradientTransformation = dataclasses.field(
      metadata={"pytree_node": False}
  )

  @classmethod
  def initial_state(
      cls,
      model: ModelPyTree,
      optimizer_def: optax.GradientTransformation,
      root_rng: PRNGKeyArray,
      loss_fn_state: LossStatePyTree = None,
  ):
    """Constructs the initial training state.

    Args:
      model: The model being optimized.
      optimizer_def: The optax optimizer to use.
      root_rng: Base random number generator; used in combination with step to
        derive per-step random numbers.
      loss_fn_state: Optional initial state for the loss function.

    Returns:
      An initial training state.
    """
    # Extract the parameters.
    pz.nn.check_no_duplicated_parameters(model)
    param_selection = pz.select(model).at_instances_of(pz.nn.Parameter)
    params = {
        param.name: param.value for param in param_selection.get_sequence()
    }
    model_without_params = param_selection.at(lambda p: p.value).set(
        pz.NotInThisPartition()
    )
    # Derive opt state from parameters.
    opt_state = optimizer_def.init(params)
    return cls(
        step=0,
        root_rng=root_rng,
        params=params,
        model_without_params=model_without_params,
        opt_state=opt_state,
        loss_fn_state=loss_fn_state,
        optimizer_def=optimizer_def,
    )

  @property
  def model(self) -> ModelPyTree:
    """The full model, including parameters and nonlearnable parts."""
    return (
        pz.select(self.model_without_params)
        .at_instances_of(pz.nn.Parameter)
        .apply(lambda p: dataclasses.replace(p, value=self.params[p.name]))
    )


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


class TrainStepFunction(Protocol):
  """Signature for a common training step function after it is built."""

  def __call__(
      self,
      train_state: TrainState,
      **kwargs,
  ) -> tuple[TrainState, AuxOutPyTree]:
    """Signature for a training step.

    Args:
      train_state: The current state.
      **kwargs: Arguments passed to the train step, usually inputs to the model
        or labels.

    Returns:
      A tuple ``(new_train_state, aux_outputs)``.
    """


def compute_training_outputs_and_updates(
    current_params: dict[str, Any],
    model_without_params: ModelPyTree,
    opt_state: OptimizerStatePyTree,
    loss_fn_state: LossStatePyTree,
    root_rng: PRNGKeyArray,
    step: int | jax.Array,
    loss_kwargs: dict[str, Any],
    loss_fn: LossFunction,
    optimizer_def: optax.GradientTransformation,
) -> tuple[ModelPyTree, OptimizerStatePyTree, LossStatePyTree, AuxOutPyTree]:
  """Runs a loss function and computes all of its outputs.

  This function runs the model and loss function and updates the corresponding
  parameters in the optimizer. It splits each component of the input into a
  separate argument to make it easy to JIT-compile, and to allow donating the
  parts that will be updated.

  Args:
    current_params: A dictionary of model parameters. These are the parts that
      WILL be updated by the optimizer.
    model_without_params: A model PyTree that includes ``pz.NotInThisPartition``
      in place of each of the learnable parameter values. These are the parts of
      the model that will NOT be updated by the optimizer.
    opt_state: State for the optimizer.
    loss_fn_state: State for the loss function.
    root_rng: Root random key for the training process.
    step: Current step of training, used to adjust root RNG.
    loss_kwargs: Keyword arguments for the loss function.
    loss_fn: The loss function.
    optimizer_def: The optimizer.

  Returns:
    Tuple of ``(new_params, new_opt_state, new_loss_fn_state, aux_outputs)``
  """
  step_rng = jax.random.fold_in(root_rng, step)

  def compute_loss(
      current_params,
  ) -> tuple[jax.Array, tuple[LossStatePyTree, AuxOutPyTree]]:
    """Computes loss, aux out, and loss state for a single example."""
    # Rebuild the model using the input parameters, which we are
    # differentiating with respect to.
    model = (
        pz.select(model_without_params)
        .at_instances_of(pz.nn.Parameter)
        .apply(lambda p: dataclasses.replace(p, value=current_params[p.name]))
    )
    loss, new_loss_fn_state, aux_outputs = loss_fn(
        model=model, state=loss_fn_state, rng=step_rng, **loss_kwargs
    )
    return loss, (new_loss_fn_state, aux_outputs)

  # Take gradients.
  grad_fn = jax.grad(compute_loss, has_aux=True)
  grads, (new_loss_fn_state, aux_outputs) = grad_fn(current_params)

  # Get updated parameters.
  updates, new_opt_state = optimizer_def.update(
      grads, opt_state, current_params
  )
  new_params = optax.apply_updates(current_params, updates)

  return new_params, new_opt_state, new_loss_fn_state, aux_outputs


def build_train_step_fn(
    loss_fn: LossFunction,
    jit: bool = True,
    donate_params_and_state: bool = False,
    train_state_shardings: TrainState | None = None,
    input_kwarg_shardings: dict[str, Any] | None = None,
    aux_output_shardings: AuxOutPyTree | None = None,
) -> TrainStepFunction:
  """Builds a train step function for a common training loop.

  For simplicity, the output of the train step function is the third output
  of the loss function alone, not including the loss value itself. If you want
  to obtain the loss value, you can return it both as the first output of the
  loss function and also as part of the third output. For more control, consider
  forking this function and modifying the logic.

  If your model has its own local state variables or stochastic layers,
  your loss function is responsible for handling those effects using its
  arguments. For instance, you could transform your model to handle the
  RandomEffect and LocalStateEffect, and pass the non-effectful transformed
  model and initial state dict to `TrainState.initial_state`. Then in your
  ``loss_fn``, you could forward the ``rng`` and ``state`` arguments to your
  non-effectful model as arguments, following the expected argument structure of
  the handlers you used. Since models are PyTrees before and after handling
  effects, you have freedom to resolve them wherever is most convenient.

  Args:
    loss_fn: Loss function taking a model, state, rng, and additional
      keyword-argument inputs, and returning ``(loss_scalar, new_state,
      outputs)``.
    jit: Whether to JIT-compile the train step.
    donate_params_and_state: Whether to donate the old parameters and states
      when JIT-compiling the train step. If True, parameter and state arrays may
      be deleted after each step, meaning that any previous references to them
      (e.g. the version of the model with initial parameters) will break. Parts
      of the model that are not learnable will not be donated.
    train_state_shardings: Optional TrainState with leaves replaced with JAX
      shardings. If provided, the train step will be compiled to shard its
      inputs and outputs according to these shardings. If not provided, allows
      JAX to infer an appropriate sharding. Ignored unless ``jit=True``.
      Shardings for ``step`` and ``root_rng`` are ignored.
    input_kwarg_shardings: Optional mapping from input keyword argument names to
      shardings. If provided, the train step will be compiled to shard its
      user-provided inputs according to these shardings. If not provided, allows
      JAX to infer an appropriate sharding. Ignored unless ``jit=True``.
    aux_output_shardings: Optional auxiliary output PyTree with leaves replaced
      with JAX shardings. If provided, the train step will be compiled to shard
      its aux outputs according to these shardings. If not provided, allows JAX
      to infer an appropriate sharding. Ignored unless ``jit=True``.

  Returns:
    A train step, which updates the model and internal states, and returns a
    new train state and the outputs of the loss function.
  """
  # Build the possibly-jitted updater function.
  if train_state_shardings is None:
    param_shardings = None
    model_without_params_shardings = None
    opt_state_shardings = None
    loss_fn_state_shardings = None
  else:
    param_shardings = train_state_shardings.params
    model_without_params_shardings = train_state_shardings.model_without_params
    opt_state_shardings = train_state_shardings.opt_state
    loss_fn_state_shardings = train_state_shardings.loss_fn_state

  if jit:
    if donate_params_and_state:
      donate_argnames = ("current_params", "opt_state", "loss_fn_state")
    else:
      donate_argnames = None
    compute_updates_fn = jax.jit(
        compute_training_outputs_and_updates,
        static_argnames=("loss_fn", "optimizer_def"),
        donate_argnames=donate_argnames,
        in_shardings=(
            param_shardings,
            model_without_params_shardings,
            opt_state_shardings,
            loss_fn_state_shardings,
            None,
            None,
            input_kwarg_shardings,
        ),
        out_shardings=(
            param_shardings,
            opt_state_shardings,
            loss_fn_state_shardings,
            aux_output_shardings,
        ),
    )
  else:
    if donate_params_and_state:
      raise ValueError("Cannot donate params and state unless jit=True.")
    compute_updates_fn = compute_training_outputs_and_updates

  def _train_step(
      train_state: TrainState[ModelPyTree], **kwargs
  ) -> tuple[TrainState[ModelPyTree], AuxOutPyTree]:

    # Run the step.
    new_params, new_opt_state, new_loss_fn_state, aux_outputs = (
        compute_updates_fn(
            train_state.params,
            train_state.model_without_params,
            train_state.opt_state,
            train_state.loss_fn_state,
            train_state.root_rng,
            train_state.step,
            kwargs,
            loss_fn,
            train_state.optimizer_def,
        )
    )

    # Build our updated train state.
    new_train_state = TrainState(
        step=train_state.step + 1,
        root_rng=train_state.root_rng,
        params=new_params,
        model_without_params=train_state.model_without_params,
        loss_fn_state=new_loss_fn_state,
        opt_state=new_opt_state,
        optimizer_def=train_state.optimizer_def,
    )

    return new_train_state, aux_outputs

  return _train_step
