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

"""Utility for capturing and analyzing a small subcomputation in a larger model.

This utility is designed to enable quickly exploring the behavior of small
parts of a large model, and allows reproducing the in-context behavior of the
subpart without actually having to run the full model. It works by capturing
the intermediate activations immediately before and after the selected submodel,
and also capturing any slots that have been bound inside it. This makes it
possible to call the submodel in isolation on the original intermediate
activation before it, and compare its output to the original intermediate
activation after it.
"""

from __future__ import annotations

from typing import Any

from penzai import pz


@pz.pytree_dataclass
class IsolatedSubmodel(pz.Struct):
  """An isolated part of a model, with saved inputs, outputs, and variables.

  Variable values will also be frozen at the state they had when the model
  was called, allowing deterministic re-execution of the submodel. To re-play
  the submodel, you can run ::

    result, final_var_values = isolated.submodel.stateless_call(
        isolated.initial_var_values,
        isolated.saved_arg,
        **isolated.saved_side_inputs
    )

  Attributes:
    submodel: An individual layer from the larger model. This will match the
      layer that was originally selected, with a few modifications: parameters
      will be frozen, and any variables will be unbound and replaced with
      variable slots.
    saved_arg: Positional argument that was passed to the submodel when we
      isolated it. If the original argument contained variables (although this
      is rare), this will contain variable slots.
    saved_side_inputs: Input that was passed to the submodel when we isolated
      it. If the original argument contained variables (e.g. for random number
      generators), this will contain variable slots.
    saved_output: Saved output that the submodel should produce when called with
      ``saved_arg``.
    initial_var_values: Saved variable values at the point when the submodel was
      called, not including parameters (which are assumed immutable).
    final_var_values: Saved variable values at the point after the submodel was
      called, ot including parameters (which are assumed immutable).
  """

  submodel: pz.nn.Layer
  saved_arg: Any
  saved_side_inputs: dict[str, Any]
  saved_output: Any
  initial_var_values: tuple[pz.StateVariableValue, ...] | None
  final_var_values: tuple[pz.StateVariableValue, ...] | None


@pz.pytree_dataclass
class IsolationCapturer(pz.nn.Layer):
  """Helper object that captures information necessary to isolate the submodel."""

  saved_calls: pz.StateVariable[list[IsolatedSubmodel]]
  wrapped: pz.nn.Layer

  def __call__(self, argument, /, **side_inputs):
    # Freeze parameters.
    wrapped_with_constant_params = pz.freeze_params(self.wrapped)

    # Extract the vars. We extract vars from the argument and side inputs as
    # well.
    (stateless_submodel, stateless_arg, stateless_side_inputs), variables = (
        pz.unbind_variables(
            (wrapped_with_constant_params, argument, side_inputs)
        )
    )
    for var in variables:
      if not isinstance(var, pz.StateVariable):
        raise ValueError(
            "Expected all variables to be either Parameters or StateVariables,"
            f" but got {var}"
        )

    # Capture var values and outputs.
    initial_var_values = pz.freeze_variables(variables)
    result = wrapped_with_constant_params(argument, **side_inputs)
    final_var_values = pz.freeze_variables(variables)

    # Save the result.
    self.saved_calls.value = self.saved_calls.value + [
        IsolatedSubmodel(
            submodel=stateless_submodel,
            saved_arg=stateless_arg,
            saved_side_inputs=stateless_side_inputs,
            saved_output=result,
            initial_var_values=initial_var_values,
            final_var_values=final_var_values,
        )
    ]

    return result


def call_and_extract_submodel(
    submodel_selection: pz.Selection[pz.nn.Layer],
    argument: Any,
    /,
    **side_inputs,
) -> IsolatedSubmodel | list[IsolatedSubmodel]:
  """Calls a model with an argument, and captures the selected submodel.

  This function is designed to enable quickly exploring the behavior of small
  parts of a larger model, and allows reproducing the in-context behavior of the
  subpart without actually having to run the full model.

  Args:
    submodel_selection: A selection of a single `pz.nn.Layer` within a larger
      model (also a `pz.nn.Layer`). Must contain exactly one selected subtree,
      and the selected subtree must be a layer.
    argument: Argument to call the full model with.
    **side_inputs: Side inputs for the model.

  Returns:
    An isolated view of the selected submodel, with the inputs, outputs, and
    states captured so that the submodel can be analyzed without invoking the
    larger model. If the submodel was called multiple times, each call will be
    captured separately and returned as a list. Note that all parameters will
    also be frozen in the result.
  """
  # Check the selection.
  if submodel_selection.count() != 1:
    raise ValueError("submodel_selection must contain exactly one sublayer.")
  selected_layer = submodel_selection.get()
  if not isinstance(selected_layer, pz.nn.Layer):
    raise ValueError("submodel_selection must select a layer.")
  if not isinstance(submodel_selection.deselect(), pz.nn.Layer):
    raise ValueError(
        "The tree that submodule_selection was built from must be a layer."
    )

  capturer = IsolationCapturer(
      wrapped=submodel_selection.get(), saved_calls=pz.StateVariable([])
  )
  model_with_capturer = submodel_selection.set(capturer)

  # Call it to extract the necessary parts.
  _ = model_with_capturer(argument, **side_inputs)

  # Return the result.
  if len(capturer.saved_calls.value) == 1:
    return capturer.saved_calls.value[0]
  else:
    return capturer.saved_calls.value
