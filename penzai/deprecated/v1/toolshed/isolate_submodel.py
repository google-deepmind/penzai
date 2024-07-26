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
and also isolating any state or shared parameters used by it. This makes it
possible to call the submodel in isolation on the original intermediate
activation before it, and compare its output to the original intermediate
activation after it.
"""

from __future__ import annotations

from typing import Any

from penzai.deprecated.v1 import pz
from penzai.deprecated.v1.data_effects import random


@pz.pytree_dataclass
class IsolatedSubmodel(pz.Struct):
  """An isolated part of a submodel, with its saved inputs and outputs.

  When initially populated by `isolate_submodel`, any states used by the
  submodel will be functionalized and treated as part of the saved input and
  output, and any shared parameters, side inputs, or random streams will be
  captured and stored in ``submodel`` as well.

  Attributes:
    submodel: An individual layer from the larger model.
    saved_input: Input that was passed to the submodel when we isolated it.
    saved_output: Saved output that the submodel should produce when called with
      ``saved_input``.
  """

  submodel: pz.Layer
  saved_input: Any
  saved_output: Any


@pz.pytree_dataclass
class IsolationCapturer(pz.Layer):
  """Helper object that captures information necessary to isolate the submodel."""

  destination: pz.de.SideOutputEffect
  wrapped: pz.Layer
  side_inputs: dict[Any, pz.de.SideInputEffect]
  states: dict[Any, pz.de.LocalStateEffect]
  random_streams: dict[Any, pz.de.RandomEffect]

  def __call__(self, argument):
    captured = {}
    captured["input"] = argument
    captured["side_inputs"] = {k: v.ask() for k, v in self.side_inputs.items()}
    captured["states_before"] = {k: v.get() for k, v in self.states.items()}
    captured["random_stream_states"] = {}
    for k, random_eff in self.random_streams.items():
      if not isinstance(random_eff, random.RandomEffectImpl):
        raise ValueError(
            "Cannot capture random states when they use an implementation other"
            " than random.RandomEffectImpl!"
        )
      captured["random_stream_states"][k] = (
          random_eff._stream.base_key,
          random_eff._stream.next_offset,
      )
    result = self.wrapped(argument)
    captured["states_after"] = {k: v.get() for k, v in self.states.items()}
    captured["output"] = result
    self.destination.tell(captured)
    return result


def call_and_extract_submodel(
    submodel_selection: pz.Selection[pz.Layer],
    argument: Any,
) -> IsolatedSubmodel | list[IsolatedSubmodel]:
  """Calls a model with an argument, and captures the selected submodel.

  This function is designed to enable quickly exploring the behavior of small
  parts of a large model, and allows reproducing the in-context behavior of the
  subpart without actually having to run the full model.

  Args:
    submodel_selection: A selection of a single `pz.Layer` within a larger model
      (also a `pz.Layer`). Must contain exactly one selected subtree, and the
      selected subtree must be a layer.
    argument: Argument to call the full model with.

  Returns:
    An isolated view of the selected submodel, with the inputs, outputs, states
    and any shared parameters captured so that the submodel can be analyzed
    without invoking the larger model. If the submodel was called multiple
    times, each call will be captured separately and returned as a list.
  """
  # Check the selection.
  if submodel_selection.count() != 1:
    raise ValueError("submodel_selection must contain exactly one sublayer.")
  selected_layer = submodel_selection.get()
  if not isinstance(selected_layer, pz.Layer):
    raise ValueError("submodel_selection must select a layer.")
  if not isinstance(submodel_selection.deselect(), pz.Layer):
    raise ValueError(
        "The tree that submodule_selection was built from must be a layer."
    )

  # Build the isolation capturer.
  def make_capturer(target):
    # Find handler IDs that are inside the target. We don't have to worry about
    # these.
    internal_handler_ids = set()
    current_selection = pz.select(target)
    while current_selection.count():
      handlers = current_selection.at_instances_of(pz.de.EffectHandler)
      internal_handler_ids.update(
          handler.handler_id for handler in handlers.get_sequence()
      )
      current_selection = handlers.at_children()
    # Find any side inputs whose handlers are outside the target and capture
    # them.
    side_input_refs = {
        (ref.handler_id, ref.tag): ref
        for ref in (
            pz.select(target)
            .at_instances_of(pz.de.HandledSideInputRef)
            .where(lambda ref: ref.handler_id not in internal_handler_ids)
            .get_sequence()
        )
    }
    # Find any states whose handlers are outside the target and capture them.
    state_refs = {
        (ref.handler_id, ref.name): ref
        for ref in (
            pz.select(target)
            .at_instances_of(pz.de.HandledLocalStateRef)
            .where(lambda ref: ref.handler_id not in internal_handler_ids)
            .get_sequence()
        )
    }
    # Find any random effects similarly.
    random_refs = {
        ref.handler_id: ref
        for ref in (
            pz.select(target)
            .at_instances_of(pz.de.HandledRandomRef)
            .where(lambda ref: ref.handler_id not in internal_handler_ids)
            .get_sequence()
        )
    }
    # Build the interceptor.
    return IsolationCapturer(
        destination=pz.de.SideOutputRequest(tag=IsolationCapturer),
        wrapped=target,
        side_inputs=side_input_refs,
        states=state_refs,
        random_streams=random_refs,
    )

  model_with_capturer = submodel_selection.apply(make_capturer)

  # Call it to extract the necessary parts.
  collector = pz.de.CollectingSideOutputs.handling(
      model_with_capturer, tag=IsolationCapturer
  )
  _, captured_call_data = collector(argument)

  results = []
  # pylint: disable=cell-var-from-loop
  for sideout in captured_call_data:
    call_data = sideout.value
    transformed_layer = selected_layer
    transformed_input = call_data["input"]
    transformed_output = call_data["output"]

    if call_data["side_inputs"]:
      # Rebind captured side inputs.
      # Bind shared parameters using constant side inputs, and put other side
      # inputs as arguments.
      captured_side_inputs = call_data["side_inputs"]
      inner = (
          pz.select(transformed_layer)
          .at_instances_of(pz.de.HandledSideInputRef)
          .where(lambda ref: (ref.handler_id, ref.tag) in captured_side_inputs)
          .apply(
              lambda ref: pz.de.SideInputRequest(tag=(ref.handler_id, ref.tag))
          )
      )
      shared_param_side_inputs = {}
      other_side_inputs = {}
      for (hid, tag), v in captured_side_inputs.items():
        if isinstance(tag, pz.nn.SharedParamTag):
          shared_param_side_inputs[(hid, tag)] = v
        else:
          other_side_inputs[(hid, tag)] = v
      if shared_param_side_inputs:
        transformed_layer = pz.de.WithConstantSideInputs.handling(
            inner, side_inputs=shared_param_side_inputs
        )
      if other_side_inputs:
        transformed_layer = pz.de.WithSideInputsFromInputTuple.handling(
            inner, captured_side_inputs.keys()
        )
        transformed_input = (transformed_input,) + tuple(
            captured_side_inputs.values()
        )

    # Rebind captured random stream states.
    captured_randoms = call_data["random_stream_states"]
    for random_handler_id, (key, offset) in captured_randoms.items():
      transformed_layer = pz.de.WithFrozenRandomState(
          handler_id=random_handler_id,
          body=transformed_layer,
          random_key=key,
          starting_offset=offset,
      )

    if call_data["states_before"]:
      # Re-handle any local states.
      captured_states = call_data["states_before"]
      new_handler_id = pz.de.infer_or_check_handler_id(
          "WithFunctionalLocalState", transformed_layer
      )
      inner = (
          pz.select(transformed_layer)
          .at_instances_of(pz.de.HandledLocalStateRef)
          .where(lambda ref: (ref.handler_id, ref.name) in captured_states)
          .apply(
              lambda ref: pz.de.HandledLocalStateRef(
                  handler_id=new_handler_id,
                  name=f"{ref.handler_id}:{ref.name}",
                  was_explicitly_named=True,
                  category=ref.category,
              )
          )
      )
      transformed_layer = pz.de.WithFunctionalLocalState(
          handler_id=new_handler_id,
          body=inner,
      )
      transformed_input = (
          transformed_input,
          {
              f"{hid}:{name}": v
              for (hid, name), v in call_data["states_before"].items()
          },
      )
      transformed_output = (
          transformed_output,
          {
              f"{hid}:{name}": v
              for (hid, name), v in call_data["states_after"].items()
          },
      )

    results.append(
        IsolatedSubmodel(
            submodel=transformed_layer,
            saved_input=transformed_input,
            saved_output=transformed_output,
        )
    )

  # pylint: enable=cell-var-from-loop

  # Return the result.
  if len(results) == 1:
    return results[0]
  else:
    return results
