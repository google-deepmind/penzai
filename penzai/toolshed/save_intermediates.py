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

"""Utilities to capture and hold intermediate values as a model runs.

This module provides a layer `SaveIntermediate` that can be inserted into a
model to capture the intermediate value passing through it when it runs. The
value is stored as an attribute on the object and can be accessed later.

It also provides a utility `saving_all_intermediates` that copies a model and
inserts new `SaveIntermediate` layers into it at every point.

Note that storing all of the intermediate activations for a large model may
use a large amount of memory. This utility is intended primarily for debugging
and analyzing small models and small parts of larger models. If you would like
to just inspect the *shapes* of the activations, you can also use the
`SaveIntermediateShape` layer instead.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from penzai import pz

# Disable false-positive pylint warning.
# pylint: disable=assigning-non-slot


@pz.pytree_dataclass
class SaveIntermediate(pz.nn.Layer):
  """A layer that captures an intermediate value as it passes through."""

  saved: pz.StateVariable[Any | None] = dataclasses.field(
      default_factory=lambda: pz.StateVariable(None)
  )

  def __call__(self, value: Any, /, **_unused_side_inputs) -> Any:
    self.saved.value = value
    return value


@pz.pytree_dataclass
class SaveIntermediateShape(pz.nn.Layer):
  """A layer that captures the shape of intermediate values passing through."""

  saved: pz.StateVariable[Any | None] = dataclasses.field(
      default_factory=lambda: pz.StateVariable(None)
  )

  def __call__(self, value: Any, /, **_unused_side_inputs) -> Any:
    # We use `as_array_structure` here so that the value we store is a PyTree
    # with no leaves; this ensures that it is plumbed out correctly as metadata.
    self.saved.value = pz.chk.as_array_structure(value)
    return value


def saving_all_intermediates(
    model: pz.nn.Layer,
    intermediates_group: str | None = "intermediates",
    shapes_only: bool = False,
) -> pz.nn.Layer:
  """Returns a copy of `model` with `SaveIntermediate` layers inserted.

  Args:
    model: The model to copy.
    intermediates_group: The name of the group to use for the intermediate value
      Variable labels. If None, the intermediates will be given unique labels.
    shapes_only: If True, only store the shapes of the intermediate values, not
      the values themselves.

  Returns:
    A copy of the layer or model with empty `SaveIntermediate` layers added
    before and after every sublayer in the model.
  """
  if shapes_only:
    inserted_cls = SaveIntermediateShape
  else:
    inserted_cls = SaveIntermediate

  # Recursively traverse the model to identify all of the places where
  # we might need to add shape annotations, and add a inserted_cls
  # layer there. We try to only save each intermediate value once.
  def add_intercept_values_inside(cur_layer: pz.nn.Layer) -> pz.nn.Layer:
    """Inserts inserted_cls layers around sublayers of a layer."""
    if isinstance(
        cur_layer, pz.nn.Sequential | pz.nn.NamedGroup | pz.nn.CheckedSequential
    ):
      sublayers = cur_layer.sublayers
      new_sublayers = []
      for i, sublayer in enumerate(sublayers):
        new_sublayers.append(add_intercept_values_inside(sublayer))
        if not isinstance(sublayer, pz.nn.Identity) and i != len(sublayers) - 1:
          new_sublayers.append(inserted_cls())
      return dataclasses.replace(cur_layer, sublayers=new_sublayers)
    else:
      return (
          pz.select(cur_layer)
          .at_children()
          .at_instances_of(pz.nn.Layer)
          .apply(add_intercept_values_around)
      )

  def add_intercept_values_around(cur_layer: pz.nn.Layer) -> pz.nn.Layer:
    """Inserts inserted_cls layers around a layer and inside sublayers."""
    if isinstance(
        cur_layer, pz.nn.Sequential | pz.nn.NamedGroup | pz.nn.CheckedSequential
    ):
      sublayers = cur_layer.sublayers
      new_sublayers = [inserted_cls()]
      for sublayer in sublayers:
        new_sublayers.append(add_intercept_values_inside(sublayer))
        if not isinstance(sublayer, pz.nn.Identity):
          new_sublayers.append(inserted_cls())
      return dataclasses.replace(cur_layer, sublayers=new_sublayers)
    elif isinstance(cur_layer, pz.nn.Identity):
      # Add intermediates before the identity but not after (since it won't
      # change them).
      return pz.nn.Sequential([
          inserted_cls(),
          add_intercept_values_inside(cur_layer),
      ])
    else:
      # Make a new sequential so we can add intermediates around it.
      return pz.nn.Sequential([
          inserted_cls(),
          add_intercept_values_inside(cur_layer),
          inserted_cls(),
      ])

  if intermediates_group is None:
    return add_intercept_values_around(model)
  else:
    with pz.scoped_auto_state_var_labels(intermediates_group):
      return add_intercept_values_around(model)
