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

"""Utility to insert intermediate values into a model's PyTree.

This module provides a utility that runs a model, collects *all* intermediate
values, and then inserts those intermediate values into the model PyTree itself
for ease of visualization and analysis.

The inserted intermediates are represented as identity layers that have an
extra array inside them. These identity layers don't change their input in any
way, so the resulting model has the same behavior as the original model.

Note that storing all of the intermediate activations for a large model may
use a large amount of memory. This utility is intended primarily for debugging
and analyzing small models and small parts of larger models.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any

import jax
from penzai.deprecated.v1 import pz


@pz.pytree_dataclass
class _InterceptValues(pz.Layer):
  """Helper object that allows us to collect intermediate values."""

  shape_logger: pz.de.SideOutputEffect

  @classmethod
  def build(cls) -> _InterceptValues:
    return cls(shape_logger=pz.de.SideOutputRequest(tag=_InterceptValues))

  def __call__(self, value: Any, /) -> Any:
    self.shape_logger.tell(value)
    return value


@pz.pytree_dataclass
class IdentityWithSavedActivations(pz.nn.Identity):
  """No-op annotation that holds onto intermediate activations."""

  saved_activations: list[Any]


def run_and_interleave_intermediates(
    root: pz.Layer, argument: Any, jit: bool = True
) -> pz.Layer:
  """Interleaves intermediate values into a model.

  Args:
    root: The layer or model to annotate.
    argument: An argument to run the model with.
    jit: Whether to run the model under `jax.jit`.

  Returns:
    A copy of the layer or model with saved intermediates added before and after
    every layer that was called during the evaluation of the model. Saved
    intermediates are represented as `IdentityWithSavedActivations` layers,
    which hold onto saved activations but do not use them.
  """
  if (
      pz.select(root)
      .at_instances_of(pz.de.SideOutputRequest)
      .where(lambda req: req.tag is _InterceptValues)
      .count()
  ):
    raise ValueError(
        "Model has SideOutputRequests with the _InterceptValues tag, which is"
        " reserved for this utility and should not be used externally!"
    )

  # Step 1: Recursively traverse the model to identify all of the places where
  # we might need to add shape annotations, and add a temporary
  # _InterceptValues layer there. We try to only save each intermediate
  # value once.
  def add_intercept_values_inside(cur_layer: pz.Layer) -> pz.Layer:
    """Inserts _InterceptValues layers around sublayers of a layer."""
    if isinstance(
        cur_layer, pz.nn.Sequential | pz.nn.NamedGroup | pz.nn.CheckedSequential
    ):
      sublayers = cur_layer.sublayers
      new_sublayers = []
      for sublayer in sublayers:
        new_sublayers.append(add_intercept_values_inside(sublayer))
        if not isinstance(sublayer, pz.nn.Identity):
          new_sublayers.append(_InterceptValues.build())
      if new_sublayers and isinstance(new_sublayers[-1], _InterceptValues):
        new_sublayers.pop()
      return dataclasses.replace(cur_layer, sublayers=new_sublayers)
    else:
      return (
          pz.select(cur_layer)
          .at_children()
          .at_instances_of(pz.Layer)
          .apply(add_intercept_values_around)
      )

  def add_intercept_values_around(cur_layer: pz.Layer) -> pz.Layer:
    """Inserts _InterceptValues layers around a layer."""
    if isinstance(
        cur_layer, pz.nn.Sequential | pz.nn.NamedGroup | pz.nn.CheckedSequential
    ):
      sublayers = cur_layer.sublayers
      new_sublayers = [_InterceptValues.build()]
      for sublayer in sublayers:
        new_sublayers.append(add_intercept_values_inside(sublayer))
        if not isinstance(sublayer, pz.nn.Identity):
          new_sublayers.append(_InterceptValues.build())
      return dataclasses.replace(cur_layer, sublayers=new_sublayers)
    elif isinstance(cur_layer, pz.nn.Identity):
      # Add intermediates before the identity but not after (since it won't
      # change them).
      return pz.nn.Sequential([
          _InterceptValues.build(),
          add_intercept_values_inside(cur_layer),
      ])
    else:
      # Make a new sequential so we can add intermediates around it.
      return pz.nn.Sequential([
          _InterceptValues.build(),
          add_intercept_values_inside(cur_layer),
          _InterceptValues.build(),
      ])

  root_with_interceptors = add_intercept_values_around(root)

  # Step 2: Use effect handling to collect the intermediate values
  # from the annotation layers.
  handled = pz.de.CollectingSideOutputs.handling(
      root_with_interceptors, tag=_InterceptValues
  )

  def go(handled, arg):
    _, out_tells = handled(arg)
    values_at_keypaths = collections.defaultdict(list)
    for tell in out_tells:
      assert tell.tag is _InterceptValues
      assert tell.keypath[-1] == jax.tree_util.GetAttrKey("shape_logger")
      values_at_keypaths[tell.keypath[:-1]].append(tell.value)
    return collections.OrderedDict(values_at_keypaths)

  if jit:
    values_at_keypaths = jax.jit(go)(handled, argument)
  else:
    values_at_keypaths = go(handled, argument)

  # Step 5: Add the final intermediate values
  def make_and_inline_shape_annotation(
      keypath, marker: _InterceptValues
  ) -> tuple[pz.Layer, ...]:
    del marker
    if keypath in values_at_keypaths:
      assert values_at_keypaths[keypath]
      return (IdentityWithSavedActivations(list(values_at_keypaths[keypath])),)
    else:
      return ()

  return (
      pz.select(root_with_interceptors)
      .at_instances_of(_InterceptValues)
      .apply_and_inline(make_and_inline_shape_annotation, with_keypath=True)
  )
