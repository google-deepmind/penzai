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

"""Utility to add shape annotations throughout a model.

This module provides a utility that walks through a model and adds shape
annotations at every point in its evaluation. This is useful for debugging
and understanding how values flow through the model.

The resulting model is specialized to a specific input shape. However, it will
behave identically to the original model as long as it is called with the
same input shape.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any

import jax
from penzai.deprecated.v1 import pz


@pz.pytree_dataclass
class _InterceptShapes(pz.Layer):
  """Helper object that allows us to collect the shapes of values."""

  shape_logger: pz.de.SideOutputEffect

  @classmethod
  def build(cls) -> _InterceptShapes:
    return cls(shape_logger=pz.de.SideOutputRequest(tag=_InterceptShapes))

  def __call__(self, value: Any, /) -> Any:
    self.shape_logger.tell(pz.chk.as_array_structure(value))
    return value


@pz.pytree_dataclass
class CalledWithManyStructures(pz.nn.Identity):
  """No-op annotation that indicates values of many different shapes."""

  structures: tuple[Any, ...]


@pz.pytree_dataclass
class Static(pz.Struct):
  """Wraps a value so that it is treated as an empty PyTree."""

  static_value: Any = dataclasses.field(metadata={"pytree_node": False})


def annotate_shapes(root: pz.Layer, dummy_input: Any) -> pz.Layer:
  """Annotates shapes in a model or layer, for inputs with this input structure.

  Args:
    root: The layer or model to annotate.
    dummy_input: A dummy input of the same structure as the actual input. Can
      contain anything with shape/dtype attributes, e.g. `jax.ShapeDtypeStruct`.

  Returns:
    A copy of the layer or model with shape annotations added before and after
    every layer that was called during the evaluation of the model.
  """
  if (
      pz.select(root)
      .at_instances_of(pz.de.SideOutputRequest)
      .where(lambda req: req.tag is _InterceptShapes)
      .count()
  ):
    raise ValueError(
        "Model has SideOutputRequests with the `_InterceptShapes` tag, which is"
        " reserved for this utility and should not be used externally!"
    )

  # Step 1: Recursively traverse the model to identify all of the places where
  # we might need to add shape annotations, and add a temporary
  # _InterceptShapes layer there.
  def add_intercept_shapes_inside(cur_layer: pz.Layer) -> pz.Layer:
    """Inserts `_InterceptShapes` layers around all sublayers of a layer."""
    if isinstance(
        cur_layer, pz.nn.Sequential | pz.nn.NamedGroup | pz.nn.CheckedSequential
    ):
      sublayers = cur_layer.sublayers
      new_sublayers = [_InterceptShapes.build()]
      for sublayer in sublayers:
        new_sublayers.append(add_intercept_shapes_inside(sublayer))
        new_sublayers.append(_InterceptShapes.build())
      return dataclasses.replace(cur_layer, sublayers=new_sublayers)
    else:
      return (
          pz.select(cur_layer)
          .at_children()
          .at_instances_of(pz.Layer)
          .apply(add_intercept_shapes_around)
      )

  def add_intercept_shapes_around(cur_layer: pz.Layer) -> pz.Layer:
    """Inserts `_InterceptShapes` layers around a layer."""
    if isinstance(
        cur_layer, pz.nn.Sequential | pz.nn.NamedGroup | pz.nn.CheckedSequential
    ):
      # Already a sequential, so it's sufficient to add shapes inside it.
      return add_intercept_shapes_inside(cur_layer)
    else:
      # Make a new sequential so we can add shapes around it.
      return pz.nn.Sequential([
          _InterceptShapes.build(),
          add_intercept_shapes_inside(cur_layer),
          _InterceptShapes.build(),
      ])

  root_with_interceptors = add_intercept_shapes_around(root)

  # Step 2: Use effect handling and eval_shape to collect the shape information
  # from the annotation layers.
  handled = pz.de.CollectingSideOutputs.handling(
      root_with_interceptors, tag_predicate=lambda tag: tag is _InterceptShapes
  )

  def go():
    _, out_tells = handled(dummy_input)
    return Static(out_tells)

  out_tells = jax.eval_shape(go).static_value

  # Step 3: Figure out the shape(s) at each keypath.
  shapes_at_keypath = collections.defaultdict(list)
  for keypath, tag, structure in out_tells:
    assert tag is _InterceptShapes
    assert keypath[-1] == jax.tree_util.GetAttrKey("shape_logger")
    shapes_at_keypath[keypath[:-1]].append(structure)

  # Step 4: Add the final shape annotations.
  def make_and_inline_shape_annotation(
      keypath, marker: _InterceptShapes
  ) -> tuple[pz.Layer, ...]:
    del marker
    if keypath in shapes_at_keypath:
      assert shapes_at_keypath[keypath]
      first = shapes_at_keypath[keypath][0]
      if all(struct == first for struct in shapes_at_keypath[keypath][1:]):
        return (pz.nn.CheckStructure(shapes_at_keypath[keypath]),)
      else:
        return (CalledWithManyStructures(tuple(shapes_at_keypath[keypath])),)
    else:
      return ()

  return (
      pz.select(root_with_interceptors)
      .at_instances_of(_InterceptShapes)
      .apply_and_inline(make_and_inline_shape_annotation, with_keypath=True)
  )
