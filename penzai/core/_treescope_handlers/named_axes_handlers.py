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

"""Handle named arrays."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import numpy as np
from penzai.core import named_axes
from penzai.core._treescope_handlers import struct_handler
from treescope import dtype_util
from treescope import lowering
from treescope import ndarray_adapters
from treescope import renderers
from treescope import rendering_parts
from treescope.external import jax_support


def named_array_and_contained_type_summary(
    named_array: named_axes.NamedArray | named_axes.NamedArrayView,
    inspect_device_data: bool = True,
) -> tuple[str, str]:
  """Summarizes a (validly constructed) named array."""
  if isinstance(named_array.data_array, np.ndarray):
    contained_type = "numpy.ndarray"
  elif isinstance(named_array.data_array, jax.Array) and not isinstance(
      named_array.data_array, jax.core.Tracer
  ):
    contained_type = "jax.Array"
  else:
    contained_type = type(named_array.data_array).__name__

  if (
      isinstance(named_array, named_axes.NamedArray)
      and inspect_device_data
      and not named_array.named_shape
  ):
    # Try to do a one-line repr, as long as we can unwrap for free.
    node_repr = repr(named_array.unwrap())
    if "\n" not in node_repr:
      return node_repr, contained_type

  # Give a short summary for our named arrays.
  summary_parts = []
  summary_parts.append(dtype_util.get_dtype_name(named_array.dtype))
  summary_parts.append("(")
  for i, size in enumerate(named_array.positional_shape):
    if i:
      summary_parts.append(", ")
    summary_parts.append(f"{size}")

  if named_array.positional_shape:
    summary_parts.append(" |")
  elif named_array.named_shape:
    summary_parts.append("|")

  for i, (name, size) in enumerate(named_array.named_shape.items()):
    if i:
      summary_parts.append(", ")
    else:
      summary_parts.append(" ")
    summary_parts.append(f"{name}:{size}")
  summary_parts.append(")")

  if (
      inspect_device_data
      and isinstance(named_array.data_array, jax.Array)
      and jax_support.safe_to_summarize(named_array.data_array)
  ):
    summary_parts.append(
        jax_support.summarize_array_data(named_array.data_array)
    )

  return "".join(summary_parts), contained_type


def handle_named_arrays(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders NamedArrays."""

  if isinstance(node, named_axes.NamedArray | named_axes.NamedArrayView):
    try:
      node.check_valid()
    except ValueError:
      # Not a valid NamedArray! Don't try to do fancy summarization.
      return NotImplemented

    def _make_label(inspect_device_data):
      summary, contained_type = named_array_and_contained_type_summary(
          node, inspect_device_data=inspect_device_data
      )
      return rendering_parts.summarizable_condition(
          summary=rendering_parts.abbreviation_color(
              rendering_parts.text(
                  f"<{type(node).__name__} {summary} (wrapping"
                  f" {contained_type})>"
              )
          ),
          detail=rendering_parts.siblings(
              rendering_parts.maybe_qualified_type_name(type(node)),
              "(",
              rendering_parts.fold_condition(
                  expanded=rendering_parts.comment_color(
                      rendering_parts.text("  # " + summary)
                  )
              ),
          ),
      )

    fields = dataclasses.fields(node)
    children = rendering_parts.build_field_children(
        node,
        path,
        subtree_renderer,
        fields_or_attribute_names=fields,
        attr_style_fn=struct_handler.struct_attr_style_fn_for_fields(fields),
    )

    indented_children = rendering_parts.indented_children(children)

    return rendering_parts.build_custom_foldable_tree_node(
        label=lowering.maybe_defer_rendering(
            main_thunk=lambda _: _make_label(inspect_device_data=True),
            placeholder_thunk=lambda: _make_label(inspect_device_data=False),
        ),
        contents=rendering_parts.summarizable_condition(
            detail=rendering_parts.siblings(indented_children, ")")
        ),
        path=path,
        expand_state=rendering_parts.ExpandState.COLLAPSED,
    )

  return NotImplemented


class NamedArrayAdapter(
    ndarray_adapters.NDArrayAdapter[named_axes.NamedArrayBase]
):
  """Array adapter for Penzai named arrays."""

  def get_axis_info_for_array_data(
      self, array: named_axes.NamedArrayBase
  ) -> tuple[ndarray_adapters.AxisInfo, ...]:
    array = array.as_namedarrayview()
    infos = {}
    for name, axis in array.data_axis_for_name.items():
      infos[axis] = ndarray_adapters.NamedPositionlessAxisInfo(
          axis_name=name,
          size=array.data_shape[axis],
      )
    for logical_axis, axis in enumerate(array.data_axis_for_logical_axis):
      infos[axis] = ndarray_adapters.PositionalAxisInfo(
          axis_logical_index=logical_axis,
          size=array.data_shape[axis],
      )
    return tuple(infos[i] for i in range(len(infos)))

  def get_array_data_with_truncation(
      self,
      array: named_axes.NamedArrayBase,
      mask: named_axes.NamedArrayBase | jax.Array | np.ndarray | None,
      edge_items_per_axis: tuple[int | None, ...],
  ) -> tuple[named_axes.NamedArrayBase, named_axes.NamedArrayBase]:
    array = array.as_namedarrayview()
    if mask is None:
      mask_data = None
    else:
      # Make sure mask is compatible.
      if not isinstance(mask, named_axes.NamedArrayBase):
        mask = named_axes.wrap(mask)
      bad_names = set(mask.named_shape.keys()) - set(array.named_shape.keys())
      if bad_names:
        raise ValueError(
            "Valid mask must be broadcastable to the shape of `array`, but it"
            f" had extra axis names {bad_names}"
        )

      vshape = mask.positional_shape
      ashape = array.positional_shape
      if np.broadcast_shapes(vshape, ashape) != ashape:
        raise ValueError(
            "Valid mask must be broadcastable to the shape of `array`, but its"
            f" positional shape {vshape} does not match (a suffix of) the"
            f" positional shape {ashape} of `array`"
        )

      # Insert new length-1 axes.
      new_names = set(array.named_shape.keys()) - set(mask.named_shape.keys())
      if new_names:
        mask = mask[{name: None for name in new_names}]
      new_positional_axis_count = len(ashape) - len(vshape)
      if new_positional_axis_count:
        mask = mask[(None,) * new_positional_axis_count + (...,)]

      # Possibly transpose the mask to match the main array, and extract its
      # data.
      mask_data = mask.order_like(array).data_array

    return jax_support.JAXArrayAdapter().get_array_data_with_truncation(
        array=array.data_array,
        mask=mask_data,
        edge_items_per_axis=edge_items_per_axis,
    )

  def get_array_summary(
      self, array: named_axes.NamedArrayBase, fast: bool
  ) -> str:
    summary, contained_type = named_array_and_contained_type_summary(
        array, inspect_device_data=not fast
    )
    return f"{type(array).__name__} {summary} (wrapping {contained_type})"

  def get_numpy_dtype(
      self, array: named_axes.NamedArrayBase
  ) -> np.dtype | None:
    if isinstance(array.dtype, np.dtype):
      return array.dtype
    else:
      return None

  def get_sharding_info_for_array_data(
      self, array: named_axes.NamedArrayBase
  ) -> ndarray_adapters.ShardingInfo | None:
    array = array.as_namedarrayview()
    if not isinstance(array.data_array, jax.Array):
      return None
    return jax_support.JAXArrayAdapter().get_sharding_info_for_array_data(
        array.data_array
    )

  def should_autovisualize(self, array: named_axes.NamedArrayBase) -> bool:
    assert isinstance(array, named_axes.NamedArray | named_axes.NamedArrayView)
    return (
        isinstance(array.data_array, jax.Array)
        and not isinstance(array.data_array, jax.core.Tracer)
        and not array.data_array.is_deleted()
    )
