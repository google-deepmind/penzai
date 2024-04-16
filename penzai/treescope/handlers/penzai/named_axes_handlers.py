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
from penzai.treescope import ndarray_summarization
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import part_interface
from penzai.treescope.handlers import builtin_structure_handler
from penzai.treescope.handlers.penzai import struct_handler


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
  summary_parts.append(ndarray_summarization.get_dtype_name(named_array.dtype))
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

  if inspect_device_data and ndarray_summarization.safe_to_summarize(
      named_array.data_array
  ):
    summary_parts.append(
        ndarray_summarization.summarize_ndarray(
            named_array.data_array, include_shape_and_dtype=False
        )
    )

  return "".join(summary_parts), contained_type


def handle_named_arrays(
    node: Any,
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
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
      return basic_parts.SummarizableCondition(
          summary=common_styles.AbbreviationColor(
              basic_parts.Text(
                  f"<{type(node).__name__} {summary} (wrapping"
                  f" {contained_type})>"
              )
          ),
          detail=basic_parts.siblings(
              common_structures.maybe_qualified_type_name(type(node)),
              "(",
              basic_parts.FoldCondition(
                  expanded=common_styles.CommentColor(
                      basic_parts.Text("  # " + summary)
                  )
              ),
          ),
      )

    fields = dataclasses.fields(node)
    children = builtin_structure_handler.build_field_children(
        node,
        path,
        subtree_renderer,
        fields_or_attribute_names=fields,
        key_path_fn=node.key_for_field,
        attr_style_fn=struct_handler.struct_attr_style_fn_for_fields(fields),
    )

    indented_children = basic_parts.IndentedChildren.build(children)

    return common_structures.build_custom_foldable_tree_node(
        label=foldable_impl.maybe_defer_rendering(
            main_thunk=lambda _: _make_label(inspect_device_data=True),
            placeholder_thunk=lambda: _make_label(inspect_device_data=False),
        ),
        contents=basic_parts.SummarizableCondition(
            detail=basic_parts.siblings(
                indented_children,
                ")",
            )
        ),
        path=path,
        expand_state=part_interface.ExpandState.COLLAPSED,
    )

  return NotImplemented
