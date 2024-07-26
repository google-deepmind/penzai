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

"""Treescope handlers for pz.Struct."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from penzai.core import struct
from treescope import dataclass_util
from treescope import formatting_util
from treescope import renderers
from treescope import rendering_parts


def render_struct_constructor(
    node: struct.Struct,
) -> rendering_parts.RenderableTreePart:
  """Renders the constructor of a Struct, with an open parenthesis."""
  if dataclass_util.init_takes_fields(type(node)):
    return rendering_parts.siblings(
        rendering_parts.maybe_qualified_type_name(type(node)), "("
    )
  else:
    return rendering_parts.siblings(
        rendering_parts.maybe_qualified_type_name(type(node)),
        rendering_parts.roundtrip_condition(
            roundtrip=rendering_parts.text(".from_attributes")
        ),
        "(",
    )


def render_short_struct_summary(
    the_struct: struct.Struct,
) -> rendering_parts.RenderableTreePart:
  """Renders a short summary of a struct.

  Can be used by other handlers that manipulate structs.

  Args:
    the_struct: Struct to render.

  Returns:
    A short, single-line summary of the struct.
  """
  background_color, background_pattern = (
      formatting_util.parse_simple_color_and_pattern_spec(
          the_struct.treescope_color(), type(the_struct).__name__
      )
  )
  return rendering_parts.build_one_line_tree_node(
      rendering_parts.text(type(the_struct).__name__ + "(...)"),
      background_color=background_color,
      background_pattern=background_pattern,
  ).renderable


def struct_attr_style_fn_for_fields(
    fields,
) -> Callable[[str], rendering_parts.RenderableTreePart]:
  """Builds a function to render attributes of a struct.

  The resulting function will render pytree node fields in a different style.

  Args:
    fields: The fields of the struct.

  Returns:
    A function that takes a field name and returns a RenderableTreePart for the
    attribute rendering.
  """
  fields_by_name = {field.name: field for field in fields}

  def attr_style_fn(field_name):
    if struct.is_pytree_node_field(fields_by_name[field_name]):
      return rendering_parts.custom_style(
          rendering_parts.text(field_name),
          css_style="font-style: italic; color: #00225f;",
      )
    else:
      return rendering_parts.text(field_name)

  return attr_style_fn


def handle_structs(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a penzai struct or layer.

  Differences in handling structs vs other dataclasses:

  * We use ``cls.from_attributes`` instead of
    ``penzai.util.dataclasses.dataclass_from_attributes(cls)`` for dataclasses
    with custom init logic.

  * Colors are retrieved from the method ``treescope_color`` instead of
    ``__treescope_color__``.

  * Dataclass fields that are pytree children are rendered in a different style.

  Args:
    node: The node to render.
    path: Optionally, a path to this node.
    subtree_renderer: A recursive renderer for subtrees of this node.

  Returns:
    A rendering, or NotImplemented.
  """
  if not isinstance(node, struct.Struct):
    return NotImplemented

  assert dataclasses.is_dataclass(node), "Every struct.Struct is a dataclass"
  constructor_open = render_struct_constructor(node)
  fields = dataclasses.fields(node)

  background_color, background_pattern = (
      formatting_util.parse_simple_color_and_pattern_spec(
          node.treescope_color(), type(node).__name__
      )
  )

  children = rendering_parts.build_field_children(
      node,
      path,
      subtree_renderer,
      fields_or_attribute_names=fields,
      attr_style_fn=struct_attr_style_fn_for_fields(fields),
  )

  return rendering_parts.build_foldable_tree_node_from_children(
      prefix=constructor_open,
      children=children,
      suffix=")",
      path=path,
      background_color=background_color,
      background_pattern=background_pattern,
  )
