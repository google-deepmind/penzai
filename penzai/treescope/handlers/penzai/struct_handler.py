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

from penzai.core import dataclass_util
from penzai.core import struct
from penzai.treescope import html_escaping
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import part_interface
from penzai.treescope.handlers import builtin_structure_handler


class PyTreeNodeFieldName(basic_parts.BaseSpanGroup):
  """Style for attributes that are pytree node children."""

  def _span_css_class(self) -> str:
    return "struct_pytree_attr"

  def _span_css_rule(
      self, setup_context: part_interface.HtmlContextForSetup
  ) -> part_interface.CSSStyleRule:
    return part_interface.CSSStyleRule(
        html_escaping.without_repeated_whitespace("""
            .struct_pytree_attr
            {
                font-style: italic;
                color: #00225f;
            }
        """)
    )


def render_struct_constructor(
    node: struct.Struct,
) -> part_interface.RenderableTreePart:
  """Renders the constructor of a Struct, with an open parenthesis."""
  if dataclass_util.init_takes_fields(type(node)):
    return basic_parts.siblings(
        common_structures.maybe_qualified_type_name(type(node)), "("
    )
  else:
    return basic_parts.siblings(
        common_structures.maybe_qualified_type_name(type(node)),
        basic_parts.RoundtripCondition(
            roundtrip=basic_parts.Text(".from_attributes")
        ),
        "(",
    )


def render_short_struct_summary(
    the_struct: struct.Struct,
) -> part_interface.RenderableTreePart:
  """Renders a short summary of a struct.

  Can be used by other handlers that manipulate structs.

  Args:
    the_struct: Struct to render.

  Returns:
    A short, single-line summary of the struct.
  """
  return common_styles.WithBlockColor(
      common_styles.ColoredSingleLineSpanGroup(
          basic_parts.Text(type(the_struct).__name__ + "(...)")
      ),
      color=the_struct.treescope_color(),
  )


def struct_attr_style_fn_for_fields(
    fields,
) -> Callable[[str], part_interface.RenderableTreePart]:
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
      return PyTreeNodeFieldName(basic_parts.Text(field_name))
    else:
      return basic_parts.Text(field_name)

  return attr_style_fn


def handle_structs(
    node: Any,
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a penzai struct or layer.

  Differences in handling structs vs other dataclasses:

  * We use ``cls.from_attributes`` instead of
    ``penzai.util.dataclasses.dataclass_from_attributes(cls)`` for dataclasses
    with custom init logic.

  * Colors are retrieved from the method ``treescope_color`` instead of
    ``__treescope_color__``.

  * Keypaths are inferred using ``key_for_field``.

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
      builtin_structure_handler.parse_color_and_pattern(
          node.treescope_color(), type(node).__name__
      )
  )

  children = builtin_structure_handler.build_field_children(
      node,
      path,
      subtree_renderer,
      fields_or_attribute_names=fields,
      key_path_fn=node.key_for_field,
      attr_style_fn=struct_attr_style_fn_for_fields(fields),
  )

  return common_structures.build_foldable_tree_node_from_children(
      prefix=constructor_open,
      children=children,
      suffix=")",
      path=path,
      background_color=background_color,
      background_pattern=background_pattern,
  )
