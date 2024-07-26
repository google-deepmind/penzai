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

"""Handler for penzai Layers."""

from __future__ import annotations

import dataclasses

from penzai.core._treescope_handlers import struct_handler
from penzai.nn import grouping
from penzai.nn import layer
from treescope import formatting_util
from treescope import renderers
from treescope import rendering_parts


def handle_layer(
    node: layer.Layer,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
):
  """Renders a penzai layer.

  Layers render like Structs in general, except that they render with input and
  output structure annotations.

  Args:
    node: The node to render.
    path: Optionally, a path to this node.
    subtree_renderer: A recursive renderer for subtrees of this node.

  Returns:
    A rendering of the layer.
  """

  assert dataclasses.is_dataclass(node), "Every Layer is a dataclass"
  constructor_open = struct_handler.render_struct_constructor(node)
  fields = dataclasses.fields(node)

  children = rendering_parts.build_field_children(
      node,
      path,
      subtree_renderer,
      fields_or_attribute_names=fields,
      attr_style_fn=struct_handler.struct_attr_style_fn_for_fields(fields),
  )

  background_color, background_pattern = (
      formatting_util.parse_simple_color_and_pattern_spec(
          node.treescope_color(), type(node).__name__
      )
  )

  # pylint: disable=unidiomatic-typecheck
  if (
      isinstance(node, grouping.Sequential)
      and type(node) is not grouping.Sequential
  ):
    first_line_annotation = rendering_parts.comment_color(
        rendering_parts.text(" # Sequential")
    )
  elif (
      isinstance(node, grouping.CheckedSequential)
      and type(node) is not grouping.CheckedSequential
  ):
    first_line_annotation = rendering_parts.comment_color(
        rendering_parts.text(" # CheckedSequential")
    )
  else:
    first_line_annotation = None
  # pylint: enable=unidiomatic-typecheck

  return rendering_parts.build_foldable_tree_node_from_children(
      prefix=constructor_open,
      children=children,
      suffix=")",
      path=path,
      background_color=background_color,
      background_pattern=background_pattern,
      first_line_annotation=first_line_annotation,
  )
