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

import contextlib
import dataclasses
from typing import Any

from penzai.core import shapecheck
from penzai.core._treescope_handlers import struct_handler
from penzai.deprecated.v1.core import layer
from penzai.deprecated.v1.data_effects import effect_base
from penzai.deprecated.v1.nn import grouping
from treescope import context
from treescope import formatting_util
from treescope import handlers
from treescope import layout_algorithms
from treescope import renderers
from treescope import rendering_parts

_already_seen_layer: context.ContextualValue[bool] = context.ContextualValue(
    module=__name__, qualname="_already_seen_layer", initial_value=False
)
"""Tracks whether we have already encountered a layer.

This is used to enable rendering effects only for the outermost layer object.
"""


def handle_layers(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
    obvious_input_output_structure_types: tuple[type[Any], ...] = (
        grouping.CheckStructure,
        grouping.CheckedSequential,
    ),
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a penzai  layer.

  Layers render like Structs in general. However:

  * The outermost layer is augmented with information about the set of free
    effects in the layer.

  * The outermost layer with a nontrivial input/output structure shows that
    structure.

  Args:
    node: The node to render.
    path: Optionally, a path to this node.
    subtree_renderer: A recursive renderer for subtrees of this node.
    obvious_input_output_structure_types: Types where the input and output
      structure are obvious from the type's ordinary pretty-printed output.

  Returns:
    A rendering, or NotImplemented.
  """
  if not isinstance(node, layer.Layer):
    return NotImplemented

  assert dataclasses.is_dataclass(node), "Every Layer is a dataclass"
  constructor_open = struct_handler.render_struct_constructor(node)
  fields = dataclasses.fields(node)

  extra_annotations = []
  with contextlib.ExitStack() as context_stack:
    try:
      input_structure = node.input_structure()
      output_structure = node.output_structure()
      exc_message = None
    except Exception as exc:  # pylint: disable=broad-exception-caught
      input_structure = None
      output_structure = None
      exc_message = str(exc)
    if (
        not isinstance(input_structure, shapecheck.Wildcard)
        or not isinstance(output_structure, shapecheck.Wildcard)
        or exc_message is not None
    ):
      if exc_message is not None:
        structure_annotation = rendering_parts.fold_condition(
            expanded=rendering_parts.floating_annotation_with_separate_focus(
                rendering_parts.in_outlined_box(
                    rendering_parts.error_color(
                        rendering_parts.text(
                            "Error while inferring input/output structure:"
                            f" {exc_message}"
                        )
                    )
                )
            )
        )
        extra_annotations.append(structure_annotation)
      elif not isinstance(node, obvious_input_output_structure_types):
        # Add input and output type annotations.
        # Don't worry about shared values while rendering these, since they
        # don't actually appear in the tree.
        with handlers.setup_shared_value_context():
          structure_annotation = rendering_parts.fold_condition(
              expanded=rendering_parts.floating_annotation_with_separate_focus(
                  rendering_parts.in_outlined_box(
                      rendering_parts.comment_color(
                          rendering_parts.on_separate_lines([
                              rendering_parts.siblings_with_annotations(
                                  "# Input: ",
                                  subtree_renderer(input_structure),
                              ),
                              rendering_parts.siblings_with_annotations(
                                  "# Output: ",
                                  subtree_renderer(output_structure),
                              ),
                          ])
                      )
                  )
              )
          )
        layout_algorithms.expand_to_depth(structure_annotation, 0)
        extra_annotations.append(structure_annotation)

    if not _already_seen_layer.get():
      context_stack.enter_context(_already_seen_layer.set_scoped(True))
      # Add effect annotations.
      free_effects = effect_base.free_effect_types(node)
      if free_effects:
        effect_type_blobs = []
        for effect_protocol in free_effects:
          effect_type_blobs.append(" ")
          effect_type_blobs.append(
              rendering_parts.maybe_qualified_type_name(effect_protocol)
          )
        extra_annotations.append(
            rendering_parts.fold_condition(
                expanded=rendering_parts.floating_annotation_with_separate_focus(
                    rendering_parts.in_outlined_box(
                        rendering_parts.comment_color(
                            rendering_parts.siblings(
                                "# Unhandled effects:", *effect_type_blobs
                            )
                        )
                    )
                )
            )
        )
      broken_refs = effect_base.broken_handler_refs(node)
      if broken_refs:
        with handlers.setup_shared_value_context():
          broken_annotation = rendering_parts.fold_condition(
              expanded=rendering_parts.floating_annotation_with_separate_focus(
                  rendering_parts.in_outlined_box(
                      rendering_parts.build_full_line_with_annotations(
                          rendering_parts.error_color(
                              rendering_parts.text("# Broken handler refs: ")
                          ),
                          subtree_renderer(broken_refs, path=None),
                      )
                  )
              )
          )
          layout_algorithms.expand_to_depth(broken_annotation, 0)
          extra_annotations.append(broken_annotation)

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
      children=extra_annotations + children,
      suffix=")",
      path=path,
      background_color=background_color,
      background_pattern=background_pattern,
      first_line_annotation=first_line_annotation,
  )
