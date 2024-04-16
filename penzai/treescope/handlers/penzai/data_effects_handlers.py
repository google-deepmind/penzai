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

"""Handle data-effects objects."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

from penzai.core import context
from penzai.core import formatting_util
from penzai.core import struct
from penzai.data_effects import effect_base
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import part_interface
from penzai.treescope.handlers import builtin_structure_handler
from penzai.treescope.handlers.penzai import layer_handler
from penzai.treescope.handlers.penzai import struct_handler

_known_handlers: context.ContextualValue[
    dict[str, tuple[effect_base.EffectHandler, tuple[Any, ...] | None]] | None
] = context.ContextualValue(
    module=__name__, qualname="_known_handlers", initial_value=None
)
"""Tracks all of the effect handlers we have seen.

This context dictionary maps known handler IDs to the keypath where that
handler was defined. This can be used to add hyperlinks from effect refs
and implementations back to the handler that is responsible for them.
"""


def handle_data_effects_objects(
    node: Any,
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Handles data effects objects."""

  def handler_id_interceptor(
      node: Any,
      path: tuple[Any, ...] | None = None,
      *,
      handler_id: str,
      hyperlink_path=None,
  ):
    if isinstance(node, str) and node == handler_id:
      child = basic_parts.Text(repr(node))
      if hyperlink_path is not None:
        child = foldable_impl.NodeHyperlink(
            child=child, target_keypath=hyperlink_path
        )
      return common_structures.build_one_line_tree_node(
          common_styles.CustomTextColor(
              child,
              color=formatting_util.color_from_string(
                  node, lightness=0.51, chroma=0.11
              ),
          ),
          path,
      )
    else:
      return subtree_renderer(node, path)

  if isinstance(node, effect_base.EffectHandler):
    cur_known = _known_handlers.get()
    if cur_known is None:
      cur_known = {}
    if node.handler_id not in cur_known:
      # Mutate a copy.
      cur_known = dict(cur_known)
      cur_known[node.handler_id] = (node, path)
    with _known_handlers.set_scoped(cur_known):
      return layer_handler.handle_layers(
          node,
          path,
          functools.partial(handler_id_interceptor, handler_id=node.handler_id),
      )

  elif isinstance(node, effect_base.HandledEffectRef):
    # Render and add comment linking it back
    cur_known = _known_handlers.get()
    if cur_known is not None and node.handler_id in cur_known:
      handler, handler_path = cur_known[node.handler_id]
      comment = [
          common_styles.CommentColor(
              basic_parts.siblings(
                  " # Handled by ",
                  foldable_impl.NodeHyperlink(
                      child=common_structures.maybe_qualified_type_name(
                          type(handler)
                      ),
                      target_keypath=handler_path,
                  ),
              )
          )
      ]
    else:
      comment = []
      handler_path = None

    assert dataclasses.is_dataclass(node), "Every struct.Struct is a dataclass"
    constructor_open = struct_handler.render_struct_constructor(node)
    fields = dataclasses.fields(node)
    children = builtin_structure_handler.build_field_children(
        node,
        path,
        functools.partial(
            handler_id_interceptor,
            handler_id=node.handler_id,
            hyperlink_path=handler_path,
        ),
        fields_or_attribute_names=fields,
        key_path_fn=node.key_for_field,
        attr_style_fn=struct_handler.struct_attr_style_fn_for_fields(fields),
    )
    background_color = node.treescope_color()
    return basic_parts.siblings_with_annotations(
        common_structures.build_foldable_tree_node_from_children(
            prefix=constructor_open,
            children=children,
            suffix=")",
            path=path,
            background_color=background_color,
            background_pattern=(
                "repeating-linear-gradient(-45deg,color-mix(in oklab,"
                f" {background_color} 20%, white) 0 0.5em,color-mix(in oklab,"
                f" {background_color} 7%, white) 0.5em 1em)"
            ),
        ),
        extra_annotations=comment,
    )

  elif isinstance(node, effect_base.EffectRuntimeImpl):
    if isinstance(node, struct.Struct):
      constructor_open = struct_handler.render_struct_constructor(node)
    elif dataclasses.is_dataclass(node):
      constructor_open = builtin_structure_handler.render_dataclass_constructor(
          node
      )
    else:
      return NotImplemented
    background_color = node.treescope_color()

    return common_structures.build_foldable_tree_node_from_children(
        prefix=constructor_open,
        children=builtin_structure_handler.build_field_children(
            node,
            path,
            functools.partial(
                handler_id_interceptor, handler_id=node.handler_id()
            ),
            fields_or_attribute_names=dataclasses.fields(node),
        ),
        suffix=")",
        path=path,
        background_color=background_color,
        background_pattern=(
            "repeating-linear-gradient(-45deg,color-mix(in oklab,"
            f" {background_color} 20%, white) 0 0.5em,color-mix(in oklab,"
            f" {background_color} 7%, white) 0.5em 1em)"
        ),
    )

  return NotImplemented
