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

from penzai.core import struct
from penzai.core._treescope_handlers import struct_handler
from penzai.deprecated.v1.core._treescope_handlers import layer_handler
from penzai.deprecated.v1.data_effects import effect_base
from treescope import context
from treescope import formatting_util
from treescope import renderers
from treescope import rendering_parts

_known_handlers: context.ContextualValue[
    dict[str, tuple[effect_base.EffectHandler, str | None]] | None
] = context.ContextualValue(
    module=__name__, qualname="_known_handlers", initial_value=None
)
"""Tracks all of the effect handlers we have seen.

This context dictionary maps known handler IDs to the keypath where that
handler was defined.
"""


def handle_data_effects_objects(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Handles data effects objects."""

  def handler_id_interceptor(
      node: Any,
      path: str | None = None,
      *,
      handler_id: str,
  ):
    if isinstance(node, str) and node == handler_id:
      child = rendering_parts.text(repr(node))
      return rendering_parts.build_one_line_tree_node(
          rendering_parts.custom_text_color(
              child,
              css_color=formatting_util.color_from_string(
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
      handler, _ = cur_known[node.handler_id]
      comment = [
          rendering_parts.comment_color(
              rendering_parts.siblings(
                  " # Handled by ",
                  rendering_parts.maybe_qualified_type_name(type(handler)),
              )
          )
      ]
    else:
      comment = []

    assert dataclasses.is_dataclass(node), "Every struct.Struct is a dataclass"
    constructor_open = struct_handler.render_struct_constructor(node)
    fields = dataclasses.fields(node)
    children = rendering_parts.build_field_children(
        node,
        path,
        functools.partial(
            handler_id_interceptor,
            handler_id=node.handler_id,
        ),
        fields_or_attribute_names=fields,
        attr_style_fn=struct_handler.struct_attr_style_fn_for_fields(fields),
    )
    background_color = node.treescope_color()
    return rendering_parts.siblings_with_annotations(
        rendering_parts.build_foldable_tree_node_from_children(
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
      constructor_open = rendering_parts.render_dataclass_constructor(node)
    else:
      return NotImplemented
    background_color = node.treescope_color()

    return rendering_parts.build_foldable_tree_node_from_children(
        prefix=constructor_open,
        children=rendering_parts.build_field_children(
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
