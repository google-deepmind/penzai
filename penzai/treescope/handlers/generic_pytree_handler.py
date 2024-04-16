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

"""Pretty-print handlers for generic pytrees."""

from typing import Any

from penzai.core import tree_util as penzai_tree_util
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import part_interface
from penzai.treescope.handlers import generic_repr_handler


def handle_arbitrary_pytrees(
    node: Any,
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Generic foldable fallback for an unrecognized pytree type."""
  # Is this a pytree?

  maybe_result = penzai_tree_util.tree_flatten_exactly_one_level(node)

  if maybe_result is None:
    # Not a pytree.
    return NotImplemented

  subtrees_with_paths, _ = maybe_result

  # First, render the object with repr.
  repr_rendering = generic_repr_handler.handle_anything_with_repr(
      node=node,
      path=path,
      subtree_renderer=subtree_renderer,
  )

  # Then add an extra block that pretty-prints its children.
  list_items = []
  for key, child in subtrees_with_paths:
    child_path = None if path is None else (path + (key,))
    list_items.append(
        basic_parts.siblings_with_annotations(
            subtree_renderer(key, path=None),
            ": ",
            subtree_renderer(child, path=child_path),
            ", ",
        )
    )

  boxed_pytree_children = basic_parts.IndentedChildren.build(
      [
          common_styles.DashedGrayOutlineBox(
              basic_parts.build_full_line_with_annotations(
                  common_structures.build_custom_foldable_tree_node(
                      label=common_styles.CommentColor(
                          basic_parts.Text("# PyTree children: ")
                      ),
                      contents=basic_parts.IndentedChildren.build(list_items),
                  )
              ),
          )
      ]
  )
  return basic_parts.siblings_with_annotations(
      repr_rendering,
      basic_parts.FoldCondition(
          expanded=basic_parts.RoundtripCondition(
              not_roundtrip=boxed_pytree_children
          )
      ),
  )
