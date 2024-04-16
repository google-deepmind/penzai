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

"""Handler that falls back to ordinary `repr`."""

from __future__ import annotations

from typing import Any

from penzai.treescope import copypaste_fallback
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import part_interface


CSSStyleRule = part_interface.CSSStyleRule
HtmlContextForSetup = part_interface.HtmlContextForSetup


def handle_anything_with_repr(
    node: Any,
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Builds a foldable from its repr."""
  del subtree_renderer
  fallback = copypaste_fallback.render_not_roundtrippable(node)
  node_repr = repr(node)
  basic_repr = object.__repr__(node)
  if "\n" in node_repr:
    # Multiline repr. Use a simpler version for the one-line summary.
    if node_repr.startswith("<") and node_repr.endswith(">"):
      # This node has an idiomatic non-roundtrippable repr.
      # Replace its newlines with markers, and add the basic summary as a
      # comment
      lines = node_repr.split("\n")
      lines_with_markers = [
          basic_parts.siblings(
              basic_parts.Text(line),
              basic_parts.FoldCondition(
                  expanded=basic_parts.Text("\n"),
                  collapsed=common_styles.CommentColor(basic_parts.Text("↩")),
              ),
          )
          for line in lines[:-1]
      ]
      lines_with_markers.append(basic_parts.Text(lines[-1]))
      return basic_parts.siblings_with_annotations(
          common_structures.build_custom_foldable_tree_node(
              contents=basic_parts.RoundtripCondition(
                  roundtrip=fallback,
                  not_roundtrip=common_styles.AbbreviationColor(
                      basic_parts.siblings(*lines_with_markers)
                  ),
              ),
              path=path,
          ),
          extra_annotations=[
              common_styles.CommentColor(basic_parts.Text("  # " + basic_repr))
          ],
      )
    else:
      # Use basic repr as the summary.
      return common_structures.build_custom_foldable_tree_node(
          label=basic_parts.RoundtripCondition(
              roundtrip=fallback,
              not_roundtrip=common_styles.AbbreviationColor(
                  common_styles.CommentColorWhenExpanded(
                      basic_parts.siblings(
                          basic_parts.FoldCondition(
                              expanded=basic_parts.Text("# ")
                          ),
                          basic_repr,
                      )
                  )
              ),
          ),
          contents=basic_parts.FoldCondition(
              expanded=basic_parts.IndentedChildren.build(
                  [common_styles.AbbreviationColor(basic_parts.Text(node_repr))]
              )
          ),
          path=path,
      )
  elif node_repr == basic_repr:
    # Just use the basic repr as the summary, since we don't have anything else.
    return common_structures.build_one_line_tree_node(
        line=basic_parts.RoundtripCondition(
            roundtrip=fallback,
            not_roundtrip=common_styles.AbbreviationColor(
                basic_parts.Text(node_repr)
            ),
        ),
        path=path,
    )
  else:
    # Use the custom repr as a one-line summary, but float the basic repr to
    # the right to tell the user what the type is in case the custom repr
    # doesn't include that info.
    return basic_parts.siblings_with_annotations(
        common_structures.build_one_line_tree_node(
            line=basic_parts.RoundtripCondition(
                roundtrip=fallback,
                not_roundtrip=common_styles.AbbreviationColor(
                    basic_parts.Text(node_repr)
                ),
            ),
            path=path,
        ),
        extra_annotations=[
            common_styles.CommentColor(basic_parts.Text("  # " + basic_repr))
        ],
    )
