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

"""Handlers for builtin atom types (e.g. constants or literals)."""
from __future__ import annotations

import enum
from typing import Any

from penzai.treescope import html_escaping
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import part_interface


CSSStyleRule = part_interface.CSSStyleRule
HtmlContextForSetup = part_interface.HtmlContextForSetup


class KeywordColor(basic_parts.BaseSpanGroup):
  """Renders its child in a color for keywords."""

  def _span_css_class(self) -> str:
    return "color_keyword"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .color_keyword
      {
        color: #0000ff;
      }
    """))


class NumberColor(basic_parts.BaseSpanGroup):
  """Renders its child in a color for numbers."""

  def _span_css_class(self) -> str:
    return "color_number"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .color_number
      {
        color: #098156;
      }
    """))


class StringLiteralColor(basic_parts.BaseSpanGroup):
  """Renders its child in a color for string literals."""

  def _span_css_class(self) -> str:
    return "color_string"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .color_string
      {
        color: #a31515;
      }
    """))


def handle_builtin_atoms(
    node: Any,
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Handles builtin atom types."""
  del subtree_renderer

  # String literals.
  if isinstance(node, (str, bytes)):
    lines = node.splitlines(keepends=True)
    if len(lines) > 1:
      # For multiline strings, we use two renderings:
      # - When collapsed, they render with ordinary `repr`,
      # - When expanded, they render as the implicit concatenation of per-line
      #   string literals.
      # Note that the `repr` for a string sometimes switches delimiters
      # depending on whether the string contains quotes or not, so we can't do
      # much manipulation of the strings themselves. This means that the safest
      # thing to do is to just embed two copies of the string into the IR,
      # one for the full string and the other for each line.
      return common_structures.build_custom_foldable_tree_node(
          contents=StringLiteralColor(
              basic_parts.FoldCondition(
                  collapsed=basic_parts.Text(repr(node)),
                  expanded=basic_parts.IndentedChildren.build(
                      children=[basic_parts.Text(repr(line)) for line in lines],
                      comma_separated=False,
                  ),
              )
          ),
          path=path,
      )
    else:
      # No newlines, so render it on a single line.
      return common_structures.build_one_line_tree_node(
          StringLiteralColor(basic_parts.Text(repr(node))), path
      )

  # Numeric literals.
  if isinstance(node, (int, float)):
    return common_structures.build_one_line_tree_node(
        NumberColor(basic_parts.Text(repr(node))), path
    )

  # Keyword objects.
  if any(
      node is literal
      for literal in (False, True, None, Ellipsis, NotImplemented)
  ):
    return common_structures.build_one_line_tree_node(
        KeywordColor(basic_parts.Text(repr(node))), path
    )

  # Enums. (Rendered roundtrippably, unlike the normal enum `repr`.)
  if isinstance(node, enum.Enum):
    cls = type(node)
    if node is getattr(cls, node.name):
      return common_structures.build_one_line_tree_node(
          basic_parts.siblings_with_annotations(
              common_structures.maybe_qualified_type_name(cls),
              "." + node.name,
              extra_annotations=[
                  common_styles.CommentColor(
                      basic_parts.Text(f"  # value: {repr(node.value)}")
                  )
              ],
          ),
          path,
      )

  return NotImplemented
