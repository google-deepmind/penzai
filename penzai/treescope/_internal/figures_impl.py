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

"""Internal definitions for figure utilities."""


from __future__ import annotations

import dataclasses
import io
from typing import Any

from penzai.treescope import lowering
from penzai.treescope._internal import html_escaping
from penzai.treescope._internal.parts import basic_parts
from penzai.treescope._internal.parts import part_interface


@dataclasses.dataclass(frozen=True)
class TreescopeFigure:
  """Wrapper that renders its child Treescope part as an IPython figure.

  This class implements the IPython display methods, so that it can be rendered
  to IPython at the top level.

  Attributes:
    treescope_part: Child to render.
  """

  treescope_part: part_interface.RenderableTreePart

  def _repr_html_(self) -> str:
    """Returns a rich HTML representation of this part."""
    return lowering.render_to_html_as_root(self.treescope_part, compressed=True)

  def _repr_pretty_(self, p, cycle):
    """Builds a representation of this part for the IPython text prettyprinter."""
    del cycle
    p.text(lowering.render_to_text_as_root(self.treescope_part))


class InlineBlock(basic_parts.BaseSpanGroup):
  """Renders an object in "inline-block" mode."""

  def _span_css_class(self) -> str:
    return "inline_block"

  def _span_css_rule(
      self, context: part_interface.HtmlContextForSetup
  ) -> part_interface.CSSStyleRule:
    return part_interface.CSSStyleRule(
        html_escaping.without_repeated_whitespace("""
          .inline_block {
            display: inline-block;
          }
        """)
    )


class AllowWordWrap(basic_parts.BaseSpanGroup):
  """Allows line breaks in its child.."""

  def _span_css_class(self) -> str:
    return "allow_wrap"

  def _span_css_rule(
      self, context: part_interface.HtmlContextForSetup
  ) -> part_interface.CSSStyleRule:
    return part_interface.CSSStyleRule(
        html_escaping.without_repeated_whitespace("""
          .allow_wrap {
            white-space: pre-wrap;
          }
        """)
    )


class PreventWordWrap(basic_parts.BaseSpanGroup):
  """Allows line breaks in its child.."""

  def _span_css_class(self) -> str:
    return "prevent_wrap"

  def _span_css_rule(
      self, context: part_interface.HtmlContextForSetup
  ) -> part_interface.CSSStyleRule:
    return part_interface.CSSStyleRule(
        html_escaping.without_repeated_whitespace("""
          .prevent_wrap {
            white-space: pre;
          }
        """)
    )


@dataclasses.dataclass(frozen=True)
class CSSStyled(basic_parts.DeferringToChild):
  """Adjusts the CSS style of its child.

  Attributes:
    child: Child to render.
    css: A CSS style string.
  """

  child: part_interface.RenderableTreePart
  style: str

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    style = html_escaping.escape_html_attribute(self.style)
    stream.write(f'<span style="{style}">')
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")
