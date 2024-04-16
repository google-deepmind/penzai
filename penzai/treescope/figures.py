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

"""Containers for conveniently rendering multiple objects together.

These containers make it possible to group together multiple children so that
they render as a single figure-like display object in IPython, and allow
embedding arbitrary treescope-renderable objects or treescope-compatible styles
as figures.
"""

from __future__ import annotations

import dataclasses
import io
from typing import Any

from penzai.treescope import default_renderer
from penzai.treescope import html_compression
from penzai.treescope import html_escaping
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import embedded_iframe
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import part_interface


class RendersAsRootInIPython(part_interface.RenderableTreePart):
  """Base class / mixin that implements ``_repr_html_`` for treescope parts.

  Subclasses of this class will render themselves as rich display objects when
  displayed in IPython, instead of having their contents formatted with a
  pretty printer or ``repr``.
  """

  def _repr_html_(self) -> str:
    """Returns a rich HTML representation of this part."""
    return html_compression.compress_html(
        foldable_impl.render_to_html_as_root(self),
        loading_message="(Loading...)",
    )

  def _repr_pretty_(self, p, cycle):
    """Builds a representation of this part for the IPython text prettyprinter."""
    del cycle
    p.text(foldable_impl.render_to_text_as_root(self))


@dataclasses.dataclass(frozen=True)
class TreescopeRenderingFigure(
    basic_parts.DeferringToChild, RendersAsRootInIPython
):
  """Wrapper that renders its child rendering as a HTML figure.

  Attributes:
    child: Child to render as a figure.
  """

  child: part_interface.RenderableTreePart


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


def wrap_as_treescope_figure(value: Any) -> part_interface.RenderableTreePart:
  """Converts an arbitrary object to a renderable treescope part if possible.

  Behavior depends on the type of `value`:

  * If ``value`` is an instance of `RendersAsRootInIPython`, returns it
    unchanged, since it knows how to render itself.
  * If ``value`` is a string, returns a rendering of that string.
  * If ``value`` has a ``_repr_html_`` method (but isn't an instance of
    `RendersAsRootInIPython`), returns an embedded iframe with the given HTML
    contents.
  * Otherwise, renders the value  using the default treescope renderer, but
    strips off any top-level comments / copy button annotations.

  The typical use is to provide helper constructors for containers to allow
  rendering lots of different objects in the "obvious" way.

  Args:
    value: Value to wrap.

  Returns:
    A renderable treescope part showing the value.
  """
  if isinstance(value, RendersAsRootInIPython):
    return value
  elif isinstance(value, str):
    return basic_parts.Text(value)
  elif isinstance(value, embedded_iframe.HasReprHtml):
    return InlineBlock(
        embedded_iframe.EmbeddedIFrame(
            embedded_iframe.to_html(value),
            fallback_in_text_mode=basic_parts.Text(object.__repr__(value)),
        )
    )
  else:
    return default_renderer.build_foldable_representation(value).renderable


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


def inline(*parts: Any, wrap: bool = False) -> RendersAsRootInIPython:
  """Returns a figure that arranges a set of displayable objects along a line.

  Args:
    *parts: Subfigures to display inline. These will be displayed using
      `wrap_as_treescope_figure`.
    wrap: Whether to wrap (insert newlines) between words at the end of a line.

  Returns:
    A figure which can be rendered in IPython or used to build more complex
    figures.
  """
  siblings = basic_parts.siblings(
      *(wrap_as_treescope_figure(part) for part in parts)
  )
  if wrap:
    return TreescopeRenderingFigure(AllowWordWrap(siblings))
  else:
    return TreescopeRenderingFigure(PreventWordWrap(siblings))


def indented(subfigure: Any) -> RendersAsRootInIPython:
  """Returns a figure object that displays a value with an indent.

  Args:
    subfigure: A value to render indented. Will be wrapped using
      `wrap_as_treescope_figure`.
  """
  return TreescopeRenderingFigure(
      basic_parts.IndentedChildren.build([wrap_as_treescope_figure(subfigure)])
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


def styled(subfigure: Any, style: str) -> RendersAsRootInIPython:
  """Returns a CSS-styled version of the first figure.

  Args:
    subfigure: A value to render. Will be wrapped using
      `wrap_as_treescope_figure`.
    style: A CSS style string.
  """
  return TreescopeRenderingFigure(
      CSSStyled(wrap_as_treescope_figure(subfigure), style)
  )


def with_font_size(subfigure: Any, size: str | float) -> RendersAsRootInIPython:
  """Returns a scaled version of the first figure.

  Args:
    subfigure: A value to render. Will be wrapped using
      `wrap_as_treescope_figure`.
    size: A multiplier for the font size (as a float) or a string giving a
      specific CSS font size (e.g. "14pt" or "2em").
  """
  if isinstance(size, str):
    style = f"font-size: {size}"
  else:
    style = f"font-size: {size}em"
  return TreescopeRenderingFigure(
      CSSStyled(wrap_as_treescope_figure(subfigure), style)
  )


def with_color(subfigure: Any, color: str) -> RendersAsRootInIPython:
  """Returns a colored version of the first figure.

  Args:
    subfigure: A value to render. Will be wrapped using
      `wrap_as_treescope_figure`.
    color: Any CSS color string.
  """
  return TreescopeRenderingFigure(
      CSSStyled(wrap_as_treescope_figure(subfigure), f"color: {color}")
  )


def bolded(subfigure: Any) -> RendersAsRootInIPython:
  """Returns a bolded version of the first figure.

  Args:
    subfigure: A value to render. Will be wrapped using
      `wrap_as_treescope_figure`.
  """
  return TreescopeRenderingFigure(
      CSSStyled(wrap_as_treescope_figure(subfigure), "font-weight: bold")
  )
