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

"""Embedding of external HTML content into treescope's IR."""
from __future__ import annotations

import dataclasses
import io
from typing import Any, Sequence

from penzai.treescope._internal import html_escaping
from penzai.treescope._internal.parts import part_interface

CSSStyleRule = part_interface.CSSStyleRule
JavaScriptDefn = part_interface.JavaScriptDefn
HtmlContextForSetup = part_interface.HtmlContextForSetup
RenderableTreePart = part_interface.RenderableTreePart
ExpandState = part_interface.ExpandState
FoldableTreeNode = part_interface.FoldableTreeNode


@dataclasses.dataclass(frozen=True)
class EmbeddedIFrame(RenderableTreePart):
  """Builds an HTML iframe containing scoped HTML for a rich display object.

  Attributes:
    embedded_html: HTML source to render in the iframe.
    fallback_in_text_mode: Fallback object to render in text mode.
    virtual_width: Number of characters wide to pretend this iframe is for
      layout purposes (since we can't infer it from `embedded_html` until it's
      actually rendered)
    virtual_height: Number of lines high to pretend this iframe is for layout
      purposes (since we can't infer it from `embedded_html` until it's actually
      rendered)
  """

  embedded_html: str
  fallback_in_text_mode: RenderableTreePart
  virtual_width: int = 80
  virtual_height: int = 2

  def _compute_collapsed_width(self) -> int:
    return self.virtual_width

  def _compute_newlines_in_expanded_parent(self) -> int:
    return self.virtual_height

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    return ()

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return frozenset()

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    # Render the fallback.
    self.fallback_in_text_mode.render_to_text(
        stream,
        expanded_parent=expanded_parent,
        indent=indent,
        roundtrip_mode=roundtrip_mode,
        render_context=render_context,
    )

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    rules = {
        # Register an interaction observer to detect when the content of the
        # iframe changes size, then update the size of the iframe element
        # itself to match its content.
        # But start with a minimum width of 80 characters.
        JavaScriptDefn(html_escaping.without_repeated_whitespace("""
        this.getRootNode().host.defns.resize_iframe_by_content = ((iframe) => {
          iframe.height = 0;
          iframe.style.width = "80ch";
          iframe.style.overflow = "hidden";
          iframe.contentDocument.scrollingElement.style.width = "fit-content";
          iframe.contentDocument.scrollingElement.style.height = "fit-content";
          iframe.contentDocument.scrollingElement.style.overflow = "hidden";
          const observer = new ResizeObserver((entries) => {
            console.log("resize", entries);
            const [entry] = entries;
            const computedStyle = getComputedStyle(
                iframe.contentDocument.scrollingElement);
            iframe.style.width = `calc(4ch + ${computedStyle['width']})`;
            iframe.style.height = `calc(${computedStyle['height']})`;
          });
          observer.observe(iframe.contentDocument.scrollingElement);
        });
        """)),
        CSSStyleRule(html_escaping.without_repeated_whitespace("""
          .embedded_html {
            display: block;
            padding-left: 1ch;
            padding-right: 1ch;
            width: max-content;
          }
          .embedded_html iframe {
            border: none;
            resize: both;
          }
        """)),
    }
    return rules

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    srcdoc = html_escaping.escape_html_attribute(
        f'<html><body style="margin: 0">{self.embedded_html}</body></html>'
    )
    stream.write(
        f'<div class="embedded_html"><iframe srcdoc="{srcdoc}"'
        ' onload="this.getRootNode().host.defns.resize_iframe_by_content(this)">'
        '</iframe></div>'
    )


def embedded_iframe(
    embedded_html: str,
    fallback_in_text_mode: RenderableTreePart,
    virtual_width: int = 80,
    virtual_height: int = 2,
) -> part_interface.RenderableTreePart:
  """Returns a wrapped child in a color for non-roundtrippable abbreviations."""
  if not isinstance(embedded_html, str):
    raise ValueError(
        '`embedded_html` must be a string, but got'
        f' {type(embedded_html).__name__}'
    )
  if not isinstance(fallback_in_text_mode, RenderableTreePart):
    raise ValueError(
        '`fallback_in_text_mode` must be a renderable part, but got'
        f' {type(fallback_in_text_mode).__name__}'
    )
  if not isinstance(virtual_width, int):
    raise ValueError(
        '`virtual_width` must be an integer, but got'
        f' {type(virtual_width).__name__}'
    )
  if not isinstance(virtual_height, int):
    raise ValueError(
        '`virtual_height` must be an integer, but got'
        f' {type(virtual_height).__name__}'
    )
  return EmbeddedIFrame(
      embedded_html=embedded_html,
      fallback_in_text_mode=fallback_in_text_mode,
      virtual_width=virtual_width,
      virtual_height=virtual_height,
  )
