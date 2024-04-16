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

"""Common styles for parts of renderings."""

from __future__ import annotations

import dataclasses
import io
from typing import Any

from penzai.treescope import html_escaping
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import part_interface


CSSStyleRule = part_interface.CSSStyleRule
JavaScriptDefn = part_interface.JavaScriptDefn
RenderableTreePart = part_interface.RenderableTreePart
HtmlContextForSetup = part_interface.HtmlContextForSetup


class AbbreviationColor(basic_parts.BaseSpanGroup):
  """Renders its child in a color for non-roundtrippable abbreviations."""

  def _span_css_class(self) -> str:
    return "color_abbrev"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .color_abbrev
      {
        color: #682a00;
      }
    """))


class CommentColor(basic_parts.BaseSpanGroup):
  """Renders its child in a color for comments."""

  def _span_css_class(self) -> str:
    return "color_comment"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .color_comment {
          color: #aaaaaa;
      }
      .color_comment a:not(:hover){
          color: #aaaaaa;
      }
    """))


class ErrorColor(basic_parts.BaseSpanGroup):
  """Renders its child in red to indicate errors / problems during rendering."""

  def _span_css_class(self) -> str:
    return "color_error"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .color_error {
          color: red;
      }
    """))


class DeferredPlaceholderStyle(basic_parts.BaseSpanGroup):
  """Renders its child in italics to indicate a deferred placeholder."""

  def _span_css_class(self) -> str:
    return "deferred_placeholder"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .deferred_placeholder {
          color: #a7a7a7;
          font-style: italic;
      }
    """))


class CommentColorWhenExpanded(basic_parts.BaseSpanGroup):
  """Renders its child in a color for comments, but only when expanded.

  This can be used in combination with another color (usually AbbreviationColor)
  to show a summary and then transform it into a comment when clicked, e.g. with

  AbbreviationColor(CommentColorWhenExpanded(...))
  """

  def _span_css_class(self) -> str:
    return "color_comment_when_expanded"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace(f"""
      .color_comment_when_expanded:not({context.collapsed_selector} *) {{
          color: #aaaaaa;
      }}
    """))


@dataclasses.dataclass(frozen=True)
class CustomTextColor(basic_parts.DeferringToChild):
  """A group that wraps its child in a span with a custom text color.

  Attributes:
    child: Contents of the group.
    color: CSS string defining the color.
  """

  child: RenderableTreePart
  color: str

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    colorstr = html_escaping.escape_html_attribute(self.color)
    stream.write(f'<span style="color:{colorstr};">')
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


class DashedGrayOutlineBox(basic_parts.BaseBoxWithOutline):
  """A highlighted box that identifies a part as being selected."""

  def _box_css_class(self) -> str:
    return "dashed_gray_outline"

  def _box_css_rule(
      self, context: HtmlContextForSetup
  ) -> part_interface.CSSStyleRule:
    return part_interface.CSSStyleRule(
        html_escaping.without_repeated_whitespace("""
            .box_with_outline.dashed_gray_outline
            {
                outline: 1px dashed #aaaaaa;
            }
        """)
    )


@dataclasses.dataclass(frozen=True)
class ColoredBorderIndentedChildren(basic_parts.IndentedChildren):
  """A sequence of children that also draws a colored line on the left.

  Assumes that the CSS variables --block-color and --block-border-scale are
  set in some container object. These are set by WithBlockColor.
  """

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    rule = html_escaping.without_repeated_whitespace(f"""
        .stacked_children.colored_border
          > .stacked_child:not({context.collapsed_selector} *)
        {{
          display: block;
          margin-left: calc(2ch - 0.4ch);
        }}
        .stacked_children.colored_border:not({context.collapsed_selector} *)
        {{
          display: block;
          border-left: solid 0.4ch var(--block-color);
          padding-top: calc(0.5lh * (1 - var(--block-border-scale)));
          margin-top: calc(-0.5lh * (1 - var(--block-border-scale)));
          padding-bottom: 0.5lh;
          margin-bottom: -0.5lh;
        }}
        .stacked_children.colored_border:not({context.collapsed_selector} *)
        {{
          border-top-left-radius: calc(1em * var(--block-border-scale));
        }}
        {context.collapsed_selector} .stacked_children.colored_border
        {{
          border-top: solid 1px var(--block-color);
          border-bottom: solid 1px var(--block-color);
        }}
        """)
    return {CSSStyleRule(rule)} | super().html_setup_parts(context)

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    stream.write('<span class="stacked_children colored_border">')
    for part in self.children:
      # All children render at the beginning of their line.
      stream.write('<span class="stacked_child colored_border">')
      part.render_to_html(
          stream,
          at_beginning_of_line=True,
          render_context=render_context,
      )
      stream.write("</span>")

    stream.write("</span>")


def _common_block_rules(
    context: HtmlContextForSetup,
) -> dict[str, CSSStyleRule]:
  return {
      "colored_line": CSSStyleRule(
          html_escaping.without_repeated_whitespace(f"""
            .colored_line
            {{
              color: black;
              background-color: var(--block-color);
              border-top: 1px solid var(--block-color);
              border-bottom: 1px solid var(--block-color);
            }}

            {context.hyperlink_hover_selector} .colored_line {{
              background-color: oklch(84% 0.145 109);
            }}

            {context.hyperlink_clicked_tick_selector} .colored_line {{
              background-color: oklch(94.1% 0.09 136.7);
            }}

            {context.hyperlink_clicked_selector}:not(
                {context.hyperlink_clicked_tick_selector}
            ) .colored_line {{
              transition: background-color 1s ease-in-out,
                  font-weight 1s ease-in-out;
            }}
          """)
      ),
      "patterned_line": CSSStyleRule(
          html_escaping.without_repeated_whitespace(f"""
            .patterned_line
            {{
              color: black;
              border: 1px solid var(--block-color);
              background-image: var(--block-background-image);
              background-clip: padding-box;
            }}

            {context.hyperlink_hover_selector} .patterned_line {{
              background-color: oklch(84% 0.145 109);
              background-image: none;
            }}

            {context.hyperlink_clicked_tick_selector} .colored_line {{
              background-color: oklch(94.1% 0.09 136.7);
              background-image: none;
            }}

            {context.hyperlink_clicked_selector}:not(
                {context.hyperlink_clicked_tick_selector}
            ) .colored_line {{
              transition: background-color 1s ease-in-out,
                  background-image 1s ease-in-out,
                  font-weight 1s ease-in-out;
            }}
          """)
      ),
      "topline": CSSStyleRule(html_escaping.without_repeated_whitespace(f"""
            .topline:not({context.collapsed_selector} *)
            {{
              padding-bottom: 0.15em;
              line-height: 1.5em;
            }}
          """)),
      "bottomline": CSSStyleRule(html_escaping.without_repeated_whitespace(f"""
            .bottomline:not({context.collapsed_selector} *)
            {{
              padding-top: 0.15em;
              line-height: 1.5em;
            }}
          """)),
      "hch_space_left": CSSStyleRule(
          html_escaping.without_repeated_whitespace("""
            .hch_space_left
            {
              padding-left: 0.5ch;
            }
          """)
      ),
      "hch_space_right": CSSStyleRule(
          html_escaping.without_repeated_whitespace("""
            .hch_space_right
            {
              padding-right: 0.5ch;
            }
          """)
      ),
  }


class ColoredTopLineSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that colors the prefix of a block."""

  def _span_css_class(self) -> str:
    return "colored_line topline hch_space_left"

  def _span_css_rule(self, context: HtmlContextForSetup) -> set[CSSStyleRule]:
    rules = _common_block_rules(context)
    return {rules["colored_line"], rules["topline"], rules["hch_space_left"]}


class ColoredSingleLineSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that colors a single line."""

  def _span_css_class(self) -> str:
    return "colored_line hch_space_left hch_space_right"

  def _span_css_rule(self, context: HtmlContextForSetup) -> set[CSSStyleRule]:
    rules = _common_block_rules(context)
    return {
        rules["colored_line"],
        rules["hch_space_left"],
        rules["hch_space_right"],
    }


class ColoredBottomLineSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that colors the suffix of a block."""

  def _span_css_class(self) -> str:
    return "colored_line bottomline hch_space_right"

  def _span_css_rule(self, context: HtmlContextForSetup) -> set[CSSStyleRule]:
    rules = _common_block_rules(context)
    return {
        rules["colored_line"],
        rules["bottomline"],
        rules["hch_space_right"],
    }


@dataclasses.dataclass
class WithBlockColor(basic_parts.DeferringToChild):
  """A group that configures block color CSS variables.

  This provides the necessary colors for ColoredBorderStackedChildren,
  ColoredTopLineSpanGroup, or ColoredBottomLineSpanGroup.

  Attributes:
    child: Contents of the group.
    color: CSS color to render.
  """

  child: RenderableTreePart
  color: str

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    style_string = f"--block-color: {self.color};"
    if at_beginning_of_line:
      style_string += "--block-border-scale: 0;"
    else:
      style_string += "--block-border-scale: 1;"
    stream.write(f'<span style="{style_string}">')
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


class PatternedTopLineSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that adds a pattern to the prefix of a block."""

  def _span_css_class(self) -> str:
    return "patterned_line topline hch_space_left"

  def _span_css_rule(self, context: HtmlContextForSetup) -> set[CSSStyleRule]:
    rules = _common_block_rules(context)
    return {rules["patterned_line"], rules["topline"], rules["hch_space_left"]}


class PatternedBottomLineSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that adds a pattern to the suffix of a block."""

  def _span_css_class(self) -> str:
    return "patterned_line bottomline hch_space_left hch_space_right"

  def _span_css_rule(self, context: HtmlContextForSetup) -> set[CSSStyleRule]:
    rules = _common_block_rules(context)
    return {
        rules["patterned_line"],
        rules["bottomline"],
        rules["hch_space_left"],
        rules["hch_space_right"],
    }


class PatternedSingleLineSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that adds a pattern to a single line."""

  def _span_css_class(self) -> str:
    return "patterned_line expand_right"

  def _span_css_rule(self, context: HtmlContextForSetup) -> set[CSSStyleRule]:
    rules = _common_block_rules(context)
    return {rules["patterned_line"], rules["expand_right"]}


@dataclasses.dataclass
class WithBlockPattern(basic_parts.DeferringToChild):
  """A group that configures block color CSS variables.

  This should be used around uses of PatternedTopLineSpanGroup,
  PatternedBottomLineSpanGroup, or PatternedSingleLineSpanGroup.

  Attributes:
    child: Contents of the group.
    color: CSS color for borders.
    pattern: CSS value for a background image pattern (e.g. using
      `repeating-linear-gradient`)
  """

  child: RenderableTreePart
  color: str
  pattern: str

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    style_string = (
        f"--block-color: {self.color}; "
        f"--block-background-image: {self.pattern};"
    )
    if at_beginning_of_line:
      style_string += "--block-border-scale: 0;"
    else:
      style_string += "--block-border-scale: 1;"
    stream.write(f'<span style="{style_string}">')
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


class QualifiedTypeNameSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that displays qualified names in a small font."""

  def _span_css_class(self) -> str:
    return "qualname_prefix"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
        .qualname_prefix {
          font-size: 0.8em;
        }
    """))
