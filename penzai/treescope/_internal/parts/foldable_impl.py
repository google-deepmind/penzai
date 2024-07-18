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

"""Low-level implementation details of the foldable system.

This module contains low-level wrappers that handle folding and unfolding
as well as hyperlinking (and unfolding) internal nodes.
"""

from __future__ import annotations

import dataclasses
import io
from typing import Any, Callable, Sequence

from penzai.treescope._internal import html_escaping
from penzai.treescope._internal.parts import basic_parts
from penzai.treescope._internal.parts import part_interface


CSSStyleRule = part_interface.CSSStyleRule
JavaScriptDefn = part_interface.JavaScriptDefn
HtmlContextForSetup = part_interface.HtmlContextForSetup
RenderableTreePart = part_interface.RenderableTreePart
ExpandState = part_interface.ExpandState
FoldableTreeNode = part_interface.FoldableTreeNode


SETUP_CONTEXT = HtmlContextForSetup(
    collapsed_selector=(
        ".foldable_node:has(>label>.foldable_node_toggle:not(:checked))"
    ),
    roundtrip_selector=".treescope_root.roundtrip_mode",
    hyperlink_hover_selector=".hyperlink_remote_hover",
    hyperlink_clicked_selector=".was_scrolled_to",
    hyperlink_clicked_tick_selector=".was_scrolled_to.first_tick",
    hyperlink_target_selector=".hyperlink_target",
)

################################################################################
# Foldable node implementation
################################################################################


@dataclasses.dataclass(frozen=False)
class FoldableTreeNodeImpl(FoldableTreeNode):
  """Concrete implementation of a node that can be expanded or collapsed.

  This is kept separate from the abstract definition of a FoldableTreeNode to
  avoid strong dependencies on its implementation as much as possible.

  Attributes:
    contents: Contents of the foldable node.
    label: Optional label for the foldable node. This appears in front of the
      contents, and clicking it expands or collapses the foldable node. This
      should not contain any other foldables and should generally be a single
      line.
    expand_state: Current expand state for the node.
  """

  contents: RenderableTreePart
  label: RenderableTreePart = basic_parts.EmptyPart()
  expand_state: ExpandState = ExpandState.WEAKLY_COLLAPSED

  def get_expand_state(self) -> ExpandState:
    """Returns the node's expand state."""
    return self.expand_state

  def set_expand_state(self, expand_state: ExpandState):
    """Sets the node's expand state."""
    self.expand_state = expand_state

  def as_expanded_part(self) -> RenderableTreePart:
    """Returns the contents of this foldable when expanded."""
    return basic_parts.siblings(self.label, self.contents)

  def html_setup_parts(
      self, setup_context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    if setup_context.collapsed_selector != SETUP_CONTEXT.collapsed_selector:
      raise ValueError(
          "FoldableTreeNodeImpl only works properly when the tree is"
          " configured using SETUP_CONTEXT.collapsed_selector"
      )
    # These CSS rules ensure that:
    # - Fold markers appear relative to the "foldable_node" HTML element
    # - The checkbox we are using to track collapsed state is hidden.
    # - Before the label, if this node isn't inside a collapsed parent node,
    #   we insert a floating expand/collapse marker depending on whether the
    #   checkbox is checked, and style it appropriately.
    # - If this node is in a collapsed parent node, don't toggle its state when
    #   clicked, since that won't do anything. If it is not, then show a pointer
    #   cursor to indicate that clicking will do something.
    # - When this is the first object on its line, shift the triangle marker
    #   left into the margin. Otherwise, shift the contents to the right.
    rule = html_escaping.without_repeated_whitespace(f"""
        .foldable_node
        {{
            position: relative;
        }}

        .foldable_node_toggle
        {{
            display: none;
        }}

        .foldable_node > label::before
        {{
            color: #cccccc;
            position: relative;
            left: -1ch;
            width: 0;
            display: inline-block;
        }}
        .foldable_node > label:hover::before
        {{
            color: darkseagreen;
        }}

        .foldable_node:not({setup_context.collapsed_selector} *)
          > label:has(>.foldable_node_toggle:checked)::before
        {{
            content: '\\25bc';
        }}
        .foldable_node:not({setup_context.collapsed_selector} *)
          > label:has(>.foldable_node_toggle:not(:checked))::before
        {{
            content: '\\25b6';
        }}

        .foldable_node:not({setup_context.collapsed_selector} *) > label:hover
        {{
            cursor: pointer;
        }}
        .foldable_node:is({setup_context.collapsed_selector} *) > label
        {{
            pointer-events: none;
        }}
        .foldable_node:is({setup_context.collapsed_selector} *)
        {{
            cursor: text;
        }}

        .foldable_node.is_first_on_line > label::before
        {{
            position: relative;
            left: -1.25ch;
        }}
        .foldable_node:not(
            {setup_context.collapsed_selector} *):not(.is_first_on_line)
        {{
            margin-left: 1ch;
        }}
        """)
    return (
        {CSSStyleRule(rule)}
        | self.contents.html_setup_parts(setup_context)
        | self.label.html_setup_parts(setup_context)
    )

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    if at_beginning_of_line:
      classname = "foldable_node is_first_on_line"
    else:
      classname = "foldable_node"

    stream.write(f'<span class="{classname}"><label>')

    stream.write('<input type="checkbox" class="foldable_node_toggle"')
    if (
        self.expand_state == ExpandState.WEAKLY_EXPANDED
        or self.expand_state == ExpandState.EXPANDED
    ):
      stream.write(" checked")
    stream.write("></input>")

    self.label.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</label>")
    self.contents.render_to_html(
        stream,
        at_beginning_of_line=isinstance(self.label, basic_parts.EmptyPart),
        render_context=render_context,
    )
    stream.write("</span>")

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    # In text mode, we just render the label and contents with no additional
    # wrapping. Note that we only expand if this node is expanded AND the
    # parents are expanded, since collapsed parents override children.
    expanded_here = expanded_parent and (
        self.expand_state == ExpandState.WEAKLY_EXPANDED
        or self.expand_state == ExpandState.EXPANDED
    )
    self.label.render_to_text(
        stream,
        expanded_parent=expanded_here,
        indent=indent,
        roundtrip_mode=roundtrip_mode,
        render_context=render_context,
    )
    self.contents.render_to_text(
        stream,
        expanded_parent=expanded_here,
        indent=indent,
        roundtrip_mode=roundtrip_mode,
        render_context=render_context,
    )


################################################################################
# Node hyperlinks implementation
################################################################################


@dataclasses.dataclass
class HyperlinkTarget(basic_parts.DeferringToChild):
  """Wraps a node so that it can be targeted by a hyperlink.

  Attributes:
    child: Child part to render.
    keypath: Keypath to this node, which can be referenced by hyperlinks.
  """

  child: RenderableTreePart
  keypath: str | None

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    if self.keypath is None:
      stream.write('<span class="hyperlink_target">')
    else:
      keypath_as_html_attr = html_escaping.escape_html_attribute(
          "".join(str(key) for key in self.keypath)
      )
      stream.write(
          html_escaping.without_repeated_whitespace(
              '<span class="hyperlink_target" '
              f'data-keypath="{keypath_as_html_attr}"'
              ">"
          )
      )
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


@dataclasses.dataclass
class NodeHyperlink(basic_parts.DeferringToChild):
  """Builds a hyperlink to another node, based on that node's JAX keypath.

  This does nothing in text rendering mode.

  Attributes:
    child: Child part to render.
    target_keypath: Keypath to the target.
  """

  child: RenderableTreePart
  target_keypath: str | None

  def html_setup_parts(
      self, setup_context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    rules = {
        JavaScriptDefn(html_escaping.without_repeated_whitespace("""
        (()=>{
          const _get_target = (root, target_path) => {
            /* Look for the requested path string. */
            let target = root.querySelector(
                `[data-keypath="${CSS.escape(target_path)}"]`
            );
            return target;
          };
          const _get_scroll_target = (target) => {
              /* Try to jump to the label. */
              const possible = target.querySelector(":scope > label");
              return possible ? possible : target;
          };

          const defns = this.getRootNode().host.defns;
          defns.expand_and_scroll_to = (
            (linkelement, target_path) => {
              const root = linkelement.getRootNode().shadowRoot;
              const target = _get_target(root, target_path);
              /* Expand all of its parents. */
              let may_need_expand = target.parentElement;
              while (may_need_expand != root) {
                if (may_need_expand.classList.contains("foldable_node")) {
                  const checkbox = may_need_expand.querySelector(
                      ":scope > label > .foldable_node_toggle");
                  checkbox.checked = true;
                }
                may_need_expand = may_need_expand.parentElement;
              }
              /* Scroll it into view. */
              _get_scroll_target(target).scrollIntoView({
                  "behavior":"smooth", "block":"center", "inline":"center"
              });
              if (!target.classList.contains("was_scrolled_to")) {
                target.classList.add("was_scrolled_to", "first_tick");
                setTimeout(() => {
                  target.classList.remove("first_tick");
                }, 100);
                setTimeout(() => {
                  target.classList.remove("was_scrolled_to");
                }, 1200);
              }
            }
          );

          defns.handle_hyperlink_mouse = (
            (linkelement, event, target_path) => {
              const root = linkelement.getRootNode().shadowRoot;
              const target = _get_target(root, target_path);
              if (event.type == "mouseover") {
                target.classList.add("hyperlink_remote_hover");
              } else {
                target.classList.remove("hyperlink_remote_hover");
              }
            }
          );
        })();
        """)),
        CSSStyleRule(html_escaping.without_repeated_whitespace("""
          .path_hyperlink {
              text-decoration: underline oklch(45.2% 0.198 264) dashed;
          }
          .path_hyperlink:hover {
              cursor: pointer;
              background-color: oklch(95.4% 0.034 109);
              text-decoration: underline oklch(84.3% 0.205 109) solid;
          }

          .hyperlink_remote_hover {
              font-weight: bold;
              background-color: oklch(95.4% 0.034 109);
              --highlight-color-light: oklch(95.4% 0.034 109);
              --highlight-color-dark: oklch(84% 0.145 109);
          }

          .was_scrolled_to {
              --highlight-color-light: oklch(84% 0.133 151.065);
              --highlight-color-dark: oklch(74% 0.133 151.065);
          }
          .was_scrolled_to:not(.first_tick) {
              transition: background-color 1s ease-in-out,
                  font-weight 1s ease-in-out;
          }
          .was_scrolled_to.first_tick {
              background-color: oklch(69.2% 0.133 151.065);
              font-weight: bold;
          }
        """)),
    }
    return rules | self.child.html_setup_parts(setup_context)

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    if self.target_keypath is None:
      stream.write("<span>")
    else:
      target_path_as_html_attr = html_escaping.escape_html_attribute(
          "".join(str(key) for key in self.target_keypath)
      )
      stream.write(
          html_escaping.without_repeated_whitespace(
              '<span class="path_hyperlink"'
              ' onClick="this.getRootNode().host.defns.expand_and_scroll_to(this,'
              ' this.dataset.targetpath)"'
              ' onMouseOver="this.getRootNode().host.defns.handle_hyperlink_mouse('
              ' this, event, this.dataset.targetpath)"'
              ' onMouseOut="this.getRootNode().host.defns.handle_hyperlink_mouse('
              ' this, event, this.dataset.targetpath)"'
              f' data-targetpath="{target_path_as_html_attr}">'
          )
      )
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


################################################################################
# Path copy button implementation
################################################################################


@dataclasses.dataclass
class StringCopyButton(RenderableTreePart):
  """Builds a button that, when clicked, copies the given path.

  This does nothing in text rendering mode, and only appears when its parent
  is expanded.

  Attributes:
    copy_string: String to copy when clicked
    annotation: Annotation to show when the button is hovered over.
  """

  copy_string: str
  annotation: str = "Copy: "

  def _compute_collapsed_width(self) -> int:
    return 0

  def _compute_newlines_in_expanded_parent(self) -> int:
    return 0

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
    # Doesn't render at all in text mode.
    pass

  def html_setup_parts(
      self, setup_context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    # https://github.com/google/material-design-icons/blob/master/symbols/web/content_copy/materialsymbolsoutlined/content_copy_grad200_20px.svg
    font_data_url = "data:font/woff2;base64,d09GMgABAAAAAAIQAAoAAAAABRwAAAHFAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAABmAAgkoKdIEBCwYAATYCJAMIBCAFgxIHLhtsBMieg3FDX2LI6YvSpuiPM5T1JXIRVMvWMztPyFFC+WgkTiBD0hiDQuJEdGj0Hb/fvIdpqK6hqWiiQuMnGhHfUtAU6BNr4AFInkE6cUuun+R5qcskwvfFl/qxgEo8gbJwG81HA/nAR5LrrJ1R+gz0Rd0AJf1gN7CwGj2g0oyuR77mE16wHX9OggpeTky4eIbz5cbrOGtaAgQINwDasysQuIIXWEFwAPQpIYdU//+g7T7X3t0fKPqAv52g0LAN7AMwAmgzRS+uZSeEXx2f6czN4RHy5uBAKzBjpFp3iHQCE0ZuP4S7nfBLEHFMmAi+8vE2hn1h7+bVwXjwHrvDGUCnjfEEgt+OcZll759CJwB8h94MMGS3GZAgmI5jBQ9tTGeH9EBBIG3Dg4R/YcybAGEAAVK/AQGaAeMClAHzEOgZtg6BPgOOIDBkiQ5eFBXCBFci0phropnQAApZED1z1kSfCfthyKnHdaFsHf0NmGEN6BdAqVVpatsSZmddai92fz94Uijq6pmr6OoYCSirGmvJG3SWS3FE2cBQfT+HlopG4Fsw5agq68iZeSNlpWnBHIedMreuWqGCm1WFrkSSx526WWswAQAA"
    rules = {
        JavaScriptDefn(html_escaping.without_repeated_whitespace("""
          this.getRootNode().host.defns.handle_copy_click = async (button) => {
            const dataToCopy = button.dataset.copy;
            try {
              await navigator.clipboard.writeText(dataToCopy);
              button.classList.add("was_clicked");
              setTimeout(() => {
                button.classList.remove("was_clicked");
              }, 2000);
            } catch (e) {
              button.classList.add("broken_copy");
              button.textContent = (
                  "  # Failed to copy! Copy manually instead: " + dataToCopy
              );
            }
          };
        """)),
        # Font-face definitions can't live inside a shadow
        # DOM node, so we need to inject them into the root document.
        JavaScriptDefn(html_escaping.without_repeated_whitespace("""
        if (
            !Array.from(document.fonts.values()).some(
              font => font.family == 'Material Symbols Outlined Content Copy"'
            )
        ) {
          const sheet = new CSSStyleSheet();
          sheet.replaceSync(`@font-face {
              font-family: 'Material Symbols Outlined Content Copy';
              font-style: normal;
              font-weight: 400;
              src: url({__FONT_DATA_URL__}) format('woff2');
          }`);
          document.adoptedStyleSheets = [...document.adoptedStyleSheets, sheet];
        }
        """.replace("{__FONT_DATA_URL__}", font_data_url))),
        CSSStyleRule(html_escaping.without_repeated_whitespace(f"""
          {setup_context.collapsed_selector} .copybutton {{
              display: none;
          }}
          .copybutton::before {{
              content: " ";
          }}
          .copybutton > span::before {{
              content: "\\e14d";
              font-family: 'Material Symbols Outlined Content Copy';
              -webkit-font-smoothing: antialiased;
              color: #e0e0e0;
              cursor: pointer;
              font-size: 0.9em;
          }}
          .copybutton:hover > span::before {{
              color: darkseagreen;
          }}
          .copybutton.was_clicked > span::after {{
              color: #cccccc;
          }}
          .copybutton:hover > span::after {{
              color: darkseagreen;
              content: " " var(--data-annotation) var(--data-copy);
              transition: color 1s ease-in-out;
          }}
          .copybutton.was_clicked > span::after {{
              content: " Copied! " var(--data-copy) !important;
          }}
          .copybutton.broken_copy:hover {{
              color: darkseagreen;
          }}
          .copybutton.broken_copy {{
              color: #e0e0e0;
          }}
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
    if self.copy_string is not None:
      copy_string_attr = html_escaping.escape_html_attribute(self.copy_string)
      copy_string_property = html_escaping.escape_html_attribute(
          repr(self.copy_string)
      )
      annotation_property = html_escaping.escape_html_attribute(
          repr(self.annotation)
      )
      attributes = html_escaping.without_repeated_whitespace(
          "class='copybutton'"
          " onClick='this.getRootNode().host.defns.handle_copy_click(this)'"
          f' data-copy="{copy_string_attr}" style="--data-copy:'
          f" {copy_string_property}; --data-annotation:"
          f' {annotation_property}" '
      )
      stream.write(f"<span {attributes}><span></span></span>")


################################################################################
# Deferrables
################################################################################


@dataclasses.dataclass(frozen=False)
class DeferredPlaceholder(basic_parts.DeferringToChild):
  """A deferred part. Renders as a placeholder which will later be replaced."""

  child: RenderableTreePart
  replacement_id: str
  saved_at_beginning_of_line: bool | None = None
  saved_render_context: dict[Any, Any] | None = None
  needs_layout_decision: bool = False

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    self.saved_at_beginning_of_line = at_beginning_of_line
    self.saved_render_context = render_context
    stream.write(f'<span id="{self.replacement_id}">')
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


@dataclasses.dataclass(frozen=True)
class DeferredWithThunk:
  """Stores a deferred placeholder along with its thunk."""

  placeholder: DeferredPlaceholder
  thunk: Callable[[ExpandState | None], RenderableTreePart]


@dataclasses.dataclass(frozen=True)
class EmptyWithHeightGuess(basic_parts.BaseContentlessLeaf):
  """Helper class that reports a guess of its height."""

  fake_newlines: int

  def _compute_newlines_in_expanded_parent(self) -> int:
    return self.fake_newlines


def fake_placeholder_foldable(
    placeholder_content: RenderableTreePart, extra_newlines_guess: int
) -> FoldableTreeNode:
  """Builds a fake placeholder for deferred renderings.

  The main use for this is is to return as the placeholder for a deferred
  rendering, so that it can participate in layout decisions. The
  `get_expand_state` method can be used to infer whether the part was
  collapsed while deferred.

  Args:
    placeholder_content: The content of the placeholder to render.
    extra_newlines_guess: The number of fake newlines to pretend this object
      has.

  Returns:
    A foldable node that does not have any actual content, but pretends to
    contain the given number of newlines for layout decisions.
  """
  return FoldableTreeNodeImpl(
      contents=basic_parts.siblings(
          placeholder_content,
          EmptyWithHeightGuess(fake_newlines=extra_newlines_guess),
      ),
      expand_state=ExpandState.WEAKLY_EXPANDED,
  )
