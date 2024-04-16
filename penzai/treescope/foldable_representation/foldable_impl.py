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

import contextlib
import dataclasses
import io
import json
from typing import Any, Callable, Iterator, Sequence
import uuid

from penzai.core import context
from penzai.treescope import html_compression
from penzai.treescope import html_escaping
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import part_interface


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
    return basic_parts.Siblings.build(self.label, self.contents)

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
            content: '▼';
        }}
        .foldable_node:not({setup_context.collapsed_selector} *)
          > label:has(>.foldable_node_toggle:not(:checked))::before
        {{
            content: '▶';
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
  keypath: tuple[Any, ...] | None

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
  target_keypath: tuple[Any, ...] | None

  def html_setup_parts(
      self, setup_context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    rules = {
        JavaScriptDefn(html_escaping.without_repeated_whitespace("""
        (()=>{
          const _get_root = (relative_to) => {
            /* Look for the root node. */
            let root = relative_to;
            while (
              root != document.body
              && !root.classList.contains("treescope_root")
            ) {
              root = root.parentElement;
            }
            return root;
          };
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

          window.treescope.expand_and_scroll_to = (
            (linkelement, target_path) => {
              const root = _get_root(linkelement);
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

          window.treescope.handle_hyperlink_mouse = (
            (linkelement, event, target_path) => {
              const root = _get_root(linkelement);
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
              ' onClick="treescope.expand_and_scroll_to(this,'
              ' this.dataset.targetpath)"'
              ' onMouseOver="treescope.handle_hyperlink_mouse(   this, event,'
              ' this.dataset.targetpath)"'
              ' onMouseOut="treescope.handle_hyperlink_mouse(   this, event,'
              ' this.dataset.targetpath)"'
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

  def _compute_tags_in_this_part(self) -> frozenset[Any]:
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
    rules = {
        JavaScriptDefn(html_escaping.without_repeated_whitespace("""
          window.treescope.handle_copy_click = async (button) => {
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
        CSSStyleRule(html_escaping.without_repeated_whitespace(f"""
          @font-face {{
              font-family: 'Material Symbols Outlined';
              font-style: normal;
              font-weight: 100 700;
              src: url(https://fonts.gstatic.com/s/materialsymbolsoutlined/v114/kJEhBvYX7BgnkSrUwT8OhrdQw4oELdPIeeII9v6oFsLjBuVY.woff2) format('woff2');
          }}
          {setup_context.collapsed_selector} .copybutton {{
              display: none;
          }}
          .copybutton > span::before {{
              content: " content_copy";
              font-family: 'Material Symbols Outlined';
              -webkit-font-feature-settings: 'liga';
              -webkit-font-smoothing: antialiased;
              font-variation-settings:
                'FILL' 0,
                'wght' 400,
                'GRAD' 200,
                'opsz' 20;
              position: relative;
              top: 0.1em;
              color: #e0e0e0;
              cursor: pointer;
              font-size: 0.8em;
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
          "class='copybutton' onClick='treescope.handle_copy_click(this)'"
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
  thunk: Callable[[RenderableTreePart | None], RenderableTreePart]


_deferrables: context.ContextualValue[list[DeferredWithThunk] | None] = (
    context.ContextualValue(
        module=__name__, qualname="_deferrables", initial_value=None
    )
)
"""An optional list of accumulated deferrables, for use by this module."""


def maybe_defer_rendering(
    main_thunk: Callable[[RenderableTreePart | None], RenderableTreePart],
    placeholder_thunk: Callable[[], RenderableTreePart],
) -> RenderableTreePart:
  """Possibly defers rendering of a part in interactive contexts.

  This function can be used by advanced handlers and autovisualizers to delay
  the rendering of "expensive" leaves such as `jax.Array` until after the tree
  structure is drawn. If run in a non-interactive context, this just calls the
  main thunk. If run in an interactive context, it instead calls the placeholder
  thunk, and enqueues the placeholder thunk to be called later.

  Rendering can be performed in a deferred context by running the handlers under
  the `collecting_deferred_renderings` context manager, and then rendered to
  a sequence of streaming HTML updates using the `display_streaming_as_root`
  function.

  Note that handlers who call this are responsible for ensuring that the
  logic in `main_thunk` is safe to run at a later point in time. In particular,
  any rendering context managers may have been exited by the time this main
  thunk is called. As a best practice, handlers should control all of the logic
  in `main_thunk` and shouldn't recursively call the subtree renderer inside it;
  subtrees should be rendered before calling `maybe_defer_rendering`.

  Args:
    main_thunk: A callable producing the main part to render. If not deferred,
      will be called with None. If deferred, will be called with the placeholder
      part, which can be inspected to e.g. infer folding state.
    placeholder_thunk: A callable producing a placeholder object, which will be
      rendered if we are deferring rendering.

  Returns:
    Either the rendered main part or a wrapped placeholder that will later be
    replaced with the main part.
  """
  deferral_list = _deferrables.get()
  if deferral_list is None:
    return main_thunk(None)
  else:
    placeholder = DeferredPlaceholder(
        child=placeholder_thunk(),
        replacement_id="deferred_" + uuid.uuid4().hex,
    )
    deferral_list.append(DeferredWithThunk(placeholder, main_thunk))
    return placeholder


@contextlib.contextmanager
def collecting_deferred_renderings() -> Iterator[list[DeferredWithThunk]]:
  # pylint: disable=g-doc-return-or-yield
  """Context manager that defers and collects `maybe_defer_rendering` calls.

  This context manager can be used by renderers that wish to render deferred
  objects in a streaming fashion. When used in a
  `with collecting_deferred_renderings() as deferreds:`
  expression, `deferreds` will be a list that is populated by calls to
  `maybe_defer_rendering`. This can later be passed to
  `display_streaming_as_root` to render the deferred object in a streaming
  fashion.

  Returns:
    A context manager in which `maybe_defer_rendering` calls will be deferred
    and collected into the result list.
  """
  # pylint: enable=g-doc-return-or-yield
  try:
    target = []
    with _deferrables.set_scoped(target):
      yield target
  finally:
    pass


################################################################################
# Top-level rendering and roundtrip mode implementation
################################################################################


def render_to_text_as_root(
    root_node: RenderableTreePart,
    roundtrip: bool = False,
    strip_whitespace_lines: bool = True,
) -> str:
  """Renders a root node to text.

  Args:
    root_node: The root node to render.
    roundtrip: Whether to render in roundtrip mode.
    strip_whitespace_lines: Whether to remove lines that are entirely
      whitespace. These lines can sometimes be generated by layout code being
      conservative about line breaks.

  Returns:
    Text for the rendered node.
  """
  stream = io.StringIO()
  root_node.render_to_text(
      stream,
      expanded_parent=True,
      indent=0,
      roundtrip_mode=roundtrip,
      render_context={},
  )
  result = stream.getvalue()

  if strip_whitespace_lines:
    postprocess_stream = io.StringIO()
    for line in result.splitlines(keepends=True):
      if line.strip():
        postprocess_stream.write(line)
    result = postprocess_stream.getvalue()

  return result


def _render_to_html_as_root_streaming(
    root_node: RenderableTreePart,
    roundtrip: bool,
    deferreds: Sequence[DeferredWithThunk],
) -> Iterator[str]:
  """Helper function: renders a root node to HTML one step at a time.

  Args:
    root_node: The root node to render.
    roundtrip: Whether to render in roundtrip mode.
    deferreds: Sequence of deferred objects to render and splice in.

  Yields:
    HTML source for the rendered node, followed by logic to substitute each
    deferred object.
  """
  all_css_styles = set()
  all_js_defns = set()

  def _render_one(
      node,
      at_beginning_of_line: bool,
      render_context: dict[Any, Any],
      stream: io.StringIO,
  ):
    # Extract setup rules.
    setup_parts = node.html_setup_parts(SETUP_CONTEXT)
    current_styles = []
    current_js_defns = []
    for part in setup_parts:
      if isinstance(part, CSSStyleRule):
        if part not in all_css_styles:
          current_styles.append(part)
          all_css_styles.add(part)
      elif isinstance(part, JavaScriptDefn):
        if part not in all_js_defns:
          current_js_defns.append(part)
          all_js_defns.add(part)
      else:
        raise ValueError(f"Invalid setup object: {part}")

    if current_styles:
      stream.write("<style>")
      for css_style in sorted(current_styles):
        stream.write(css_style.rule)
      stream.write("</style>")

    if current_js_defns:
      stream.write("<script>")
      for js_defn in sorted(current_js_defns):
        stream.write(js_defn.source)
      stream.write("</script>")

    # Render the node itself.
    node.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )

  # Set up the styles and scripts for the root object.
  stream = io.StringIO()
  stream.write("<style>")
  stream.write(html_escaping.without_repeated_whitespace("""
    .treescope_root {
      position: relative;
      font-family: monospace;
      white-space: pre;
      list-style-type: none;
      background-color: white;
      color: black;
      width: fit-content;
      margin-left: 2ch;
      line-height: 1.5;
    }
  """))
  stream.write("</style>")
  # These scripts allow us to defer execution of javascript blocks until after
  # the content is loaded, avoiding locking up the browser rendering process.
  stream.write(html_escaping.without_repeated_whitespace("""
  <script>
    if (window.treescope === undefined) {
      window.treescope = {};
      window.treescope._pendingActions = [];
      window.treescope._pendingActionHandle = null;
      window.treescope.runSoon = (work) => {
          const doWork = () => {
              const tick = performance.now();
              while (performance.now() - tick < 32) {
                if (window.treescope._pendingActions.length == 0) {
                    window.treescope._pendingActionHandle = null;
                    return;
                } else {
                    const thunk = window.treescope._pendingActions.shift();
                    thunk();
                }
              }
              window.treescope._pendingActionHandle = (
                  window.requestAnimationFrame(doWork));
          };
          window.treescope._pendingActions.push(work);
          if (window.treescope._pendingActionHandle === null) {
              window.treescope._pendingActionHandle = (
                  window.requestAnimationFrame(doWork));
          }
      };
      window.treescope.enqueueDeferredTemplates = () => {
          const tpls = document.querySelectorAll("template.treescope_run_soon");
          for (let tpl of tpls) {
              tpl.classList.remove("treescope_run_soon");
              window.treescope.runSoon(() => {
                  let frozen = tpl.content.querySelector("script");
                  let scr = document.createElement("script");
                  scr.type = frozen.type;
                  scr.textContent = frozen.textContent;
                  tpl.parentNode.replaceChild(scr, tpl);
              });
          }
      };
      window.treescope.toggle_root_roundtrip = (rootelt, event) => {
          if (event.key == "r") {
              rootelt.classList.toggle("roundtrip_mode");
          }
      };
    }
  </script>
  """))

  # Render the root node.
  classnames = "treescope_root"
  if roundtrip:
    classnames += " roundtrip_mode"
  stream.write(
      f'<div class="{classnames}" tabindex="0"'
      ' onkeydown="treescope.toggle_root_roundtrip(this, event)">'
  )
  _render_one(root_node, True, {}, stream)
  stream.write("</div>")
  stream.write("<script>window.treescope.enqueueDeferredTemplates()</script>")

  yield stream.getvalue()

  # Render any deferred parts. We insert each part into a hidden element, then
  # move them all out to their appropriate positions.
  if deferreds:
    stream = io.StringIO()
    for deferred in deferreds:
      stream.write(
          '<div style="display: none"'
          f' id="for_{deferred.placeholder.replacement_id}"><span>'
      )
      if (
          deferred.placeholder.saved_at_beginning_of_line is None
          or deferred.placeholder.saved_render_context is None
      ):
        replacement_part = common_styles.ErrorColor(
            basic_parts.Text("<deferred rendering error>")
        )
      else:
        replacement_part = deferred.thunk(deferred.placeholder.child)
      _render_one(
          replacement_part,
          deferred.placeholder.saved_at_beginning_of_line,
          deferred.placeholder.saved_render_context,
          stream,
      )
      stream.write("</span></div>")

    stream.write("<script>(() => {")
    all_ids = [deferred.placeholder.replacement_id for deferred in deferreds]
    stream.write(f"const targetIds = {json.dumps(all_ids)};")
    stream.write(html_escaping.without_repeated_whitespace("""
        for (let i = 0; i < targetIds.length; i++) {
            let target = document.getElementById(targetIds[i]);
            let sourceDiv = document.getElementById("for_" + targetIds[i]);
            target.parentNode.replaceChild(sourceDiv.firstElementChild, target);
            sourceDiv.remove();
        }
      })();
      </script>
    """))
    stream.write("<script>window.treescope.enqueueDeferredTemplates()</script>")
    yield stream.getvalue()


def render_to_html_as_root(
    root_node: RenderableTreePart,
    roundtrip: bool = False,
    compressed: bool = False,
) -> str:
  """Renders a root node to HTML.

  This handles collecting styles and JS definitions and inserting the root
  HTML element.

  Args:
    root_node: The root node to render.
    roundtrip: Whether to render in roundtrip mode.
    compressed: Whether to compress the HTML for display.

  Returns:
    HTML source for the rendered node.
  """
  html_src = "".join(
      _render_to_html_as_root_streaming(root_node, roundtrip, [])
  )
  if compressed:
    html_src = html_compression.compress_html(
        html_src, include_preamble=True, loading_message="(Loading...)"
    )
  return html_src


def display_streaming_as_root(
    root_node: RenderableTreePart,
    deferreds: Sequence[DeferredWithThunk],
    roundtrip: bool = False,
    compressed: bool = True,
):
  """Displays a root node in an IPython notebook in a streaming fashion.

  Args:
    root_node: The root node to render.
    deferreds: Deferred objects to render and splice in.
    roundtrip: Whether to render in roundtrip mode.
    compressed: Whether to compress the HTML for display.
  """
  import IPython.display  # pylint: disable=g-import-not-at-top

  render_iterator = _render_to_html_as_root_streaming(
      root_node, roundtrip, deferreds
  )
  for i, step in enumerate(render_iterator):
    if compressed:
      if i == 0:
        step = html_compression.compress_html(
            step, include_preamble=True, loading_message="(Loading...)"
        )
      else:
        step = html_compression.compress_html(step, include_preamble=False)
    IPython.display.display(IPython.display.HTML(step))
