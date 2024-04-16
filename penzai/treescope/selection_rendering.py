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

"""Rendering logic for selections and selected objects in treescope."""
from __future__ import annotations

import dataclasses
from typing import Any
import warnings

from penzai.core import context
from penzai.core import selectors
from penzai.treescope import default_renderer
from penzai.treescope import html_escaping
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import layout_algorithms
from penzai.treescope.foldable_representation import part_interface


@dataclasses.dataclass
class SelectedNodeTracker:
  """Tracks the set of selected nodes, and makes sure we render them  correctly.

  Attributes:
    selected_node_paths: The set of key paths for selected nodes.
    rendered_node_paths: The set of key paths we have rendered.
    visible_boundary: Whether we should actually visualize the boundary. If
      False, inserts boundary tags for layout purposes but does not actually
      modify the rendered HTML tree.
  """

  selected_node_paths: set[tuple[Any, ...]]
  rendered_node_paths: set[tuple[Any, ...]]
  visible_boundary: bool


_selected_nodes: context.ContextualValue[SelectedNodeTracker | None] = (
    context.ContextualValue(
        module=__name__,
        qualname="_selected_nodes",
        initial_value=None,
    )
)
"""The active tracker for the selected nodes we are rendering.

We use this to postprocess renderings of selections to insert markers around the
selected nodes.

This context manager is accessed only inside `render_selection_to_html`
and the selection wrapper.
"""


@dataclasses.dataclass(frozen=True)
class SelectionBoundaryTag:
  """A tag that can be used to identify selected nodes."""


class SelectionTaggedGroup(basic_parts.BaseTaggedGroup):
  """Tags its child as being selected."""

  def _tags(self) -> frozenset[Any]:
    return frozenset({SelectionBoundaryTag()})


class SelectionBoundaryBox(basic_parts.BaseBoxWithOutline):
  """A highlighted box that identifies a part as being selected."""

  def _box_css_class(self) -> str:
    return "selection_boundary"

  def _box_css_rule(
      self, setup_context: part_interface.HtmlContextForSetup, /
  ) -> part_interface.CSSStyleRule:
    return part_interface.CSSStyleRule(
        html_escaping.without_repeated_whitespace("""
            .selection_boundary
            {
                border: 2px solid cyan;
                border-left: 2ch solid cyan;
            }
        """)
    )


class SelectionBoundaryLabel(basic_parts.BaseSpanGroup):
  """A comment identifying this part as being selected."""

  def _span_css_class(self) -> str:
    return "selection_label"

  def _span_css_rule(
      self, setup_context: part_interface.HtmlContextForSetup
  ) -> part_interface.CSSStyleRule:
    return part_interface.CSSStyleRule(
        html_escaping.without_repeated_whitespace("""
            .selection_label
            {
                color: darkcyan;
                font-weight: bold;
            }
        """)
    )


def _wrap_selected_nodes(
    node: Any,
    path: tuple[Any, ...] | None,
    node_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Custom wrapper hook that intercepts selected nodes."""
  tracker = _selected_nodes.get()

  assert tracker is not None
  if path is not None and path in tracker.selected_node_paths:
    if path in tracker.rendered_node_paths:
      warnings.warn(
          "Rendering a single selected node multiple times! This likely means"
          " the current treescope handlers are not assigning keypaths"
          " consistently with JAX for some custom PyTree type.\nRepeated"
          f" keypath: {path}"
      )
    else:
      tracker.rendered_node_paths.add(path)

    # Tag the child, and possibly annotate its visualization.
    rendering = node_renderer(node, path)

    tagged_rendering = part_interface.RenderableAndLineAnnotations(
        renderable=SelectionTaggedGroup(rendering.renderable),
        annotations=rendering.annotations,
    )

    if tracker.visible_boundary:
      wrapped_rendering = basic_parts.siblings_with_annotations(
          common_structures.build_custom_foldable_tree_node(
              contents=SelectionBoundaryBox(
                  basic_parts.OnSeparateLines.build([
                      basic_parts.FoldCondition(
                          expanded=SelectionBoundaryLabel(
                              basic_parts.Text("# Selected:")
                          )
                      ),
                      tagged_rendering.renderable,
                  ])
              )
          ),
          extra_annotations=[tagged_rendering.annotations],
      )
    else:
      wrapped_rendering = tagged_rendering

    return wrapped_rendering

  return NotImplemented


def render_selection_to_foldable_representation(
    selection: selectors.Selection,
    visible_selection: bool = True,
    ignore_exceptions: bool = False,
) -> part_interface.RenderableTreePart:
  """Renders a top-level selection object to its foldable representation.

  This function produces a rendering of either the selection
  itself, or its selected value. However, instead of directly outputting the
  attributes of the selection, we draw the selected regions as highlighted boxes
  around selected subtrees.

  The output is still a round-trippable "repr"-like output. When
  `visible_selection` is True, copying it will reproduce the original selection.
  When `visible_selection` is False, copying it will reproduce the selected
  value.

  Args:
    selection: A selection to render.
    visible_selection: Whether to render the selection itself rather than the
      selected value. If True, renders code to build the selection itself, and
      highlights selected nodes in blue. If False, renders the value but expands
      the selected nodes.
    ignore_exceptions: Whether to catch errors during rendering of subtrees and
      show a fallback for those subtrees, instead of failing the entire
      renderer. Best used in contexts where
      `render_selection_to_foldable_representation` is not being called
      directly, e.g. when registering this as a default pretty-printer.

  Returns:
    A representation of the selection, expanded accordingly.
  """
  # Fetch the default renderer, and add selection-specific handlers.
  base_renderer = default_renderer.active_renderer.get()
  extended_renderer = base_renderer.extended_with(
      wrapper_hooks=[_wrap_selected_nodes],
  )
  # Enter a scope where we know which values to select, and render with our
  # extended renderer.
  tracker = SelectedNodeTracker(
      selected_node_paths=set(selection.selected_by_path.keys()),
      rendered_node_paths=set(),
      visible_boundary=visible_selection,
  )
  with _selected_nodes.set_scoped(tracker):
    rendered_ir = basic_parts.build_full_line_with_annotations(
        extended_renderer.to_foldable_representation(
            selection.deselect(),
            ignore_exceptions=ignore_exceptions,
        )
    )

  unrendered_node_paths = {
      path
      for path in tracker.selected_node_paths
      if path not in tracker.rendered_node_paths
  }
  if unrendered_node_paths:
    warnings.warn(
        "Some selected nodes were not rendered! This likely means the current"
        " treescope handlers are not assigning keypaths consistently with JAX"
        " for some custom PyTree type.\nMissing keypaths:"
        f" {unrendered_node_paths}"
    )

  # Expand the selected nodes.
  layout_algorithms.expand_to_tags(
      rendered_ir, tags=[SelectionBoundaryTag()], collapse_weak_others=True
  )

  # For convenience, make a default balanced layout for the remaining nodes.
  layout_algorithms.expand_for_balanced_layout(rendered_ir)

  if visible_selection:
    # Render the keypaths:
    keypath_rendering = basic_parts.build_full_line_with_annotations(
        base_renderer.to_foldable_representation(
            tuple(key for key in selection.selected_by_path.keys()),
            ignore_exceptions=ignore_exceptions,
            root_keypath=None,
        )
    )
    layout_algorithms.expand_for_balanced_layout(keypath_rendering)

    # Combine everything into a rendering of the selection itself.
    count = len(selection)
    return basic_parts.Siblings.build(
        common_styles.CommentColor(basic_parts.Text("pz.select(")),
        basic_parts.IndentedChildren.build([rendered_ir]),
        common_styles.CommentColor(basic_parts.Text(").at_keypaths(")),
        common_structures.build_custom_foldable_tree_node(
            label=common_styles.CommentColor(
                basic_parts.FoldCondition(
                    collapsed=basic_parts.Text(
                        f"<{count} subtrees, highlighted above>"
                    ),
                    expanded=basic_parts.Text(
                        f"# {count} subtrees, highlighted above"
                    ),
                )
            ),
            contents=basic_parts.FoldCondition(
                expanded=basic_parts.IndentedChildren.build([keypath_rendering])
            ),
            expand_state=part_interface.ExpandState.COLLAPSED,
        ).renderable,
        common_styles.CommentColor(basic_parts.Text(")")),
    )
  else:
    # Just return our existing rendering.
    return rendered_ir


def display_selection_streaming(
    selection: selectors.Selection,
    visible_selection: bool = True,
    ignore_exceptions: bool = False,
):
  """Displays a top-level selection object in IPython.

  This function produces a rendering of either the selection
  itself, or its selected value. However, instead of directly outputting the
  attributes of the selection, we draw the selected regions as highlighted boxes
  around selected subtrees.

  The output is still a round-trippable "repr"-like output. When
  ``visible_selection`` is True, copying it will reproduce the original
  selection. When ``visible_selection`` is False, copying it will reproduce the
  selected value.

  This function is used to implement `.show_selection()` and `.show_value()` on
  selection objects.

  Args:
    selection: A selection to render.
    visible_selection: Whether to render the selection itself rather than the
      selected value. If True, renders code to build the selection itself, and
      highlights selected nodes in blue. If False, renders the value but expands
      the selected nodes.
    ignore_exceptions: Whether to catch errors during rendering of subtrees and
      show a fallback for those subtrees, instead of failing the entire
      renderer. Best used in contexts where `render_selection_to_html` is not
      being called directly, e.g. when registering this as a default
      pretty-printer.
  """
  with foldable_impl.collecting_deferred_renderings() as deferreds:
    rendered_ir = render_selection_to_foldable_representation(
        selection,
        visible_selection=visible_selection,
        ignore_exceptions=ignore_exceptions,
    )
  foldable_impl.display_streaming_as_root(
      rendered_ir, deferreds, roundtrip=False
  )
