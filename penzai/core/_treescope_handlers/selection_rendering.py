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

import jax
from penzai.core import selectors
import treescope
from treescope import context
from treescope import layout_algorithms
from treescope import lowering
from treescope import renderers
from treescope import rendering_parts


@dataclasses.dataclass
class SelectedNodeTracker:
  """Tracks the set of selected nodes, and makes sure we render them  correctly.

  Attributes:
    selected_node_paths: The set of key paths for selected nodes.
    rendered_node_paths: The set of key paths we have rendered.
    repeated_node_paths: The set of key paths we have rendered more than once;
      this should usually be empty, and we emit a warning if it is not.
    visible_boundary: Whether we should actually visualize the boundary. If
      False, inserts boundary tags for layout purposes but does not actually
      modify the rendered HTML tree.
  """

  selected_node_paths: set[str]
  rendered_node_paths: set[str]
  repeated_node_paths: set[str]
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


def is_rendering_a_selection() -> bool:
  """Returns whether we are currently rendering a selection."""
  return _selected_nodes.get() is not None


@dataclasses.dataclass(frozen=True)
class SelectionBoundaryLayoutMark:
  """A layout mark that can be used to identify selected nodes."""


def _wrap_selected_nodes(
    node: Any,
    path: str | None,
    node_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Custom wrapper hook that intercepts selected nodes."""
  tracker = _selected_nodes.get()

  assert tracker is not None
  if path is not None and path in tracker.selected_node_paths:
    if path in tracker.rendered_node_paths:
      tracker.repeated_node_paths.add(path)
    else:
      tracker.rendered_node_paths.add(path)

    # Tag the child, and possibly annotate its visualization.
    rendering = node_renderer(node, path)

    tagged_rendering = rendering_parts.RenderableAndLineAnnotations(
        renderable=rendering_parts.with_layout_mark(
            rendering.renderable, SelectionBoundaryLayoutMark()
        ),
        annotations=rendering.annotations,
    )

    if tracker.visible_boundary:
      wrapped_rendering = rendering_parts.siblings_with_annotations(
          rendering_parts.build_custom_foldable_tree_node(
              contents=rendering_parts.in_outlined_box(
                  rendering_parts.on_separate_lines([
                      rendering_parts.fold_condition(
                          expanded=rendering_parts.custom_style(
                              rendering_parts.text("# Selected:"),
                              css_style="color: darkcyan; font-weight: bold;",
                          )
                      ),
                      tagged_rendering.renderable,
                  ]),
                  css_style=(
                      "border: 2px solid cyan; border-left: 2ch solid cyan;"
                  ),
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
) -> rendering_parts.RenderableTreePart:
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
  base_renderer = treescope.active_renderer.get()
  extended_renderer = base_renderer.extended_with(
      wrapper_hooks=[_wrap_selected_nodes],
  )
  # Enter a scope where we know which values to select, and render with our
  # extended renderer.
  tracker = SelectedNodeTracker(
      selected_node_paths=set(
          jax.tree_util.keystr(keypath)
          for keypath in selection.selected_by_path.keys()
      ),
      rendered_node_paths=set(),
      repeated_node_paths=set(),
      visible_boundary=visible_selection,
  )
  with _selected_nodes.set_scoped(tracker):
    rendered_ir = rendering_parts.build_full_line_with_annotations(
        extended_renderer.to_foldable_representation(
            selection.deselect(),
            ignore_exceptions=ignore_exceptions,
        )
    )

  warnings = []

  unrendered_node_paths = {
      path
      for path in tracker.selected_node_paths
      if path not in tracker.rendered_node_paths
  }
  if unrendered_node_paths:
    warnings.extend([
        "# Some selected nodes were not rendered!",
        "# This likely means the current treescope handlers are not assigning ",
        "# keypaths consistently with JAX for some custom PyTree type.",
        f"# Missing keypaths: {unrendered_node_paths}",
    ])

  if tracker.repeated_node_paths:
    warnings.extend([
        "# Rendered a single selected node multiple times!",
        "# This likely means the current treescope handlers are not assigning ",
        "# keypaths consistently with JAX for some custom PyTree type.",
        f"# Repeated keypaths: {tracker.repeated_node_paths}",
    ])

  # Expand the selected nodes.
  layout_algorithms.expand_to_layout_marks(
      rendered_ir,
      marks=[SelectionBoundaryLayoutMark()],
      collapse_weak_others=True,
  )

  # For convenience, make a default balanced layout for the remaining nodes.
  layout_algorithms.expand_for_balanced_layout(rendered_ir)

  if visible_selection:
    # Render the keypaths:
    keypath_rendering = rendering_parts.build_full_line_with_annotations(
        base_renderer.to_foldable_representation(
            tuple(key for key in selection.selected_by_path.keys()),
            ignore_exceptions=ignore_exceptions,
            root_keypath=None,
        )
    )
    layout_algorithms.expand_for_balanced_layout(keypath_rendering)

    # Combine everything into a rendering of the selection itself.
    count = len(selection)
    result = rendering_parts.siblings(
        rendering_parts.comment_color(rendering_parts.text("pz.select(")),
        rendering_parts.indented_children([rendered_ir]),
        rendering_parts.comment_color(rendering_parts.text(").at_keypaths(")),
        rendering_parts.build_custom_foldable_tree_node(
            label=rendering_parts.comment_color(
                rendering_parts.fold_condition(
                    collapsed=rendering_parts.text(
                        f"<{count} subtrees, highlighted above>"
                    ),
                    expanded=rendering_parts.text(
                        f"# {count} subtrees, highlighted above"
                    ),
                )
            ),
            contents=rendering_parts.fold_condition(
                expanded=rendering_parts.indented_children([keypath_rendering])
            ),
            expand_state=rendering_parts.ExpandState.COLLAPSED,
        ).renderable,
        rendering_parts.comment_color(rendering_parts.text(")")),
    )
  else:
    # Just return our existing rendering.
    result = rendered_ir

  if warnings:
    result = rendering_parts.on_separate_lines([
        rendering_parts.error_color(
            rendering_parts.on_separate_lines(warnings)
        ),
        result,
    ])

  return result


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
  with lowering.collecting_deferred_renderings() as deferreds:
    rendered_ir = render_selection_to_foldable_representation(
        selection,
        visible_selection=visible_selection,
        ignore_exceptions=ignore_exceptions,
    )
  lowering.display_streaming_as_root(rendered_ir, deferreds, roundtrip=False)
