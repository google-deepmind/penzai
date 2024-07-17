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

"""Handler for user-specified autovisualizers."""

from __future__ import annotations

from typing import Any

from penzai.treescope import autovisualize
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import embedded_iframe
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import part_interface

IPythonVisualization = autovisualize.IPythonVisualization
CustomTreescopeVisualization = autovisualize.CustomTreescopeVisualization
ChildAutovisualizer = autovisualize.ChildAutovisualizer


def use_autovisualizer_if_present(
    node: Any,
    path: str | None,
    node_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Treescope wrapper hook that runs the active autovisualizer."""
  autoviz = autovisualize.active_autovisualizer.get()
  result = autoviz(node, path)
  if result is None:
    # Continue as normal.
    return NotImplemented

  elif isinstance(result, IPythonVisualization | CustomTreescopeVisualization):
    # We should use this visualization instead.

    # Fallback: Render normally for round-trip mode, and if this wasn't a
    # replaced object.
    with autovisualize.active_autovisualizer.set_scoped(
        lambda value, path: None
    ):
      ordinary_result = node_renderer(node, path)

    if isinstance(result, IPythonVisualization):
      if isinstance(result.display_object, embedded_iframe.HasReprHtml):
        obj = result.display_object

        def _thunk(_):
          html_rendering = embedded_iframe.to_html(obj)
          if html_rendering:
            return embedded_iframe.EmbeddedIFrame(
                embedded_html=html_rendering,
                fallback_in_text_mode=common_styles.AbbreviationColor(
                    basic_parts.Text("<rich HTML visualization>")
                ),
            )
          else:
            return common_styles.ErrorColor(
                basic_parts.Text(
                    "<Autovisualizer returned a Visualization with an invalid"
                    f" display object {result.display_object}>"
                )
            )

        ipy_rendering = foldable_impl.maybe_defer_rendering(
            _thunk,
            lambda: basic_parts.Text("<rich HTML visualization loading...>"),
        )
      else:
        # Bad display object
        ipy_rendering = common_structures.build_one_line_tree_node(
            line=common_styles.ErrorColor(
                basic_parts.Text(
                    "<Autovisualizer returned a Visualization with an invalid"
                    f" display object {result.display_object}>"
                )
            ),
            path=path,
        )
      if result.replace:
        replace = True
        rendering_and_annotations = (
            common_structures.build_custom_foldable_tree_node(
                label=common_styles.AbbreviationColor(
                    basic_parts.Text(f"<Visualization of {type(node).__name__}")
                ),
                contents=basic_parts.siblings(
                    basic_parts.FoldCondition(
                        expanded=basic_parts.siblings(
                            common_styles.AbbreviationColor(
                                basic_parts.Text(":")
                            ),
                            basic_parts.IndentedChildren.build([ipy_rendering]),
                        )
                    ),
                    common_styles.AbbreviationColor(basic_parts.Text(">")),
                ),
                path=path,
                expand_state=part_interface.ExpandState.EXPANDED,
            )
        )
      else:
        replace = False
        rendering_and_annotations = part_interface.RenderableAndLineAnnotations(
            renderable=basic_parts.ScopedSelectableAnnotation(
                common_styles.DashedGrayOutlineBox(ipy_rendering)
            ),
            annotations=basic_parts.EmptyPart(),
        )
    else:
      assert isinstance(result, CustomTreescopeVisualization)
      replace = True
      rendering_and_annotations = result.rendering

    if replace:
      in_roundtrip_with_annotations = basic_parts.siblings_with_annotations(
          ordinary_result,
          extra_annotations=[
              common_styles.CommentColor(
                  basic_parts.Text("  # Visualization hidden in roundtrip mode")
              )
          ],
      )
      return part_interface.RenderableAndLineAnnotations(
          renderable=basic_parts.RoundtripCondition(
              roundtrip=in_roundtrip_with_annotations.renderable,
              not_roundtrip=rendering_and_annotations.renderable,
          ),
          annotations=basic_parts.RoundtripCondition(
              roundtrip=in_roundtrip_with_annotations.annotations,
              not_roundtrip=rendering_and_annotations.annotations,
          ),
      )
    else:
      return basic_parts.siblings_with_annotations(
          ordinary_result, rendering_and_annotations
      )

  elif isinstance(result, ChildAutovisualizer):
    # Use a different autovisualizer while rendering this object.
    with autovisualize.active_autovisualizer.set_scoped(result.autovisualizer):
      return node_renderer(node, path)

  else:
    return common_structures.build_one_line_tree_node(
        line=common_styles.ErrorColor(
            basic_parts.Text(
                f"<Autovizualizer returned an invalid value {result}; expected"
                " IPythonVisualization, CustomTreescopeVisualization,"
                " ChildAutovisualizer, or None>"
            )
        ),
        path=path,
    )
