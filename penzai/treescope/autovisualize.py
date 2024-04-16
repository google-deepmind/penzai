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

"""User-configurable automatic visualization of subtrees.

This module provides a quick but flexible interface for customizing the
output of Treescope by replacing subtrees by rendered figures. The intended
use case is to make it easy to generate visualizations for arrays, parameters,
or other information, even if they are deeply nested inside larger data
structures.

This module also defines a particular automatic visualizer that uses Arrayviz
to visualize all arrays in a tree, which can be useful for exploratory analysis
of models and data.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Protocol

from penzai.core import context
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import embedded_iframe
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import part_interface


@dataclasses.dataclass
class IPythonVisualization:
  """Used by autovisualizers to replace a subtree with a display object.

  Attributes:
    display_object: An object to render, e.g. a figure.
    replace: Whether to replace the subtree commpletely with this display
      object. If False, the display object will be rendered below the object.
  """

  display_object: embedded_iframe.HasReprHtml
  replace: bool = False


@dataclasses.dataclass
class CustomTreescopeVisualization:
  """Used by autovisualizers to directly insert Treescope content.

  Attributes:
    rendering: A custom treescope rendering which will completely replace the
      subtree.
  """

  rendering: part_interface.RenderableAndLineAnnotations


@dataclasses.dataclass
class ChildAutovisualizer:
  """Used by autovisualizers to switch to a different autovisualizer.

  This can be used to enable conditional logic, e.g. "when rendering an object
  of type Foo, visualize objects of type Bar". Note that the returned child
  autovisualizer is also allowed to close over things from the current call,
  e.g. you can return a lambda function that checks if it is rendering a
  particular child of the original node.

  Attributes:
    autovisualizer: The new autovisualizer to use while rendering this value's
      children.
  """

  autovisualizer: Autovisualizer


class Autovisualizer(Protocol):
  """Protocol for autovisualizers.

  An autovisualizer is a callable object that can be used to customize
  Treescope for rendering purposes. The active autovisualizer is called once
  for each subtree of a rendered PyTree, and can choose to:

  * render it normally (but possibly customize its children),

  * replace the rendering of that PyTree with a custom rich display object,

  * or replace the current autovisualizer with a new one that will be active
    while rendering the children.
  """

  @abc.abstractmethod
  def __call__(
      self, value: Any, path: tuple[Any, ...] | None
  ) -> (
      IPythonVisualization
      | CustomTreescopeVisualization
      | ChildAutovisualizer
      | None
  ):
    """Runs the autovisualizer on a node at a given path.

    Args:
      value: A value being rendered in treescope.
      path: Path to this value from the root, as a JAX keypath. May be None if
        this object isn't part of the root PyTree and so treescope doesn't know
        how to access it.

    Returns:
      A visualization for this subtree, a child autovisualizer to use while
      rendering the node's children, or None to continue rendering the node
      normally using the current autovisualizer.
    """
    raise NotImplementedError()


active_autovisualizer: context.ContextualValue[Autovisualizer] = (
    context.ContextualValue(
        module=__name__,
        qualname="active_autovisualizer",
        initial_value=(lambda value, path: None),
    )
)
"""The active autovisualizer to use when rendering a tree to HTML.

This can be overridden interactively to enable rich visualizations in
treescope. Users are free to set this to an arbitrary renderer of their
choice; a common choice is arrayviz's `ArrayAutovisualizer()`.
"""


def use_autovisualizer_if_present(
    node: Any,
    path: tuple[Any, ...] | None,
    node_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Treescope wrapper hook that runs the active autovisualizer."""
  autoviz = active_autovisualizer.get()
  result = autoviz(node, path)
  if result is None:
    # Continue as normal.
    return NotImplemented

  elif isinstance(result, IPythonVisualization | CustomTreescopeVisualization):
    # We should use this visualization instead.

    # Fallback: Render normally for round-trip mode, and if this wasn't a
    # replaced object.
    with active_autovisualizer.set_scoped(lambda value, path: None):
      ordinary_result = node_renderer(node, path)

    if isinstance(result, IPythonVisualization):
      if isinstance(result.display_object, embedded_iframe.HasReprHtml):
        obj = result.display_object

        def _thunk(_):
          return embedded_iframe.EmbeddedIFrame(
              embedded_html=embedded_iframe.to_html(obj),
              fallback_in_text_mode=common_styles.AbbreviationColor(
                  basic_parts.Text("<rich HTML visualization>")
              ),
          )

        ipy_rendering = foldable_impl.maybe_defer_rendering(
            _thunk,
            lambda: basic_parts.Text("<rich HTML visualization loading...>"),
        )
      else:
        # Bad display object
        ipy_rendering = common_structures.build_one_line_tree_node(
            line=common_styles.ErrorColor(
                "<Autovisualizer returned a Visualization with an invalid"
                f" display object {result.display_object}>"
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
            )
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
    with active_autovisualizer.set_scoped(result.autovisualizer):
      return node_renderer(node, path)

  else:
    return common_structures.build_one_line_tree_node(
        line=common_styles.ErrorColor(
            f"<Autovizualizer returned an invalid value {result}; expected"
            " IPythonVisualization, CustomTreescopeVisualization,"
            " ChildAutovisualizer, or None>"
        ),
        path=path,
    )
