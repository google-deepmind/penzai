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

"""Handler for any object with a _repr_html_.

`_repr_html_` is a method standardized by IPython, which is used to register
a pretty HTML representation of an object. This means we can support rendering
any rich object with a custom notebook rendering strategy.

See
https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display
for details on the purpose of the `_repr_html_` method.
"""

from __future__ import annotations

from typing import Any

from penzai.core import context
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import embedded_iframe
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import part_interface


_already_processing_repr_html: context.ContextualValue[bool] = (
    context.ContextualValue(
        module=__name__,
        qualname="_already_processing_repr_html",
        initial_value=False,
    )
)
"""Tracks whether we are already rendering this subtree in some parent.

This is used to prevent situations where a node defines ``_repr_html_`` but
also includes children that define ``_repr_html_``. We only want to use the
rich HTML representation of the outermost element.
"""


def append_repr_html_when_present(
    node: Any,
    path: tuple[Any, ...] | None,
    node_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Appends rich HTML representations of objects that have them."""
  if _already_processing_repr_html.get():
    # We've processed the repr_html for a parent of this node already.
    return NotImplemented

  if not isinstance(node, embedded_iframe.HasReprHtml):
    return NotImplemented

  # Make sure we don't try to call _repr_html_ on the children of this node,
  # since this node's _repr_html_ probably handles that already.
  with _already_processing_repr_html.set_scoped(True):
    # In "auto" mode, collapse the plaintext repr of objects with a _repr_html_,
    # and expand the HTML view instead.
    node_rendering = node_renderer(node, path=path)

  def _thunk(_):
    return embedded_iframe.EmbeddedIFrame(
        embedded_html=embedded_iframe.to_html(node),
        fallback_in_text_mode=common_styles.AbbreviationColor(
            basic_parts.Text("# (not shown in text mode)")
        ),
    )

  iframe_rendering = foldable_impl.maybe_defer_rendering(
      main_thunk=_thunk,
      placeholder_thunk=lambda: common_styles.DeferredPlaceholderStyle(
          basic_parts.Text("...")
      ),
  )

  boxed_html_repr = basic_parts.IndentedChildren.build([
      basic_parts.ScopedSelectableAnnotation(
          common_styles.DashedGrayOutlineBox(
              basic_parts.build_full_line_with_annotations(
                  common_structures.build_custom_foldable_tree_node(
                      label=common_styles.CommentColor(
                          basic_parts.Text("# Rich HTML representation")
                      ),
                      contents=basic_parts.FoldCondition(
                          expanded=iframe_rendering
                      ),
                  )
              ),
          )
      )
  ])
  return basic_parts.siblings_with_annotations(
      node_rendering,
      basic_parts.FoldCondition(
          expanded=basic_parts.RoundtripCondition(not_roundtrip=boxed_html_repr)
      ),
  )
