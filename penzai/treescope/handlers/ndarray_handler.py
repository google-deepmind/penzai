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

"""Handler for NDArrays."""

from typing import Any

import jax
import numpy as np
from penzai.treescope import canonical_aliases
from penzai.treescope import copypaste_fallback
from penzai.treescope import ndarray_summarization
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import part_interface


def handle_ndarrays(
    node: Any,
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a NDArray."""
  del subtree_renderer
  if not isinstance(node, (np.ndarray, jax.Array)):
    return NotImplemented

  if isinstance(node, jax.core.Tracer):
    return NotImplemented

  # What to call it?
  if isinstance(node, np.ndarray):
    np_name = canonical_aliases.maybe_local_module_name(np)
    prefix = f"{np_name}."
    short = f"{np_name}.ndarray"
  else:
    jax_name = canonical_aliases.maybe_local_module_name(jax)
    prefix = f"{jax_name}."
    short = f"{jax_name}.Array"

    if node.is_deleted():
      return common_styles.ErrorColor(
          basic_parts.Text(
              f"<{short} {ndarray_summarization.get_dtype_name(node.dtype)}{repr(node.shape)} -"
              " deleted!>"
          )
      )

  def _placeholder() -> part_interface.RenderableTreePart:
    short_summary = (
        f"<{short} {ndarray_summarization.get_dtype_name(node.dtype)}{repr(node.shape)} ... >"
    )
    return common_structures.fake_placeholder_foldable(
        common_styles.DeferredPlaceholderStyle(basic_parts.Text(short_summary)),
        extra_newlines_guess=8,
    )

  def _thunk(placeholder):
    # Is this array simple enough to render without a summary?
    node_repr = ndarray_summarization.faster_array_repr(node)
    if "\n" not in node_repr and "..." not in node_repr:
      rendering = common_styles.AbbreviationColor(
          basic_parts.Text(f"<{prefix}{node_repr}>")
      )
      repr_summary = node_repr
    else:
      if node_repr.count("\n") <= 15:
        if isinstance(placeholder, part_interface.FoldableTreeNode):
          default_expand_state = placeholder.get_expand_state()
        else:
          assert placeholder is None
          default_expand_state = part_interface.ExpandState.WEAKLY_EXPANDED
      else:
        # Always start big NDArrays in collapsed mode to hide irrelevant detail.
        default_expand_state = part_interface.ExpandState.COLLAPSED

      # Render it with a summary.
      summarized = ndarray_summarization.summarize_ndarray(node)
      repr_summary = f"<{short} {summarized}>"
      rendering = common_structures.build_custom_foldable_tree_node(
          label=common_styles.AbbreviationColor(
              common_styles.CommentColorWhenExpanded(
                  basic_parts.siblings(
                      basic_parts.FoldCondition(
                          expanded=basic_parts.Text("# "),
                          collapsed=basic_parts.Text("<"),
                      ),
                      f"{short} " + summarized,
                      basic_parts.FoldCondition(
                          collapsed=basic_parts.Text(">")
                      ),
                  )
              )
          ),
          contents=basic_parts.FoldCondition(
              expanded=basic_parts.IndentedChildren.build(
                  [basic_parts.Text(node_repr)]
              )
          ),
          path=path,
          expand_state=default_expand_state,
      ).renderable

    fallback_rendering = copypaste_fallback.render_not_roundtrippable(
        node, repr_override=repr_summary
    )
    return basic_parts.RoundtripCondition(
        roundtrip=fallback_rendering,
        not_roundtrip=rendering,
    )

  return basic_parts.RenderableAndLineAnnotations(
      renderable=foldable_impl.maybe_defer_rendering(
          main_thunk=_thunk, placeholder_thunk=_placeholder
      ),
      annotations=common_structures.build_copy_button(path),
  )


def handle_dtype_instances(
    node: Any,
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a np.dtype, adding the `np.` qualifier."""
  del subtree_renderer
  if not isinstance(node, np.dtype):
    return NotImplemented

  dtype_name = node.name
  if dtype_name in np.sctypeDict and node is np.dtype(
      np.sctypeDict[dtype_name]
  ):
    # Use the named type. (Sometimes extended dtypes don't print in a
    # roundtrippable way otherwise.)
    dtype_string = f"dtype({repr(dtype_name)})"
  else:
    # Hope that `repr` is already round-trippable (true for builtin numpy types)
    # and add the "numpy." prefix as needed.
    dtype_string = repr(node)

  # Use an alias for numpy if one is defined, since people often import numpy
  # as np.
  np_name = canonical_aliases.maybe_local_module_name(np)

  return common_structures.build_one_line_tree_node(
      line=basic_parts.siblings(
          basic_parts.RoundtripCondition(
              roundtrip=basic_parts.Text(f"{np_name}.")
          ),
          dtype_string,
      ),
      path=path,
  )
