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

"""Handlers for certain "well-known" objects, such as module functions.

Taking the `repr` of a function or callable usually produces something like

  <function vmap at 0x7f98bf556440>

and in some cases produces something like

  <jax.custom_derivatives.custom_jvp object at 0x7f98c0b5f130>

This can make it hard to determine what object this actually is from a user
perspective, and it would likely be more user-friendly to just output `jax.vmap`
or `jax.nn.relu` as the representation of this object.

See `canonical_aliases` for details of how we determine aliases for a subset
of common functions.
"""

from __future__ import annotations

from typing import Any

from penzai.treescope import canonical_aliases
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import part_interface


def replace_with_canonical_aliases(
    node: Any,
    path: tuple[Any, ...] | None,
    node_renderer: renderer.TreescopeSubtreeRenderer,
    summarization_threshold: int = 20,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Rewrites objects to use well-known aliases when known.

  This rendering hook checks if the object is a well-known object exported by
  a public API, and if so, makes its short representation be its path in that
  API rather than its repr.

  Objects are looked up in canonical_aliases to determine if there is a
  specific alias registered for them. If not, but the object defines
  `__module__` and `__qualname__` attributes, these are used to determine the
  canonical alias instead.

  For objects whose fully-qualified name is longer than
  `summarization_threshold`, we shorten it when collapsed or not in roundtrip
  mode.

  Args:
    node: The node that has been rendered
    path: Optionally, a path to this node as a string.
    node_renderer: The inner renderer for this node. This should be used to
      render `node` itself into HTML tags.
    summarization_threshold: Threshold at which to omit the fully-qualified name
      except in roundtrip mode.

  Returns:
    A possibly-modified representation of this object, or NotImplemented.
  """
  maybe_alias = canonical_aliases.lookup_alias(
      node, infer_from_attributes=True, allow_relative=True
  )
  if not maybe_alias:
    # No alias.
    return NotImplemented

  qualified_name = str(maybe_alias)
  if "." not in qualified_name:
    name_rendering = qualified_name
  else:
    prefix, suffix = qualified_name.rsplit(".", 1)
    if len(qualified_name) > summarization_threshold:
      name_rendering = basic_parts.siblings(
          basic_parts.SummarizableCondition(
              detail=common_styles.QualifiedTypeNameSpanGroup(
                  basic_parts.Text(prefix + ".")
              )
          ),
          suffix,
      )
    else:
      name_rendering = basic_parts.siblings(
          common_styles.QualifiedTypeNameSpanGroup(
              basic_parts.Text(prefix + ".")
          ),
          suffix,
      )

  original_rendering = node_renderer(node, path=path)

  return common_structures.build_custom_foldable_tree_node(
      label=common_styles.CommentColorWhenExpanded(
          basic_parts.siblings(
              basic_parts.FoldCondition(expanded=basic_parts.Text("# ")),
              name_rendering,
          )
      ),
      contents=basic_parts.FoldCondition(
          expanded=basic_parts.IndentedChildren.build([original_rendering])
      ),
      # Aliases should start collapsed regardless of layout.
      expand_state=basic_parts.ExpandState.COLLAPSED,
      path=path,
  )
