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

"""Handle shapecheck structures and variables."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from penzai.core import shapecheck
from treescope import dtype_util
from treescope import renderers
from treescope import rendering_parts


def _wrap_dimvar(msg: str) -> rendering_parts.RenderableTreePart:
  return rendering_parts.custom_style(
      rendering_parts.text(msg), css_style="color: #1bb2c3; font-style: italic;"
  )


def _arraystructure_summary(
    structure: shapecheck.ArraySpec,
) -> rendering_parts.RenderableTreePart:
  """Creates a summary line for an array structure."""

  # Give a short summary for our named arrays.
  summary_parts = []
  if structure.dtype is np.generic:
    summary_parts.append("any")
  else:
    summary_parts.append(dtype_util.get_dtype_name(structure.dtype))
  summary_parts.append("(")
  for i, dim in enumerate(structure.shape):
    if i:
      summary_parts.append(", ")
    if isinstance(dim, shapecheck.DimVar):
      if isinstance(dim.name, tuple):
        summary_parts.append(_wrap_dimvar(f"{{{dim.name[0]}/{dim.name[1]}}}"))
      else:
        summary_parts.append(_wrap_dimvar(f"{{{dim.name}}}"))
    elif isinstance(dim, shapecheck.KnownDim):
      summary_parts.append(f"{dim.size}")
    elif isinstance(dim, shapecheck.MultiDimVar):
      summary_parts.append(_wrap_dimvar(f"*{{{dim.name}…}}"))
    else:
      summary_parts.append(f"{dim}")

  if structure.named_shape:
    if structure.shape:
      summary_parts.append(" |")
    else:
      summary_parts.append("|")

  for i, (name, dim) in enumerate(structure.named_shape.items()):
    if i:
      summary_parts.append(", ")
    else:
      summary_parts.append(" ")

    if isinstance(name, shapecheck.MultiDimVar) and isinstance(
        dim, shapecheck.RemainingAxisPlaceholder
    ):
      summary_parts.append(_wrap_dimvar(f"**{{{name.name}:…}}"))
    elif isinstance(dim, shapecheck.DimVar):
      summary_parts.append(f"{name}:")
      if isinstance(dim.name, tuple):
        summary_parts.append(_wrap_dimvar(f"{{{dim.name[0]}/{dim.name[1]}}}"))
      else:
        summary_parts.append(_wrap_dimvar(f"{{{dim.name}}}"))
    elif isinstance(dim, shapecheck.KnownDim):
      summary_parts.append(f"{name}:{dim.size}")
    else:
      summary_parts.append(f"{name}:{dim}")
  summary_parts.append(")")

  return rendering_parts.siblings(*summary_parts)


def handle_arraystructures(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders ArraySpec and contained variables."""
  if isinstance(node, shapecheck.Wildcard):
    summary = "*" if node.description is None else f"<{node.description}>"
    return rendering_parts.build_one_line_tree_node(
        line=rendering_parts.roundtrip_condition(
            roundtrip=rendering_parts.siblings(
                rendering_parts.maybe_qualified_type_name(type(node)),
                f"({repr(node.description)})",
            ),
            not_roundtrip=_wrap_dimvar(summary),
        ),
        path=path,
    )

  elif isinstance(node, shapecheck.ArraySpec):
    summary = _arraystructure_summary(node)
    children = rendering_parts.build_field_children(
        node,
        path,
        subtree_renderer,
        fields_or_attribute_names=dataclasses.fields(node),
    )
    indented_children = rendering_parts.indented_children(children)

    return rendering_parts.build_custom_foldable_tree_node(
        label=rendering_parts.summarizable_condition(
            summary=rendering_parts.custom_text_color(
                rendering_parts.siblings("<ArraySpec ", summary, ">"), "#1b3e73"
            ),
            detail=rendering_parts.siblings(
                rendering_parts.maybe_qualified_type_name(type(node)),
                "(",
            ),
        ),
        contents=rendering_parts.summarizable_condition(
            detail=rendering_parts.siblings(
                indented_children,
                ")",
            )
        ),
        path=path,
        expand_state=rendering_parts.ExpandState.COLLAPSED,
    )

  else:
    return NotImplemented
