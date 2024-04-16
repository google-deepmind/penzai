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

"""Reflection-based handlers for functions, closures, and classes.

These handlers provide:
- deep-linking to jump to the place a particular object was defined,
- visualization of closure variables for closures
"""

import dataclasses
import functools
import inspect
import io
import re
from typing import Any

from penzai.treescope import html_escaping
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import part_interface


@functools.cache
def _get_filepath_and_lineno(value) -> tuple[str, int] | tuple[None, None]:
  """Retrieves a filepath and line number for an inspectable value."""
  try:
    filepath = inspect.getsourcefile(value)
    if filepath is None:
      return None, None
    # This also returns the source lines, but we don't need those.
    _, lineno = inspect.findsource(value)
    # Add 1, since findsource is zero-indexed.
    return filepath, lineno + 1
  except (TypeError, OSError):
    return None, None


def format_source_location(
    filepath: str, lineno: int
) -> part_interface.RenderableTreePart:
  """Formats a reference to a given filepath and line number."""

  # Try to match it as an IPython file
  ipython_output_path = re.fullmatch(
      r"<ipython-input-(?P<cell_number>\d+)-.*>", filepath
  )
  if ipython_output_path:
    cell_number = ipython_output_path.group("cell_number")
    return basic_parts.Text(f"line {lineno} of output cell {cell_number}")

  return basic_parts.Text(f"line {lineno} of {filepath}")


def handle_code_objects_with_reflection(
    node: Any,
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
    show_closure_vars: bool = False,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders code objects using source-code reflection and closure inspection."""
  if inspect.isclass(node):
    # Render class.
    closure_vars = None
  elif inspect.isfunction(node):
    # Render function.
    if show_closure_vars:
      closure_vars = inspect.getclosurevars(node).nonlocals
    else:
      closure_vars = None
  else:
    # Not a supported object.
    return NotImplemented

  annotations = []
  filepath, lineno = _get_filepath_and_lineno(node)
  if filepath is not None:
    annotations.append(
        common_styles.CommentColor(
            basic_parts.siblings(
                basic_parts.Text("  # Defined at "),
                format_source_location(filepath, lineno),
            )
        )
    )

  if closure_vars:
    boxed_closure_var_rendering = common_styles.DashedGrayOutlineBox(
        basic_parts.OnSeparateLines.build([
            common_styles.CommentColor(
                basic_parts.Text("# Closure variables:")
            ),
            subtree_renderer(closure_vars),
        ])
    )
    return basic_parts.siblings_with_annotations(
        common_structures.build_custom_foldable_tree_node(
            label=common_styles.AbbreviationColor(basic_parts.Text(repr(node))),
            contents=basic_parts.FoldCondition(
                expanded=basic_parts.IndentedChildren.build(
                    [boxed_closure_var_rendering]
                )
            ),
            path=path,
        ),
        extra_annotations=annotations,
    )

  else:
    return basic_parts.siblings_with_annotations(
        common_structures.build_one_line_tree_node(
            line=common_styles.AbbreviationColor(basic_parts.Text(repr(node))),
            path=path,
        ),
        extra_annotations=annotations,
    )
