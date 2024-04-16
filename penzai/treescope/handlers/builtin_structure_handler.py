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

"""Handler for built-in collection types (lists, sets, dicts, etc)."""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Callable, Optional, Sequence
import warnings

import jax
from penzai.core import dataclass_util
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import layout_algorithms
from penzai.treescope.foldable_representation import part_interface


CSSStyleRule = part_interface.CSSStyleRule
HtmlContextForSetup = part_interface.HtmlContextForSetup


def _dict_to_foldable(
    node: dict[Any, Any],
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> part_interface.RenderableAndLineAnnotations:
  """Renders a dictionary."""

  children = []
  for i, (key, child) in enumerate(node.items()):
    if i < len(node) - 1:
      # Not the last child. Always show a comma, and add a space when
      # collapsed.
      comma_after = basic_parts.siblings(
          ",", basic_parts.FoldCondition(collapsed=basic_parts.Text(" "))
      )
    else:
      # Last child: only show the comma when the node is expanded.
      comma_after = basic_parts.FoldCondition(expanded=basic_parts.Text(","))

    child_path = (
        None if path is None else (path + (jax.tree_util.DictKey(key),))
    )
    # Figure out whether this key is simple enough to render inline with
    # its value.
    key_rendering = subtree_renderer(key)
    value_rendering = subtree_renderer(child, path=child_path)

    if (
        key_rendering.renderable.collapsed_width < 40
        and not key_rendering.renderable.foldables_in_this_part()
        and key_rendering.annotations.collapsed_width == 0
    ):
      # Simple enough to render on one line.
      children.append(
          basic_parts.siblings_with_annotations(
              key_rendering, ": ", value_rendering, comma_after
          )
      )
    else:
      # Should render on multiple lines.
      children.append(
          basic_parts.siblings(
              basic_parts.build_full_line_with_annotations(
                  key_rendering,
                  ":",
                  basic_parts.FoldCondition(collapsed=basic_parts.Text(" ")),
              ),
              basic_parts.IndentedChildren.build([
                  basic_parts.siblings_with_annotations(
                      value_rendering, comma_after
                  ),
                  basic_parts.FoldCondition(
                      expanded=basic_parts.VerticalSpace("0.5em")
                  ),
              ]),
          )
      )

  if type(node) is dict:  # pylint: disable=unidiomatic-typecheck
    start = "{"
    end = "}"
  else:
    start = basic_parts.siblings(
        common_structures.maybe_qualified_type_name(type(node)), "({"
    )
    end = "})"

  if not children:
    return common_structures.build_one_line_tree_node(
        line=basic_parts.siblings(start, end), path=path
    )
  else:
    return common_structures.build_foldable_tree_node_from_children(
        prefix=start,
        children=children,
        suffix=end,
        path=path,
    )


def _sequence_or_set_to_foldable(
    sequence: dict[Any, Any],
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> part_interface.RenderableAndLineAnnotations:
  """Renders a sequence or set to a foldable."""

  children = []
  for i, child in enumerate(sequence):
    child_path = (
        None if path is None else (path + (jax.tree_util.SequenceKey(i),))
    )
    children.append(subtree_renderer(child, path=child_path))

  force_trailing_comma = False
  if isinstance(sequence, tuple):
    before = "("
    after = ")"
    if type(sequence) is not tuple:  # pylint: disable=unidiomatic-typecheck
      # Unusual situation: this is a subclass of `tuple`, but it shouldn't be
      # a namedtuple because we look for _fields already.
      assert not hasattr(type(sequence), "_fields")
      # It's unclear what the constructor will be; try calling it with a single
      # ordinary tuple as an argument.
      before = basic_parts.siblings(
          common_structures.maybe_qualified_type_name(type(sequence)),
          "(" + before,
      )
      after = after + ")"
    force_trailing_comma = len(sequence) == 1
  elif isinstance(sequence, list):
    before = "["
    after = "]"
    if type(sequence) is not list:  # pylint: disable=unidiomatic-typecheck
      before = basic_parts.siblings(
          common_structures.maybe_qualified_type_name(type(sequence)),
          "(" + before,
      )
      after = after + ")"
  elif isinstance(sequence, set):
    if not sequence:
      before = "set("
      after = ")"
    else:  # pylint: disable=unidiomatic-typecheck
      before = "{"
      after = "}"

    if type(sequence) is not set:  # pylint: disable=unidiomatic-typecheck
      before = basic_parts.siblings(
          common_structures.maybe_qualified_type_name(type(sequence)),
          "(" + before,
      )
      after = after + ")"
  elif isinstance(sequence, frozenset):
    before = "frozenset({"
    after = "})"
    if type(sequence) is not frozenset:  # pylint: disable=unidiomatic-typecheck
      before = basic_parts.siblings(
          common_structures.maybe_qualified_type_name(type(sequence)),
          "(" + before,
      )
      after = after + ")"
  else:
    raise ValueError(f"Unrecognized sequence {sequence}")

  if not children:
    return common_structures.build_one_line_tree_node(
        line=basic_parts.siblings(before, after), path=path
    )
  else:
    return common_structures.build_foldable_tree_node_from_children(
        prefix=before,
        children=children,
        suffix=after,
        path=path,
        comma_separated=True,
        force_trailing_comma=force_trailing_comma,
    )


def build_field_children(
    node: dict[Any, Any],
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
    fields_or_attribute_names: Sequence[dataclasses.Field[Any] | str],
    key_path_fn: Callable[[str], Any] = jax.tree_util.GetAttrKey,
    attr_style_fn: (
        Callable[[str], part_interface.RenderableTreePart] | None
    ) = None,
) -> list[part_interface.RenderableTreePart]:
  """Renders a set of fields/attributes into a list of comma-separated children.

  This is a helper function used for rendering dataclasses, namedtuples, and
  similar objects, of the form ::

    ClassName(
        field_name_one=value1,
        field_name_two=value2,
    )

  If `fields_or_attribute_names` includes dataclass fields:

  * Metadata for the fields will be visible on hover,

  * Fields with ``repr=False`` will be hidden unless roundtrip mode is enabled.

  Args:
    node: Node to render.
    path: Path to this node.
    subtree_renderer: How to render subtrees (see `TreescopeSubtreeRenderer`)
    fields_or_attribute_names: Sequence of fields or attribute names to render.
      Any field with the metadata key "treescope_always_collapse" set to True
      will always render collapsed.
    key_path_fn: Optional function which maps field names to their JAX keys, if
      applicable. This should match their registered keypaths in the PyTree
      registry when applicable (although it will also be called for fields that
      are not necessarily PyTree children).
    attr_style_fn: Optional function which makes attributes to a part that
      should render them. If not provided, all parts are rendered as plain text.

  Returns:
    A list of child objects. This can be passed to
    `common_structures.build_foldable_tree_node_from_children` (with
    ``comma_separated=False``)
  """
  if attr_style_fn is None:
    attr_style_fn = basic_parts.Text

  field_names = []
  fields: list[Optional[dataclasses.Field[Any]]] = []
  for field_or_name in fields_or_attribute_names:
    if isinstance(field_or_name, str):
      field_names.append(field_or_name)
      fields.append(None)
    else:
      field_names.append(field_or_name.name)
      fields.append(field_or_name)

  children = []
  for i, (field_name, maybe_field) in enumerate(zip(field_names, fields)):
    child_path = None if path is None else (path + (key_path_fn(field_name),))

    if i < len(fields) - 1:
      # Not the last child. Always show a comma, and add a space when
      # collapsed.
      comma_after = basic_parts.siblings(
          ",", basic_parts.FoldCondition(collapsed=basic_parts.Text(" "))
      )
    else:
      # Last child: only show the comma when the node is expanded.
      comma_after = basic_parts.FoldCondition(expanded=basic_parts.Text(","))

    if maybe_field is not None:
      hide_except_in_roundtrip = not maybe_field.repr
      force_collapsed = maybe_field.metadata.get(
          "treescope_always_collapse", False
      )
    else:
      hide_except_in_roundtrip = False
      force_collapsed = False

    field_name_rendering = attr_style_fn(field_name)

    try:
      field_value = getattr(node, field_name)
    except AttributeError:
      child = basic_parts.FoldCondition(
          expanded=common_styles.CommentColor(
              basic_parts.siblings("# ", field_name_rendering, " is missing")
          )
      )
    else:
      child = basic_parts.siblings_with_annotations(
          field_name_rendering,
          "=",
          subtree_renderer(field_value, path=child_path),
      )

    child_line = basic_parts.build_full_line_with_annotations(
        child, comma_after
    )
    if force_collapsed:
      layout_algorithms.expand_to_depth(child_line, 0)
    if hide_except_in_roundtrip:
      child_line = basic_parts.RoundtripCondition(roundtrip=child_line)

    children.append(child_line)

  return children


def render_dataclass_constructor(
    node: Any,
) -> part_interface.RenderableTreePart:
  """Renders the constructor for a dataclass, including the open parenthesis."""
  assert dataclasses.is_dataclass(node) and not isinstance(node, type)
  if not dataclass_util.init_takes_fields(type(node)):
    constructor_open = basic_parts.siblings(
        basic_parts.RoundtripCondition(
            roundtrip=basic_parts.Text("pz.dataclass_from_attributes(")
        ),
        common_structures.maybe_qualified_type_name(type(node)),
        basic_parts.RoundtripCondition(
            roundtrip=basic_parts.Text(", "),
            not_roundtrip=basic_parts.Text("("),
        ),
    )
  else:
    constructor_open = basic_parts.siblings(
        common_structures.maybe_qualified_type_name(type(node)), "("
    )
  return constructor_open


def parse_color_and_pattern(
    requested_color: str | tuple[str, str], typename_for_warning: str | None
) -> tuple[str | None, str | None]:
  """Parses a background color and pattern from a user-provided color request."""
  if isinstance(requested_color, str):
    background_color = requested_color
    background_pattern = None
  elif isinstance(requested_color, tuple) and len(requested_color) == 2:
    background_color = requested_color[0]
    background_pattern = (
        f"linear-gradient({requested_color[1]},{requested_color[1]})"
    )
  else:
    if typename_for_warning:
      warnings.warn(
          f"{typename_for_warning} requested an invalid color:"
          f" {requested_color} (not a string or a tuple)"
      )
    background_color = None
    background_pattern = None
  return background_color, background_pattern


def handle_builtin_structures(
    node: Any,
    path: tuple[Any, ...] | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders builtin structure types."""
  if dataclasses.is_dataclass(node) and not isinstance(node, type):
    constructor_open = render_dataclass_constructor(node)

    if hasattr(node, "__treescope_color__") and callable(
        node.__treescope_color__
    ):
      background_color, background_pattern = parse_color_and_pattern(
          node.__treescope_color__(), type(node).__name__
      )
    else:
      background_color = None
      background_pattern = None

    return common_structures.build_foldable_tree_node_from_children(
        prefix=constructor_open,
        children=build_field_children(
            node,
            path,
            subtree_renderer,
            fields_or_attribute_names=dataclasses.fields(node),
        ),
        suffix=")",
        path=path,
        background_color=background_color,
        background_pattern=background_pattern,
    )

  elif isinstance(node, tuple) and hasattr(type(node), "_fields"):
    # Namedtuple class.
    return common_structures.build_foldable_tree_node_from_children(
        prefix=basic_parts.siblings(
            common_structures.maybe_qualified_type_name(type(node)), "("
        ),
        children=build_field_children(
            node,
            path,
            subtree_renderer,
            fields_or_attribute_names=type(node)._fields,
        ),
        suffix=")",
        path=path,
    )

  if isinstance(node, dict):
    return _dict_to_foldable(node, path, subtree_renderer)

  if isinstance(node, (tuple, list, set, frozenset)):
    # Sequence or set. (Not a namedtuple; those are handled above.)
    return _sequence_or_set_to_foldable(node, path, subtree_renderer)

  elif isinstance(node, types.SimpleNamespace):
    return common_structures.build_foldable_tree_node_from_children(
        prefix=basic_parts.siblings(
            common_structures.maybe_qualified_type_name(type(node)), "("
        ),
        children=build_field_children(
            node,
            path,
            subtree_renderer,
            fields_or_attribute_names=tuple(node.__dict__.keys()),
        ),
        suffix=")",
        path=path,
    )

  return NotImplemented
