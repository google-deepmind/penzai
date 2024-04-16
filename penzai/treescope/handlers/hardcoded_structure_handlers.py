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

"""Handles a hardcoded list of dataclass-like structures."""
from __future__ import annotations

import dataclasses
import functools
import inspect
from typing import Any, Sequence

from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import part_interface
from penzai.treescope.handlers import builtin_structure_handler


@dataclasses.dataclass(frozen=True)
class HasFieldsInClassAttr:
  """Marks a type as having fields listed as a class attribute.

  We assume that the constructor for such a type takes each of the fields as
  keyword arguments, and that the fields are also attributes.

  Attributes:
    fields_class_attr: Attribute on the class object that specifies what the
      fields are, sometimes called "_fields", "__slots__"
    render_subclasses: Whether to also render subclasses of the class.
  """

  fields_class_attr: str
  render_subclasses: bool = False


@dataclasses.dataclass(frozen=True)
class HasExplicitFields:
  """Marks a type as having an explicit set of fields.

  We assume that the constructor for such a type takes each of the fields as
  keyword arguments, and that the fields are also attributes.

  Attributes:
    fields: Collection of fields to render.
    render_subclasses: Whether to also render subclasses of the class.
  """

  fields: Sequence[str]
  render_subclasses: bool = False


@dataclasses.dataclass(frozen=True)
class HasFieldsLikeInit:
  """Marks a type as having fields based on the signature of `__init__`.

  We assume that every argument to __init__ is also an attribute.

  Attributes:
    render_subclasses: Whether to also render subclasses of the class.
  """

  render_subclasses: bool = False


@dataclasses.dataclass(frozen=True)
class IsEnumLike:
  """Marks a type as behaving like an enum.

  Instances of enum-like types are assumed to have `name` and `value`
  attributes, and those instances should be accessible through attribute lookup
  on the class.

  Attributes:
    render_subclasses: Whether to also render subclasses of the class.
  """

  render_subclasses: bool = False


def _dataclass_like(
    fields: Sequence[str],
    node: Any,
    path: tuple[Any, ...],
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
):
  """Renders a dataclass-like object."""
  return common_structures.build_foldable_tree_node_from_children(
      prefix=basic_parts.siblings(
          common_structures.maybe_qualified_type_name(type(node)), "("
      ),
      children=builtin_structure_handler.build_field_children(
          node,
          path,
          subtree_renderer,
          fields_or_attribute_names=fields,
      ),
      suffix=")",
      path=path,
  )


def _enum_like(
    node: Any,
    path: tuple[Any, ...],
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
):
  """Renders a enum-like object."""
  del subtree_renderer
  cls = type(node)
  if node == getattr(cls, node.name):
    return common_structures.build_one_line_tree_node(
        basic_parts.siblings_with_annotations(
            common_structures.maybe_qualified_type_name(cls),
            "." + node.name,
            extra_annotations=[
                common_styles.CommentColor(
                    basic_parts.Text(f"  # value: {repr(node.value)}")
                )
            ],
        ),
        path,
    )
  else:
    return NotImplemented


@functools.cache
def _get_init_args(cls: type[Any]) -> Sequence[str]:
  return tuple(inspect.signature(cls).parameters.keys())


@dataclasses.dataclass(frozen=True)
class HardcodedStructureHandler:
  """A handler for a specific hardcoded list of dataclass-like/enum-like types.

  Each of these types will be shown like a dataclass or an enum. This is
  intended to support structures that act like dataclasses, namedtuples, or
  enums, but are not implemented as such (e.g. JAX's ShapeDtypeStruct.)

  Attributes:
    known_structure_types: Mapping from handled types to a tuple of their
      attribute names.
  """

  known_structure_types: dict[
      type[Any], HasFieldsInClassAttr | HasExplicitFields | IsEnumLike
  ]

  def __call__(
      self,
      node: Any,
      path: tuple[Any, ...] | None,
      subtree_renderer: renderer.TreescopeSubtreeRenderer,
  ) -> (
      part_interface.RenderableTreePart
      | part_interface.RenderableAndLineAnnotations
      | type(NotImplemented)
  ):
    """Renders the hardcoded types from `known_structure_types`."""
    for candidate_type, spec in self.known_structure_types.items():
      if spec.render_subclasses:
        matched = isinstance(node, candidate_type)
      else:
        matched = type(node) is candidate_type  # pylint: disable=unidiomatic-typecheck

      if matched:
        if isinstance(spec, HasFieldsInClassAttr):
          fields = getattr(type(node), spec.fields_class_attr)
          return _dataclass_like(fields, node, path, subtree_renderer)
        elif isinstance(spec, HasExplicitFields):
          fields = spec.fields
          return _dataclass_like(fields, node, path, subtree_renderer)
        elif isinstance(spec, HasFieldsLikeInit):
          fields = _get_init_args(type(node))
          return _dataclass_like(fields, node, path, subtree_renderer)
        elif isinstance(spec, IsEnumLike):
          return _enum_like(node, path, subtree_renderer)

    else:
      return NotImplemented
