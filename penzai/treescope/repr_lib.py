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

"""Stable high-level interface for building object reprs.

These functions simplify the process of implmenting `__penzai_repr__` for
custom types, allowing them to integrate with treescope. This interface will be
stable across penzai releases, and may be expanded in the future to support
additional customization.

Note that the exact types of `path` and `subtree_renderer` are subject to
change in future releases. These should always be passed directly from
`__penzai_repr__`.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import part_interface


def render_object_constructor(
    object_type: type[Any],
    attributes: Mapping[str, Any],
    path: part_interface.NodePath | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
    roundtrippable: bool = False,
    color: str | None = None,
) -> part_interface.Rendering:
  """Renders an object in "constructor format", similar to a dataclass.

  This produces a rendering like `Foo(bar=1, baz=2)`, where Foo identifies the
  type of the object, and bar and baz are the names of the attributes of the
  object. It is a *requirement* that these are the actual attributes of the
  object, which can be accessed via `obj.bar` or similar; otherwise, the
  path renderings will break.

  This can be used from within a `__penzai_repr__` implementation via ::

    def __penzai_repr__(self, path, subtree_renderer):
      return repr_lib.render_object_constructor(
          object_type=type(self),
          attributes=<dict of attributes here>,
          path=path,
          subtree_renderer=subtree_renderer,
      )

  Args:
    object_type: The type of the object.
    attributes: The attributes of the object, which will be rendered as keyword
      arguments to the constructor.
    path: The path to the object. When `render_object_constructor` is called
      from `__penzai_repr__`, this should come from the `path` argument to
      `__penzai_repr__`.
    subtree_renderer: The renderer to use to render subtrees. When
      `render_object_constructor` is called from `__penzai_repr__`, this
      should come from the `subtree_renderer` argument to `__penzai_repr__`.
    roundtrippable: Whether evaluating the rendering as Python code will produce
      an object that is equal to the original object. This implies that the
      keyword arguments are actually the keyword arguments to the constructor,
      and not some other attributes of the object.
    color: The background color to use for the object rendering. If None, does
      not use a background color. A utility for assigning a random color based
      on a string key is given in `penzai.treescope.formatting_util`.

  Returns:
    A rendering of the object, suitable for returning from `__penzai_repr__`.
  """
  if roundtrippable:
    constructor = basic_parts.siblings(
        common_structures.maybe_qualified_type_name(object_type), "("
    )
    closing_suffix = basic_parts.Text(")")
  else:
    constructor = basic_parts.siblings(
        basic_parts.RoundtripCondition(roundtrip=basic_parts.Text("<")),
        common_structures.maybe_qualified_type_name(object_type),
        "(",
    )
    closing_suffix = basic_parts.siblings(
        ")",
        basic_parts.RoundtripCondition(roundtrip=basic_parts.Text(">")),
    )

  children = []
  for i, (name, value) in enumerate(attributes.items()):
    child_path = (
        None if path is None else path + (jax.tree_util.GetAttrKey(name),)
    )

    if i < len(attributes) - 1:
      # Not the last child. Always show a comma, and add a space when
      # collapsed.
      comma_after = basic_parts.siblings(
          ",", basic_parts.FoldCondition(collapsed=basic_parts.Text(" "))
      )
    else:
      # Last child: only show the comma when the node is expanded.
      comma_after = basic_parts.FoldCondition(expanded=basic_parts.Text(","))

    child_line = basic_parts.build_full_line_with_annotations(
        basic_parts.siblings_with_annotations(
            f"{name}=",
            subtree_renderer(value, path=child_path),
        ),
        comma_after,
    )
    children.append(child_line)

  return common_structures.build_foldable_tree_node_from_children(
      prefix=constructor,
      children=children,
      suffix=closing_suffix,
      path=path,
      background_color=color,
  )


def render_dictionary_wrapper(
    object_type: type[Any],
    wrapped_dict: Mapping[str, Any],
    path: part_interface.NodePath | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
    roundtrippable: bool = False,
    color: str | None = None,
) -> part_interface.Rendering:
  """Renders an object in "wrapped dictionary format".

  This produces a rendering like `Foo({"bar": 1, "baz": 2})`, where Foo
  identifies the type of the object, and "bar" and "baz" are the keys in the
  dictionary that Foo acts like. It is a *requirement* that these are accessible
  through `__getitem__`, e.g. as `obj["bar"]` or similar; otherwise, the path
  renderings will break.

  This can be used from within a `__penzai_repr__` implementation via ::

    def __penzai_repr__(self, path, subtree_renderer):
      return repr_lib.render_dictionary_wrapper(
          object_type=type(self),
          wrapped_dict=<dict of items here>,
          path=path,
          subtree_renderer=subtree_renderer,
      )

  Args:
    object_type: The type of the object.
    wrapped_dict: The dictionary that the object wraps.
    path: The path to the object. When `render_object_constructor` is called
      from `__penzai_repr__`, this should come from the `path` argument to
      `__penzai_repr__`.
    subtree_renderer: The renderer to use to render subtrees. When
      `render_object_constructor` is called from `__penzai_repr__`, this
      should come from the `subtree_renderer` argument to `__penzai_repr__`.
    roundtrippable: Whether evaluating the rendering as Python code will produce
      an object that is equal to the original object. This implies that the
      constructor for `object_type` takes a single argument, which is a
      dictionary, and that `object_type` then acts like that dictionary.
    color: The background color to use for the object rendering. If None, does
      not use a background color. A utility for assigning a random color based
      on a string key is given in `penzai.treescope.formatting_util`. (By
      convention, wrapped dictionaries aren't usually assigned a color in
      Penzai.)

  Returns:
    A rendering of the object, suitable for returning from `__penzai_repr__`.
  """
  if roundtrippable:
    constructor = basic_parts.siblings(
        common_structures.maybe_qualified_type_name(object_type), "({"
    )
    closing_suffix = basic_parts.Text("})")
  else:
    constructor = basic_parts.siblings(
        basic_parts.RoundtripCondition(roundtrip=basic_parts.Text("<")),
        common_structures.maybe_qualified_type_name(object_type),
        "({",
    )
    closing_suffix = basic_parts.siblings(
        "})",
        basic_parts.RoundtripCondition(roundtrip=basic_parts.Text(">")),
    )

  children = []
  for i, (key, value) in enumerate(wrapped_dict.items()):
    child_path = None if path is None else path + (jax.tree_util.DictKey(key),)

    if i < len(wrapped_dict) - 1:
      # Not the last child. Always show a comma, and add a space when
      # collapsed.
      comma_after = basic_parts.siblings(
          ",", basic_parts.FoldCondition(collapsed=basic_parts.Text(" "))
      )
    else:
      # Last child: only show the comma when the node is expanded.
      comma_after = basic_parts.FoldCondition(expanded=basic_parts.Text(","))

    key_rendering = subtree_renderer(key)
    value_rendering = subtree_renderer(value, path=child_path)

    if (
        key_rendering.renderable.collapsed_width < 40
        and not key_rendering.renderable.foldables_in_this_part()
        and (
            key_rendering.annotations is None
            or key_rendering.annotations.collapsed_width == 0
        )
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

  return common_structures.build_foldable_tree_node_from_children(
      prefix=constructor,
      children=children,
      suffix=closing_suffix,
      path=path,
      background_color=color,
  )
