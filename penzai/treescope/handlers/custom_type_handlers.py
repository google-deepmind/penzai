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

"""Handler for custom types via __penzai_repr__ or the global registry."""

from __future__ import annotations

from typing import Any

from penzai.treescope import object_inspection
from penzai.treescope import renderer
from penzai.treescope import type_registries
from penzai.treescope.foldable_representation import part_interface


def handle_via_penzai_repr_method(
    node: Any,
    path: str | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a type by calling its __penzai_repr__ method, if it exists.

  The __penzai_repr__ method can be used to add treescope support to custom
  classes. The method is expected to return a rendering in treescope's internal
  intermediate representation.

  Currently, the exact structure of the intermediate representation is an
  implementation detail and may change in future releases. Instead of building
  a rendering directly, most types should use the construction helpers in
  `penzai.treescope.repr_lib` to implement this method.

  A useful pattern is to only import `penzai.treescope` inside
  `__penzai_repr__`. This allows a library to support treescope without
  requiring treescope to be a direct dependency of the library.

  Args:
    node: The node to render.
    path: An optional path to this node from the root.
    subtree_renderer: The renderer for sutrees of this node.

  Returns:
    A rendering of this node, if it implements the __penzai_repr__ extension
    method.
  """
  penzai_repr_method = object_inspection.safely_get_real_method(
      node, "__penzai_repr__"
  )
  if penzai_repr_method:
    return penzai_repr_method(path, subtree_renderer)
  else:
    return NotImplemented


def handle_via_global_registry(
    node: Any,
    path: str | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    part_interface.RenderableTreePart
    | part_interface.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a type by looking it up in the global handler registry.

  If it is not feasible to define ``__treescope_repr__`` for a type, it can
  instead be registered in the global handler registry. This is a dictionary
  mapping types to functions that render a node of that type.

  Currently, the exact structure of the intermediate representation is an
  implementation detail and may change in future releases. Instead of building
  a rendering directly, most types should use the construction helpers in
  `penzai.treescope.repr_lib` to implement this method.

  Args:
    node: The node to render.
    path: An optional path to this node from the root.
    subtree_renderer: The renderer for sutrees of this node.

  Returns:
    A rendering of this node, if it was found in the global registry.
  """
  maybe_handler = type_registries.lookup_by_mro(
      type_registries.TREESCOPE_HANDLER_REGISTRY, type(node)
  )
  if maybe_handler:
    return maybe_handler(node, path, subtree_renderer)
  else:
    return NotImplemented
