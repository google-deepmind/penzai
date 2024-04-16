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

"""Fallback definitions for copying non-roundtrippable objects."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import part_interface
from penzai.treescope.handlers import builtin_atom_handler
from penzai.treescope.handlers import builtin_structure_handler
from penzai.treescope.handlers import canonical_alias_postprocessor
from penzai.treescope.handlers import function_reflection_handlers


@dataclasses.dataclass(frozen=True)
class NotRoundtrippable:
  """A placeholder for a non-roundtrippable object in roundtrip mode.

  Many objects can be round-trippably rendered by Treescope, but some do not
  support full round-tripping (e.g. JAX arrays or unrecognized PyTree leaves).
  For these objects, it is still sometimes convenient to render something
  that is valid Python and identifies the original object. This placeholder
  object serves that purpose.

  Attributes:
    original_repr: The raw ``repr`` of the object.
    original_id: The ID of the original object.
    original_type: The type of the original object.
  """

  original_repr: str
  original_id: int
  original_type: type[Any]

  @classmethod
  def from_object(
      cls, obj: Any, repr_override: str | None = None
  ) -> NotRoundtrippable:
    """Constructs a NotRoundtrippable from an object."""
    if isinstance(obj, jax.Array) and not isinstance(obj, jax.core.Tracer):
      # Don't output the internal implementation type.
      ty = jax.Array
    else:
      ty = type(obj)
    if repr_override is None:
      repr_value = repr(obj)
    else:
      repr_value = repr_override
    return cls(repr_value, id(obj), ty)

  def __treescope_color__(self):
    return "#db845a"


def render_not_roundtrippable(
    obj: NotRoundtrippable,
    repr_override: str | None = None,
) -> part_interface.RenderableTreePart:
  """Renders an object as a `NotRoundtrippable` instance.

  This can be used inside handlers for non-roundtrippable objects to render
  them as a `NotRoundtrippable`, independent of the renderer being used for
  normal rendering.

  Args:
    obj: The object to render.
    repr_override: Optional override for the repr of the object.

  Returns:
    A RenderableTreePart rendering of a `NotRoundtrippable` for this object.
  """
  fallback_renderer = renderer.TreescopeRenderer(
      handlers=[
          builtin_atom_handler.handle_builtin_atoms,
          builtin_structure_handler.handle_builtin_structures,
          function_reflection_handlers.handle_code_objects_with_reflection,
      ],
      wrapper_hooks=[
          canonical_alias_postprocessor.replace_with_canonical_aliases,
      ],
      context_builders=[],
  )
  return fallback_renderer.to_foldable_representation(
      NotRoundtrippable.from_object(obj, repr_override=repr_override)
  ).renderable
