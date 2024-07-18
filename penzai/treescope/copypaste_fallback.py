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
import sys
from typing import Any

from penzai.treescope import handlers
from penzai.treescope import renderer
from penzai.treescope import rendering_parts


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
    ty = type(obj)
    # Hide implementation details of JAX arrays if JAX is imported.
    if "jax" in sys.modules:
      jax = sys.modules["jax"]
      if isinstance(obj, jax.Array) and not isinstance(obj, jax.core.Tracer):
        # Don't output the internal implementation type.
        ty = jax.Array
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
) -> rendering_parts.RenderableTreePart:
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
          handlers.handle_basic_types,
          handlers.handle_code_objects_with_reflection,
      ],
      wrapper_hooks=[
          handlers.replace_with_canonical_aliases,
      ],
      context_builders=[],
  )
  return fallback_renderer.to_foldable_representation(
      NotRoundtrippable.from_object(obj, repr_override=repr_override)
  ).renderable
