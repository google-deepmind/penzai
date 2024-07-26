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

"""Utilities for JIT compilation of Penzai models.

Directly transforming a Penzai model with `jax.jit` is allowed, but it makes
the model difficult to manipulate because the resulting function is an opaque
closure. This module provides wrappers that preserve the structure of the
original model, and re-express JIT compilation using Penzai conventions.

The intended use of these wrappers is to enable interactive exploration of
models in e.g. Colab notebooks, while still taking advantage of JIT compilation.
"""

from __future__ import annotations

from typing import Any

import jax
from penzai.deprecated.v1 import pz


@jax.jit
def _flat_jit_call_layer(layer, arg):
  """Helper to call a callable pytree with an argument."""
  return layer(arg)


@pz.pytree_dataclass
class Jitted(pz.Layer):
  """Wraps a pure layer to run under JIT compliation.

  The Jitted wrapper has the same behavior and input/output structure as the
  layer it wraps, but modifies ``__call__`` so that every call is run under JIT
  compilation. Since `jax.jit` operates on pure functions, the layer wrapped
  by `Jitted` should not have any unhandled effects (i.e. ``EffectRequest``
  instances from `penzai.deprecated.v1.data_effects`).

  Since Jitted is an ordinary Layer, you can still inspect the contained layer
  and make modifications to it. This will automatically trigger a recompile,
  since the `jax.jit` call depends on the PyTree structure of the Jitted block.

  Attributes:
    body: The layer that should run under JIT compilation.
  """

  body: pz.LayerLike

  def __call__(self, argument: Any, /) -> Any:
    try:
      return _flat_jit_call_layer(self.body, argument)
    except TypeError as exc:
      impls = pz.select(self.body).at_instances_of(pz.de.EffectRuntimeImpl)
      if impls.is_empty():
        raise
      else:
        raise TypeError(
            "Detected data-effect runtime implementations inside a Jitted"
            " block, indicating that this Jitted block is in between the"
            " effect references and their handler. This is not"
            " supported.\nEffect implementations found:"
            f" {repr(impls.get_sequence())}"
        ) from exc
