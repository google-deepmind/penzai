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

This module provides a JIT-compilation wrapper that preserves the structure of
the original model, and also supports mutable variables inside the jitted block.

The intended use of these wrappers is to enable interactive exploration of
models in e.g. Colab notebooks, while still taking advantage of JIT compilation.
"""

from __future__ import annotations

from typing import Any

from penzai import pz


@pz.variable_jit
def _flat_jit_call_layer(layer, arg, side_inputs):
  """Helper to call a callable pytree with an argument."""
  return layer(arg, **side_inputs)


@pz.pytree_dataclass
class Jitted(pz.nn.Layer):
  """Wraps a pure layer to run under JIT compliation.

  The Jitted wrapper has the same behavior and input/output structure as the
  layer it wraps, but modifies ``__call__`` so that every call is run under JIT
  compilation. Variables in the layer will be updated correctly using
  `pz.variable_jit`.

  Since Jitted is an ordinary Layer, you can still inspect the contained layer
  and make modifications to it. This will automatically trigger a recompile,
  since the `jax.jit` call depends on the PyTree structure of the Jitted block.

  Attributes:
    body: The layer that should run under JIT compilation.
  """

  body: pz.nn.Layer

  def __call__(self, argument: Any, /, **side_inputs) -> Any:
    return _flat_jit_call_layer(self.body, argument, side_inputs)
