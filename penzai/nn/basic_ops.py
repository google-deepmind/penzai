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

"""Basic primitives for neural networks."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
from penzai.core import named_axes
from penzai.core import struct
from penzai.nn import layer


@struct.pytree_dataclass
class Elementwise(layer.Layer):
  """A layer that runs an elementwise operation on its `NamedArray` argument.

  The function is treated as a static object with no parameters, and
  automatically maps its argument over named axes of its input. It is useful
  for applying activation functions or other elementwise operations.

  Attributes:
    fn: The function to call.
  """

  fn: Callable[[Any], Any] = dataclasses.field(metadata={"pytree_node": False})

  def __call__(
      self,
      value: jax.Array | named_axes.NamedArrayBase,
      **_unused_side_inputs,
  ) -> jax.Array | named_axes.NamedArrayBase:
    if isinstance(value, named_axes.NamedArrayBase):
      return named_axes.nmap(self.fn)(value)
    else:
      return self.fn(value)

  def treescope_color(self) -> str:
    from treescope import formatting_util  # pylint: disable=g-import-not-at-top

    return formatting_util.color_from_string(repr(self.fn))


@struct.pytree_dataclass
class Softmax(layer.Layer):
  """Layer that applies a softmax along a given set of axes."""

  axes: str | tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def __call__(
      self, inputs: named_axes.NamedArrayBase, **_unused_side_inputs
  ) -> named_axes.NamedArrayBase:
    axes = (self.axes,) if isinstance(self.axes, str) else self.axes
    return named_axes.nmap(jax.nn.softmax)(
        inputs.untag(*axes), axis=tuple(range(len(axes)))
    ).tag(*axes)


@struct.pytree_dataclass
class CastToDType(layer.Layer):
  """Casts an input to a given dtype.

  Attributes:
    dtype: The dtype to cast to.
  """

  dtype: jax.typing.DTypeLike = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def __call__(self, value: Any, **_unused_side_inputs) -> Any:
    return jax.tree_util.tree_map(lambda x: x.astype(self.dtype), value)
