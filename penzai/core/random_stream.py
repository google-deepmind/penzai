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

"""A stateful random stream helper."""

from __future__ import annotations

import jax
from penzai.core import struct
from penzai.core import variables


@struct.pytree_dataclass
class RandomStream(struct.Struct):
  """A stateful random stream object.

  This object can be used to generate a sequence of random numbers inside a
  Penzai model. It uses a Variable to track its state.

  Attributes:
    base_key: The base key to use for this random stream. This does not change,
      and determines which sequence of random numbers will be generated.
    offset: The number of random numbers that have been generated so far.
  """

  base_key: jax.Array
  offset: variables.StateVariable[int | jax.Array]

  @classmethod
  def from_base_key(
      cls,
      base_key: jax.Array,
      offset_label: variables.VariableLabel = "random_stream_offset",
  ) -> RandomStream:
    """Returns a new random stream with the given base key."""
    return cls(
        base_key=base_key, offset=variables.StateVariable(0, label=offset_label)
    )

  def next_key(self) -> jax.Array:
    """Returns the next key in the sequence, and advances the stream."""
    old_offset = self.offset.value
    self.offset.value = self.offset.value + 1
    return jax.random.fold_in(self.base_key, old_offset)
