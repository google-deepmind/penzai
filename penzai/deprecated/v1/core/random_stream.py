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

"""A simple random stream abstraction."""
from __future__ import annotations

import dataclasses
from typing import Literal

import jax


@dataclasses.dataclass(frozen=False)
class RandomStream:
  """Helper object to construct a stream of random numbers.

  This object is unsafe to pass across a JAX transformation boundary, and should
  only be used locally. To ensure they do not escape the scope in which they
  were defined, RandomStreams must be used as a context manager, e.g.::

    with RandomStream(key) as stream:
      # do something with stream.next_key()

    # stream can no longer be used here

  Attributes:
    base_key: Base key used to generate the stream.
    next_offset: Offset to use when generating the next key.
    state: Whether this random stream has "expired" and should no longer be
      used. Random streams expire once their context manager is exited.
  """

  base_key: jax.Array
  next_offset: int | jax.Array = 0
  state: Literal["pending", "active", "expired"] = dataclasses.field(
      default="pending", init=False
  )

  def next_key(self) -> jax.Array:
    """Gets the next key from this stream, mutating the stream in place."""
    if self.state != "active":
      if self.state == "pending":
        raise ValueError(
            "Cannot use a random stream that is not yet active! For safety,"
            " random streams should be activated by using them as a context"
            " manager, e.g. `with RandomStream(key) as stream:`, and they are"
            " only active within that context.\n\nIf you want to use a random"
            " stream at the top level (e.g. in a notebook), you can call"
            " `.unsafe_mark_active()`. However, in that case, make sure not to"
            " use this random stream inside a function being transformed by"
            " JAX, since this may lead to unexpected results."
        )
      elif self.state == "expired":
        raise ValueError(
            "Cannot use a random stream that has expired! For safety, random"
            " streams are only active within a scope defined by a context"
            " manager, e.g. `with RandomStream(key) as stream:`. This random"
            " stream has now been accessed outside the context it was"
            " defined in, which is not allowed."
        )
      else:
        raise ValueError(f"Unexpected state for RandomStream: {self.state}")
    offset = self.next_offset
    self.next_offset += 1
    return jax.random.fold_in(self.base_key, offset)

  def unsafe_mark_active(self) -> "RandomStream":
    """Activates the random stream, returning itself for convenience."""
    if self.state != "pending":
      raise ValueError(
          "RandomStream instances can only be used as context managers once,"
          f" in state 'pending'. Got state {repr(self.state)}"
      )
    self.state = "active"
    return self

  def __enter__(self) -> "RandomStream":
    """Activates the random stream in a context."""
    return self.unsafe_mark_active()

  def __exit__(self, exc_type, exc_value, traceback):
    """Deactivates the random stream."""
    del exc_type, exc_value, traceback  # Don't suppress any exceptions.
    self.state = "expired"
