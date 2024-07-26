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

"""Parameters and state variables in Penzai neural networks.

Parameters in Penzai models are allowed to be any object that has a `.value`
attribute, as described in the `ParameterLike` protocol. This makes it possible
to wrap parameters or apply preprocessing logic when values are retrieved.
Usually, however, Penzai parameters are instances of `Parameter`, which is a
subclass of `Variable`. This allows them to be easily shared and updated in
place.

By convention, neural network initializers take an argument `init_base_rng`,
which is a JAX PRNG key that is shared across the full model. They then
initialize their parameters by combining the `init_base_rng` with the unique
string name of each parameter.
"""

from __future__ import annotations

import abc
import hashlib
from typing import Any, Callable, Generic, Protocol, TypeVar

import jax
from penzai.core import selectors
from penzai.core import variables

T = TypeVar("T")


class ParameterLike(Protocol, Generic[T]):
  """Protocol for a parameter-like object.

  ``ParameterLike`` defines the common API for parameters in `penzai.nn`. Any
  parameterized network layer should annotate its parameters as
  ``ParameterLike``, and should not make assumptions about the exact
  implementation of ``ParameterLike`` used.

  ``ParameterLike`` is a protocol, which means implementations of it do not have
  to explicitly subclass it. If you define ``value`` as a dataclass attribute
  rather than a property, you should avoid subclassing ``ParameterLike``
  directly, since Python's ABC system will not see that as an implementation of
  the method.
  """

  @property
  @abc.abstractmethod
  def value(self) -> T:
    """The value of the parameter. May raise an error for some instances."""
    raise NotImplementedError


def derive_param_key(base_rng: jax.Array, name: str) -> jax.Array:
  """Derives a PRNG key for a parameter from a base key and a name.

  Args:
    base_rng: The base PRNG key.
    name: The name of the parameter.

  Returns:
    A unique PRNG key for the parameter with this name.
  """
  hashed_bytes = hashlib.sha1(name.encode("utf-8")).digest()
  offset = int.from_bytes(hashed_bytes[:4], byteorder="big")
  return jax.random.fold_in(base_rng, offset)


def make_parameter(
    name: str,
    init_base_rng: jax.Array | None,
    initializer: Callable[..., T],
    *init_args,
    metadata: dict[Any, Any] | None = None,
    **init_kwargs,
) -> variables.Parameter[T] | variables.ParameterSlot:
  """Makes a parameter variable (or slot) with a given name and initializer.

  Args:
    name: The name of the parameter.
    init_base_rng: The base PRNG key to use for initialization, which will be
      combined with the name to obtain a unique key for this parameter. If
      `None`, the parameter will not be initialized; instead it will be set to a
      `ParameterSlot` to be filled in later.
    initializer: The initializer function to use. The first argument should be a
      PRNG key, which will
    *init_args: Positional arguments to pass to the initializer.
    metadata: Metadata for the variable.
    **init_kwargs: Keyword arguments to pass to the initializer.

  Returns:
    A `Parameter` if `init_base_rng` is not `None`, or a `ParameterSlot`
    otherwise.
  """
  if init_base_rng is None:
    return variables.ParameterSlot(name)
  else:
    if metadata is None:
      metadata = {}
    return variables.Parameter(
        value=initializer(
            derive_param_key(init_base_rng, name), *init_args, **init_kwargs
        ),
        label=name,
        metadata=metadata,
    )


def assert_no_parameter_slots(model: Any):
  """Asserts that the given model has no ParameterSlot subtrees."""
  bad_slots = (
      selectors.select(model)
      .at_instances_of(variables.ParameterSlot)
      .get_sequence()
  )
  if bad_slots:
    raise ValueError(f"Unexpected ParameterSlots found in model: {bad_slots}")
