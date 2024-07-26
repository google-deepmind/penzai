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

"""Random number generation effect."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Protocol

import jax
import numpy as np
from penzai.core import selectors
from penzai.core import shapecheck
from penzai.core import struct
from penzai.deprecated.v1.core import layer as layer_base
from penzai.deprecated.v1.core import random_stream
from penzai.deprecated.v1.data_effects import effect_base
from penzai.deprecated.v1.data_effects import local_state


@effect_base.register_effect_color(color="#df982b")
class RandomEffect(Protocol):
  """Protocol for the random number generation effect."""

  def next_key(self) -> jax.Array:
    """Returns a new random key."""
    ...


@struct.pytree_dataclass
class RandomRequest(effect_base.EffectRequest):
  """Random number generation request."""

  @classmethod
  def effect_protocol(cls):
    return RandomEffect


@struct.pytree_dataclass
class TaggedRandomRequest(effect_base.EffectRequest):
  """Random number generation request, with a tag.

  Attributes:
    tag: A tag for this random number generation effect. This can be used to
      distinguish between multiple random number generation effects in the same
      model.
  """

  tag: Any = dataclasses.field(metadata={"pytree_node": False})

  @classmethod
  def effect_protocol(cls):
    return RandomEffect


@struct.pytree_dataclass(has_implicitly_inherited_fields=True)  # pytype: disable=wrong-keyword-args
class HandledRandomRef(effect_base.HandledEffectRef):
  """Reference for a handled random effect."""

  @classmethod
  def effect_protocol(cls):
    return RandomEffect


@dataclasses.dataclass
class RandomEffectImpl(effect_base.EffectRuntimeImpl):
  """Implementation of the random number generation effect."""

  _stream: random_stream.RandomStream
  _handler_id: effect_base.HandlerId

  def next_key(self) -> jax.Array:
    return self._stream.next_key()

  def handler_id(self) -> effect_base.HandlerId:
    return self._handler_id

  @classmethod
  def effect_protocol(cls):
    return RandomEffect


def _is_untagged_hole(hole: RandomRequest | TaggedRandomRequest):
  """Default predicate that selects only untagged holes."""
  return isinstance(hole, RandomRequest)


@struct.pytree_dataclass
class WithRandomKeyFromArg(effect_base.EffectHandler):
  """RandomEffect handler that expects a random seed as its second argument.

  ``WithRandomKeyFromArg`` "functionalizes" the `RandomEffect` effect by
  deriving randomness from its second argument, a key. The model is then
  deterministic and will produce the same output when given the same key.
  """

  handler_id: effect_base.HandlerId = dataclasses.field(
      metadata={"pytree_node": False}
  )
  body: layer_base.LayerLike

  def input_structure(self):
    return (
        shapecheck.Wildcard("input to body"),
        shapecheck.ArraySpec(shape=(), dtype=jax.dtypes.prng_key),
    )

  def output_structure(self):
    return shapecheck.Wildcard("output from body")

  @layer_base.checked_layer_call
  def __call__(self, argument: tuple[Any, jax.Array]):
    inner_arg, key = argument
    with random_stream.RandomStream(key) as stream:
      impl = RandomEffectImpl(_stream=stream, _handler_id=self.handler_id)
      handled_body = (
          selectors.select(self.body)
          .at_equal_to(HandledRandomRef(handler_id=self.handler_id))
          .set(impl)
      )
      return handled_body(inner_arg)

  @classmethod
  def handling(
      cls,
      body: layer_base.LayerLike,
      hole_predicate: Callable[
          [RandomRequest | TaggedRandomRequest], bool
      ] = _is_untagged_hole,
      handler_id: str | None = None,
  ) -> WithRandomKeyFromArg:
    """Builds a ``WithRandomKeyFromArg`` that handles effects in this layer.

    Args:
      body: The layer to wrap. Usually will contain random effects in the form
        of `RandomRequest` or `TaggedRandomRequest`.
      hole_predicate: Callable that determines whether we should handle a given
        random effect hole. By default, handles all instances of `RandomRequest`
        but no instances of `TaggedRandomRequest`.
      handler_id: ID to use for the handler. If None, will be inferred.

    Returns:
      A `WithRandomKeyFromArg` handler wrapping ``body``, with its random
      effect holes replaced with references to this handler (whenever allowed
      by the predicate).
    """
    handler_id = effect_base.infer_or_check_handler_id(
        "WithRandomKeyFromArg", body, explicit_id=handler_id
    )
    ref = HandledRandomRef(handler_id=handler_id)
    adjusted_body = (
        selectors.select(body)
        .at_instances_of(RandomRequest | TaggedRandomRequest)
        .where(hole_predicate)
        .set(ref)
    )
    return cls(handler_id=handler_id, body=adjusted_body)

  @classmethod
  def effect_protocol(cls):
    return RandomEffect


@struct.pytree_dataclass
class WithStatefulRandomKey(effect_base.EffectHandler):
  """`RandomEffect` handler that tracks a random seed as a local state.

  ``WithStatefulRandomKey`` transforms RandomEffect effect into a
  `LocalStateEffect`, allowing it to be statefully updated using the existing
  state manipulation features. It does not change the input or output types of
  the model, since the random state is managed externally.

  Attributes:
    handler_id: The ID of this handler.
    body: The layer that this handler wraps.
    random_state: The local state holding the current random key.
  """

  handler_id: effect_base.HandlerId = dataclasses.field(
      metadata={"pytree_node": False}
  )
  body: layer_base.LayerLike
  random_state: local_state.LocalStateEffect[jax.Array]

  def __call__(self, argument: Any):
    old_key = self.random_state.get()
    body_key, new_key = jax.random.split(old_key, 2)
    with random_stream.RandomStream(body_key) as stream:
      impl = RandomEffectImpl(_stream=stream, _handler_id=self.handler_id)
      handled_body = (
          selectors.select(self.body)
          .at_equal_to(HandledRandomRef(handler_id=self.handler_id))
          .set(impl)
      )
      result = handled_body(argument)
    self.random_state.set(new_key)
    return result

  @classmethod
  def handling(
      cls,
      body: layer_base.LayerLike,
      initial_key: jax.Array,
      hole_predicate: Callable[
          [RandomRequest | TaggedRandomRequest], bool
      ] = _is_untagged_hole,
      state_category: Any = "random",
      handler_id: str | None = None,
  ) -> WithStatefulRandomKey:
    """Builds a `WithStatefulRandomKey` that handles effects in this layer.

    Args:
      body: The layer to wrap. Usually will contain random effects in the form
        of `RandomRequest` or `TaggedRandomRequest`.
      initial_key: Initial key to use for the state.
      hole_predicate: Callable that determines whether we should handle a given
        random effect hole. By default, handles all instances of `RandomRequest`
        but no instances of `TaggedRandomRequest`.
      state_category: Type to use when configuring the state effect.
      handler_id: ID to use for the handler. If None, will be inferred.

    Returns:
      A `WithStatefulRandomKey` handler wrapping `body`, with its random
      effect holes replaced with references to this handler (whenever allowed
      by the predicate).
    """
    shapecheck.check_structure(
        initial_key,
        shapecheck.ArraySpec(shape=(), dtype=jax.dtypes.prng_key),
    )
    handler_id = effect_base.infer_or_check_handler_id(
        "WithStatefulRandomKey", body, explicit_id=handler_id
    )
    ref = HandledRandomRef(handler_id=handler_id)
    adjusted_body = (
        selectors.select(body)
        .at_instances_of(RandomRequest | TaggedRandomRequest)
        .where(hole_predicate)
        .set(ref)
    )
    return cls(
        handler_id=handler_id,
        body=adjusted_body,
        random_state=local_state.FrozenLocalStateRequest(
            state=initial_key, category=state_category
        ),
    )

  @classmethod
  def effect_protocol(cls):
    return RandomEffect


@struct.pytree_dataclass
class WithFrozenRandomState(effect_base.EffectHandler):
  """`RandomEffect` handler that uses a fixed random state.

  ``WithFrozenRandomState`` can be used to freeze the random state of a model
  at a given point in time, allowing it to be deterministic and reproducible.
  It is most useful for debugging the behavior of a stochastic model while
  holding the random seed constant.

  Attributes:
    handler_id: The ID of this handler.
    body: The layer that this handler wraps.
    random_key: The constant random key to use.
    starting_offset: The starting offset at which to generate random numbers
      using the random key. This can be used to advance the random stream as if
      there were previous calls to `next_key`.
  """

  handler_id: effect_base.HandlerId = dataclasses.field(
      metadata={"pytree_node": False}
  )
  body: layer_base.LayerLike
  random_key: jax.Array
  starting_offset: int | jax.Array

  def __call__(self, argument: Any):
    with random_stream.RandomStream(
        self.random_key, self.starting_offset
    ) as stream:
      impl = RandomEffectImpl(_stream=stream, _handler_id=self.handler_id)
      handled_body = (
          selectors.select(self.body)
          .at_equal_to(HandledRandomRef(handler_id=self.handler_id))
          .set(impl)
      )
      result = handled_body(argument)
    return result

  @classmethod
  def handling(
      cls,
      body: layer_base.LayerLike,
      random_key: jax.Array,
      starting_offset: int = 0,
      hole_predicate: Callable[
          [RandomRequest | TaggedRandomRequest], bool
      ] = _is_untagged_hole,
      handler_id: str | None = None,
  ) -> WithStatefulRandomKey:
    """Builds a `WithFrozenRandomState` that handles effects in this layer.

    Args:
      body: The layer to wrap. Usually will contain random effects in the form
        of `RandomRequest` or `TaggedRandomRequest`.
      random_key: Initial key to use for the state.
      starting_offset: Offset to use for the key.
      hole_predicate: Callable that determines whether we should handle a given
        random effect hole. By default, handles all instances of `RandomRequest`
        but no instances of `TaggedRandomRequest`.
      handler_id: ID to use for the handler. If None, will be inferred.

    Returns:
      A `WithFrozenRandomState` handler wrapping ``body``, with its random
      effect holes replaced with references to this handler (whenever allowed
      by the predicate).
    """
    shapecheck.check_structure(
        random_key,
        shapecheck.ArraySpec(shape=(), dtype=jax.dtypes.prng_key),
    )
    shapecheck.check_structure(
        starting_offset,
        shapecheck.ArraySpec(shape=(), dtype=np.integer),
    )
    handler_id = effect_base.infer_or_check_handler_id(
        "WithFrozenRandomState", body, explicit_id=handler_id
    )
    ref = HandledRandomRef(handler_id=handler_id)
    adjusted_body = (
        selectors.select(body)
        .at_instances_of(RandomRequest | TaggedRandomRequest)
        .where(hole_predicate)
        .set(ref)
    )
    return cls(
        handler_id=handler_id,
        body=adjusted_body,
        random_key=random_key,
        starting_offset=starting_offset,
    )

  @classmethod
  def effect_protocol(cls):
    return RandomEffect
