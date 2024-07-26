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

"""Local layer state effect.

This effect allows layers to maintain and update their own state variables.
The local state effect is a complicated effect relative to others, since state
is used in different ways by different models.

Penzai's local state effect is designed to support two type of state:

* explicitly-named state variables, which act like parameters but are updated
  and checkpointed separately from parameters, and aren't updated by gradient
  descent. This could include e.g. batch norm statistics.

* unnamed state variables, which are used for things like sampling state or
  decoding state, which are locally updated in some context but are rarely
  checkpointed or serialized.

Both types of state use the same basic mechanism. The difference is that
explicitly-named state variables will be renamed by parameter-renaming
transformations (but not be affected by being moved around in the model PyTree),
whereas unnamed state variables will have an inferred name based on their
position in the PyTree at the moment that the state was handled (but will not
be affected by parameter naming).

In general, stateful models should introduce state by adding an attribute with
type `LocalStateEffect` and initial value that is an instance of
`InitialLocalStateRequest`. Stateful models can then be turned into a stateful
form using `handle_local_states`, which also produces an initial state
dictionary. The state dictionary can then be combined with the model again
using `freeze_local_states`, which embeds the current state variables as
`FrozenLocalStateRequest` instances; the model can then be checkpointed if
desired.
"""

from __future__ import annotations

import collections
from collections.abc import Hashable
import dataclasses
import typing
from typing import Any, Callable, Generic, Literal, Protocol

from penzai.core import selectors
from penzai.core import shapecheck
from penzai.core import struct
from penzai.core import tree_util
from penzai.deprecated.v1.core import layer as layer_base
from penzai.deprecated.v1.data_effects import effect_base
from penzai.deprecated.v1.nn import parameters

_T = typing.TypeVar("_T")

Category: typing.TypeAlias = Hashable


@effect_base.register_effect_color(color="#f58566")
class LocalStateEffect(Protocol[_T]):
  """Protocol for a local state effect."""

  def get(self) -> _T:
    """Gets the current state of the local variable."""
    ...

  def set(self, value: _T):
    """Sets the current state of the local variable."""
    ...


@struct.pytree_dataclass
class InitialLocalStateRequest(
    Generic[_T],
    effect_base.EffectRequest,
    parameters.SupportsParameterRenaming,
):
  """Effect request for local state, with a state initializer.

  This can be used to configure the initial state when initializing a model
  or transforming a model into a stateful configuration.

  Typically, if this state is something that should be updated and checkpointed
  during training, each ``InitialLocalStateRequest`` should be created when
  the model is built, and given a name similar to a parameter. If this state
  is something that will only be used temporarily (e.g. decoding state while
  sampling or doing a per-example rollout), it's not necessary to give it a
  name.

  Attributes:
    state_initializer: Callable that builds the initial state.
    category: Category tag identifying the kind of state.
    name: Optional name for this state. If provided, it will be renamed as if
      this state was a parameter, and used as the key for this state in the
      state dictionary. If not provided, a name will be inferred from the PyTree
      structure at the time that the state is used. States with the same
      explicit name will share the same value.
  """

  state_initializer: Callable[[], _T] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  category: Category = dataclasses.field(metadata={"pytree_node": False})
  name: str | None = dataclasses.field(
      default=None, metadata={"pytree_node": False}
  )

  @classmethod
  def effect_protocol(cls):
    return LocalStateEffect

  def with_renamed_parameters(
      self, rename_fn: Callable[[str], str]
  ) -> InitialLocalStateRequest:
    if self.name is None:
      return self
    else:
      return dataclasses.replace(self, name=rename_fn(self.name))

  def as_frozen(self) -> FrozenLocalStateRequest[_T]:
    """Initializes the parameter, returning an equivalent frozen one."""
    return FrozenLocalStateRequest(
        state=self.state_initializer(),
        name=self.name,
        category=self.category,
    )


@struct.pytree_dataclass
class FrozenLocalStateRequest(Generic[_T], effect_base.EffectRequest):
  """Effect request for local state with a frozen value.

  This can be used to store the state for a stateful model after it has been
  initialized and possibly updated. As a convenience, it's possible to call
  `get` on a ``FrozenLocalStateRequest``, but not `set`.

  Attributes:
    state: Frozen version of the state for this effect.
    category: Category tag identifying the kind of state.
    name: Optional name for this state. If provided, it will be renamed as if
      this state was a parameter, and used as the key for this state in the
      state dictionary. If not provided, a name will be inferred from the PyTree
      structure.
  """

  state: _T
  category: Category = dataclasses.field(metadata={"pytree_node": False})
  name: str | None = dataclasses.field(
      default=None, metadata={"pytree_node": False}
  )

  def get(self) -> _T:
    return self.state

  @classmethod
  def effect_protocol(cls):
    return LocalStateEffect

  def with_renamed_parameters(
      self, rename_fn: Callable[[str], str]
  ) -> FrozenLocalStateRequest:
    if self.name is None:
      return self
    else:
      return dataclasses.replace(self, name=rename_fn(self.name))


@struct.pytree_dataclass
class SharedLocalStateRequest(Generic[_T], effect_base.EffectRequest):
  """Effect request for local state that is shared.

  A ``SharedLocalStateRequest`` can be used to share an explicitly-named state
  variable between multiple layers. Shared states should have exactly one
  version that is a FrozenLocalStateRequest or InitialLocalStateRequest, and
  potentially multiple versions that are each ``SharedLocalStateRequest``
  instances. Furthermore, the ``SharedLocalStateRequest`` must appear after the
  version with a value in PyTree flattening order.

  Attributes:
    name: Name for this state, which will be renamed as if this state was a
      parameter, and used as the key for this state in the state dictionary.
    category: Category tag identifying the kind of state.
  """

  name: str = dataclasses.field(metadata={"pytree_node": False})
  category: Category = dataclasses.field(metadata={"pytree_node": False})

  @classmethod
  def effect_protocol(cls):
    return LocalStateEffect

  def with_renamed_parameters(
      self, rename_fn: Callable[[str], str]
  ) -> FrozenLocalStateRequest:
    return dataclasses.replace(self, name=rename_fn(self.name))


@struct.pytree_dataclass
class HandledLocalStateRef(effect_base.HandledEffectRef):
  """Marker for a handled local state effect.

  Attributes:
    handler_id: The ID of the handler that is responsible for handling this
      effect.
    category: Category tag identifying the kind of state.
    name: Name of this state variable. Used to identify it in the state dict.
    was_explicitly_named: True if this name was provided explicitly by the user
      and should be preserved as-is.
  """

  handler_id: effect_base.HandlerId = dataclasses.field(
      metadata={"pytree_node": False}
  )
  category: Category = dataclasses.field(metadata={"pytree_node": False})
  name: str = dataclasses.field(metadata={"pytree_node": False})
  was_explicitly_named: bool = dataclasses.field(
      metadata={"pytree_node": False}
  )

  @classmethod
  def effect_protocol(cls):
    return LocalStateEffect


@dataclasses.dataclass
class LocalStateEffectImpl(Generic[_T], effect_base.EffectRuntimeImpl):
  """Implementation of the local state effect.

  Attributes:
    _state: Mutable (!) state tracked by the implementation.
    _handler_id: ID of the handler that is managing this state.
  """

  _state: _T
  _handler_id: effect_base.HandlerId

  def handler_id(self) -> effect_base.HandlerId:
    return self._handler_id

  @classmethod
  def effect_protocol(cls):
    return LocalStateEffect

  def get(self) -> _T:
    return self._state

  def set(self, value: _T):
    self._state = value


@struct.pytree_dataclass
class WithFunctionalLocalState(effect_base.EffectHandler):
  """`LocalState` effect handler that functionalizes local states.

  ``WithFunctionalLocalState`` transforms the body layer so that it takes a
  dictionary of states as an argument and returns a dictionary of states as a
  result.

  The standard way to construct a ``WithFunctionalLocalState`` handler is to use
  `handle_local_states`, which returns a functional wrapper and also the
  initial state callable. Conversely, you can re-embed local states into the
  model using `freeze_local_states`.
  """

  handler_id: effect_base.HandlerId = dataclasses.field(
      metadata={"pytree_node": False}
  )
  body: layer_base.LayerLike

  @layer_base.checked_layer_call
  def __call__(
      self, argument: tuple[Any, dict[str, Any]]
  ) -> tuple[Any, dict[str, Any]]:
    inner_arg, states = argument
    impls = {
        k: LocalStateEffectImpl(_state=v, _handler_id=self.handler_id)
        for k, v in states.items()
    }
    handled_body = (
        selectors.select(self.body)
        .at_instances_of(HandledLocalStateRef)
        .where(lambda ref: ref.handler_id == self.handler_id)
        .apply(lambda ref: impls[ref.name])
    )
    result = handled_body(inner_arg)
    new_states = {k: impl._state for k, impl in impls.items()}
    return result, new_states

  def _state_structure(self, desc):
    result = {}
    refs = (
        selectors.select(self.body)
        .at_instances_of(HandledLocalStateRef)
        .where(lambda ref: ref.handler_id == self.handler_id)
        .get_sequence()
    )
    for i, ref in enumerate(refs):
      result[ref.name] = shapecheck.Wildcard(f"{desc} {i}")
    return result

  def input_structure(self):
    return (
        shapecheck.Wildcard("input to body"),
        self._state_structure("old state"),
    )

  def output_structure(self):
    return (
        shapecheck.Wildcard("output from body"),
        self._state_structure("new state"),
    )

  @classmethod
  def effect_protocol(cls):
    return LocalStateEffect


@typing.overload
def handle_local_states(
    body: layer_base.LayerLike,
    category: Category | None = None,
    category_predicate: Callable[[Category], bool] | None = None,
    lazy: Literal[False] = False,
    state_sharing: Literal["forbidden", "allowed", "unsafe"] = "forbidden",
    handler_id: str | None = None,
) -> tuple[WithFunctionalLocalState, dict[str, Any]]:
  ...


@typing.overload
def handle_local_states(
    body: layer_base.LayerLike,
    category: Category | None = None,
    category_predicate: Callable[[Category], bool] | None = None,
    lazy: Literal[True] = False,  # pytype: disable=annotation-type-mismatch
    state_sharing: Literal["forbidden", "allowed", "unsafe"] = "forbidden",
    handler_id: str | None = None,
) -> tuple[WithFunctionalLocalState, Callable[[], dict[str, Any]]]:
  ...


def handle_local_states(
    body: layer_base.LayerLike,
    category: Category | None = None,
    category_predicate: Callable[[Category], bool] | None = None,
    lazy: bool = False,
    state_sharing: Literal["forbidden", "allowed", "unsafe"] = "forbidden",
    handler_id: str | None = None,
) -> tuple[
    WithFunctionalLocalState, dict[str, Any] | Callable[[], dict[str, Any]]
]:
  """Extracts local states from a stateful model.

  This method the primary way to transform a stateful model into a functional
  form that can be run.

  Args:
    body: A model or submodel with local state.
    category: The category of states to extract. Not needed if
      category_predicate is provided.
    category_predicate: An optional callable that returns True for categories we
      should take ownership of. Note that states with different categories must
      still have unique names if they are being handled by the same handler. Not
      needed if category is provided.
    lazy: If True, returns a callable that initializes the state, instead of
      returning the state itself.
    state_sharing: Strictness for sharing of states. If "forbidden", state
      sharing is strictly not allowed. If "allowed", state sharing is allowed
      between `InitialLocalStateRequest` states with identical initializers, and
      between `SharedLocalStateRequest` and any other state with the same name.
      If "unsafe", any states with the same name will be shared, with the value
      coming from whichever one was seen last.
    handler_id: ID to use for the handler. If None, will be inferred.

  Returns:
    A handler wrapping the model to handle the given states, and an initial
    state dictionary to pass as the second argument to that handler (or a
    callable producing that dictionary if `lazy` was True).
  """
  handler_id = effect_base.infer_or_check_handler_id(
      "WithFunctionalLocalState", body, explicit_id=handler_id
  )
  initial_state_thunks = {}

  if category is None and category_predicate is None:
    raise ValueError(
        "One of `category` and `category_predicate` must be specified. (If you"
        " want to handle states whose category is exactly None, use a category"
        " predicate `lambda x: x is None`.)"
    )
  elif category is not None and category_predicate is not None:
    raise ValueError(
        "Only one of `category` and `category_predicate` may be specified."
    )
  elif category_predicate is None:
    category_predicate = lambda c: c == category

  def _make_ref(
      keypath,
      hole: (
          InitialLocalStateRequest
          | FrozenLocalStateRequest
          | SharedLocalStateRequest
      ),
  ):
    if isinstance(hole, SharedLocalStateRequest):
      if state_sharing == "forbidden":
        raise ValueError(
            "Found a SharedLocalStateRequest for state variable"
            f" {hole.name} when state_sharing is set to 'forbidden'."
            " SharedLocalStateRequest must only be used when state_sharing is"
            " set to 'allowed' or 'unsafe'."
        )
      elif hole.name in initial_state_thunks:
        return HandledLocalStateRef(
            handler_id=handler_id,
            name=hole.name,
            was_explicitly_named=True,
            category=hole.category,
        )
      else:
        raise ValueError(
            "Found a SharedLocalStateRequest for state variable"
            f" {hole.name} before seeing a corresponding value."
            " SharedLocalStateRequest must appear after some other state"
            " request with the same name and an explicit value."
        )
    if isinstance(hole, InitialLocalStateRequest):
      thunk = hole.state_initializer
    else:
      thunk = lambda: hole.state
    if hole.name is None:
      auto_name = tree_util.pretty_keystr(keypath, body)
      ref = HandledLocalStateRef(
          handler_id=handler_id,
          name=auto_name,
          was_explicitly_named=False,
          category=hole.category,
      )
    else:
      ref = HandledLocalStateRef(
          handler_id=handler_id,
          name=hole.name,
          was_explicitly_named=True,
          category=hole.category,
      )
    if ref.name in initial_state_thunks:
      if state_sharing == "forbidden":
        raise ValueError(
            "Detected two local states with the same explicit name"
            f" {repr(ref.name)}, which is not allowed when state_sharing is"
            " set to 'forbidden'."
        )
      elif state_sharing == "allowed":
        if not (
            isinstance(hole, InitialLocalStateRequest)
            and initial_state_thunks[ref.name] is hole.state_initializer
        ):
          raise ValueError(
              "Detected two local states with the same explicit name"
              f" {repr(ref.name)} but different initializers! This is only"
              " allowed when state_sharing is set to 'unsafe'."
          )
      elif state_sharing == "unsafe":
        pass
      else:
        raise ValueError(f"Bad state sharing setting: {state_sharing}")
    initial_state_thunks[ref.name] = thunk
    return ref

  adjusted_body = (
      selectors.select(body)
      .at_instances_of(
          InitialLocalStateRequest
          | FrozenLocalStateRequest
          | SharedLocalStateRequest
      )
      .where(lambda req: category_predicate(req.category))
      .apply(_make_ref, with_keypath=True)
  )

  if lazy:
    states_out = lambda: {k: v() for k, v in initial_state_thunks.items()}
  else:
    states_out = {k: v() for k, v in initial_state_thunks.items()}

  handler = WithFunctionalLocalState(handler_id=handler_id, body=adjusted_body)
  return handler, states_out


def freeze_local_states(
    handled: WithFunctionalLocalState, states: dict[str, Any]
) -> Any:
  """Embeds the given states into a handled model, and removes the handler.

  This function converts a functionalized stateful model into a single effectful
  one, making it easier to inspect, patch, or serialize. It is roughly the
  inverse of `handle_local_states`, and `handle_local_states` can be used
  to re-install the handler.

  Args:
    handled: A WithFunctionalLocalState wrapping a model, as constructed using
      `handle_local_states`.
    states: A state dictionary, either from `handle_local_states` or from
      calling the handled model to get updated states.

  Returns:
    A copy of the original model, without the `WithFunctionalLocalState`
    wrapper, and with each of the states embedded as a `FrozenLocalStateRequest`
    inside the tree.
  """
  already_made_request_for_state = set()

  def _hole_for_ref(ref: effect_base.HandledEffectRef):
    if ref.was_explicitly_named:
      name = ref.name
      if name in already_made_request_for_state:
        return SharedLocalStateRequest(name=name, category=ref.category)
      else:
        already_made_request_for_state.add(name)
    else:
      name = None
    return FrozenLocalStateRequest(
        state=states[ref.name], name=name, category=ref.category
    )

  return (
      selectors.select(handled.body)
      .at_instances_of(effect_base.HandledEffectRef)
      .where(lambda ref: ref.handler_id == handled.handler_id)
      .apply(_hole_for_ref)
  )


def hoist_shared_state_requests(
    tree: Any,
    unsafe: bool = False,
) -> tuple[Any, dict[str, InitialLocalStateRequest | FrozenLocalStateRequest]]:
  r"""Hoists out the value of shared states in a pytree.

  This function is a helper for manipulating Penzai models that contain shared
  states. Ordinarily, shared states in a Penzai model are represented as some
  combination of:

  * exactly one of:

    * a single `FrozenLocalStateRequest` with a value

    * multiple `InitialLocalStateRequest` nodes with identical initializers

  * followed by one or more `SharedLocalStateRequest` nodes

  where all such requests have the same explicit name. This is convenient for
  manipulating the model as a whole, but can make it somewhat annoying to
  extract a small part of a model that uses a shared state defined elsewhere.

  This function takes a tree of this form and returns a new tree that only
  contains a `SharedLocalStateRequest` whenever there is a state that is used in
  multiple places, along with a dictionary mapping each state name to the
  concrete definition it uses. The new tree can be freely manipulated, and then
  a single copy of the state can be re-embedded using `embed_shared_states`.

  Args:
    tree: A tree where each shared variable appears either once as a
      `FrozenLocalStateRequest` or multiple times as identical
      `InitialLocalStateRequest` nodes, followed by some number of
      `SharedLocalStateRequest` nodes.
    unsafe: If True, `tree` can have multiple `FrozenLocalStateRequest` or
      `InitialLocalStateRequest` nodes with different initializers, and one will
      be picked arbitrarily.

  Returns:
    A tuple of ``(new_tree, state_defs)``, where ``new_tree`` is a copy of
    ``tree`` where all shared states have been replaced by a
    `SharedLocalStateRequest`, and ``state_defs`` is a dictionary mapping each
    shared state name to the corresponding `InitialLocalStateRequest` or
    `FrozenLocalStateRequest`. These can be passed to
    `embed_shared_state_requests` to rebuild the original tree.
  """
  state_counter = collections.Counter(
      req.name
      for req in (
          selectors.select(tree)
          .at_instances_of(
              InitialLocalStateRequest
              | FrozenLocalStateRequest
              | SharedLocalStateRequest
          )
          .get_sequence()
      )
  )

  hoisted_vals = {}

  def _hoist_it(request: InitialLocalStateRequest | FrozenLocalStateRequest):
    if request.name not in hoisted_vals:
      hoisted_vals[request.name] = request
    elif (
        isinstance(request, InitialLocalStateRequest)
        and isinstance(hoisted_vals[request.name], InitialLocalStateRequest)
        and request.state_initializer
        is hoisted_vals[request.name].state_initializer
    ):
      pass
    elif not unsafe:
      raise ValueError(
          "Detected two local states with the same explicit name"
          f" {repr(request.name)} but different initializers or values! This is"
          " only allowed `unsafe` is set to `True`."
      )
    return SharedLocalStateRequest(name=request.name, category=request.category)

  hoisted_tree = (
      selectors.select(tree)
      .at_instances_of(InitialLocalStateRequest | FrozenLocalStateRequest)
      .where(
          lambda request: (
              request.name is not None and state_counter[request.name] > 1
          )
      )
      .apply(_hoist_it)
  )
  return hoisted_tree, hoisted_vals


def embed_shared_state_requests(
    tree: Any,
    state_requests: dict[
        str, InitialLocalStateRequest | FrozenLocalStateRequest
    ],
) -> Any:
  """Embeds shared state requests into a tree.

  This function is the inverse of hoist_shared_state_requests, and can be used
  to re-embed the initial value of explicitly-named shared states into a tree
  so that they can be handled using an ordinary handler.

  The intended use of `embed_shared_state_requests` is when you want to extract
  parts of a model with shared states, or embed a model with shared states into
  a larger model. This function ensures that there is only one initial value
  for each named state, regardless of how many times it is used.

  Args:
    tree: A tree where each shared variable appears as a
      `SharedLocalStateRequest`.
    state_requests: A dictionary mapping state names to the corresponding
      `InitialLocalStateRequest` or `FrozenLocalStateRequest`.

  Returns:
    A tree where the first appearance of each shared state is replaced by the
    corresponding `InitialLocalStateRequest` or `FrozenLocalStateRequest`, and
    all other appearances are kept as `SharedLocalStateRequest`.
  """
  pending = dict(state_requests)

  def _maybe_embed_it(request: SharedLocalStateRequest):
    if request.name in pending:
      replacement = pending.pop(request.name)
      if request.category != replacement.category:
        raise ValueError(
            "Tried to substitute values with mismatched categories! Got"
            f" {replacement.category}, expected {request.category}"
        )

      return replacement
    else:
      return request

  return (
      selectors.select(tree)
      .at_instances_of(SharedLocalStateRequest)
      .apply(_maybe_embed_it)
  )
