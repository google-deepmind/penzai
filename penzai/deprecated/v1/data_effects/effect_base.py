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

"""Base types for the ``data_effects`` system.

Penzai's effect system is based on representing effects as dataclass PyTree
nodes, and handling them by substituting abstract effect nodes with concrete
handler nodes. This module contains the base types for the system.

To use an existing effect, you should:

1. Add an attribute to the layer that will use the effect, and annotate its
   type as ``{EffectName}Effect``.

2. In the builder for your layer (e.g. inside the ``from_config`` method or
   other class method), set that attribute to an instance of the corresponding
   effect request, e.g. ``{EffectName}Request(*args, **kwargs)``. Alternatively
   you can set this as the default value for the dataclass attribute.

3. In the ``__call__`` method of your layer, assume that the effect request has
   been replaced with some concrete instance that implements all of the methods
   of ``{EffectName}Effect``, and call them as normal.

4. To use your layer, wrap it in an effect handler for that effect. The effect
   handler will replace the effect request with an effect reference (usually of
   type ``HandledEffectRef``), and then when it is called it will further
   replace that reference with a temporary implementation node implementing the
   effect.

To define a new effect, you should:

1. Create a new ``Protocol`` (e.g. a new class with ``Protocol`` as the base
   class), by convention called ``{EffectName}Effect``, whose methods define the
   interface for that effect. For instance, a state effect would define methods
   for getting and setting values.

2. Create a subclass of `EffectRequest`, by convention called
   ``{EffectName}Request`` and override ``effect_protocol`` to return
   the protocol you defined in step 1. Optionally, add attributes that provide
   information to the handler that is needed to handle the effect.

3. Create a subclass of `HandledEffectRef`, by convention called
   ``Handled{EffectName}Ref`` and override ``effect_protocol`` to return the
   protocol you defined in step 1. Optionally, add attributes that store
   information needed by the handler to handle the effect.

4. Create a subclass of `EffectRuntimeImpl` that implements the protocol you
   defined in step 1, by convention called ``{EffectName}EffectImpl``. This does
   not have to be a PyTree dataclass node (although it can be). It is allowed to
   store a reference to some external state that it can modify. Note that you
   are also allowed to define multiple different implementations if needed.

5. Create one or more subclasses of `EffectHandler`, by convention called
   something like ``With{EffectDescription}``, and configure them to replace the
   effect requests with refs when the handler is created, and then replaces the
   refs with a temporary implementation node when it is called.
"""

from __future__ import annotations

import abc
from collections.abc import Collection
import dataclasses
import functools
import hashlib
import typing
from typing import Any, Callable
import weakref

import jax
import ordered_set
from penzai.core import selectors
from penzai.core import struct
from penzai.deprecated.v1.core import layer as layer_base


T = typing.TypeVar("T")


class UnhandledEffectError(Exception):
  """Exception raised when a method is called on an unhandled effect."""


_EFFECT_COLORS = weakref.WeakKeyDictionary()


def get_effect_color(effect_protocol: type[Any]) -> str:
  """Gets the default color for a given effect (for treescope rendering)."""
  if effect_protocol in _EFFECT_COLORS:
    return _EFFECT_COLORS[effect_protocol]
  from treescope import formatting_util  # pylint: disable=g-import-not-at-top

  return formatting_util.color_from_string(effect_protocol.__qualname__)


def register_effect_color(color: str) -> Callable[[type[Any]], type[Any]]:
  """Decorator to register a treescope-rendering color for a given effect.

  Args:
    color: The color to assign, as a CSS string.

  Returns:
    A decorator that can be applied to a protocol class to associate it with
    the given color.
  """

  def color_registration_decorator(cls):
    _EFFECT_COLORS[cls] = color
    return cls

  return color_registration_decorator


class EffectRequest(struct.Struct, abc.ABC):
  """Base class for "effect requests", which represent unhandled effects.

  Effect requests are used as placeholders for effects in a PyTree. They are
  substituted by effect handlers when the effect is handled, and they can be
  used to provide information to the handler that the handler needs.

  Each new effect type should have at least one ``EffectRequest`` subclass, and
  each ``EffectRequest`` subclass should handle a unique effect type.
  ``EffectRequest`` subclasses should declare which effect type they handle by
  overriding the `effect_protocol` method. Overriding the actual methods in that
  protocol is optional; ``EffectRequest`` will automatically raise an exception
  when missing methods are called on an un-filled request.

  If there is a reasonable "no-op" implementation of an effect method even when
  the effect isn't handled, it is allowed to implement that method and do
  nothing when it is called. This can in some cases make it possible to call
  models that have unhandled effects, while still allowing handlers to override
  that default behavior as needed.

  If there is no other information you need to provide to the handler, you can
  leave the rest of the definition empty. Otherwise, you can add attributes to
  ``EffectRequest`` to provide information to the handler.
  """

  @classmethod
  @abc.abstractmethod
  def effect_protocol(cls) -> type[Any]:
    """Returns the protocol that this effect request handles."""
    raise NotImplementedError(
        "Subclasses of EffectRequest must override the effect_protocol class"
        " method"
    )

  def __getattr__(self, attr: str):
    """Fallback to get a missing method from an ``EffectRequest``."""
    eff_proto = self.effect_protocol()
    if hasattr(eff_proto, attr) and callable(getattr(eff_proto, attr)):
      # This attribute is a protocol method of the effect. This probably means
      # that some layer is trying to access an effect that hasn't been handled.
      def unhandled_effect_stub(*args, **kwargs):
        del args, kwargs
        raise UnhandledEffectError(
            f"Effect method {repr(attr)} was called on a non-handled effect"
            f" request of type {type(self).__name__}! This usually means that"
            " an effectful layer has been called without wrapping that layer"
            " in an effect handler that knows how to interpret the effect."
            " Instances of (subclasses of) EffectRequest must be replaced"
            " before the layer is called."
        )

      return unhandled_effect_stub
    else:
      raise AttributeError(
          f"'{type(self).__name__}' object has no attribute '{attr}'",
          name=attr,
          obj=self,
      )

  def treescope_color(self):
    return get_effect_color(self.effect_protocol())


def free_effect_types(model_tree: Any) -> list[type[Any]]:
  """Collects the effect types of all `EffectRequest` nodes in a (sub)model.

  Args:
    model_tree: A pytree for a model or submodel.

  Returns:
    A list of the effect protocol types for all `EffectRequest` nodes that
    appear in a model or submodel.
  """
  results = ordered_set.OrderedSet()

  def _go(eff_request: EffectRequest):
    results.add(type(eff_request).effect_protocol())

  selectors.select(model_tree).at_instances_of(EffectRequest).apply(_go)
  return list(results)


def broken_handler_refs(model_tree: Any) -> list[HandledEffectRef]:
  """Collects the effect types of each broken `HandledEffectRef` in a (sub)model.

  A broken handler reference occurs when a model contains a `HandledEffectRef`
  and is then removed from the handler for that ref. This is OK if you are
  making a targeted modification and will insert it back into the handler,
  but it can cause problems if the subtree is extracted and used in a different
  context.

  Args:
    model_tree: A pytree for a model or submodel.

  Returns:
    A list of all broken `HandledEffectRef` nodes in the submodel.
  """
  results = ordered_set.OrderedSet()

  def _go(node: EffectHandler | HandledEffectRef, known_hids: set[HandlerId]):
    if isinstance(node, HandledEffectRef):
      if node.handler_id not in known_hids:
        results.add(node)
    elif isinstance(node, EffectHandler):
      sub_known_hids = known_hids | {node.handler_id}
      selectors.select(node).at_children().at_instances_of(
          EffectHandler | HandledEffectRef
      ).apply(functools.partial(_go, known_hids=sub_known_hids))

  selectors.select(model_tree).at_instances_of(
      EffectHandler | HandledEffectRef
  ).apply(functools.partial(_go, known_hids=set()))
  return list(results)


HandlerId: typing.TypeAlias = str


def all_handler_ids(model_tree: Any) -> list[HandlerId]:
  """Collects the set of all handler IDs inside a model or submodel.

  Args:
    model_tree: A pytree for a model or submodel.

  Returns:
    A list of all handler IDs that appear in the model or submodel.
  """
  results = ordered_set.OrderedSet()

  def _go(thing_with_id: HandledEffectRef | EffectHandler):
    results.add(thing_with_id.handler_id)
    if isinstance(thing_with_id, EffectHandler):
      selectors.select(thing_with_id).at_children().at_instances_of(
          HandledEffectRef | EffectHandler
      ).apply(_go)

  selectors.select(model_tree).at_instances_of(
      HandledEffectRef | EffectHandler
  ).apply(_go)

  return list(results)


def infer_or_check_handler_id(
    tag: str,
    subtree: Any,
    explicit_id: str | None = None,
    min_hash_length: int = 5,
) -> HandlerId:
  """Tries to generate a unique handler ID from the structure of a subtree.

  Each handler in Penzai's effect system needs to have a unique ID that
  identifies which effects it is responsible for handling. Since effect requests
  and effect handlers are both PyTree nodes that appear inside the PyTree
  structure, we can generate handler IDs by hashing the structure of the
  subtree, which should almost always result in a unique ID for each handler.
  (Failing this would mean there exists a hash such that hashing that hash
  inside its structure yields the structure again.)

  Args:
    tag: User-readable tag for this handler ID. Mostly to help identify what
      kind of handler this is for debugging purposes.
    subtree: Subtree with effects that is being wrapped by the effect handler.
    explicit_id: An explicit ID requested by the user. We will use this if
      provided and if it does not already appear in the structure.
    min_hash_length: Minimum number of hash characters to append to try to make
      IDs unique. Note that we may append more characters as needed to ensure
      uniqueness.

  Returns:
    A unique handler ID which does not already appear in this structure.

  Raises:
    RuntimeError: If there is a hash collision, and increasing the length of
      the hash did not resolve the collision.
    ValueError: If there was an explicit ID provided, but it conflicts with an
      existing ID in the model.
  """
  existing = all_handler_ids(subtree)
  if explicit_id is not None:
    if explicit_id in existing:
      raise ValueError(
          f"Explicit handler ID {explicit_id} already exists in the subtree!"
      )
    return explicit_id
  # Extract and hash the structure. We just hash the string representation of
  # the pytreedef, which isn't necessarily unique but should be close enough
  # for our purposes. Note that any already-bound effects will have their
  # handler IDs actually shown inside this structure, since handler IDs are not
  # PyTree nodes.
  structure = jax.tree_util.tree_structure(subtree)
  digest = hashlib.sha256(repr(structure).encode("utf-8")).hexdigest()
  long_id = f"{tag}_{digest}"
  # Avoid conflicts.
  potential_conflicts = {hid for hid in existing if long_id.startswith(hid)}
  for length in range(min_hash_length, len(digest)):
    handler_id = f"{tag}_{digest[:length]}"
    if handler_id not in potential_conflicts:
      return handler_id
  # If we get here, we couldn't find a unique ID.
  raise RuntimeError(
      f"Hash collision while generating a unique ID for tag {tag} with subtree"
      f" hash {digest}! This usually means you have a very large number of"
      " handlers and are also extremely unlucky. Try providing an explicit"
      " name for the handler or making a small change to your model's"
      " structure."
  )


@struct.pytree_dataclass
class HandledEffectRef(struct.Struct, abc.ABC):
  """Base class for references to a handler that handles this effect.

  `HandledEffectRef` nodes are used as sentinel markers in PyTrees to indicate
  which handler is responsible for handling a given effect. They replace
  EffectRequest nodes when a handler is constructed, and are in turn replaced by
  implementation nodes when that handler runs. Each effect handler should
  define its own effect reference type, subclassing this type.

  Handler refs are required to be hashable, which means they shouldn't contain
  array data.

  Attributes:
    handler_id: The ID of the handler that is responsible for handling this
      effect.
  """

  handler_id: HandlerId = dataclasses.field(metadata={"pytree_node": False})

  @classmethod
  @abc.abstractmethod
  def effect_protocol(cls) -> type[Any]:
    """Returns the protocol that this effect request handles."""
    raise NotImplementedError(
        "Subclasses of HandledEffectRef must override the effect_protocol class"
        " method"
    )

  def __getattr__(self, attr: str):
    """Fallback to get a missing method from an HandledEffectRef."""
    eff_proto = self.effect_protocol()
    if hasattr(eff_proto, attr) and callable(getattr(eff_proto, attr)):
      # This attribute is a protocol method of the effect. This probably means
      # that some layer is trying to access an effect that we haven't yet
      # actually replaced with an implementation.
      def unhandled_effect_stub(*args, **kwargs):
        del args, kwargs
        raise UnhandledEffectError(
            f"Effect method {repr(attr)} was called on an unbound effect"
            f" reference {self}! Layers with side"
            " effects should only be called using the handler that handles"
            " those effects. This error usually means that an effectful node"
            " was removed from its handler without actually handling the"
            " effect first."
        )

      return unhandled_effect_stub
    else:
      raise AttributeError(
          f"'{type(self).__name__}' object has no attribute '{attr}'",
          name=attr,
          obj=self,
      )

  def treescope_color(self):
    return get_effect_color(self.effect_protocol())

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    from penzai.deprecated.v1.data_effects import _treescope_handlers  # pylint: disable=g-import-not-at-top

    return _treescope_handlers.handle_data_effects_objects(
        self, path, subtree_renderer
    )


class EffectRuntimeImpl(abc.ABC):
  """Base class for runtime effect implementations.

  Subclasses of ``EffectRuntimeImpl`` are created by effect handlers and are
  inserted into any layer that needs access to the effect while the model is
  actually being called at runtime. They are responsible for providing the
  implementation of the effect protocol according to the handler's
  configuration.

  ``EffectRuntimeImpl`` should not usually be kept around outside of the
  handler's ``__call__`` method, as they may contain mutable state that is
  incompatible with JAX. In some cases they can be used temporarily outside of
  their handler, but models containing ``EffectRuntimeImpl`` should never be the
  "source of truth" for a model's structure or parameters.

  Each implementation must implement all of the methods of the effect protocol
  that it is replacing, but the details are up to the handler as long as it is
  a subclass of this class. Note that implementations are not required to be
  valid PyTree nodes and can thus contain e.g. mutable state.

  Advanced: Higher-order layers that require their sublayers to be valid PyTrees
  (e.g. if they use scan or map over their sublayers) are responsible for
  swapping these out for something compatible. The semantics of this are
  determined by the higher-order layer, and it's allowed to just raise an
  exception if there is an unrecognized implementation that cannot be handled in
  a generic way.
  """

  @classmethod
  @abc.abstractmethod
  def effect_protocol(cls) -> type[Any]:
    """Returns the protocol that this effect request handles."""
    raise NotImplementedError(
        "Subclasses of EffectRuntimeImpl must override effect_protocol"
    )

  @abc.abstractmethod
  def handler_id(self) -> HandlerId:
    """Returns the ID of the handler that created this implementation."""
    raise NotImplementedError(
        "Subclasses of EffectRuntimeImpl must override handler_id"
    )

  def treescope_color(self):
    return get_effect_color(self.effect_protocol())

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    from penzai.deprecated.v1.data_effects import _treescope_handlers  # pylint: disable=g-import-not-at-top

    return _treescope_handlers.handle_data_effects_objects(
        self, path, subtree_renderer
    )


@struct.pytree_dataclass
class EffectHandler(layer_base.Layer, abc.ABC):
  """A handler for a particular effect.

  Subclasses of ``EffectHandler`` are responsible for replacing effect requests
  with effect references when constructed, and then replacing those references
  with concrete implementations when called.
  Subclasses of this class are by convention called "With{X}" where X is some
  description of the effect that occurs when the handler is called (e.g.
  "WithRandomStream", "WithMutableLocalState", "WithSideInputs").

  Handlers must define the attributes ``handler_id`` and ``body`` but may also
  define additional attributes as needed.

  Implementers of new handlers should implement a constructor class method
  (usually not ``__init__`` but an explicit class method called something like
  ``handling`` or ``from_submodel``). This constructor class method should
  generate a new handler ID using `infer_or_check_handler_id`, set that as its
  handler ID, and replace all `EffectRequest` instances that it knows how to
  handle with `HandledEffectRef` instances that reference this handler ID. They
  should also override ``__call__`` to create a temporary copy of the submodel
  ``body`` where each of the `HandledEffectRef` instances are replaced with some
  instance of `EffectRuntimeImpl`, then call that copy.

  Handlers are free to modify the inputs or outputs of the submodel as needed,
  e.g. by accepting extra information as an input or returning additional
  information as output. However, the inputs should be in the form of a single
  Python argument (e.g. a tuple, dictionary, or dataclass) and similarly the
  output should be a single Python value. This ensures that multiple handlers
  can be nested together without conflicts.

  Attributes:
    handler_id: The ID of this handler.
    body: The layer that this handler wraps.
  """

  handler_id: HandlerId = dataclasses.field(metadata={"pytree_node": False})
  body: layer_base.LayerLike

  @classmethod
  @abc.abstractmethod
  def effect_protocol(cls) -> type[Any] | Collection[type[Any]] | None:
    """Returns the effect protocol(s) that this handler handles.

    Advanced handlers are allowed to handle multiple effects, and the specific
    effect interfaces are determined by the references inside ``body``. This
    method is used primarily to aid debugging and visualization.

    Returns:
      A single effect protocol if applicable. Can also return a collection of
      protocols or None if the handler is not associated with a specific
      effect.
    """
    raise NotImplementedError(
        "Subclasses of EffectHandler must override effect_protocol"
    )

  def treescope_color(self):
    from treescope import formatting_util  # pylint: disable=g-import-not-at-top

    protocol = self.effect_protocol()
    if isinstance(protocol, type):
      return get_effect_color(protocol)
    else:
      return formatting_util.color_from_string(type(self).__qualname__)

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    from penzai.deprecated.v1.data_effects import _treescope_handlers  # pylint: disable=g-import-not-at-top

    return _treescope_handlers.handle_data_effects_objects(
        self, path, subtree_renderer
    )
