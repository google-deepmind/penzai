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

"""Side-output effect, which allows layers to output intermediate values.

Side outputs are sometimes referred to as "Writer" effects in functional
programming. They allow inner functions to write values to an environment
without explicitly passing it in.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Callable, Generic, Protocol

from penzai.core import selectors
from penzai.core import shapecheck
from penzai.core import struct
from penzai.deprecated.v1.core import layer as layer_base
from penzai.deprecated.v1.data_effects import effect_base

_T = typing.TypeVar("_T")

SideOutputTag = Any


@effect_base.register_effect_color(color="#3fbfb6")
class SideOutputEffect(Protocol[_T]):
  """Protocol for a side output effect."""

  def tell(self, value: _T, /):
    """Writes a value to the side output."""
    ...


@struct.pytree_dataclass
class SideOutputRequest(Generic[_T], effect_base.EffectRequest):
  """Effect request for a side output.

  Since side outputs do not change the behavior of the model,
  ``SideOutputRequest`` implements `tell` (but in a way that it doesn't do
  anything).

  Attributes:
    tag: A tag for this side output, identifying what it is.
  """

  tag: SideOutputTag = dataclasses.field(metadata={"pytree_node": False})

  def tell(self, value: _T, /):
    pass

  @classmethod
  def effect_protocol(cls):
    return SideOutputEffect


@struct.pytree_dataclass
class HandledSideOutputRef(effect_base.HandledEffectRef):
  """Marker for a handled local state effect.

  Attributes:
    handler_id: The ID of the handler that is responsible for handling this
      effect.
    tag: Tag for this side output.
  """

  handler_id: effect_base.HandlerId = dataclasses.field(
      metadata={"pytree_node": False}
  )
  tag: SideOutputTag = dataclasses.field(metadata={"pytree_node": False})

  @classmethod
  def effect_protocol(cls):
    return SideOutputEffect


@struct.pytree_dataclass
class SideOutputValue(struct.Struct):
  """A value written to a side output.

  Attributes:
    keypath: The keypath of the side output.
    tag: The tag of the side output.
    value: The value written to the side output.
  """

  keypath: tuple[Any, ...] = dataclasses.field(metadata={"pytree_node": False})
  tag: SideOutputTag = dataclasses.field(metadata={"pytree_node": False})
  value: Any


@dataclasses.dataclass
class SideOutputEffectImpl(Generic[_T], effect_base.EffectRuntimeImpl):
  """Implementation of the side output effect."""

  _keypath: tuple[Any, ...]
  _tag: SideOutputTag
  _destination: list[_T]
  _handler_id: effect_base.HandlerId

  def tell(self, value: _T, /):
    self._destination.append(SideOutputValue(self._keypath, self._tag, value))

  def handler_id(self) -> effect_base.HandlerId:
    return self._handler_id

  @classmethod
  def effect_protocol(cls):
    return SideOutputEffect


@struct.pytree_dataclass
class CollectingSideOutputs(effect_base.EffectHandler):
  """`SideOutput` handler that collects all side outputs into a list.

  ``CollectingSideOutputs`` takes the same arguments as the wrapped layer, and
  returns the same output, but also returns a list of all side outputs written
  by the handled side output effect objects. Each side output is returned as a
  tuple containing the keypath, tag, and value.
  """

  handler_id: effect_base.HandlerId = dataclasses.field(
      metadata={"pytree_node": False}
  )
  body: layer_base.LayerLike

  def input_structure(self):
    return shapecheck.Wildcard("input to body")

  def output_structure(self):
    return (
        shapecheck.Wildcard("output from body"),
        shapecheck.Wildcard("list of side outputs"),
    )

  @classmethod
  def effect_protocol(cls):
    return SideOutputEffect

  @layer_base.checked_layer_call
  def __call__(self, argument: Any) -> tuple[Any, list[SideOutputValue]]:
    all_outputs = []

    def _make_impl(keypath, ref: HandledSideOutputRef):
      return SideOutputEffectImpl(
          _keypath=keypath,
          _tag=ref.tag,
          _destination=all_outputs,
          _handler_id=self.handler_id,
      )

    handled_body = (
        selectors.select(self.body)
        .at_instances_of(HandledSideOutputRef)
        .where(lambda ref: ref.handler_id == self.handler_id)
        .apply(_make_impl, with_keypath=True)
    )
    result = handled_body(argument)
    return result, all_outputs

  @classmethod
  def handling(
      cls,
      body: layer_base.LayerLike,
      tag: SideOutputTag | None = None,
      tag_predicate: Callable[[SideOutputTag], bool] | None = None,
      handler_id: str | None = None,
  ) -> CollectingSideOutputs:
    """Builds a ``CollectingSideOutputs`` that handles effects in this layer.

    Args:
      body: The layer to wrap. Usually will contain `SideOutputRequest` nodes.
      tag: A tag to collect. If None, defers to ``tag_predicate``.
      tag_predicate: A predicate to use to select which side outputs to collect.
        Should return True for tags to collect. If neither ``tag`` nor
        ``tag_predicate`` is specified, all side outputs are collected.
      handler_id: ID to use for the handler. If None, will be inferred.

    Returns:
      A ``CollectingSideOutputs`` handler wrapping ``body``, with its side
      output holes with the specified tasg replaced with references to this
      handler.
    """
    handler_id = effect_base.infer_or_check_handler_id(
        "WithSideInputFromArg", body, explicit_id=handler_id
    )

    if tag is None and tag_predicate is None:
      tag_predicate = lambda _: True
    elif tag is not None and tag_predicate is not None:
      raise ValueError(
          "Only one of `tag` and `tag_predicate` may be specified."
      )
    elif tag_predicate is None:
      tag_predicate = lambda t: t == tag

    def _make_ref(hole: SideOutputRequest):
      return HandledSideOutputRef(handler_id=handler_id, tag=hole.tag)

    handled_body = (
        selectors.select(body)
        .at_instances_of(SideOutputRequest)
        .where(lambda hole: tag_predicate(hole.tag))
        .apply(_make_ref)
    )
    return cls(handler_id=handler_id, body=handled_body)


@struct.pytree_dataclass
class TellIntermediate(layer_base.Layer):
  """Helper layer that writes its intermediate value to a side output."""

  side_out: SideOutputEffect = SideOutputRequest(tag="intermediate")

  def __call__(self, intermediate_value):
    self.side_out.tell(intermediate_value)
    return intermediate_value

  @classmethod
  def from_config(cls, tag: SideOutputTag) -> TellIntermediate:
    """Builds a TellIntermediate layer that writes to the given tag."""
    return cls(side_out=SideOutputRequest(tag=tag))
