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

"""Side-input effect, which allows extra values to be injected into layers.

Side inputs are sometimes referred to as "Reader" effects in functional
programming. They allow inner functions to read a value from an environment
without explicitly passing it in.
"""

from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Any, Callable, Generic, Protocol, Sequence

import jax
from penzai.core import selectors
from penzai.core import shapecheck
from penzai.core import struct
from penzai.core import tree_util
from penzai.deprecated.v1.core import layer as layer_base
from penzai.deprecated.v1.data_effects import effect_base

_T = typing.TypeVar("_T")

Tag: typing.TypeAlias = Any


@effect_base.register_effect_color(color="#5eb3e5")
class SideInputEffect(Protocol[_T]):
  """Protocol for a side input effect."""

  def ask(self) -> _T:
    """Retrieves the value for the side input."""
    ...


@struct.pytree_dataclass
class SideInputRequest(Generic[_T], effect_base.EffectRequest):
  """Effect request for a side input.

  Every side input must be associated with a unique identifier.

  Attributes:
    tag: A tag for this side input, identifying what it is.
  """

  tag: Tag = dataclasses.field(metadata={"pytree_node": False})

  @classmethod
  def effect_protocol(cls):
    return SideInputEffect


@struct.pytree_dataclass(has_implicitly_inherited_fields=True)  # pytype: disable=wrong-keyword-args
class HandledSideInputRef(effect_base.HandledEffectRef):
  """Reference for a handled side input effect."""

  tag: Tag = dataclasses.field(metadata={"pytree_node": False})

  @classmethod
  def effect_protocol(cls):
    return SideInputEffect


@dataclasses.dataclass(frozen=True)
class SideInputEffectImpl(Generic[_T], effect_base.EffectRuntimeImpl):
  """Implementation of the side input effect."""

  _value: _T
  _handler_id: effect_base.HandlerId

  def ask(self) -> _T:
    return self._value

  def handler_id(self) -> effect_base.HandlerId:
    return self._handler_id

  @classmethod
  def effect_protocol(cls):
    return SideInputEffect


@struct.pytree_dataclass
class WithSideInputsFromInputTuple(effect_base.EffectHandler):
  """`SideInput` handler that unpacks side inputs from a tuple argument.

  ``WithSideInputsFromInputTuple`` "functionalizes" the `SideInputEffect` effect
  by calling its body with the first element of its argument (a tuple), and then
  passing the remaining elements of the tuple as side inputs.

  Attributes:
    handler_id: The ID of this handler.
    body: The layer that this handler wraps.
    side_input_tags: The tags for each side input.
  """

  handler_id: effect_base.HandlerId = dataclasses.field(
      metadata={"pytree_node": False}
  )
  body: layer_base.LayerLike
  side_input_tags: tuple[Tag, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def input_structure(self):
    return (shapecheck.Wildcard("input to body"),) + tuple(
        shapecheck.Wildcard(f"side input: {tag}")
        for tag in self.side_input_tags
    )

  def output_structure(self):
    return shapecheck.Wildcard("output from body")

  @classmethod
  def effect_protocol(cls):
    return SideInputEffect

  @layer_base.checked_layer_call
  def __call__(self, argument: tuple[Any, ...]):
    inner_arg = argument[0]
    side_inputs = argument[1:]
    impls = {
        tag: SideInputEffectImpl(_value=val, _handler_id=self.handler_id)
        for tag, val in zip(self.side_input_tags, side_inputs)
    }
    handled_body = (
        selectors.select(self.body)
        .at_instances_of(HandledSideInputRef)
        .where(lambda ref: ref.handler_id == self.handler_id)
        .apply(lambda ref: impls[ref.tag])
    )
    return handled_body(inner_arg)

  @classmethod
  def handling(
      cls,
      body: layer_base.LayerLike,
      tags: Sequence[Tag],
      handler_id: str | None = None,
  ) -> WithSideInputsFromInputTuple:
    """Builds a `WithSideInputsFromInputTuple` that handles effects in this layer.

    Args:
      body: The layer to wrap. Usually will contain `SideInputRequest` nodes. If
        any `SideInputRequest` has an explicit structure, all such structures
        must agree and will be used for shapechecking.
      tags: The tags of each of the side inputs we are providing.
      handler_id: ID to use for the handler. If None, will be inferred.

    Returns:
      A ``WithSideInputsFromInputTuple`` handler wrapping `body`, with its side
      input requests with the given tags replaced with references to this
      handler.
    """
    if isinstance(tags, str):
      raise ValueError(
          "WithSideInputsFromInputTuple expects a sequence of tags as its"
          " `tags` argument, but got a string. Strings are technically"
          " sequences of characters, but this is rarely the desired behavior."
          " Please pass an explicit sequence (e.g. a list or tuple)."
      )
    handler_id = effect_base.infer_or_check_handler_id(
        "WithSideInputsFromInputTuple", body, explicit_id=handler_id
    )
    tags = tuple(tags)
    tag_set = set(tags)
    selected_holes = (
        selectors.select(body)
        .at_instances_of(SideInputRequest)
        .where(lambda hole: hole.tag in tag_set)
    )
    return cls(
        handler_id=handler_id,
        body=selected_holes.apply(
            lambda hole: HandledSideInputRef(
                handler_id=handler_id, tag=hole.tag
            )
        ),
        side_input_tags=tags,
    )


@struct.pytree_dataclass
class WithConstantSideInputs(effect_base.EffectHandler):
  """`SideInput` handler that provides side inputs using its own attribute.

  Attributes:
    handler_id: The ID of this handler.
    body: The layer that this handler wraps.
    side_inputs: The value for the side inputs that the handler provides.
  """

  handler_id: effect_base.HandlerId = dataclasses.field(
      metadata={"pytree_node": False}
  )
  body: layer_base.LayerLike
  side_inputs: dict[Tag, Any]

  @classmethod
  def effect_protocol(cls):
    return SideInputEffect

  def __call__(self, argument: tuple[Any, Any]):
    impls = {
        tag: SideInputEffectImpl(_value=val, _handler_id=self.handler_id)
        for tag, val in self.side_inputs.items()
    }
    handled_body = (
        selectors.select(self.body)
        .at_instances_of(HandledSideInputRef)
        .where(lambda ref: ref.handler_id == self.handler_id)
        .apply(lambda ref: impls[ref.tag])
    )
    return handled_body(argument)

  @classmethod
  def handling(
      cls,
      body: layer_base.LayerLike,
      side_inputs: dict[Tag, Any],
      handler_id: str | None = None,
      keep_unused: bool = False,
  ) -> WithConstantSideInputs:
    """Builds a ``WithConstantSideInputs`` that handles effects in this layer.

    Args:
      body: The layer to wrap. Usually will contain `SideInputRequest` nodes.
      side_inputs: The constant values to provide for each tag that we should
        handle.
      handler_id: ID to use for the handler. If None, will be inferred.
      keep_unused: Whether to keep unused side inputs. If False, then any tag
        that isn't actually used by a `SideInputRequest` in the layer will be
        omitted from the handler's attributes.

    Returns:
      A ``WithConstantSideInputs`` handler wrapping ``body``, with its side
      input holes with the given tag replaced with references to this handler.
    """
    handler_id = effect_base.infer_or_check_handler_id(
        "WithConstantSideInputs", body, explicit_id=handler_id
    )
    selected_requests = (
        selectors.select(body)
        .at_instances_of(SideInputRequest)
        .where(lambda req: req.tag in side_inputs)
    )
    used_tags = set()
    for req in selected_requests.get_sequence():
      used_tags.add(req.tag)

    if keep_unused:
      side_inputs_attr = dict(side_inputs)
    else:
      side_inputs_attr = {
          tag: val for tag, val in side_inputs.items() if tag in used_tags
      }
    return cls(
        handler_id=handler_id,
        body=selected_requests.apply(
            lambda req: HandledSideInputRef(handler_id=handler_id, tag=req.tag)
        ),
        side_inputs=side_inputs_attr,
    )


@dataclasses.dataclass(frozen=True, order=True)
class HoistedTag:
  """A tag that has been hoisted out of a handler.

  HoistedTag is used to indicate side inputs that used to be located in some
  other location in a tree and have been moved out of it by
  `hoist_constant_side_inputs`. This ensures that distinct side inputs with
  the same tag do not conflict.

  Attributes:
    source: A string representation of the place this tag was hoisted from.
    tag: The original tag.
  """

  source: str
  tag: Tag


def hoist_constant_side_inputs(
    tree: Any,
    hoist_predicate: Callable[[Tag, Any], bool] = lambda tag, val: True,
    scoped: bool = True,
) -> tuple[Any, dict[Tag, Any]]:
  """Extracts all constant side inputs from a tree so they can be re-handled.

  This function finds all `WithConstantSideInputs` handlers in a tree, and
  extracts the side inputs from those handlers, and replaces the bound side
  input references with unhandled requests. You can then manipulate the tree
  and finally re-handle the side inputs using `WithConstantSideInputs.handling`.

  This function is useful when a tree contains constant side inputs inside a
  subtree, and you want to pull out an even smaller subtree that uses those
  side inputs, or move around subtrees that use side inputs. A particular
  instance of this is shared parameters, which can be represented in Penzai as
  side inputs with a constant value provided by a `WithConstantSideInputs`
  handler. If you want to extract a layer that uses shared parameters, you can
  first hoist out the constant values of all shared parameters, extract the
  submodel you want, and then re-bind the shared parameters of that subtree
  alone.

  Calling `hoist_constant_side_inputs` on a layer and then immediately calling
  `WithConstantSideInputs.handling` on the results will usually produce a layer
  with the same observable behavior, although the tree structure may differ.
  However, if ``scoped`` is False, it is possible that this function will raise
  an error due to tag conflicts.

  Args:
    tree: A tree to hoist constant side inputs from. Usually will contain
      WithConstantSideInputs handlers somewhere inside it.
    hoist_predicate: Optional predicate to determine which side inputs to try to
      hoist, given the tag and value. Defaults to hoisting all side inputs.
    scoped: Whether to modify the tags for the hoisted side inputs so that they
      remain unique even if different side inputs use different tags.

  Returns:
    Tuple `(effectful_tree, hoisted_values)` where `effectful_tree` is a copy
    of `tree` with all hoisted side inputs replaced by `SideInputRequest`
    instances and all `WithConstantSideInputs` removed or modified to not
    include the hoisted side inputs, and `hoisted_values` are the values that
    were bound by those constant side inputs, as needed by
    `WithConstantSideInputs.handling`.
  """
  hoisted_values = {}
  handler_remaps = {}

  def _prepare(subtree, keypath_prefix):
    for keypath, handler in (
        selectors.select(subtree)
        .at_instances_of(WithConstantSideInputs)
        .get_by_path()
        .items()
    ):
      if scoped:
        source = tree_util.pretty_keystr(keypath_prefix + keypath, tree)
        wrap_tag = functools.partial(HoistedTag, source)
      else:
        wrap_tag = lambda tag: tag

      cur_remap = {}
      for tag, value in handler.side_inputs.items():
        if hoist_predicate(tag, value):
          new_tag = wrap_tag(tag)
          if new_tag in hoisted_values:
            raise ValueError(
                f"Tag conflict while hoisting tag {repr(new_tag)}! Try using"
                " `scoped` to ensure the tags are unique."
            )
          hoisted_values[new_tag] = value
          cur_remap[tag] = new_tag
      handler_remaps[handler.handler_id] = cur_remap

      _prepare(
          handler.body,
          keypath_prefix + keypath + (jax.tree_util.GetAttrKey("body"),),
      )

  _prepare(tree, ())

  def _find_and_hoist(subtree):
    return (
        selectors.select(subtree)
        .at_instances_of(
            SideInputRequest | HandledSideInputRef | WithConstantSideInputs
        )
        .apply(_hoister)
    )

  def _hoister(
      node: SideInputRequest | HandledSideInputRef | WithConstantSideInputs,
  ):
    if isinstance(node, SideInputRequest):
      if node.tag in hoisted_values:
        raise ValueError(
            "Cannot hoist a tag that is already used by an unhandled"
            f" SideInputRequest! Found an unhandled request with tag {node.tag}"
        )
      else:
        return node
    elif isinstance(node, HandledSideInputRef):
      if (
          node.handler_id in handler_remaps
          and node.tag in handler_remaps[node.handler_id]
      ):
        return SideInputRequest(handler_remaps[node.handler_id][node.tag])
      else:
        return node
    elif isinstance(node, WithConstantSideInputs):
      assert node.handler_id in handler_remaps
      cur_remap = handler_remaps[node.handler_id]
      # Process the body recursively.
      processed_body = _find_and_hoist(node.body)
      # Hoist any relevant side inputs.
      new_side_inputs = {}
      for tag, value in node.side_inputs.items():
        if tag not in cur_remap:
          new_side_inputs[tag] = value
      if new_side_inputs:
        return WithConstantSideInputs(
            handler_id=node.handler_id,
            body=processed_body,
            side_inputs=new_side_inputs,
        )
      else:
        # Every tag this handler provides has been hoisted.
        return processed_body
    else:
      raise TypeError(f"unexpected node {node}")

  final_tree = _find_and_hoist(tree)
  return final_tree, hoisted_values
