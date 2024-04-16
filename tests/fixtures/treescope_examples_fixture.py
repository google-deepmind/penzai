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

"""Defines some classes for use in rendering tests.

This is in a separate module so we can test treescope's handling of module
names during qualified name lookup.
"""

from __future__ import annotations

import dataclasses
import enum
import typing
from typing import Any

import jax
from penzai import pz


class MyTestEnum(enum.Enum):
  FOO = 1
  BAR = 2


@pz.pytree_dataclass
class StructWithOneChild(pz.Struct):
  foo: Any


@dataclasses.dataclass
class DataclassWithOneChild:
  foo: Any


@dataclasses.dataclass
class DataclassWithTwoChildren:
  foo: Any
  bar: Any


@dataclasses.dataclass(frozen=True)
class EmptyDataclass:
  pass


class SomeNamedtupleClass(typing.NamedTuple):
  foo: Any
  bar: Any


class SomeOuterClass:

  @dataclasses.dataclass
  class NestedClass:
    foo: Any


def make_class_with_weird_qualname():
  @dataclasses.dataclass
  class ClassDefinedInAFunction:  # pylint: disable=redefined-outer-name
    foo: Any

  return ClassDefinedInAFunction


ClassDefinedInAFunction = make_class_with_weird_qualname()


class RedefinedClass:
  """A class that will later be redefined."""

  pass


OriginalRedefinedClass = RedefinedClass


class RedefinedClass:  # pylint: disable=function-redefined
  """The redefined class; no longer the same as `_OriginalRedefinedClass`."""

  pass


class _PrivateClass:

  def some_function(self):
    pass


def _private_function():
  pass


class SomeFunctionLikeWrapper:
  func: Any

  def __init__(self, func):
    self.func = func

  def __call__(self, *args, **kwargs):
    return self.func(*args, **kwargs)

  @property
  def __wrapped__(self):
    return self.func


@SomeFunctionLikeWrapper
def wrapped_function():
  pass


immutable_constant = (1, 2, 3)
mutable_constant = [1, 2, 3]


@pz.pytree_dataclass
class ExampleLayer(pz.Layer):
  foo: int

  def __call__(self, value: int) -> int:
    return value + self.foo


# pytype is confused about the dataclass transform here
some_callable_block = ExampleLayer(foo=0)  # pytype: disable=wrong-keyword-args


@jax.tree_util.register_pytree_with_keys_class
class UnknownPytreeNode:
  """A Pytree node treescope doesn't know."""

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __repr__(self):
    return f"<custom repr for UnknownPytreeNode: x={self.x}, y={self.y}>"

  def tree_flatten_with_keys(self):
    return (
        ((jax.tree_util.GetAttrKey("x"), self.x), ("custom_key", self.y)),
        "example_pytree_aux_data",
    )

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


class UnknownObjectWithBuiltinRepr:
  pass


class UnknownObjectWithOneLineRepr:

  def __repr__(self):
    return "<custom repr for UnknownObjectWithOneLineRepr>"


class UnknownObjectWithMultiLineRepr:

  def __repr__(self):
    return "<custom repr\n  for\n  UnknownObjectWithMultiLineRepr\n>"


class UnknownObjectWithBadMultiLineRepr:

  def __repr__(self):
    return "Non-idiomatic\nmultiline\nobject"


@pz.pytree_dataclass
class LayerThatHoldsStuff(pz.Layer):
  stuff: Any

  def input_structure(self):
    return {"input": pz.chk.ArraySpec(shape=(1, 2, 3))}

  def output_structure(self):
    return {"output": pz.chk.ArraySpec(named_shape={"foo": 5})}

  @pz.checked_layer_call
  def __call__(self, value: int) -> int:
    return value
