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

"""Defines some classes for use in rendering tests."""

from __future__ import annotations

from typing import Any

from penzai import pz
from penzai.deprecated.v1 import pz as pz_v1


@pz.pytree_dataclass
class StructWithOneChild(pz.Struct):
  foo: Any


@pz.pytree_dataclass
class ExampleLayer(pz_v1.Layer):
  foo: int

  def __call__(self, value: int) -> int:
    return value + self.foo


@pz.pytree_dataclass
class LayerThatHoldsStuff(pz_v1.Layer):
  stuff: Any

  def input_structure(self):
    return {"input": pz.chk.ArraySpec(shape=(1, 2, 3))}

  def output_structure(self):
    return {"output": pz.chk.ArraySpec(named_shape={"foo": 5})}

  @pz_v1.checked_layer_call
  def __call__(self, value: int) -> int:
    return value
