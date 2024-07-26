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
"""Tests for auto_order_types."""

import dataclasses
from absl.testing import absltest
import jax
from penzai.core import auto_order_types


class AutoOrderTypesTest(absltest.TestCase):

  def test_auto_order_types(self):

    @dataclasses.dataclass(frozen=True)
    class Foo(auto_order_types.AutoOrderedAcrossTypes):
      a: int
      b: int

    @dataclasses.dataclass(frozen=True)
    class Bar(auto_order_types.AutoOrderedAcrossTypes):
      name: str

    items = [
        Foo(a=1, b=2),
        Bar(name="bar"),
        Foo(a=3, b=4),
        Bar(name="bar2"),
        "some_string",
        Foo(a=5, b=6),
        Bar(name="bar3"),
    ]

    # Check that we can sort the items.
    _ = sorted(items)

    # Check that they can be used as keys in a dict and passed through JAX
    # pytree operations.
    my_dict = {val: i for i, val in enumerate(items)}
    leaves = jax.tree_util.tree_leaves(my_dict)
    self.assertSameElements(leaves, range(len(items)))


if __name__ == "__main__":
  absltest.main()
