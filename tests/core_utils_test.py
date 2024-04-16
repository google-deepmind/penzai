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

"""Tests for penzai's core utilities.."""
import dataclasses
import re
from absl.testing import absltest
import jax
from penzai.core import context
from penzai.core import dataclass_util
from penzai.core import tree_util


class ContextualValueTest(absltest.TestCase):
  """Tests for core/context.py."""

  def setUp(self):
    super().setUp()
    # Make sure we aren't in an interactive context.
    context.disable_interactive_context()

  def tearDown(self):
    # Make sure we exit any interactive context from the test.
    context.disable_interactive_context()
    super().tearDown()

  def test_set_scoped(self):
    ctx_value = context.ContextualValue(
        initial_value=1, module=__name__, qualname=None
    )
    self.assertEqual(ctx_value.get(), 1)

    with ctx_value.set_scoped(2):
      self.assertEqual(ctx_value.get(), 2)
      with ctx_value.set_scoped(3):
        self.assertEqual(ctx_value.get(), 3)
      self.assertEqual(ctx_value.get(), 2)

    self.assertEqual(ctx_value.get(), 1)

  def test_set_interactive(self):
    ctx_value = context.ContextualValue(
        initial_value=1, module=__name__, qualname=None
    )
    with self.assertRaisesRegex(
        RuntimeError,
        re.escape(
            "`set_interactive` should only be used in an interactive setting."
        ),
    ):
      ctx_value.set_interactive(2)

    self.assertEqual(ctx_value.get(), 1)

    context.enable_interactive_context()
    ctx_value.set_interactive(2)
    self.assertEqual(ctx_value.get(), 2)

    with ctx_value.set_scoped(3):
      self.assertEqual(ctx_value.get(), 3)

    self.assertEqual(ctx_value.get(), 2)

    context.disable_interactive_context()
    self.assertEqual(ctx_value.get(), 1)


class DataclassUtilTest(absltest.TestCase):
  """Tests for core/dataclass_util.py."""

  def test_dataclass_from_attributes(self):
    @dataclasses.dataclass(frozen=True)
    class MyWeirdInitClass:
      foo: int
      bar: int

      def __init__(self, weird_arg):
        raise NotImplementedError("shouldn't be called")

    value = dataclass_util.dataclass_from_attributes(
        MyWeirdInitClass, foo=3, bar=4
    )
    self.assertEqual(value.foo, 3)
    self.assertEqual(value.bar, 4)

    with self.assertRaisesRegex(
        ValueError, re.escape("Incorrect fields provided to `from_attributes`")
    ):
      _ = dataclass_util.dataclass_from_attributes(MyWeirdInitClass, qux=5)

  def test_init_takes_fields_normal_init(self):
    @dataclasses.dataclass(frozen=True)
    class MyDataclass:
      foo: int
      bar: int = 7

    self.assertTrue(dataclass_util.init_takes_fields(MyDataclass))

  def test_init_takes_fields_custom_compatible_init(self):
    @dataclasses.dataclass
    class MyDataclass:
      foo: int
      bar: int

      def __init__(self, foo: int = 3, bar: int = 4):
        self.foo = foo
        self.bar = bar

    self.assertTrue(dataclass_util.init_takes_fields(MyDataclass))

  def test_init_takes_fields_for_init_with_custom_args(self):
    @dataclasses.dataclass
    class MyDataclass:
      foo: int
      bar: int

      def __init__(self, weird_arg):
        self.foo = weird_arg
        self.bar = weird_arg

    self.assertFalse(dataclass_util.init_takes_fields(MyDataclass))


class TreeUtilTest(absltest.TestCase):
  """Tests for core/tree_util.py."""

  def test_tree_flatten_exactly_one_level_nested_pytree(self):
    result = tree_util.tree_flatten_exactly_one_level(
        {"a": [1, 2], "b": [3, 4]}
    )
    self.assertIsNotNone(result)
    keys_and_subtrees, treedef = result
    self.assertEqual(
        keys_and_subtrees,
        [
            (jax.tree_util.DictKey("a"), [1, 2]),
            (jax.tree_util.DictKey("b"), [3, 4]),
        ],
    )
    self.assertEqual(treedef.unflatten(["x", "y"]), {"a": "x", "b": "y"})

  def test_tree_flatten_exactly_one_level_leaf(self):
    self.assertIsNone(tree_util.tree_flatten_exactly_one_level(42))


if __name__ == "__main__":
  absltest.main()
