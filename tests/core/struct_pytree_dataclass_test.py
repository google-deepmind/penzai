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

"""Tests for core pytree_dataclass and Struct abstractions."""

import collections
import dataclasses
import re
from absl.testing import absltest
import jax
import penzai.core.selectors
import penzai.core.struct
import treescope.dataclass_util

# pylint: disable=unused-variable


class StructTest(absltest.TestCase):

  def test_pytree_dataclass_properties(self):

    @penzai.core.struct.pytree_dataclass
    class SomePytreeDataclass:
      foo: int
      bar: int

      def tree_flatten_with_keys(self):
        children = (("key_for_foo", self.foo),)
        aux_data = {"bar": self.bar}
        return children, aux_data

      @classmethod
      def tree_unflatten(cls, aux_data, children):
        """Unflattens this tree node."""
        return cls(foo=children[0], bar=aux_data["bar"])

    some_instance = SomePytreeDataclass(3, 4)

    with self.subTest("is_pytree_with_keys"):
      keys_and_values, treedef = jax.tree_util.tree_flatten_with_path(
          some_instance
      )
      self.assertEqual(
          keys_and_values,
          [(
              ("key_for_foo",),
              3,
          )],
      )

      rebuilt = treedef.unflatten([value for _, value in keys_and_values])
      self.assertEqual(rebuilt, some_instance)

    with self.subTest("is_dataclass"):
      self.assertTrue(dataclasses.is_dataclass(SomePytreeDataclass))
      self.assertTrue(dataclasses.is_dataclass(some_instance))
      self.assertEqual(
          [field.name for field in dataclasses.fields(SomePytreeDataclass)],
          ["foo", "bar"],
      )

    with self.subTest("is_pytree_dataclass"):
      self.assertTrue(
          penzai.core.struct.is_pytree_dataclass_type(SomePytreeDataclass)
      )
      self.assertFalse(
          penzai.core.struct.is_pytree_dataclass_type(some_instance)
      )

    with self.subTest("has_generated_repr"):
      self.assertTrue(hasattr(SomePytreeDataclass, "__repr__"))
      self.assertEndsWith(
          SomePytreeDataclass.__repr__.__qualname__,
          "SomePytreeDataclass.__repr__",
      )

    with self.subTest("subclass_is_not_pytree_dataclass"):

      class SomePytreeDataclassSubclass(SomePytreeDataclass):
        pass

      # not a pytree dataclass without a decorator (in particular, not a pytree)
      self.assertFalse(
          penzai.core.struct.is_pytree_dataclass_type(
              SomePytreeDataclassSubclass
          )
      )

  def test_pytree_dataclass_requires_pytree_methods(self):
    with self.assertRaisesRegex(
        AttributeError,
        re.escape(
            "IncompletePytreeDataclass must define both"
            " `tree_flatten_with_keys` and `tree_unflatten`"
        ),
    ):

      @penzai.core.struct.pytree_dataclass
      class IncompletePytreeDataclass:
        foo: int
        bar: int

  def test_struct_pytree_structure(self):
    @penzai.core.struct.pytree_dataclass
    class SomeStruct(penzai.core.struct.Struct):
      foo: int
      bar: int = dataclasses.field(metadata={"pytree_node": False})
      baz: int
      qux: str = dataclasses.field(metadata={"pytree_node": False})

    some_instance = SomeStruct(3, 4, 5, "six")

    keys_and_values, treedef = jax.tree_util.tree_flatten_with_path(
        some_instance
    )
    foo_key = jax.tree_util.GetAttrKey("foo")
    baz_key = jax.tree_util.GetAttrKey("baz")
    self.assertEqual(
        keys_and_values,
        [((foo_key,), 3), ((baz_key,), 5)],
    )

    rebuilt = treedef.unflatten([value for _, value in keys_and_values])
    self.assertEqual(some_instance, rebuilt)

  def test_pytree_dataclass_field_inheritance_disallowed_by_default(self):

    @penzai.core.struct.pytree_dataclass
    class BaseStruct(penzai.core.struct.Struct):
      foo: int
      bar: int

    with self.subTest("disallowed_by_default"):
      with self.assertRaisesRegex(
          penzai.core.struct.PyTreeDataclassSafetyError,
          re.escape(
              "BadSubStruct has missing or badly-ordered dataclass attributes:"
              " the explicit field annotations ['qux', 'bar'] don't match the"
              " inferred dataclass fields ['foo', 'bar', 'qux']."
          ),
      ):

        @penzai.core.struct.pytree_dataclass
        class BadSubStruct(BaseStruct):
          qux: int
          bar: int

    with self.subTest("opt_in"):

      @penzai.core.struct.pytree_dataclass(has_implicitly_inherited_fields=True)
      class OptedInSubStruct(BaseStruct):
        qux: int
        bar: int

      self.assertEqual(
          [field.name for field in dataclasses.fields(OptedInSubStruct)],
          ["foo", "bar", "qux"],
      )
      self.assertEqual(
          OptedInSubStruct(1, 2, 3),
          OptedInSubStruct(foo=1, bar=2, qux=3),
      )

    with self.subTest("must_have_inheritance_if_opted_in"):
      with self.assertRaisesRegex(
          penzai.core.struct.PyTreeDataclassSafetyError,
          re.escape(
              "UnnecessaryOptInSubStruct was constructed with"
              " `has_implicitly_inherited_fields=True`, but it does not have"
              " any implicitly inherited fields."
          ),
      ):

        @penzai.core.struct.pytree_dataclass(
            has_implicitly_inherited_fields=True
        )
        class UnnecessaryOptInSubStruct(BaseStruct):
          foo: int
          bar: int
          qux: int

  def test_pytree_dataclass_mutable_init_proxy_enabled(self):
    testcase_self = self

    @penzai.core.struct.pytree_dataclass
    class StructWithMutableInitProxy(penzai.core.struct.Struct):
      foo: int
      bar: int

      def some_instance_method(self):
        return self.foo + self.bar

      def __init__(self, init_arg: int):
        self.foo = init_arg
        self.bar = init_arg + 1
        self._bad_attribute_that_wont_be_saved = 1234
        # in __init__, proxy `self` type is NOT the same as the original type
        testcase_self.assertIsNot(type(self), StructWithMutableInitProxy)
        testcase_self.assertEndsWith(
            type(self).__qualname__,
            "StructWithMutableInitProxy._MutableInitProxy",
        )
        # It's not a pytree, but it is an instance of (a subclass of)
        # StructWithMutableInitProxy
        testcase_self.assertFalse(
            penzai.core.struct.is_pytree_dataclass_type(self)
        )
        testcase_self.assertSequenceEqual(
            jax.tree_util.tree_leaves(self), [self]
        )
        testcase_self.assertIsInstance(self, StructWithMutableInitProxy)
        testcase_self.assertEqual(
            self.some_instance_method(), self.foo + self.bar
        )

    instance = StructWithMutableInitProxy(init_arg=10)

    self.assertIsInstance(instance, StructWithMutableInitProxy)
    self.assertIs(type(instance), StructWithMutableInitProxy)
    self.assertFalse(hasattr(instance, "_bad_attribute_that_wont_be_saved"))
    self.assertSequenceEqual(jax.tree_util.tree_leaves(instance), [10, 11])
    # repr also includes some unnecessary qualname stuff
    self.assertEndsWith(
        repr(instance), "StructWithMutableInitProxy(foo=10, bar=11)"
    )

  def test_pytree_dataclass_mutable_init_proxy_disabled(self):
    testcase_self = self

    @penzai.core.struct.pytree_dataclass(use_mutable_proxy_in_init=False)
    class StructWithoutMutableInitProxy(penzai.core.struct.Struct):
      foo: int
      bar: int

      def __init__(self, init_arg: int):
        with testcase_self.assertRaises(dataclasses.FrozenInstanceError):
          self.foo = init_arg

        testcase_self.assertIs(type(self), StructWithoutMutableInitProxy)
        object.__setattr__(self, "foo", init_arg)
        object.__setattr__(self, "bar", init_arg + 1)
        testcase_self.assertSequenceEqual(
            jax.tree_util.tree_leaves(self), [10, 11]
        )

    instance = StructWithoutMutableInitProxy(init_arg=10)
    self.assertEndsWith(
        repr(instance), "StructWithoutMutableInitProxy(foo=10, bar=11)"
    )

  def test_pytree_dataclass_overwrite_parent_init(self):

    @penzai.core.struct.pytree_dataclass
    class StructWithCustomInit(penzai.core.struct.Struct):
      foo: int
      bar: int

      def __init__(self, init_arg: int):
        self.foo = init_arg
        self.bar = init_arg + 1

    with self.subTest("disallowed_by_default"):
      with self.assertRaisesRegex(
          penzai.core.struct.PyTreeDataclassSafetyError,
          re.escape(
              "@pytree_dataclass decorator for CustomInitSubclassDefault is"
              " about to overwrite an inherited custom __init__ "
          )
          + r"<.*StructWithCustomInit\.__init__.*>",
      ):

        @penzai.core.struct.pytree_dataclass
        class CustomInitSubclassDefault(StructWithCustomInit):
          foo: int
          bar: int

    with self.subTest("opt_in"):

      @penzai.core.struct.pytree_dataclass(overwrite_parent_init=True)
      class CustomInitSubclassOptIn(StructWithCustomInit):
        foo: int
        bar: int

      instance = CustomInitSubclassOptIn(foo=3, bar=2)  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
      self.assertEqual(instance.foo, 3)
      self.assertEqual(instance.bar, 2)

    with self.subTest("opt_out"):

      @penzai.core.struct.pytree_dataclass(init=False)
      class CustomInitSubclassOptOut(StructWithCustomInit):
        foo: int
        bar: int

      instance = CustomInitSubclassOptOut(init_arg=10)  # pylint: disable=unexpected-keyword-arg
      self.assertEqual(instance.foo, 10)
      self.assertEqual(instance.bar, 11)

    with self.subTest("manual_override"):

      @penzai.core.struct.pytree_dataclass
      class CustomInitSubclassManual(StructWithCustomInit):
        foo: int
        bar: int

        def __init__(self, new_init_arg: int):  # pylint: disable=super-init-not-called
          self.foo = new_init_arg
          self.bar = new_init_arg + 2

      instance = CustomInitSubclassManual(new_init_arg=10)
      self.assertEqual(instance.foo, 10)
      self.assertEqual(instance.bar, 12)

  def test_pytree_dataclass_overwrite_parent_warning_ignores_generated(self):

    @penzai.core.struct.pytree_dataclass
    class StructWithGeneratedInit(penzai.core.struct.Struct):
      foo: int
      bar: int

    # these are the defaults, but written out explicitly here for clarity:
    @penzai.core.struct.pytree_dataclass(init=True, overwrite_parent_init=False)
    class StructWithGeneratedInit2(StructWithGeneratedInit):
      foo: int
      bar: int
      baz: int

    instance = StructWithGeneratedInit2(1, 2, 3)
    self.assertEqual(instance.foo, 1)
    self.assertEqual(instance.bar, 2)
    self.assertEqual(instance.baz, 3)

  def test_undecorated_struct_is_abstract(self):
    # no pytree_dataclass decorator:
    class MyAbstractStruct(penzai.core.struct.Struct):
      foo: int
      bar: int

    with self.assertRaisesRegex(
        TypeError,
        re.escape("Can't instantiate abstract Struct subclass ")
        + r"\<class .*MyAbstractStruct.*\>",
    ):
      _ = MyAbstractStruct(1, 2)

  def test_struct_attribute_helpers(self):
    @penzai.core.struct.pytree_dataclass
    class StructWithCustomInit(penzai.core.struct.Struct):
      foo: int
      bar: int

      def __init__(self, init_arg: int):
        self.foo = init_arg
        self.bar = init_arg + 1

    instance = StructWithCustomInit(init_arg=10)
    with self.subTest("attributes_dict"):
      self.assertEqual(instance.attributes_dict(), {"foo": 10, "bar": 11})
    with self.subTest("from_attributes"):
      self.assertEqual(
          StructWithCustomInit.from_attributes(foo=10, bar=11), instance
      )

  def test_struct_select(self):
    @penzai.core.struct.pytree_dataclass
    class SomeStruct(penzai.core.struct.Struct):
      foo: int
      bar: int
      name: str = dataclasses.field(metadata={"pytree_node": False})

    self.assertEqual(
        SomeStruct(1, 2, "my_block").select(),
        penzai.core.selectors.Selection(
            selected_by_path=collections.OrderedDict(
                {(): SomeStruct(foo=1, bar=2, name="my_block")}
            ),
            remainder=penzai.core.selectors.SelectionHole(path=()),
        ),
    )

  def test_struct_repr_inherited(self):
    @penzai.core.struct.pytree_dataclass
    class SomeStruct(penzai.core.struct.Struct):
      foo: int

    self.assertIs(SomeStruct.__repr__, penzai.core.struct.Struct.__repr__)

  def test_struct_keypath_custom_keys(self):
    @penzai.core.struct.pytree_dataclass
    class StructWithCustomKeys(penzai.core.struct.Struct):
      foo: int
      bar: int
      name: str = dataclasses.field(metadata={"pytree_node": False})

      def key_for_field(self, field_name: str):
        return f"my_custom_key_for_{field_name}"

    some_instance = StructWithCustomKeys(3, 4, "my_block")
    keys_and_values, treedef = jax.tree_util.tree_flatten_with_path(
        some_instance
    )
    self.assertEqual(
        keys_and_values,
        [
            (("my_custom_key_for_foo",), 3),
            (("my_custom_key_for_bar",), 4),
        ],
    )

  def test_dataclass_util(self):

    # Not a pytree dataclass necessarily, just an ordinary one
    @dataclasses.dataclass
    class MyDataclass:
      foo: int
      bar: int

      def __init__(self, init_arg: int):
        self.foo = init_arg
        self.bar = init_arg + 1

    with self.subTest("dataclass_from_attributes"):
      self.assertEqual(
          treescope.dataclass_util.dataclass_from_attributes(
              MyDataclass, foo=10, bar=11
          ),
          MyDataclass(init_arg=10),
      )

    with self.subTest("init_takes_fields_false"):
      self.assertFalse(treescope.dataclass_util.init_takes_fields(MyDataclass))

    @dataclasses.dataclass
    class MyDataclass2:
      foo: int
      bar: int

    with self.subTest("init_takes_fields_true"):
      self.assertTrue(treescope.dataclass_util.init_takes_fields(MyDataclass2))


if __name__ == "__main__":
  absltest.main()
