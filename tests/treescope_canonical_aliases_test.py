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

"""Tests for treescope's canonical_aliases."""

import types

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy
import penzai.core.named_axes
import penzai.core.struct
from tests import fixtures as fixture_parent
from tests.fixtures import treescope_examples_fixture as fixture_lib
from penzai.treescope import canonical_aliases


def fresh_canonical_aliases():
  return canonical_aliases._alias_environment.set_scoped(
      canonical_aliases.CanonicalAliasEnvironment({}, [])
  )


class TreescopeCanonicalAliasesTest(parameterized.TestCase):

  def test_module_attribute_path_string(self):
    self.assertEqual(
        str(canonical_aliases.ModuleAttributePath("foo.bar", ("baz", "Qux"))),
        "foo.bar.baz.Qux",
    )
    self.assertEqual(
        str(canonical_aliases.ModuleAttributePath("__main__", ("Foo",))),
        "Foo",
    )

  def test_module_attribute_path_retrieve(self):
    path = canonical_aliases.ModuleAttributePath(
        fixture_lib.__name__, ("DataclassWithTwoChildren",)
    )
    self.assertIs(path.retrieve(), fixture_lib.DataclassWithTwoChildren)

  def test_module_attribute_path_retrieve_nested(self):
    path = canonical_aliases.ModuleAttributePath(
        fixture_lib.__name__, ("SomeOuterClass", "NestedClass")
    )
    self.assertIs(path.retrieve(), fixture_lib.SomeOuterClass.NestedClass)

  def test_module_attribute_path_retrieve_error(self):
    path = canonical_aliases.ModuleAttributePath(
        fixture_lib.__name__, "NonexistentClass"
    )
    with self.assertRaises(ValueError):
      path.retrieve()

    self.assertIsNone(path.retrieve(forgiving=True))

  def test_local_name_path_str(self):
    self.assertEqual(
        str(canonical_aliases.LocalNamePath("foo", ("baz", "Qux"))),
        "foo.baz.Qux",
    )
    self.assertEqual(str(canonical_aliases.LocalNamePath("bar", ())), "bar")

  def test_local_name_path_retrieve(self):
    path = canonical_aliases.LocalNamePath("foo", ("bar", "baz"))
    self.assertEqual(
        path.retrieve(
            {"foo": types.SimpleNamespace(bar=types.SimpleNamespace(baz=1234))}
        ),
        1234,
    )

  def test_add_and_look_up_alias(self):
    with fresh_canonical_aliases():
      self.assertIsNone(
          canonical_aliases.lookup_alias(
              fixture_lib.SomeOuterClass.NestedClass,
              infer_from_attributes=False,
          ),
      )
      canonical_aliases.add_alias(
          fixture_lib.SomeOuterClass.NestedClass,
          canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__,
              ("SomeOuterClass", "NestedClass"),
          ),
      )
      self.assertEqual(
          canonical_aliases.lookup_alias(
              fixture_lib.SomeOuterClass.NestedClass,
              infer_from_attributes=False,
          ),
          canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__,
              ("SomeOuterClass", "NestedClass"),
          ),
      )

  def test_infer_alias_from_attributes(self):
    with fresh_canonical_aliases():
      self.assertEqual(
          canonical_aliases.lookup_alias(
              fixture_lib.SomeOuterClass.NestedClass,
              infer_from_attributes=True,
          ),
          canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__,
              ("SomeOuterClass", "NestedClass"),
          ),
      )

  def test_add_uninferabble_class(self):
    # ClassDefinedInAFunction is defined inside a function, so it's __qualname__
    # includes "<locals>"
    with fresh_canonical_aliases():
      self.assertIsNone(
          canonical_aliases.lookup_alias(
              fixture_lib.ClassDefinedInAFunction,
              infer_from_attributes=True,
          ),
      )
      canonical_aliases.add_alias(
          fixture_lib.ClassDefinedInAFunction,
          canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__,
              ("ClassDefinedInAFunction",),
          ),
      )
      self.assertEqual(
          canonical_aliases.lookup_alias(
              fixture_lib.ClassDefinedInAFunction,
              infer_from_attributes=True,
          ),
          canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__,
              ("ClassDefinedInAFunction",),
          ),
      )

  def test_add_bad_alias(self):
    with fresh_canonical_aliases():
      with self.assertRaises(ValueError):
        canonical_aliases.add_alias(
            fixture_lib.SomeOuterClass.NestedClass,
            canonical_aliases.ModuleAttributePath(
                fixture_lib.__name__,
                ("not_a_correct_path", "NestedClass"),
            ),
        )

  def test_outdated_alias(self):
    # sanity checks
    self.assertEqual(
        fixture_lib.OriginalRedefinedClass.__name__,
        "RedefinedClass",
    )
    self.assertIsNot(
        fixture_lib.RedefinedClass,
        fixture_lib.OriginalRedefinedClass,
    )
    # Make sure canonical aliases don't get confused. (This can happen in some
    # cases due to module reloads, for instance.)
    with fresh_canonical_aliases():
      self.assertIsNone(
          canonical_aliases.lookup_alias(
              fixture_lib.OriginalRedefinedClass,
              infer_from_attributes=True,
          )
      )
      self.assertEqual(
          canonical_aliases.lookup_alias(
              fixture_lib.OriginalRedefinedClass,
              infer_from_attributes=True,
              allow_outdated=True,
          ),
          canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__,
              ("RedefinedClass",),
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="ordinary_class",
          candidate=fixture_lib.DataclassWithOneChild,
          expected_result=True,
      ),
      dict(
          testcase_name="ordinary_function",
          candidate=fixture_lib.make_class_with_weird_qualname,
          expected_result=True,
      ),
      dict(
          testcase_name="moved_internal_class",
          candidate=fixture_lib.ClassDefinedInAFunction,
          alias=canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__, ("ClassDefinedInAFunction",)
          ),
          expected_result=True,
      ),
      dict(
          testcase_name="confusingly_renamed_internal_class",
          candidate=fixture_lib.RedefinedClass,
          alias=canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__, ("OriginalRedefinedClass",)
          ),
          expected_result=False,
      ),
      dict(
          testcase_name="private_class",
          candidate=fixture_lib._PrivateClass,
          expected_result=False,
      ),
      dict(
          testcase_name="private_function",
          candidate=fixture_lib._private_function,
          expected_result=False,
      ),
      dict(
          testcase_name="attribute_of_private_object",
          candidate=fixture_lib._PrivateClass.some_function,
          alias=canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__,
              ("_PrivateClass", "some_function"),
          ),
          expected_result=False,
      ),
      dict(
          testcase_name="wrapped_functionlike_object",
          candidate=fixture_lib.wrapped_function,
          alias=canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__, ("wrapped_function",)
          ),
          expected_result=True,
      ),
      dict(
          testcase_name="callable_object_but_not_wrapped_function",
          candidate=fixture_lib.some_callable_block,
          alias=canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__, ("some_callable_block",)
          ),
          expected_result=False,
      ),
      dict(
          testcase_name="immutable_constant",
          candidate=fixture_lib.immutable_constant,
          alias=canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__, ("immutable_constant",)
          ),
          expected_result=False,
      ),
      dict(
          testcase_name="mutable_constant",
          candidate=fixture_lib.mutable_constant,
          alias=canonical_aliases.ModuleAttributePath(
              fixture_lib.__name__, ("mutable_constant",)
          ),
          expected_result=True,
      ),
  )
  def test_default_well_known_filter(
      self, candidate, expected_result, alias=None
  ):
    with fresh_canonical_aliases():
      if alias is None:
        alias = canonical_aliases.lookup_alias(
            candidate, infer_from_attributes=True
        )
        assert alias is not None

      self.assertEqual(
          canonical_aliases.default_well_known_filter(candidate, alias),
          expected_result,
      )

  def test_local_aliases(self):
    with fresh_canonical_aliases():
      with canonical_aliases.relative_alias_names({
          "DataclassOne": fixture_lib.DataclassWithOneChild,
          "SomeOuterClass": fixture_lib.SomeOuterClass,
          "fixture_lib": fixture_lib,
      }):
        with self.subTest("class_alias"):
          self.assertEqual(
              canonical_aliases.lookup_alias(
                  fixture_lib.DataclassWithOneChild,
                  allow_relative=True,
              ),
              canonical_aliases.LocalNamePath(
                  local_name="DataclassOne", attribute_path=()
              ),
          )
        with self.subTest("parent_class_alias"):
          self.assertEqual(
              canonical_aliases.lookup_alias(
                  fixture_lib.SomeOuterClass.NestedClass,
                  allow_relative=True,
              ),
              canonical_aliases.LocalNamePath(
                  local_name="SomeOuterClass",
                  attribute_path=("NestedClass",),
              ),
          )
        with self.subTest("module_alias"):
          self.assertEqual(
              canonical_aliases.lookup_alias(
                  fixture_lib.DataclassWithTwoChildren,
                  allow_relative=True,
              ),
              canonical_aliases.LocalNamePath(
                  local_name="fixture_lib",
                  attribute_path=("DataclassWithTwoChildren",),
              ),
          )

      with canonical_aliases.relative_alias_names({
          "fixture_parent": fixture_parent,
      }):
        with self.subTest("parent_module_alias"):
          self.assertEqual(
              canonical_aliases.lookup_alias(
                  fixture_lib.DataclassWithTwoChildren,
                  allow_relative=True,
              ),
              canonical_aliases.LocalNamePath(
                  local_name="fixture_parent",
                  attribute_path=(
                      "treescope_examples_fixture",
                      "DataclassWithTwoChildren",
                  ),
              ),
          )
        with self.subTest("local_module_name_helper"):
          self.assertEqual(
              canonical_aliases.maybe_local_module_name(fixture_lib),
              "fixture_parent.treescope_examples_fixture",
          )

      with self.subTest("magic_aliases"):
        # Will detect "fixture_lib" in scope using stack inspection
        with canonical_aliases.relative_alias_names("magic"):
          self.assertEqual(
              canonical_aliases.lookup_alias(
                  fixture_lib.DataclassWithTwoChildren,
                  allow_relative=True,
              ),
              canonical_aliases.LocalNamePath(
                  local_name="fixture_lib",
                  attribute_path=("DataclassWithTwoChildren",),
              ),
          )

  @parameterized.named_parameters(
      dict(
          testcase_name="penzai.core.struct.Struct",
          target=penzai.core.struct.Struct,
          alias_string="penzai.core.struct.Struct",
      ),
      dict(
          testcase_name="penzai.core.named_axes",
          target=penzai.core.named_axes,
          alias_string="penzai.core.named_axes",
      ),
      dict(
          testcase_name="penzai.core.named_axes.NamedArray",
          target=penzai.core.named_axes.NamedArray,
          alias_string="penzai.core.named_axes.NamedArray",
      ),
      dict(
          testcase_name="penzai.core.named_axes.nmap",
          target=penzai.core.named_axes.nmap,
          alias_string="penzai.core.named_axes.nmap",
      ),
      dict(
          testcase_name="numpy.ndarray",
          target=numpy.ndarray,
          alias_string="numpy.ndarray",
      ),
      dict(
          testcase_name="jax.ShapeDtypeStruct",
          target=jax.ShapeDtypeStruct,
          alias_string="jax.ShapeDtypeStruct",
      ),
      dict(
          testcase_name="jax.vmap",
          target=jax.vmap,
          alias_string="jax.vmap",
      ),
      dict(
          testcase_name="jax.nn.relu",
          target=jax.nn.relu,
          alias_string="jax.nn.relu",
      ),
      dict(
          testcase_name="jax.nn.softmax",
          target=jax.nn.softmax,
          alias_string="jax.nn.softmax",
      ),
      dict(
          testcase_name="jax.numpy.sum",
          target=jax.numpy.sum,
          alias_string="jax.numpy.sum",
      ),
      dict(
          testcase_name="jax.scipy.special.logsumexp",
          target=jax.scipy.special.logsumexp,
          alias_string="jax.scipy.special.logsumexp",
      ),
  )
  def test_default_canonical_aliases(self, target, alias_string):
    canonical_aliases.update_lazy_aliases()
    self.assertEqual(str(canonical_aliases.lookup_alias(target)), alias_string)


if __name__ == "__main__":
  absltest.main()
