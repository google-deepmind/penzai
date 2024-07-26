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
"""Tests for variables, slots, and layers."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from penzai.core import variables


class VariablesTest(parameterized.TestCase):

  @parameterized.parameters(variables.Parameter, variables.StateVariable)
  def test_variable_basic(self, var_cls):
    var_1 = var_cls(value=1, label="var1", metadata={"a": "b"})
    self.assertEqual(var_1.value, 1)
    self.assertEqual(var_1.label, "var1")
    self.assertEqual(var_1.metadata, {"a": "b"})
    var_1.value = [1, 2, 3]
    self.assertEqual(var_1.value, [1, 2, 3])

  def test_variable_auto_label(self):
    var_1 = variables.StateVariable(value=1)
    self.assertIsInstance(var_1.label, variables.AutoStateVarLabel)
    var_2 = variables.StateVariable(value=2)
    self.assertNotEqual(var_1.label, var_2.label)

  def test_variable_auto_label_scoped(self):
    with variables.scoped_auto_state_var_labels():
      var_1 = variables.StateVariable(value=1)
      with variables.scoped_auto_state_var_labels(group="mygroup"):
        var_2 = variables.StateVariable(value=2)
        var_3 = variables.StateVariable(value=3)
      var_4 = variables.StateVariable(value=4)

    self.assertEqual(
        var_1.label, variables.ScopedStateVarLabel(group=None, index=0)
    )
    self.assertEqual(
        var_2.label, variables.ScopedStateVarLabel(group="mygroup", index=0)
    )
    self.assertEqual(
        var_3.label, variables.ScopedStateVarLabel(group="mygroup", index=1)
    )
    self.assertEqual(
        var_4.label, variables.ScopedStateVarLabel(group=None, index=1)
    )

  def test_variable_under_transform(self):
    def with_local_var(x):
      var = variables.StateVariable(value=100)
      var.value = var.value + x
      return var.value

    stuff = jax.vmap(with_local_var)(jnp.arange(10))
    chex.assert_trees_all_equal(stuff, jnp.arange(100, 110))

  def test_unbind_variables(self):
    var_1 = variables.StateVariable(value=1, label="foo")
    var_2 = variables.StateVariable(value=2, label="bar")
    param = variables.Parameter(value=1, label="foo")
    thing_with_vars = {
        "var_1": var_1,
        "var_2": var_2,
        "param": param,
        "inner": [var_1, var_2],
        "something_else": "something else",
    }
    with self.subTest("unbind_all"):
      thing_with_slots, vars_list = variables.unbind_variables(thing_with_vars)
      self.assertEqual(
          thing_with_slots,
          {
              "var_1": variables.StateVariableSlot("foo"),
              "var_2": variables.StateVariableSlot("bar"),
              "param": variables.ParameterSlot("foo"),
              "inner": [
                  variables.StateVariableSlot("foo"),
                  variables.StateVariableSlot("bar"),
              ],
              "something_else": "something else",
          },
      )
      self.assertEqual(vars_list, (var_1, var_2, param))

    with self.subTest("unbind_frozen"):
      thing_with_slots, vars_list = variables.unbind_variables(
          thing_with_vars, freeze=True
      )
      self.assertEqual(
          thing_with_slots,
          {
              "var_1": variables.StateVariableSlot("foo"),
              "var_2": variables.StateVariableSlot("bar"),
              "param": variables.ParameterSlot("foo"),
              "inner": [
                  variables.StateVariableSlot("foo"),
                  variables.StateVariableSlot("bar"),
              ],
              "something_else": "something else",
          },
      )
      self.assertEqual(
          vars_list, (var_1.freeze(), var_2.freeze(), param.freeze())
      )

    with self.subTest("unbind_some"):
      thing_with_slots, vars_list = variables.unbind_state_vars(
          thing_with_vars, lambda var: var.label == "foo"
      )
      self.assertEqual(
          thing_with_slots,
          {
              "var_1": variables.StateVariableSlot("foo"),
              "var_2": var_2,
              "param": param,
              "inner": [variables.StateVariableSlot("foo"), var_2],
              "something_else": "something else",
          },
      )
      self.assertEqual(vars_list, (var_1,))

  def test_variable_unbind_conflict(self):
    # Not allowed: repeated use of the same label.
    var_1a = variables.StateVariable(value=1, label="var_1")
    var_1b = variables.StateVariable(value=2, label="var_1")
    thing_with_vars = {
        "var_1a": var_1a,
        "var_1b": var_1b,
        "inner": [var_1a, var_1b],
        "something_else": "something else",
    }
    with self.assertRaises(variables.VariableConflictError):
      variables.unbind_variables(thing_with_vars)

  def test_bind_variables(self):
    var_1 = variables.StateVariable(value=1, label="var_1")
    var_2 = variables.Parameter(value=2, label="var_2")
    thing_with_slots = {
        "var_1": variables.StateVariableSlot("var_1"),
        "var_2": variables.ParameterSlot("var_2"),
        "inner": [
            variables.StateVariableSlot("var_1"),
            variables.ParameterSlot("var_2"),
        ],
        "something_else": "something else",
    }
    self.assertEqual(
        variables.bind_variables(thing_with_slots, [var_1, var_2]),
        {
            "var_1": var_1,
            "var_2": var_2,
            "inner": [var_1, var_2],
            "something_else": "something else",
        },
    )

  def test_bind_and_unfreeze(self):
    var_1 = variables.StateVariableValue(value=1, label="var_1")
    var_2 = variables.ParameterValue(value=2, label="var_2")
    thing_with_slots = {
        "var_1": variables.StateVariableSlot("var_1"),
        "var_2": variables.ParameterSlot("var_2"),
        "inner": [
            variables.StateVariableSlot("var_1"),
            variables.ParameterSlot("var_2"),
        ],
        "something_else": "something else",
    }
    unfrozen = variables.bind_variables(
        thing_with_slots, [var_1, var_2], unfreeze_as_copy=True
    )
    self.assertIsInstance(unfrozen["var_1"], variables.StateVariable)
    self.assertIsInstance(unfrozen["var_2"], variables.Parameter)
    self.assertEqual(
        unfrozen,
        {
            "var_1": unfrozen["var_1"],
            "var_2": unfrozen["var_2"],
            "inner": [unfrozen["var_1"], unfrozen["var_2"]],
            "something_else": "something else",
        },
    )

  def test_variable_freeze_unfreeze(self):
    var_1 = variables.StateVariable(
        value=1.0, label="var_1", metadata={"foo": "bar"}
    )
    frozen_var_1 = var_1.freeze()
    chex.assert_trees_all_equal(
        frozen_var_1,
        variables.StateVariableValue(
            value=1.0, label="var_1", metadata={"foo": "bar"}
        ),
    )
    unfrozen_var_1 = frozen_var_1.unfreeze_as_copy()
    self.assertEqual(unfrozen_var_1.value, 1.0)
    self.assertEqual(unfrozen_var_1.label, "var_1")
    self.assertEqual(unfrozen_var_1.metadata, {"foo": "bar"})
    self.assertIsNot(var_1, unfrozen_var_1)

  def test_freeze_all_variables(self):
    var_1 = variables.StateVariable(value=1, label="var_1")
    var_2 = variables.Parameter(value=2, label="var_2")
    thing_with_vars = {
        "var_1": var_1,
        "var_2": var_2,
        "inner": [var_1, var_2],
        "something_else": "something else",
    }

    with self.subTest("freeze_all"):
      frozen = variables.freeze_variables(thing_with_vars)
      chex.assert_trees_all_equal_comparator(
          lambda a, b: a == b,
          lambda a, b: f"mismatch: {a} != {b}",
          frozen,
          {
              "var_1": var_1.freeze(),
              "var_2": var_2.freeze(),
              "inner": [var_1.freeze(), var_2.freeze()],
              "something_else": "something else",
          },
      )

    with self.subTest("freeze_some"):
      frozen = variables.freeze_state_vars(
          thing_with_vars, lambda var: var.label == "var_1"
      )
      chex.assert_trees_all_equal_comparator(
          lambda a, b: a == b,
          lambda a, b: f"mismatch: {a} != {b}",
          frozen,
          {
              "var_1": var_1.freeze(),
              "var_2": var_2,
              "inner": [var_1.freeze(), var_2],
              "something_else": "something else",
          },
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="jit", jit=True, donate_vars=False, donate_other=False
      ),
      dict(
          testcase_name="jit_donate_vars",
          jit=True,
          donate_vars=True,
          donate_other=False,
      ),
      dict(
          testcase_name="jit_donate_vars_and_other",
          jit=True,
          donate_vars=True,
          donate_other=True,
      ),
      dict(
          testcase_name="nojit",
          jit=False,
          donate_vars=False,
          donate_other=False,
      ),
  )
  def test_variable_jit(self, jit, donate_vars, donate_other):
    var_1 = variables.StateVariable(value=1, label="var_1")
    var_2 = variables.StateVariable(value=2, label="var_2")
    thing_with_vars = {
        "var_1": var_1,
        "var_2": var_2,
        "inner": [var_1, var_2],
        "some_value": jnp.ones([10]),
    }

    def my_var_fun(thing_with_vars, increment, something_else):
      self.assertEqual(something_else, "something else")
      thing_with_vars["var_1"].value += increment
      thing_with_vars["var_2"].value += 10 * increment
      return (
          thing_with_vars["inner"][0].value + thing_with_vars["inner"][1].value
      )

    if jit:
      if donate_other:
        extra_kwargs = {"donate_argnames": ["increment"]}
      else:
        extra_kwargs = {}
      my_var_fun = variables.variable_jit(
          my_var_fun,
          static_argnames="something_else",
          donate_variables=donate_vars,
          **extra_kwargs,
      )
    result = my_var_fun(thing_with_vars, 100, "something else")
    self.assertEqual(thing_with_vars["var_1"].value, 101)
    self.assertEqual(thing_with_vars["var_2"].value, 1002)
    self.assertEqual(result, 1103)

  def test_variable_jit_disallows_returning_vars(self):

    def bad(x):
      return [1, 2, variables.Parameter(label="foo", value=x)]

    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda exc: "Returning a variable" in str(exc)
    ):
      variables.variable_jit(bad)(10)


if __name__ == "__main__":
  absltest.main()
