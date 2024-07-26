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

"""Tests for neural network parameters and associated utilities."""

from __future__ import annotations

import dataclasses
import re
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from penzai.deprecated.v1 import pz


@pz.pytree_dataclass
class CallMockHelper(pz.Struct):
  """Helper object to call a function with it's value and a child.

  Attributes:
    child: A PyTree child. Can be traversed to handle effects.
    fn: A function, which will be called with two keyword arguments "value" and
      "child", corresponding to the input argument and the current PyTree child.
  """

  child: Any
  fn: Any = dataclasses.field(metadata={"pytree_node": False})

  def __call__(self, value):
    return self.fn(value=value, child=self.child)


class NNParametersTest(parameterized.TestCase):

  def test_uninitialized_parameter_initialization(self):
    param = pz.nn.UninitializedParameter(
        initializer=lambda key: {"foo": jax.random.uniform(key, (2, 3, 4))},
        name="my_parameter",
    )

    with self.subTest("infers_value_structure"):
      self.assertEqual(
          param.value_structure,
          {"foo": pz.chk.ArraySpec(shape=(2, 3, 4), dtype=np.dtype("float32"))},
      )

    with self.subTest("initializes"):
      initialized = param.initialize(jax.random.PRNGKey(0))
      chex.assert_trees_all_equal_shapes_and_dtypes(
          initialized,
          pz.nn.Parameter(
              value={
                  "foo": jax.ShapeDtypeStruct((2, 3, 4), np.dtype("float32"))
              },
              name="my_parameter",
          ),
      )
      self.assertIsInstance(initialized.value["foo"], jax.Array)

    with self.subTest("can_initialize_manually"):
      the_value = {"foo": jnp.zeros((2, 3, 4))}
      initialized = param.initialize_with_value(the_value)
      chex.assert_trees_all_equal(
          initialized,
          pz.nn.Parameter(value=the_value, name="my_parameter"),
      )

    with self.subTest("as_empty_parameter"):
      initialized_empty = param.as_empty_parameter()
      self.assertEqual(
          initialized_empty,
          pz.nn.Parameter(
              value={
                  "foo": jax.ShapeDtypeStruct((2, 3, 4), np.dtype("float32"))
              },
              name="my_parameter",
          ),
      )

  def test_uninitialized_parameter_checks_structure(self):
    param = pz.nn.UninitializedParameter(
        initializer=lambda key: {"foo": jax.random.uniform(key, (2, 3, 4))},
        name="my_parameter",
        value_structure={
            "bar": pz.chk.ArraySpec(shape=(2, 3, 4), dtype=np.dtype(np.float32))
        },
    )
    with self.assertRaises(ValueError):
      param.initialize(jax.random.PRNGKey(0))

    with self.assertRaises(ValueError):
      param.initialize_with_value(param.initializer(jax.random.PRNGKey(0)))

  def test_uninitialized_parameter_checks_shape(self):
    param = pz.nn.UninitializedParameter(
        initializer=lambda key: {"foo": jax.random.uniform(key, (2, 3, 4))},
        name="my_parameter",
        value_structure={
            "foo": pz.chk.ArraySpec(shape=(1, 2, 3), dtype=np.dtype(np.float32))
        },
    )
    with self.assertRaises(pz.chk.StructureMismatchError):
      param.initialize(jax.random.PRNGKey(0))

    with self.assertRaises(pz.chk.StructureMismatchError):
      param.initialize_with_value(param.initializer(jax.random.PRNGKey(0)))

  def test_uninitialized_parameter_checks_dtype(self):
    param = pz.nn.UninitializedParameter(
        initializer=lambda key: {"foo": jax.random.uniform(key, (2, 3, 4))},
        name="my_parameter",
        value_structure={
            "foo": pz.chk.ArraySpec(shape=(2, 3, 4), dtype=np.dtype(np.int32))
        },
    )
    with self.assertRaises(pz.chk.StructureMismatchError):
      param.initialize(jax.random.PRNGKey(0))

    with self.assertRaises(pz.chk.StructureMismatchError):
      param.initialize_with_value(param.initializer(jax.random.PRNGKey(0)))

  def test_parameter_prefix(self):
    example_value = jnp.arange(6).reshape((2, 3))
    example_initializer = lambda _: example_value
    initial_object = {
        "a_parameter": pz.nn.Parameter(name="foo", value=example_value),
        "an_unintialized_parameter": pz.nn.UninitializedParameter(
            name="foo", initializer=example_initializer
        ),
        "a_container_of_parameters": [
            pz.nn.Parameter(name="bar", value=example_value),
            pz.nn.Parameter(name="baz", value=example_value),
        ],
        "a_frozen_parameter": pz.nn.FrozenParameter(
            name="foo", value=example_value
        ),
    }
    chex.assert_trees_all_equal(
        pz.nn.add_parameter_prefix("Namespace", initial_object),
        {
            "a_container_of_parameters": [
                pz.nn.Parameter(value=example_value, name="Namespace.bar"),
                pz.nn.Parameter(value=example_value, name="Namespace.baz"),
            ],
            "a_frozen_parameter": pz.nn.FrozenParameter(
                value=example_value, name="Namespace.foo"
            ),
            "a_parameter": pz.nn.Parameter(
                value=example_value, name="Namespace.foo"
            ),
            "an_unintialized_parameter": pz.nn.UninitializedParameter(
                initializer=example_initializer,
                name="Namespace.foo",
                value_structure=pz.chk.ArraySpec(
                    shape=(2, 3), dtype=np.dtype("int32")
                ),
            ),
        },
    )

  def test_initialize_parameters_success(self):
    object_with_unitialized = {
        "foo": pz.nn.UninitializedParameter(
            initializer=lambda key: jax.random.uniform(key, (2, 3, 4)),
            name="param_foo",
        ),
        "bar": [
            pz.nn.UninitializedParameter(
                initializer=lambda key: jax.random.uniform(key, (5,)),
                name="param_bar",
            )
        ],
    }
    chex.assert_trees_all_equal_shapes_and_dtypes(
        pz.nn.initialize_parameters(
            object_with_unitialized, prng_key=jax.random.PRNGKey(0)
        ),
        {
            "foo": pz.nn.Parameter(
                value=jax.ShapeDtypeStruct((2, 3, 4), np.dtype("float32")),
                name="param_foo",
            ),
            "bar": [
                pz.nn.Parameter(
                    value=jax.ShapeDtypeStruct((5,), np.dtype("float32")),
                    name="param_bar",
                )
            ],
        },
    )

  def test_initialize_parameters_detects_duplicate_names(self):
    object_with_unitialized = {
        "foo": pz.nn.UninitializedParameter(
            initializer=lambda key: jax.random.uniform(key, (2, 3, 4)),
            name="param_foo",
        ),
        "bar": [
            pz.nn.UninitializedParameter(
                initializer=lambda key: jax.random.uniform(key, (5,)),
                name="param_bar",
            ),
            pz.nn.UninitializedParameter(
                initializer=lambda key: jax.random.uniform(key, (5,)),
                name="param_bar",
            ),
        ],
    }
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Found multiple parameters with the same name! Repeated names:"
        " ['param_bar']",
    ):
      pz.nn.initialize_parameters(
          object_with_unitialized, prng_key=jax.random.PRNGKey(0)
      )

  def test_shared_parameters(self):
    foo_block = pz.nn.add_parameter_prefix(
        "foo",
        CallMockHelper(
            child={
                "foo": pz.nn.UninitializedParameter(
                    initializer=lambda key: jax.random.uniform(key, (1, 3, 5)),
                    name="param_foo",
                )
            },
            fn=mock.MagicMock(),
        ),
    )
    foo_block.fn.return_value = "foo_return"

    bar_block = pz.nn.add_parameter_prefix(
        "bar",
        CallMockHelper(
            child={
                "bar1": pz.nn.UninitializedParameter(
                    initializer=lambda key: jax.random.uniform(key, (2, 3, 4)),
                    name="param_bar1",
                ),
                "bar2": pz.nn.UninitializedParameter(
                    initializer=lambda key: jax.random.uniform(key, (1, 2, 3)),
                    name="param_bar2",
                ),
            },
            fn=mock.MagicMock(),
        ),
    )
    bar_block.fn.return_value = "bar_return"

    shareable_bar = pz.nn.mark_shareable(bar_block)
    self.assertIsInstance(
        shareable_bar.child["bar1"], pz.nn.ShareableUninitializedParameter
    )
    self.assertEqual(shareable_bar.child["bar1"].name, "bar.param_bar1")
    self.assertIsInstance(
        shareable_bar.child["bar2"], pz.nn.ShareableUninitializedParameter
    )
    self.assertEqual(shareable_bar.child["bar2"].name, "bar.param_bar2")

    wrapped = pz.nn.add_parameter_prefix(
        "outer",
        pz.nn.attach_shared_parameters(
            pz.nn.Sequential([
                shareable_bar,
                foo_block,
                shareable_bar,
            ])
        ),
    )

    self.assertIsInstance(wrapped, pz.de.WithConstantSideInputs)
    handler_id = wrapped.handler_id
    chex.assert_trees_all_equal(
        wrapped.body,
        pz.nn.Sequential(
            sublayers=[
                CallMockHelper(
                    child={
                        "bar1": pz.nn.SharedParameterLookup(
                            ref=pz.de.HandledSideInputRef(
                                handler_id=handler_id,
                                tag=pz.nn.SharedParamTag(name="bar.param_bar1"),
                            ),
                            value_structure=(
                                shareable_bar.child["bar1"].value_structure
                            ),
                        ),
                        "bar2": pz.nn.SharedParameterLookup(
                            ref=pz.de.HandledSideInputRef(
                                handler_id=handler_id,
                                tag=pz.nn.SharedParamTag(name="bar.param_bar2"),
                            ),
                            value_structure=(
                                shareable_bar.child["bar2"].value_structure
                            ),
                        ),
                    },
                    fn=bar_block.fn,
                ),
                CallMockHelper(
                    child={
                        "foo": pz.nn.UninitializedParameter(
                            initializer=foo_block.child["foo"].initializer,
                            name="outer.foo.param_foo",
                            value_structure=pz.chk.ArraySpec(
                                shape=(1, 3, 5),
                                dtype=np.dtype("float32"),
                                named_shape={},
                            ),
                        )
                    },
                    fn=foo_block.fn,
                ),
                CallMockHelper(
                    child={
                        "bar1": pz.nn.SharedParameterLookup(
                            ref=pz.de.HandledSideInputRef(
                                handler_id=handler_id,
                                tag=pz.nn.SharedParamTag(name="bar.param_bar1"),
                            ),
                            value_structure=(
                                shareable_bar.child["bar1"].value_structure
                            ),
                        ),
                        "bar2": pz.nn.SharedParameterLookup(
                            ref=pz.de.HandledSideInputRef(
                                handler_id=handler_id,
                                tag=pz.nn.SharedParamTag(name="bar.param_bar2"),
                            ),
                            value_structure=(
                                shareable_bar.child["bar2"].value_structure
                            ),
                        ),
                    },
                    fn=bar_block.fn,
                ),
            ],
        ),
    )
    chex.assert_trees_all_equal(
        wrapped.side_inputs,
        {
            pz.nn.SharedParamTag(
                name="bar.param_bar1"
            ): pz.nn.UninitializedParameter(
                initializer=bar_block.child["bar1"].initializer,
                name="outer.bar.param_bar1",
                value_structure=pz.chk.ArraySpec(
                    shape=(2, 3, 4), dtype=np.dtype("float32")
                ),
            ),
            pz.nn.SharedParamTag(
                name="bar.param_bar2"
            ): pz.nn.UninitializedParameter(
                initializer=bar_block.child["bar2"].initializer,
                name="outer.bar.param_bar2",
                value_structure=pz.chk.ArraySpec(
                    shape=(1, 2, 3), dtype=np.dtype("float32")
                ),
            ),
        },
    )

    # Initialize it.
    init_wrapped = pz.nn.initialize_parameters(
        wrapped, prng_key=jax.random.PRNGKey(0)
    )

    # Call it and check that, when called, the shared parameters are substituted
    # correctly.
    final_result = init_wrapped("initial_input")
    self.assertEqual(final_result, "bar_return")
    self.assertEqual(
        [call.kwargs["value"] for call in foo_block.fn.call_args_list],
        ["bar_return"],
    )
    self.assertLen(foo_block.fn.call_args_list, 1)
    for call in foo_block.fn.call_args_list:
      chex.assert_trees_all_equal_shapes_and_dtypes(
          call.kwargs["child"]["foo"].value,
          jax.ShapeDtypeStruct((1, 3, 5), np.dtype("float32")),
      )
    self.assertEqual(
        [call.kwargs["value"] for call in bar_block.fn.call_args_list],
        ["initial_input", "foo_return"],
    )
    self.assertLen(bar_block.fn.call_args_list, 2)
    for call in bar_block.fn.call_args_list:
      chex.assert_trees_all_equal_shapes_and_dtypes(
          {k: param.value for k, param in call.kwargs["child"].items()},
          {
              "bar1": jax.ShapeDtypeStruct((2, 3, 4), np.dtype("float32")),
              "bar2": jax.ShapeDtypeStruct((1, 2, 3), np.dtype("float32")),
          },
      )

  def test_shared_parameter_safety_checks(self):

    compatible_but_different = {
        "foo": pz.nn.UninitializedParameter(
            initializer=lambda key: jax.random.uniform(key, (2, 3, 4)),
            name="same_name",
        ),
        "bar": pz.nn.UninitializedParameter(
            initializer=lambda key: jnp.zeros((2, 3, 4)),
            name="same_name",
        ),
    }

    with self.subTest("strict_initializer_conflict"):
      with self.assertRaisesRegex(
          ValueError,
          re.escape(
              "Detected non-identical initializers for two shareable parameters"
              " with name 'same_name'"
          ),
      ):
        _ = pz.nn.attach_shared_parameters(
            pz.nn.mark_shareable(compatible_but_different)
        )

    with self.subTest("nonstrict_initializer"):
      owner = pz.nn.attach_shared_parameters(
          pz.nn.mark_shareable(compatible_but_different), strict=False
      )
      init_owner = pz.nn.initialize_parameters(
          owner, prng_key=jax.random.PRNGKey(0)
      )
      chex.assert_trees_all_equal_shapes_and_dtypes(
          init_owner.side_inputs,
          {
              pz.nn.SharedParamTag("same_name"): pz.nn.Parameter(
                  name="same_name",
                  value=jax.ShapeDtypeStruct((2, 3, 4), jnp.dtype("float32")),
              )
          },
      )

    incompatible = {
        "foo": pz.nn.UninitializedParameter(
            initializer=lambda key: jax.random.uniform(key, (2, 3, 4)),
            name="same_name",
        ),
        "bar": pz.nn.UninitializedParameter(
            initializer=lambda key: jnp.zeros((3, 4, 5)),
            name="same_name",
        ),
    }

    with self.subTest("strict_initializer_conflict"):
      with self.assertRaisesRegex(
          ValueError,
          re.escape(
              "Detected incompatible value structures for two shared parameters"
              " with name 'same_name'"
          ),
      ):
        _ = pz.nn.attach_shared_parameters(
            pz.nn.mark_shareable(incompatible), strict=False
        )


if __name__ == "__main__":
  absltest.main()
