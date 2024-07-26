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

"""Tests for side input effect."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from penzai.deprecated.v1 import pz


@pz.pytree_dataclass
class MyEffectfulLayer(pz.Layer):
  one: pz.de.SideInputEffect[jax.Array] = pz.de.SideInputRequest(tag="x")
  two: pz.de.SideInputEffect[jax.Array] = pz.de.SideInputRequest(tag="y")
  three: pz.de.SideInputEffect[jax.Array] = pz.de.SideInputRequest(tag="x")

  @pz.checked_layer_call
  def __call__(self, argument: Any, /) -> Any:
    assert argument == {"foo": None}
    return (self.one.ask(), self.two.ask(), self.three.ask())

  def input_structure(self):
    return {"foo": None}

  def output_structure(self):
    return (
        pz.chk.ArraySpec(shape=(2,), dtype=jnp.float32),
        pz.chk.ArraySpec(shape=(), dtype=jnp.int32),
        pz.chk.ArraySpec(shape=(2,), dtype=jnp.float32),
    )


class SideInputTest(parameterized.TestCase):

  def test_side_input_from_input_tuple_nested_pairs(self):
    layer = MyEffectfulLayer()
    wrapped = pz.de.WithSideInputsFromInputTuple.handling(
        pz.de.WithSideInputsFromInputTuple.handling(layer, tags=["y"]),
        tags=["x"],
    )
    out = wrapped((
        ({"foo": None}, jnp.array(1, dtype=jnp.int32)),
        jnp.array([2.0, 3.0], dtype=jnp.float32),
    ))
    chex.assert_trees_all_equal(
        out,
        (
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            jnp.array(1, dtype=jnp.int32),
            jnp.array([2.0, 3.0], dtype=jnp.float32),
        ),
    )

  def test_side_input_from_input_tuple_triple(self):
    layer = MyEffectfulLayer()
    wrapped = pz.de.WithSideInputsFromInputTuple.handling(
        layer, tags=["y", "x"]
    )
    out = wrapped((
        {"foo": None},
        jnp.array(1, dtype=jnp.int32),
        jnp.array([2.0, 3.0], dtype=jnp.float32),
    ))
    chex.assert_trees_all_equal(
        out,
        (
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            jnp.array(1, dtype=jnp.int32),
            jnp.array([2.0, 3.0], dtype=jnp.float32),
        ),
    )

  @parameterized.named_parameters(
      dict(testcase_name="keep_unused", keep_unused=True),
      dict(testcase_name="drop_unused", keep_unused=False),
  )
  def test_side_input_constant(self, keep_unused):
    layer = MyEffectfulLayer()
    wrapped = pz.de.WithConstantSideInputs.handling(
        layer,
        side_inputs={
            "x": jnp.array([2.0, 3.0], dtype=jnp.float32),
            "y": jnp.array(1, dtype=jnp.int32),
            "z": jnp.array(2, dtype=jnp.int32),
        },
        keep_unused=keep_unused,
    )
    if keep_unused:
      self.assertIn("z", wrapped.side_inputs)
    else:
      self.assertNotIn("z", wrapped.side_inputs)
    out = wrapped({"foo": None})
    chex.assert_trees_all_equal(
        out,
        (
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            jnp.array(1, dtype=jnp.int32),
            jnp.array([2.0, 3.0], dtype=jnp.float32),
        ),
    )

  def test_hoist_constant_side_inputs(self):
    tree_with_side_inputs = {
        "a": pz.de.WithConstantSideInputs.handling(
            {
                "x": 100,
                "y": pz.de.WithConstantSideInputs.handling(
                    {
                        1: pz.de.SideInputRequest("foo"),
                        2: pz.de.SideInputRequest("bar"),
                        3: pz.de.SideInputRequest("baz"),
                    },
                    side_inputs={
                        "foo": 123,
                    },
                ),
            },
            side_inputs={
                "bar": 456,
                "baz": 789,
            },
        ),
        "b": pz.de.WithConstantSideInputs.handling(
            {
                4: pz.de.SideInputRequest("foo"),
                5: pz.de.SideInputRequest("bar"),
            },
            side_inputs={"foo": 1000, "bar": 456},
        ),
    }
    hoisted, side_inputs = pz.de.hoist_constant_side_inputs(
        tree_with_side_inputs,
        hoist_predicate=lambda tag, val: tag == "foo" or val == 456,
    )
    self.assertEqual(
        hoisted,
        {
            "a": pz.de.WithConstantSideInputs(
                handler_id=tree_with_side_inputs["a"].handler_id,
                body={
                    "x": 100,
                    "y": {
                        1: pz.de.SideInputRequest(
                            pz.de.HoistedTag(
                                source="['a']/WithConstantSideInputs.body['y']",
                                tag="foo",
                            )
                        ),
                        2: pz.de.SideInputRequest(
                            pz.de.HoistedTag(source="['a']", tag="bar")
                        ),
                        3: pz.de.HandledSideInputRef(
                            handler_id=tree_with_side_inputs["a"].handler_id,
                            tag="baz",
                        ),
                    },
                },
                side_inputs={"baz": 789},
            ),
            "b": {
                4: pz.de.SideInputRequest(
                    pz.de.HoistedTag(source="['b']", tag="foo")
                ),
                5: pz.de.SideInputRequest(
                    pz.de.HoistedTag(source="['b']", tag="bar")
                ),
            },
        },
    )
    self.assertEqual(
        side_inputs,
        {
            pz.de.HoistedTag(source="['a']", tag="bar"): 456,
            pz.de.HoistedTag(
                source="['a']/WithConstantSideInputs.body['y']", tag="foo"
            ): 123,
            pz.de.HoistedTag(source="['b']", tag="foo"): 1000,
            pz.de.HoistedTag(source="['b']", tag="bar"): 456,
        },
    )


if __name__ == "__main__":
  absltest.main()
