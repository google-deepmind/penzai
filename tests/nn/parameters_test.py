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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from penzai import pz


class NNParametersTest(parameterized.TestCase):

  def test_make_parameter_keyed(self):
    param = pz.nn.make_parameter(
        "foo",
        jax.random.key(2),
        lambda key, a, b: {"v": jax.random.uniform(key), "s": a + b},
        100,
        metadata={"foo": "bar"},
        b=20,
    )
    self.assertIsInstance(param, pz.Parameter)
    self.assertEqual(param.label, "foo")
    self.assertEqual(param.value["s"], 120)
    self.assertEqual(param.metadata, {"foo": "bar"})
    chex.assert_shape(param.value["v"], ())

    param2 = pz.nn.make_parameter(
        "bar",
        jax.random.key(2),
        lambda key, a, b: {"v": jax.random.uniform(key), "s": a + b},
        100,
        b=20,
    )
    self.assertNotEqual(param.value["v"], param2.value["v"])

  def test_make_parameter_slot(self):
    param = pz.nn.make_parameter(
        "foo",
        None,
        lambda key, a, b: {"v": jax.random.uniform(key), "s": a + b},
        100,
        metadata={"foo": "bar"},
        b=20,
    )
    self.assertEqual(param, pz.ParameterSlot("foo"))

  def test_assert_no_parameter_slots(self):
    pz.nn.assert_no_parameter_slots({
        "foo": pz.Parameter(value=3, label="a"),
        "bar": pz.Parameter(value=3, label="b"),
        "baz": pz.StateVariableSlot(label="c"),
    })
    with self.assertRaises(ValueError):  # pylint: disable=g-error-prone-assert-raises
      pz.nn.assert_no_parameter_slots({
          "bar": pz.ParameterSlot(label="b"),
      })


if __name__ == "__main__":
  absltest.main()
