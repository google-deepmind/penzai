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

"""Tests for layer shape checking."""

import re

from absl.testing import absltest
import jax
import jax.numpy as jnp
from penzai.deprecated.v1 import pz


class LayerShapecheckTest(absltest.TestCase):

  def test_checked_layer_call(self):
    @pz.pytree_dataclass
    class MyLayer(pz.Layer):

      @pz.checked_layer_call
      def __call__(self, argument):
        return argument

      def input_structure(self):
        return (
            pz.chk.ArraySpec(shape=(3, pz.chk.var("four"))),
            pz.chk.ANY,
        )

      def output_structure(self):
        return (
            pz.chk.ANY,
            pz.chk.ArraySpec(shape=(pz.chk.var("four"), 5)),
        )

    layer = MyLayer()
    with self.subTest("well_typed"):
      value_in = (
          jax.ShapeDtypeStruct(shape=(3, 4), dtype=jnp.float32),
          jax.ShapeDtypeStruct(shape=(4, 5), dtype=jnp.float32),
      )
      value_out = layer(value_in)
      self.assertIs(value_in, value_out)

    with self.subTest("bad_input_structure"):
      with self.assertRaises(pz.chk.StructureMismatchError):
        value_in = (
            jax.ShapeDtypeStruct(shape=(7, 4), dtype=jnp.float32),
            jax.ShapeDtypeStruct(shape=(4, 5), dtype=jnp.float32),
        )
        _ = layer(value_in)

    with self.subTest("bad_output_structure"):
      with self.assertRaises(pz.chk.StructureMismatchError):
        value_in = (
            jax.ShapeDtypeStruct(shape=(3, 4), dtype=jnp.float32),
            jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32),
        )
        _ = layer(value_in)

  def test_layer_call_input_structure_decorator_required(self):
    with self.assertRaisesRegex(
        TypeError,
        re.escape("should decorate `__call__` with `checked_layer_call`"),
    ):

      @pz.pytree_dataclass
      class MyLayer(pz.Layer):

        def __call__(self, argument):
          return argument

        def input_structure(self):
          return (pz.chk.ANY, pz.chk.ANY)

      del MyLayer

  def test_layer_call_output_structure_decorator_required(self):
    with self.assertRaisesRegex(
        TypeError,
        re.escape("should decorate `__call__` with `checked_layer_call`"),
    ):

      @pz.pytree_dataclass
      class MyLayer(pz.Layer):

        def __call__(self, argument):
          return argument

        def output_structure(self):
          return (pz.chk.ANY, pz.chk.ANY)

      del MyLayer

  def test_layer_call_opt_out_decorator(self):
    @pz.pytree_dataclass
    class MyLayer(pz.Layer):

      @pz.unchecked_layer_call
      def __call__(self, argument):
        return argument

      def input_structure(self):
        return (
            pz.chk.ArraySpec(shape=(3, pz.chk.var("four"))),
            pz.chk.ANY,
        )

      def output_structure(self):
        return (
            pz.chk.ANY,
            pz.chk.ArraySpec(shape=(pz.chk.var("four"), 5)),
        )

    # Doesn't match the provided structures, but this won't cause an error
    # because it's unchecked. (In real use, the implementer is responsible for
    # doing the check.)
    layer = MyLayer()
    value_in = (
        jax.ShapeDtypeStruct(shape=(7, 8), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(9, 10), dtype=jnp.float32),
    )
    value_out = layer(value_in)
    self.assertIs(value_in, value_out)


if __name__ == "__main__":
  absltest.main()
