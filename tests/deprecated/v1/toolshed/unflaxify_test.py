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

"""Tests for flax-to-penzai conversion."""

from absl.testing import absltest
import chex
import flax.linen
import jax
from penzai.deprecated.v1 import pz
from penzai.deprecated.v1.toolshed import unflaxify


class UnflaxifyTest(absltest.TestCase):

  def test_simple_flax(self):
    class MLP(flax.linen.Module):
      out_dims: int

      @flax.linen.compact
      def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = flax.linen.Dense(128)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(self.out_dims)(x)
        return x

    model_def = MLP(out_dims=10)
    x = jax.random.normal(jax.random.key(42), shape=(4, 8))
    variables = model_def.init(jax.random.key(43), x)

    penzai_model = unflaxify.unflaxify_apply(model_def, variables, x)
    pz.nn.check_no_duplicated_parameters(penzai_model)

    flax_out = model_def.apply(variables, x)
    penzai_out = penzai_model(unflaxify.ArgsAndKwargs.capture(x))

    chex.assert_trees_all_equal(flax_out, penzai_out)

  def test_setup_flax(self):
    class WeirdDense(flax.linen.Module):
      out_dims: int

      def setup(self):
        self.dense = flax.linen.Dense(self.out_dims)
        self.w = self.param("w", flax.linen.initializers.zeros_init(), ())

      def __call__(self, x):
        return self.dense(x) + self.w

    class MLPWithVar(flax.linen.Module):
      out_dims: int

      def setup(self):
        self.dense_0 = WeirdDense(128)
        self.dense_1 = WeirdDense(self.out_dims)
        self.w = self.param("w", flax.linen.initializers.zeros_init(), ())

      def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = self.dense_0(x)
        x = flax.linen.relu(x)
        x = self.dense_1(x)
        return x + self.w

    model_def = MLPWithVar(out_dims=10)
    x = jax.random.normal(jax.random.key(42), shape=(4, 8))
    variables = model_def.init(jax.random.key(43), x)

    penzai_model = unflaxify.unflaxify_apply(model_def, variables, x)
    pz.nn.check_no_duplicated_parameters(penzai_model)

    flax_out = model_def.apply(variables, x)
    penzai_out = penzai_model(unflaxify.ArgsAndKwargs.capture(x))

    chex.assert_trees_all_equal(flax_out, penzai_out)

  def test_shared_variables_and_states(self):
    class WeirdModel(flax.linen.Module):

      @flax.linen.compact
      def __call__(self, x):
        assert x.shape[-1] == 32
        shared_dense = flax.linen.Dense(32, name="shared_dense")
        x = shared_dense(x)
        x = flax.linen.relu(x)
        x = shared_dense(x)
        x = flax.linen.Dense(32)(x)
        mv = self.variable("muts", "mv", lambda: 10.0)
        iv = self.variable("immuts", "iv", lambda: 100.0)
        mv.value += 1
        return x + mv.value + iv.value

    class WeirdOuterModel(flax.linen.Module):

      @flax.linen.compact
      def __call__(self, x):
        assert x.shape[-1] == 32
        weird_one = WeirdModel(name="weird_shared")
        x = weird_one(x)
        x = WeirdModel()(x)
        x = WeirdModel()(x)
        x = weird_one(x)
        return x

    model_def = WeirdOuterModel()
    x = jax.random.normal(jax.random.key(42), shape=(3, 32))
    variables = model_def.init(jax.random.key(43), x)

    penzai_model = unflaxify.unflaxify_apply(
        model_def, variables, x, mutable="muts"
    )
    pz.nn.check_no_duplicated_parameters(penzai_model)

    pz_model_pure, pz_state_dict = pz.de.handle_local_states(
        penzai_model,
        category_predicate=lambda cat: cat in ("muts", "immuts"),
        state_sharing="allowed",
    )
    flax_out, flax_new_states = model_def.apply(variables, x, mutable="muts")
    pz_out, pz_new_states = pz_model_pure(
        (unflaxify.ArgsAndKwargs.capture(x), pz_state_dict)
    )

    chex.assert_trees_all_equal(flax_out, pz_out)
    self.assertEqual(
        flax_new_states,
        {
            "muts": {
                "weird_shared": {"mv": 14.0},
                "WeirdModel_0": {"mv": 12.0},
                "WeirdModel_1": {"mv": 12.0},
            }
        },
    )
    self.assertEqual(
        pz_new_states,
        {
            "muts:weird_shared.mv": 14.0,
            "muts:WeirdModel_0.mv": 12.0,
            "muts:WeirdModel_1.mv": 12.0,
        },
    )


if __name__ == "__main__":
  absltest.main()
