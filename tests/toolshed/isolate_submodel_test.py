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

"""Tests for submodule isolation utility."""

from absl.testing import absltest
import chex
import jax
from penzai import pz
from penzai.models import simple_mlp
from penzai.toolshed import isolate_submodel


class IsolateSubmodelTest(absltest.TestCase):

  def test_simple_capture(self):
    mlp = simple_mlp.MLP.from_config(
        name="mlp",
        init_base_rng=jax.random.key(42),
        feature_sizes=[8, 16, 32, 32],
    )
    # root.sublayers[2].sublayers[0] is the linear layer inside
    # the second dense block
    submodel_selection = mlp.select().at(
        (lambda root: root.sublayers[2].sublayers[0])
    )
    self.assertIsInstance(submodel_selection.get(), pz.nn.Linear)
    isolated = isolate_submodel.call_and_extract_submodel(
        submodel_selection,
        pz.nx.ones({"batch": 4, "features": 8}),
    )
    self.assertIsInstance(isolated, isolate_submodel.IsolatedSubmodel)
    self.assertIsInstance(isolated.submodel, pz.nn.Linear)
    self.assertEqual(isolated.initial_var_values, ())
    self.assertEqual(isolated.final_var_values, ())
    replay_result = isolated.submodel(isolated.saved_arg)
    chex.assert_trees_all_close(replay_result, isolated.saved_output)

  def test_capture_with_states_shared_params_and_random(self):
    @pz.pytree_dataclass
    class StateIncrementLayer(pz.nn.Layer):
      state: pz.StateVariable[int]

      def __call__(self, x, **_unused_side_inputs):
        self.state.value = self.state.value + 1
        return x

    share_dropout_dense = simple_mlp.DropoutMLP.from_config(
        name="shared_mlp",
        init_base_rng=jax.random.key(42),
        feature_sizes=[8, 16, 8],
        drop_rate=0.2,
    )
    state_inc = StateIncrementLayer(pz.StateVariable(0, label="counter"))

    weird_model = pz.nn.Sequential([
        simple_mlp.MLP.from_config(
            name="mlp0",
            init_base_rng=jax.random.key(1),
            feature_sizes=[8, 16, 8],
        ),
        share_dropout_dense,
        state_inc,
        pz.nn.NamedGroup("target", [share_dropout_dense, state_inc]),
        state_inc,
        simple_mlp.MLP.from_config(
            name="mlp1",
            init_base_rng=jax.random.key(2),
            feature_sizes=[8, 16, 8],
        ),
        share_dropout_dense,
    ])
    rng_stream = pz.RandomStream(
        jax.random.key(42), offset=pz.StateVariable(0, label="rng_offset")
    )

    # Select the inner NamedGroup
    submodel_selection = weird_model.select().at_subtrees_where(
        lambda x: isinstance(x, pz.nn.NamedGroup) and x.name == "target"
    )

    isolated = isolate_submodel.call_and_extract_submodel(
        submodel_selection,
        pz.nx.ones({"batch": 4, "features": 8}),
        random_stream=rng_stream,
    )
    self.assertIsInstance(isolated, isolate_submodel.IsolatedSubmodel)
    self.assertIsInstance(isolated.submodel, pz.nn.NamedGroup)
    self.assertEqual(
        isolated.saved_side_inputs,
        {
            "random_stream": pz.RandomStream(
                jax.random.key(42),
                offset=pz.StateVariableSlot(label="rng_offset"),
            )
        },
    )

    chex.assert_trees_all_equal(
        {var.label: var for var in isolated.initial_var_values},
        {
            "counter": pz.StateVariableValue(label="counter", value=1),
            "rng_offset": pz.StateVariableValue(label="rng_offset", value=1),
        },
    )
    chex.assert_trees_all_equal(
        {var.label: var for var in isolated.final_var_values},
        {
            "counter": pz.StateVariableValue(label="counter", value=2),
            "rng_offset": pz.StateVariableValue(label="rng_offset", value=2),
        },
    )

    replay_result, replay_vars = isolated.submodel.stateless_call(
        isolated.initial_var_values,
        isolated.saved_arg,
        **isolated.saved_side_inputs,
    )
    chex.assert_trees_all_close(replay_result, isolated.saved_output)
    chex.assert_trees_all_close(replay_vars, isolated.final_var_values)


if __name__ == "__main__":
  absltest.main()
