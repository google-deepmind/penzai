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
from penzai.deprecated.v1 import pz
from penzai.deprecated.v1.example_models import simple_mlp
from penzai.deprecated.v1.toolshed import isolate_submodel


class IsolateSubmodelTest(absltest.TestCase):

  def test_simple_capture(self):
    mlp = pz.nn.initialize_parameters(
        simple_mlp.MLP.from_config(feature_sizes=[8, 16, 32, 32]),
        jax.random.key(42),
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
    assert isinstance(isolated, isolate_submodel.IsolatedSubmodel)
    assert isinstance(isolated.submodel, pz.nn.Linear)
    replay_result = isolated.submodel(isolated.saved_input)
    chex.assert_trees_all_close(replay_result, isolated.saved_output)

  def test_capture_with_states_shared_params_and_random(self):
    @pz.pytree_dataclass
    class StateIncrementLayer(pz.Layer):
      state: pz.de.LocalStateEffect = pz.de.InitialLocalStateRequest(
          lambda: 0, name="foo", category="counter"
      )

      def __call__(self, x):
        self.state.set(self.state.get() + 1)
        return x

    share_dropout_dense = pz.nn.add_parameter_prefix(
        "SharedDropoutMLP",
        pz.nn.mark_shareable(
            simple_mlp.DropoutMLP.from_config(
                feature_sizes=[8, 16, 8],
                drop_rate=0.2,
            )
        ),
    )
    state_inc = StateIncrementLayer()
    weird_model_def = pz.nn.attach_shared_parameters(
        pz.nn.Sequential([
            pz.nn.add_parameter_prefix(
                "MLP0", simple_mlp.MLP.from_config(feature_sizes=[8, 16, 8])
            ),
            share_dropout_dense,
            state_inc,
            pz.nn.NamedGroup(
                "target",
                [
                    share_dropout_dense,
                    state_inc,
                ],
            ),
            state_inc,
            pz.nn.add_parameter_prefix(
                "MLP1", simple_mlp.MLP.from_config(feature_sizes=[8, 16, 8])
            ),
            share_dropout_dense,
        ])
    )
    weird_model = pz.nn.initialize_parameters(
        weird_model_def, jax.random.key(42)
    )
    weird_model_handled, initial_state_dict = pz.de.handle_local_states(
        pz.de.WithRandomKeyFromArg.handling(weird_model),
        category="counter",
        state_sharing="allowed",
    )
    # This selection selects the inner NamedGroup
    submodel_selection = weird_model_handled.select().at(
        (lambda root: root.body.body.body.sublayers[3])
    )
    selected_submodel = submodel_selection.get()
    self.assertIsInstance(selected_submodel, pz.nn.NamedGroup)
    self.assertEqual(selected_submodel.name, "target")

    isolated = isolate_submodel.call_and_extract_submodel(
        submodel_selection,
        (
            (pz.nx.ones({"batch": 4, "features": 8}), jax.random.key(42)),
            initial_state_dict,
        ),
    )
    assert isinstance(isolated, isolate_submodel.IsolatedSubmodel)
    # Replaying this model is complex, because it requires:
    # - rewinding the state to be the state for the second increment
    # - rewinding the random key to match the calls after the first dropout
    # - rebinding the shared parameters for the dropout submodel
    # This assertion checks to make sure the values are the same when we replay
    # it v.s. when it was initially called, which requires all of the above
    # steps to be done correctly.
    replay_result = isolated.submodel(isolated.saved_input)
    chex.assert_trees_all_close(replay_result, isolated.saved_output)


if __name__ == "__main__":
  absltest.main()
