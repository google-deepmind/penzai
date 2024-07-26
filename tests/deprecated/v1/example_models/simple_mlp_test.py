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

"""Tests for simple MLP example model."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import optax
from penzai.deprecated.v1 import pz
from penzai.deprecated.v1.example_models import simple_mlp
from penzai.deprecated.v1.toolshed import basic_training
from penzai.deprecated.v1.toolshed import check_layers_by_tracing


class SimpleMlpTest(absltest.TestCase):

  def test_build_deterministic_mlp(self):
    mlp = simple_mlp.MLP.from_config(feature_sizes=[8, 16, 32, 32])
    check_layers_by_tracing.check_layer(
        mlp, pz.nx.zeros({"batch": 3, "features": 8})
    )

  def test_build_dropout_mlp(self):
    mlp = simple_mlp.DropoutMLP.from_config(
        feature_sizes=[8, 16, 32, 32], drop_rate=0.2
    )
    handled_mlp = pz.de.WithRandomKeyFromArg.handling(mlp)
    check_layers_by_tracing.check_layer(
        handled_mlp,
        (pz.nx.zeros({"batch": 3, "features": 8}), jax.random.key(10)),
    )

  def test_train_deterministic_mlp(self):
    # Train the deterministic MLP to 100% accuracy on XOR.
    mlp = simple_mlp.MLP.from_config(feature_sizes=[2, 32, 32, 2])
    init_mlp = pz.nn.initialize_parameters(mlp, jax.random.key(10))

    xor_inputs = pz.nx.wrap(
        jnp.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=jnp.float32),
        "batch",
        "features",
    )
    xor_labels = jnp.array([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=jnp.float32)

    def loss_fn(model, rng, state):
      scale = 1 + jax.random.uniform(rng, shape=(1,))
      model_out = model(xor_inputs)
      log_probs = jax.nn.log_softmax(
          model_out.unwrap("batch", "features"), axis=-1
      )
      losses = -scale * log_probs * xor_labels
      loss = jnp.sum(losses) / 4
      return (loss, state + 1, {"loss": loss, "count": state})

    train_step = jax.jit(basic_training.build_train_step_fn(loss_fn))
    train_state = basic_training.TrainState.initial_state(
        model=init_mlp,
        optimizer_def=optax.adam(0.1),
        root_rng=jax.random.key(42),
        loss_fn_state=100,
    )

    outputs = []
    for _ in range(20):
      train_state, out = train_step(train_state)
      outputs.append(out)
    self.assertEqual(outputs[0]["count"], 100)
    self.assertGreater(outputs[0]["loss"], 0.8)
    self.assertEqual(outputs[-1]["count"], 119)
    self.assertLess(outputs[-1]["loss"], 0.0001)


if __name__ == "__main__":
  absltest.main()
