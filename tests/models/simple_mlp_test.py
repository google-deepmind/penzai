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
from penzai import pz
from penzai.models import simple_mlp
from penzai.toolshed import basic_training


class SimpleMlpTest(absltest.TestCase):

  def test_build_deterministic_mlp(self):

    mlp = simple_mlp.MLP.from_config(
        name="mlp",
        init_base_rng=jax.random.key(0),
        feature_sizes=[8, 16, 32, 32],
    )
    result = mlp(pz.nx.zeros({"batch": 3, "features": 8}))
    pz.chk.check_structure(
        result, pz.chk.ArraySpec(named_shape={"batch": 3, "features": 32})
    )

  def test_build_dropout_mlp(self):
    mlp = simple_mlp.DropoutMLP.from_config(
        name="mlp",
        init_base_rng=jax.random.key(0),
        feature_sizes=[8, 16, 32, 32],
        drop_rate=0.2,
    )
    result = mlp(
        pz.nx.zeros({"batch": 3, "features": 8}),
        random_stream=pz.RandomStream.from_base_key(jax.random.key(0)),
    )
    pz.chk.check_structure(
        result, pz.chk.ArraySpec(named_shape={"batch": 3, "features": 32})
    )

  def test_train_deterministic_mlp(self):
    # Train the deterministic MLP to 100% accuracy on XOR.
    # Also tests the basic training utility.
    mlp = simple_mlp.MLP.from_config(
        name="mlp",
        init_base_rng=jax.random.key(0),
        feature_sizes=[2, 32, 32, 2],
    )

    const_xor_inputs = pz.nx.wrap(
        jnp.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=jnp.float32),
        "batch",
        "features",
    )
    const_xor_labels = jnp.array(
        [[0, 1], [1, 0], [1, 0], [0, 1]], dtype=jnp.float32
    )

    def loss_fn(model, rng, state, xor_inputs, xor_labels):
      scale = 1 + jax.random.uniform(rng, shape=(1,))
      model_out = model(xor_inputs)
      log_probs = jax.nn.log_softmax(
          model_out.unwrap("batch", "features"), axis=-1
      )
      losses = -scale * log_probs * xor_labels
      loss = jnp.sum(losses) / 4
      return (loss, state + 1, {"loss": loss, "count": state})

    trainer = basic_training.StatefulTrainer.build(
        root_rng=jax.random.key(42),
        model=mlp,
        optimizer_def=optax.adam(0.1),
        loss_fn=loss_fn,
        initial_loss_fn_state=100,
    )

    outputs = []
    for _ in range(20):
      outputs.append(
          trainer.step(xor_inputs=const_xor_inputs, xor_labels=const_xor_labels)
      )

    self.assertEqual(outputs[0]["count"], 100)
    self.assertGreater(outputs[0]["loss"], 0.7)
    self.assertEqual(outputs[-1]["count"], 119)
    self.assertLess(outputs[-1]["loss"], 0.0001)


if __name__ == "__main__":
  absltest.main()
