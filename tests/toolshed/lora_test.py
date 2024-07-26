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

"""Tests for low-rank adaptation utilities."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import optax
from penzai import pz
from penzai.models import simple_mlp
from penzai.toolshed import basic_training
from penzai.toolshed import lora


class LoraTest(absltest.TestCase):

  def test_build_lora(self):
    mlp = simple_mlp.MLP.from_config(
        name="mlp",
        init_base_rng=jax.random.key(10),
        feature_sizes=[8, 16, 32, 32],
    )
    loraified_mlp = lora.loraify_linears_in_selection(
        pz.select(mlp), rank=2, init_base_rng=jax.random.key(20)
    )
    (
        pz.select(loraified_mlp)
        .at_instances_of(lora.LowRankAdapter)
        .assert_count_is(3)
    )
    for param in (
        pz.select(loraified_mlp)
        .at_instances_of(lora.LowRankAdapter)
        .at_instances_of(pz.nn.NamedGroup)
        .where(lambda ng: ng.name == "Update")
        .at_instances_of(pz.Parameter)
        .assert_count_is(6)
        .get_sequence()
    ):
      self.assertIn("lowrank", param.value.named_shape)
      self.assertEqual(param.value.named_shape["lowrank"], 2)

    _ = loraified_mlp(pz.nx.zeros({"batch": 3, "features": 8}))

  def test_can_finetune_with_lora(self):
    mlp = simple_mlp.MLP.from_config(
        name="mlp",
        init_base_rng=jax.random.key(10),
        feature_sizes=[2, 32, 32, 2],
    )
    # Freeze the MLP at its random initialization.
    frozen_mlp = pz.freeze_variables(mlp)
    # LoRA-ify it and learn the LoRA parameters.
    lora_mlp = lora.loraify_linears_in_selection(
        pz.select(frozen_mlp), rank=4, init_base_rng=jax.random.key(20)
    )

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

    trainer = basic_training.StatefulTrainer.build(
        root_rng=jax.random.key(42),
        model=lora_mlp,
        optimizer_def=optax.adam(0.1),
        loss_fn=loss_fn,
        initial_loss_fn_state=100,
    )

    outputs = []
    for _ in range(20):
      outputs.append(trainer.step())

    self.assertEqual(outputs[0]["count"], 100)
    self.assertGreater(outputs[0]["loss"], 0.7)
    self.assertEqual(outputs[-1]["count"], 119)
    self.assertLess(outputs[-1]["loss"], 0.0001)


if __name__ == "__main__":
  absltest.main()
