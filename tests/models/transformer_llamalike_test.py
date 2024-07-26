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

"""Tests for the Llama-like transformer variants."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from penzai import pz
from penzai.models.transformer import sampling_mode
from penzai.models.transformer import simple_decoding_loop
from penzai.models.transformer.variants import llamalike_common


class LlamalikeTransformerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="full_b16",
          num_kv_heads=2,
          query_head_multiplier=1,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="full_32",
          num_kv_heads=2,
          query_head_multiplier=1,
          parameter_dtype=jnp.float32,
          activation_dtype=jnp.float32,
      ),
      dict(
          testcase_name="full_mixed",
          num_kv_heads=2,
          query_head_multiplier=1,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.float32,
      ),
      dict(
          testcase_name="multi_query",
          num_kv_heads=2,
          query_head_multiplier=1,
          parameter_dtype=jnp.float32,
          activation_dtype=jnp.float32,
      ),
      dict(
          testcase_name="grouped_query",
          num_kv_heads=2,
          query_head_multiplier=2,
          parameter_dtype=jnp.float32,
          activation_dtype=jnp.float32,
      ),
      dict(
          testcase_name="single_swiglu",
          num_kv_heads=1,
          query_head_multiplier=1,
          parameter_dtype=jnp.float32,
          activation_dtype=jnp.float32,
          mlp_variant="swiglu",
      ),
  )
  def test_build_and_run_gemma(
      self,
      num_kv_heads: int,
      query_head_multiplier: int,
      parameter_dtype,
      activation_dtype,
      mlp_variant="geglu_approx",
  ):
    def run_traced(rng_key):

      model = llamalike_common.build_llamalike_transformer(
          llamalike_common.LlamalikeTransformerConfig(
              num_kv_heads=num_kv_heads,
              query_head_multiplier=query_head_multiplier,
              embedding_dim=16,
              projection_dim=4,
              mlp_hidden_dim=32,
              num_decoder_blocks=2,
              vocab_size=11,
              parameter_dtype=parameter_dtype,
              activation_dtype=activation_dtype,
              mlp_variant=mlp_variant,
              rope_wavelength=10_000,
              tie_embedder_and_logits=True,
          ),
          init_base_rng=rng_key,
      )
      tokens = pz.nx.ones({"batch": 3, "seq": 13}, dtype=jnp.int32)
      out = model(tokens, token_positions=pz.nx.arange("seq", 13))
      pz.chk.check_structure(
          out,
          pz.chk.ArraySpec(
              named_shape={"batch": 3, "seq": 13, "vocabulary": 11},
              dtype=activation_dtype,
          ),
      )

    jax.eval_shape(run_traced, jax.random.key(2))

  @parameterized.named_parameters(
      dict(
          testcase_name="full_b16",
          num_kv_heads=2,
          query_head_multiplier=1,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="full_32",
          num_kv_heads=2,
          query_head_multiplier=1,
          parameter_dtype=jnp.float32,
          activation_dtype=jnp.float32,
      ),
      dict(
          testcase_name="full_mixed",
          num_kv_heads=2,
          query_head_multiplier=1,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.float32,
      ),
      dict(
          testcase_name="multi_query",
          num_kv_heads=2,
          query_head_multiplier=1,
          parameter_dtype=jnp.float32,
          activation_dtype=jnp.float32,
      ),
      dict(
          testcase_name="grouped_query",
          num_kv_heads=2,
          query_head_multiplier=2,
          parameter_dtype=jnp.float32,
          activation_dtype=jnp.float32,
      ),
  )
  def test_build_and_run_sampling_mode(
      self,
      num_kv_heads: int,
      query_head_multiplier: int,
      parameter_dtype,
      activation_dtype,
  ):

    model = llamalike_common.build_llamalike_transformer(
        llamalike_common.LlamalikeTransformerConfig(
            num_kv_heads=num_kv_heads,
            query_head_multiplier=query_head_multiplier,
            embedding_dim=16,
            projection_dim=4,
            mlp_hidden_dim=32,
            num_decoder_blocks=2,
            vocab_size=11,
            mlp_variant="geglu_approx",
            rope_wavelength=10_000,
            parameter_dtype=parameter_dtype,
            activation_dtype=activation_dtype,
            tie_embedder_and_logits=True,
        ),
        init_base_rng=jax.random.key(2),
    )

    sampler = sampling_mode.KVCachingTransformerLM.from_uncached(
        model, cache_len=20, batch_axes={"batch": 3}
    )
    out = sampler(pz.nx.ones({"batch": 3, "seq": 13}, dtype=jnp.int32))
    pz.chk.check_structure(
        out,
        pz.chk.ArraySpec(
            named_shape={"batch": 3, "seq": 13, "vocabulary": 11},
            dtype=activation_dtype,
        ),
    )

    sampler = sampling_mode.KVCachingTransformerLM.from_uncached(
        model, cache_len=20, batch_axes={"batch": 3}
    )
    sampleout = simple_decoding_loop.temperature_sample_pyloop(
        sampler,
        prompt=pz.nx.ones({"batch": 3, "seq": 5}, dtype=jnp.int32),
        rng=jax.random.key(1),
        max_sampling_steps=12,
    )
    pz.chk.check_structure(
        sampleout,
        pz.chk.ArraySpec(
            named_shape={"batch": 3, "seq": 12},
            dtype=jnp.int32,
        ),
    )

  def test_build_and_run_layer_stack(self):
    def run_traced(rng_key):

      stacked_model = llamalike_common.build_llamalike_transformer(
          llamalike_common.LlamalikeTransformerConfig(
              num_kv_heads=2,
              query_head_multiplier=1,
              embedding_dim=16,
              projection_dim=4,
              mlp_hidden_dim=32,
              num_decoder_blocks=2,
              vocab_size=11,
              mlp_variant="geglu_approx",
              rope_wavelength=10_000,
              parameter_dtype=jnp.bfloat16,
              activation_dtype=jnp.float32,
              use_layer_stack=True,
              tie_embedder_and_logits=True,
          ),
          init_base_rng=rng_key,
      )
      tokens = pz.nx.ones({"batch": 3, "seq": 13}, dtype=jnp.int32)
      out = stacked_model(tokens, token_positions=pz.nx.arange("seq", 13))
      pz.chk.check_structure(
          out,
          pz.chk.ArraySpec(
              named_shape={"batch": 3, "seq": 13, "vocabulary": 11},
              dtype=jnp.float32,
          ),
      )

      sampler = sampling_mode.KVCachingTransformerLM.from_uncached(
          stacked_model, cache_len=20, batch_axes={"batch": 3}
      )
      kv_cache_var = (
          pz.select(sampler)
          .at_instances_of(pz.StateVariable)
          .where(lambda v: v.label == "sampler/cache_0")
          .get()
      )
      self.assertEqual(
          kv_cache_var.metadata,
          {
              "layerstack_axes": {
                  "blocks": pz.nn.LayerStackVarBehavior.PER_LAYER
              }
          },
      )
      for i in range(2):
        self.assertEqual(
            kv_cache_var.value[i].named_shape,
            {
                "blocks": 2,
                "batch": 3,
                "heads": 2,
                "projection": 4,
                "seq": 20,
            },
        )

      out = sampler(pz.nx.ones({"batch": 3, "seq": 13}, dtype=jnp.int32))
      pz.chk.check_structure(
          out,
          pz.chk.ArraySpec(
              named_shape={"batch": 3, "seq": 13, "vocabulary": 11},
              dtype=jnp.float32,
          ),
      )

    jax.eval_shape(run_traced, jax.random.key(2))


if __name__ == "__main__":
  absltest.main()
