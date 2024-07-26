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

"""Tests for the Gemma model."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from penzai.deprecated.v1 import pz
from penzai.deprecated.v1.example_models import gemma
from penzai.deprecated.v1.toolshed import check_layers_by_tracing
from penzai.deprecated.v1.toolshed import jit_wrapper


class SimpleMlpTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="single_b16",
          single_kv_head=True,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="single_b32",
          single_kv_head=True,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="single_mixed",
          single_kv_head=True,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.float32,
      ),
      dict(
          testcase_name="many_b16",
          single_kv_head=False,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="many_b32",
          single_kv_head=False,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="many_mixed",
          single_kv_head=False,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.float32,
      ),
  )
  def test_build_and_run_gemma(
      self, single_kv_head: bool, parameter_dtype, activation_dtype
  ):
    model = gemma.model_core.GemmaTransformer.from_config(
        gemma.model_core.GemmaTransformerConfig(
            num_heads=2,
            embedding_dim=16,
            projection_dim=4,
            single_kv_head=single_kv_head,
            mlp_hidden_dim=32,
            num_decoder_blocks=2,
            vocab_size=11,
            parameter_dtype=parameter_dtype,
            activation_dtype=activation_dtype,
        )
    )
    out = check_layers_by_tracing.check_layer(
        model,
        gemma.model_core.GemmaInputs.from_basic_segments(
            pz.nx.ones({"batch": 3, "seq": 13}, dtype=jnp.int32)
        ),
    )
    pz.chk.check_structure(
        out,
        pz.chk.ArraySpec(
            named_shape={"batch": 3, "seq": 13, "vocabulary": 11},
            dtype=activation_dtype,
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="single_b16",
          single_kv_head=True,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="single_b32",
          single_kv_head=True,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="single_mixed",
          single_kv_head=True,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.float32,
      ),
      dict(
          testcase_name="many_b16",
          single_kv_head=False,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="many_b32",
          single_kv_head=False,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name="many_mixed",
          single_kv_head=False,
          parameter_dtype=jnp.bfloat16,
          activation_dtype=jnp.float32,
      ),
  )
  def test_build_and_run_sampling_mode(
      self, single_kv_head: bool, parameter_dtype, activation_dtype
  ):
    model = gemma.model_core.GemmaTransformer.from_config(
        gemma.model_core.GemmaTransformerConfig(
            num_heads=2,
            embedding_dim=16,
            projection_dim=4,
            single_kv_head=single_kv_head,
            mlp_hidden_dim=32,
            num_decoder_blocks=2,
            vocab_size=11,
            parameter_dtype=parameter_dtype,
            activation_dtype=activation_dtype,
        )
    )
    sampling_model, initial_state = (
        gemma.sampling_mode.GemmaKVCachingTransformer.from_uncached(
            model, cache_len=20, batch_axes={"batch": 3}
        )
    )
    out = check_layers_by_tracing.check_layer(
        sampling_model,
        gemma.sampling_mode.GemmaKVCachingInputs.from_basic_subsegments(
            pz.nx.ones({"batch": 3, "seq": 13}, dtype=jnp.int32),
            initial_state,
        ),
    )
    pz.chk.check_structure(
        out,
        (
            pz.chk.ArraySpec(
                named_shape={"batch": 3, "seq": 13, "vocabulary": 11},
                dtype=activation_dtype,
            ),
            pz.chk.as_array_structure(initial_state),
        ),
    )

    sampling_model_init = pz.nn.initialize_parameters(
        sampling_model, jax.random.key(2)
    )
    sampleout = gemma.simple_decoding_loop.temperature_sample_pyloop(
        jit_wrapper.Jitted(sampling_model_init),
        initial_state,
        prompt=pz.nx.ones({"batch": 3, "seq": 5}, dtype=jnp.int32),
        rng=jax.random.key(1),
        pad_id=0,
        max_sampling_steps=12,
    )
    pz.chk.check_structure(
        sampleout,
        pz.chk.ArraySpec(
            named_shape={"batch": 3, "seq": 12},
            dtype=jnp.int32,
        ),
    )


if __name__ == "__main__":
  absltest.main()
