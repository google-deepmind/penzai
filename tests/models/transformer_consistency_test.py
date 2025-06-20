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

"""Tests for consistency between official and Penzai model implementations."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
import numpy as np
from penzai import pz
from penzai.models.transformer.variants import gpt_neox
from penzai.models.transformer.variants import llama
from penzai.models.transformer.variants import mistral
import torch
import transformers


class TransformerConsistencyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="full", num_attention_heads=4, num_key_value_heads=4),
      dict(testcase_name="mqa", num_attention_heads=4, num_key_value_heads=1),
      dict(testcase_name="gqa", num_attention_heads=4, num_key_value_heads=2),
  )
  def test_llama_consistency(self, num_attention_heads, num_key_value_heads):
    cfg = transformers.LlamaConfig(
        name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
        vocab_size=11,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=0,
        eos_token_id=1,
        hidden_act="silu",
        initializer_range=0.02,
        max_position_embeddings=2048,
        mlp_bias=False,
        model_type="llama",
        pad_token_id=-1,
        pretraining_tp=1,
        rms_norm_eps=1e-06,
        rope_scaling=None,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        torch_dtype="float32",
        transformers_version="4.44.2",
        use_cache=True,
    )

    torch.manual_seed(0)
    hf_model = transformers.LlamaForCausalLM(cfg)

    tokens = pz.nx.wrap(jnp.tile(jnp.arange(11), 3)[None, :], "batch", "seq")

    hf_arg = torch.tensor(np.array(tokens.unwrap("batch", "seq")))
    hf_out = pz.nx.wrap(hf_model(hf_arg).logits.detach().numpy()).tag(
        "batch", "seq", "vocabulary"
    )

    for layer_stack in (False, True):
      with self.subTest(f"layer_stack={layer_stack}"):
        pz_model = llama.llama_from_huggingface_model(
            hf_model, use_layer_stack=layer_stack
        )

        pz_out = pz_model(
            tokens,
            token_positions=pz.nx.arange("seq", tokens.named_shape["seq"]),
        )

        chex.assert_trees_all_close(
            pz_out, hf_out.order_like(pz_out), atol=1e-6
        )

  @parameterized.named_parameters(
      dict(testcase_name="full", num_attention_heads=4, num_key_value_heads=4),
      dict(testcase_name="mqa", num_attention_heads=4, num_key_value_heads=1),
      dict(testcase_name="gqa", num_attention_heads=4, num_key_value_heads=2),
  )
  def test_mistral_consistency(self, num_attention_heads, num_key_value_heads):
    cfg = transformers.MistralConfig(
        name_or_path="hf-internal-testing/tiny-random-MistralForCausalLM",
        is_decoder=True,
        vocab_size=11,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        attention_dropout=0.0,
        attention_probs_dropout_prob=0.1,
        bos_token_id=1,
        eos_token_id=2,
        head_dim=16,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        max_position_embeddings=512,
        model_type="mistral",
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        sliding_window=4096,
        tie_word_embeddings=False,
        torch_dtype="float32",
        transformers_version="4.44.2",
        type_vocab_size=16,
        use_cache=True,
    )

    torch.manual_seed(0)
    hf_model = transformers.MistralForCausalLM(cfg)

    tokens = pz.nx.wrap(jnp.tile(jnp.arange(11), 3)[None, :], "batch", "seq")

    hf_arg = torch.tensor(np.array(tokens.unwrap("batch", "seq")))
    hf_out = pz.nx.wrap(hf_model(hf_arg).logits.detach().numpy()).tag(
        "batch", "seq", "vocabulary"
    )

    for layer_stack in (False, True):
      with self.subTest(f"layer_stack={layer_stack}"):
        pz_model = mistral.mistral_from_huggingface_model(
            hf_model, use_layer_stack=layer_stack
        )
        pz_out = pz_model(
            tokens,
            token_positions=pz.nx.arange("seq", tokens.named_shape["seq"]),
        )

        chex.assert_trees_all_close(
            pz_out, hf_out.order_like(pz_out), atol=1e-6
        )

  def test_gpt_neox_consistency(self):
    cfg = transformers.GPTNeoXConfig(
        name_or_path="organization-name/model-name",
        is_decoder=True,
        vocab_size=11,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        type_vocab_size=16,
        hidden_act="gelu",
        attention_bias=True,
        attention_dropout=0.0,
        bos_token_id=0,
        classifier_dropout=0.1,
        eos_token_id=0,
        hidden_dropout=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        max_position_embeddings=512,
        model_type="gpt_neox",
        rope_scaling=None,
        rotary_emb_base=10000,
        rotary_pct=0.25,
        tie_word_embeddings=False,
        torch_dtype="float32",
        transformers_version="4.44.2",
        use_cache=True,
        use_parallel_residual=True,
    )

    torch.manual_seed(0)
    hf_model = transformers.GPTNeoXForCausalLM(cfg)

    tokens = pz.nx.wrap(jnp.tile(jnp.arange(11), 3)[None, :], "batch", "seq")

    hf_arg = torch.tensor(np.array(tokens.unwrap("batch", "seq")))
    hf_out = pz.nx.wrap(hf_model(hf_arg).logits.detach().numpy()).tag(
        "batch", "seq", "vocabulary"
    )

    for layer_stack in (False, True):
      with self.subTest(f"layer_stack={layer_stack}"):
        pz_model = gpt_neox.gpt_neox_from_huggingface_model(
            hf_model, use_layer_stack=layer_stack
        )
        pz_out = pz_model(
            tokens,
            token_positions=pz.nx.arange("seq", tokens.named_shape["seq"]),
        )

        chex.assert_trees_all_close(
            pz_out, hf_out.order_like(pz_out), atol=1e-3
        )
        chex.assert_trees_all_close(
            pz_out, hf_out.order_like(pz_out), rtol=3e-3
        )


if __name__ == "__main__":
  absltest.main()
