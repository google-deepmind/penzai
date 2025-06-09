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
        # Adjusted architecture parameters for a smaller version of Llama.
        vocab_size=11,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        # Extra parameters that are set when loading official models from
        # HuggingFace but aren't set by default in LlamaConfig.
        max_position_embeddings=8192,
        rms_norm_eps=1e-05,
        rope_theta=500000.0,
        torch_dtype="bfloat16",
        architectures=["LlamaForCausalLM"],
        bos_token_id=128000,
        eos_token_id=128001,
        _name_or_path="meta-llama/Meta-Llama-3-8B",
        _attn_implementation_autoset=True,
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
      dict(
          testcase_name="act_gelu",
          num_attention_heads=4,
          num_key_value_heads=4,
          hidden_act="gelu",
      ),
  )
  def test_mistral_consistency(
      self, num_attention_heads, num_key_value_heads, hidden_act="silu"
  ):
    cfg = transformers.MistralConfig(
        # Adjusted architecture parameters for a smaller version of Mistral.
        vocab_size=11,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_act=hidden_act,
        # Extra parameters that are set when loading official models from
        # HuggingFace but aren't set by default in MistralConfig.
        max_position_embeddings=32768,
        sliding_window=None,
        rms_norm_eps=1e-05,
        rope_theta=1000000.0,
        torch_dtype="bfloat16",
        architectures=["MistralForCausalLM"],
        _name_or_path="fake_org/fake-Mistral-version",
        _attn_implementation_autoset=True,
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
        # Adjusted architecture parameters for a smaller version of GPT-NeoX.
        vocab_size=11,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        # Extra parameters that are set when loading official models from
        # HuggingFace but aren't set by default in GPTNeoXConfig.
        torch_dtype="float16",
        architectures=["GPTNeoXForCausalLM"],
        eos_token_id=0,
        _name_or_path="fake_org/fake-GPTNeoX-version",
        _attn_implementation_autoset=True,
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
