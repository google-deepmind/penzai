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

"""The Gemma architecture transformer variant.

See the Gemma technical report at
https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf
and the accompanying reference implementation at
https://github.com/google-deepmind/gemma.
"""

from __future__ import annotations

import itertools
from typing import Any

import jax.numpy as jnp
from penzai import pz
from penzai.models.transformer import model_parts
from penzai.models.transformer.variants import llamalike_common


def gemma_from_pretrained_checkpoint(
    ckpt_params: dict[str, Any],
    upcast_activations_to_float32: bool = False,
    use_layer_stack: bool = False,
) -> model_parts.TransformerLM:
  """Builds a Gemma model from a pretrained checkpoint.

  The parameters of the loaded ``Transformer`` will be close to those in
  the original checkpoint with a few modifications:

  * Query, key, and value heads are stored in three separate matrices instead
    of being stored either as a single matrix (qkv_einsum) or as two (q_einsum
    and kv_einsum).

  * `RMSLayerNorm` weights have their values increased by one, instead of
    adding one at call time.

  * Axes of parameters are identified by name instead of by position.

  Args:
    ckpt_params: Nested dictionary of weights from the Gemma checkpoint.
    upcast_activations_to_float32: Whether to cast activations to float32 when
      the model runs. This allows analyzing activations at higher precision
      without consuming additional memory for parameters.
    use_layer_stack: Whether to use a layer stack for the decoder blocks.

  Returns:
    A Transformer model containing the loaded parameters.
  """
  params = {k.removeprefix("transformer/"): v for k, v in ckpt_params.items()}
  num_layers = 0
  for i in itertools.count():
    if f"layer_{i}/mlp/linear" not in params:
      num_layers = i
      break
  hidden_dim, embed_dim = params["layer_0/mlp/linear"]["w"].shape
  attn_0_einsum_param = params["layer_0/attn/attn_vec_einsum"]["w"]
  num_heads, proj_dim, _ = attn_0_einsum_param.shape
  single_kv_head = "layer_0/attn/qkv_einsum" not in params
  vocab_size = params["embedder"]["input_embedding"].shape[0]

  if upcast_activations_to_float32:
    activation_dtype = jnp.float32
  else:
    activation_dtype = attn_0_einsum_param.dtype

  config = llamalike_common.LlamalikeTransformerConfig(
      num_kv_heads=1 if single_kv_head else num_heads,
      query_head_multiplier=num_heads if single_kv_head else 1,
      embedding_dim=embed_dim,
      projection_dim=proj_dim,
      mlp_hidden_dim=hidden_dim,
      num_decoder_blocks=num_layers,
      vocab_size=vocab_size,
      parameter_dtype=attn_0_einsum_param.dtype,
      mlp_variant="geglu_approx",
      rope_wavelength=10_000,
      tie_embedder_and_logits=True,
      activation_dtype=activation_dtype,
      use_layer_stack=use_layer_stack,
  )
  model_def = llamalike_common.build_llamalike_transformer(
      config, init_base_rng=None, name="transformer"
  )
  parameter_mapping = {
      "embedder.embeddings": pz.nx.NamedArray.wrap(
          params["embedder"]["input_embedding"]
      ).tag("vocabulary", "embedding"),
      "final_norm/scale.weights": pz.nx.NamedArray.wrap(
          1 + params["final_norm"]["scale"]
      ).tag("embedding"),
  }

  all_block_params = []

  for i in range(config.num_decoder_blocks):
    cur_block_params = {}
    all_block_params.append(cur_block_params)

    cur_block_params["pre_attention_norm/scale.weights"] = (
        pz.nx.NamedArray.wrap(
            1 + params[f"layer_{i}/pre_attention_norm"]["scale"]
        ).tag("embedding")
    )
    cur_block_params["pre_ffw_norm/scale.weights"] = pz.nx.NamedArray.wrap(
        1 + params[f"layer_{i}/pre_ffw_norm"]["scale"]
    ).tag("embedding")
    cur_block_params["mlp/gating_linear.weights"] = pz.nx.NamedArray.wrap(
        params[f"layer_{i}/mlp/gating_einsum"]["w"][0]
    ).tag("embedding", "neurons")
    cur_block_params["mlp/value_linear.weights"] = pz.nx.NamedArray.wrap(
        params[f"layer_{i}/mlp/gating_einsum"]["w"][1]
    ).tag("embedding", "neurons")
    cur_block_params["mlp/out_linear.weights"] = pz.nx.NamedArray.wrap(
        params[f"layer_{i}/mlp/linear"]["w"]
    ).tag("neurons", "embedding")

    if single_kv_head:
      cur_block_params["attention/query.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/q_einsum"]["w"]
      ).tag("query_heads", "embedding", "projection")
      cur_block_params["attention/key.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/kv_einsum"]["w"][0].squeeze(0)
      ).tag("embedding", "projection")
      cur_block_params["attention/value.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/kv_einsum"]["w"][1].squeeze(0)
      ).tag("embedding", "projection")
      cur_block_params["attention/output.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/attn_vec_einsum"]["w"]
      ).tag("query_heads", "projection", "embedding")
    else:
      cur_block_params["attention/query.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/qkv_einsum"]["w"][0]
      ).tag("heads", "embedding", "projection")
      cur_block_params["attention/key.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/qkv_einsum"]["w"][1]
      ).tag("heads", "embedding", "projection")
      cur_block_params["attention/value.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/qkv_einsum"]["w"][2]
      ).tag("heads", "embedding", "projection")
      cur_block_params["attention/output.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/attn_vec_einsum"]["w"]
      ).tag("heads", "projection", "embedding")

  if use_layer_stack:
    for key in all_block_params[0].keys():
      vals = [
          all_block_params[i][key] for i in range(config.num_decoder_blocks)
      ]
      parameter_mapping[f"blocks/{key}"] = pz.nx.stack(vals, "blocks")
  else:
    for i in range(config.num_decoder_blocks):
      for key, value in all_block_params[i].items():
        parameter_mapping[f"block_{i}/{key}"] = value

  # Create parameter objects for each parameter, and bind them to the model's
  # slots.
  model = pz.bind_variables(
      model_def,
      [
          pz.Parameter(value=v, label=f"transformer/{k}")
          for k, v in parameter_mapping.items()
      ],
  )
  pz.nn.assert_no_parameter_slots(model)
  return model
