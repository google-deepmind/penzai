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

"""Transformer variant for GPT-NeoX models.

The GPT-NeoX architecture is used by the GPT-NeoX-20B model (Black et al., 2022)
and the Pythia model scaling suite (Biderman et al., 2023).

Features of the architecture:

- Full multi-head attention
- Rotary positional embeddings (Su et al., 2021), but only applied to a subset
  of positions,
- Parallel Transformer layer formulation (Wang & Komatsuzaki, 2021)
- Biases for all kernels
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Literal

import jax
import jax.numpy as jnp
from penzai import pz
from penzai.models.transformer import model_parts


@dataclasses.dataclass
class GPTNeoXTransformerConfig:
  """Configuration parameters for a GPT Neo-X transformer.

  These are held in a single configuration object to simplify argument passing
  during construction of the model.

  Attributes:
    num_attention_heads: The number of attention heads.
    embedding_dim: Dimension of the embedding vectors and residual stream.
    projection_dim: Dimension of the query, key, and value projections. Usually
      ``embedding_dim // num_heads``.
    mlp_hidden_dim: Dimensionality of the hidden layer of the MLP blocks in each
      layer (the "neurons" axis).
    num_decoder_blocks: Number of transformer decoder blocks in the model.
    vocab_size: Number of tokens in the vocabulary.
    activation_fn: Activation function
    rope_subset_size: Number of projection dimensions to allocate to rotary
      position embeddings.
    rope_wavelength: Wavelength for RoPE layers.
    layernorm_epsilon: Epsilon for layer normalization layers.
    parameter_dtype: Floating dtype to use for all parameters.
    activation_dtype: Floating dtype to use for activations and KV cache tables.
    use_layer_stack: Whether to stack the blocks together using a LayerStack.
  """

  num_attention_heads: int
  embedding_dim: int
  projection_dim: int
  mlp_hidden_dim: int
  num_decoder_blocks: int
  vocab_size: int
  activation_fn: Literal["relu", "selu", "gelu_exact", "gelu_approx"]
  rope_subset_size: int
  rope_wavelength: float
  layernorm_epsilon: float
  parameter_dtype: jax.typing.DTypeLike
  activation_dtype: jax.typing.DTypeLike
  use_layer_stack: bool = False


def build_gpt_neox_feedforward(
    name: str,
    init_base_rng: jax.Array | None,
    config: GPTNeoXTransformerConfig,
) -> model_parts.TransformerFeedForward:
  """Creates a feedforward block.

  The GPT-NeoX model uses a standard MLP configuration.

  Args:
    name: Name of the feedforward block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.

  Returns:
    An instance of TransformerFeedForward containing the MLP.
  """
  act_fn = {
      "gelu_exact": functools.partial(jax.nn.gelu, approximate=False),
      "gelu_approx": functools.partial(jax.nn.gelu, approximate=True),
      "silu": jax.nn.silu,
      "relu": jax.nn.relu,
  }[config.activation_fn]
  return model_parts.TransformerFeedForward([
      pz.nn.Affine.from_config(
          name=f"{name}/in",
          init_base_rng=init_base_rng,
          input_axes={"embedding": config.embedding_dim},
          output_axes={"neurons": config.mlp_hidden_dim},
      ),
      pz.nn.Elementwise(act_fn),
      pz.nn.Affine.from_config(
          name=f"{name}/out",
          init_base_rng=init_base_rng,
          input_axes={"neurons": config.mlp_hidden_dim},
          output_axes={"embedding": config.embedding_dim},
      ),
  ])


def build_gpt_neox_attention(
    name: str,
    init_base_rng: jax.Array | None,
    config: GPTNeoXTransformerConfig,
) -> pz.nn.Attention:
  """Builds an attention block from a configuration.

  Args:
    name: Name of the attention block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.

  Returns:
    An Attention block.
  """
  num_heads = config.num_attention_heads
  embedding_dim = config.embedding_dim
  projection_dim = config.projection_dim

  # As used in https://github.com/google-deepmind/gemma.
  # (This exact value is probably not important.)
  masked_out_value = jnp.array(-2.3819763e38, dtype=config.activation_dtype)

  return pz.nn.Attention(
      input_to_query=pz.nn.Sequential([
          pz.nn.Affine.from_config(
              name=f"{name}/query",
              init_base_rng=init_base_rng,
              input_axes={"embedding": embedding_dim},
              output_axes={"heads": num_heads, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
          pz.nn.ApplyRoPEToSubset(
              positions_input_name="token_positions",
              embedding_axis="projection",
              max_wavelength=config.rope_wavelength,
              rope_subset_size=config.rope_subset_size,
          ),
          pz.nn.ConstantRescale(
              by=jnp.array(projection_dim**-0.5, dtype=config.activation_dtype)
          ),
      ]),
      input_to_key=pz.nn.Sequential([
          pz.nn.Affine.from_config(
              name=f"{name}/key",
              init_base_rng=init_base_rng,
              input_axes={"embedding": embedding_dim},
              output_axes={"heads": num_heads, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
          pz.nn.ApplyRoPEToSubset(
              positions_input_name="token_positions",
              embedding_axis="projection",
              max_wavelength=config.rope_wavelength,
              rope_subset_size=config.rope_subset_size,
          ),
      ]),
      input_to_value=pz.nn.Sequential([
          pz.nn.Affine.from_config(
              name=f"{name}/value",
              init_base_rng=init_base_rng,
              input_axes={"embedding": embedding_dim},
              output_axes={"heads": num_heads, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
      ]),
      query_key_to_attn=pz.nn.Sequential([
          pz.nn.NamedEinsum(
              (
                  {"seq": "tq", "heads": "h", "projection": "p"},
                  {"seq": "tkv", "heads": "h", "projection": "p"},
              ),
              {"seq": "tq", "heads": "h", "kv_seq": "tkv"},
          ),
          pz.nn.ApplyCausalAttentionMask(
              masked_out_value=masked_out_value,
          ),
          pz.nn.Softmax("kv_seq"),
      ]),
      attn_value_to_output=pz.nn.Sequential([
          pz.nn.NamedEinsum(
              (
                  {"seq": "tq", "heads": "h", "kv_seq": "tkv"},
                  {"seq": "tkv", "heads": "h", "projection": "p"},
              ),
              {"seq": "tq", "heads": "h", "projection": "p"},
          ),
          pz.nn.Affine.from_config(
              name=f"{name}/output",
              init_base_rng=init_base_rng,
              input_axes={"heads": num_heads, "projection": projection_dim},
              output_axes={"embedding": embedding_dim},
              dtype=config.parameter_dtype,
          ),
      ]),
  )


def build_gpt_neox_block(
    name: str,
    init_base_rng: jax.Array | None,
    config: GPTNeoXTransformerConfig,
) -> model_parts.TransformerBlock:
  """Builds a GPT-NeoX "parallel" transformer block from a configuration.

  GPT-NeoX uses a parallel formulation of transformer blocks, where the input of
  the previous block is fed to the attention and feedforward components at the
  same time:

    y = x + MLP(LayerNorm(x)) + Attention(LayerNorm(x))

  Args:
    name: Name of the block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.

  Returns:
    A full transformer block.
  """
  return model_parts.TransformerBlock(
      sublayers=[
          pz.nn.BranchAndAddTogether([
              pz.nn.Identity(),
              pz.nn.Sequential([
                  pz.nn.LayerNorm.from_config(
                      name=f"{name}/pre_attention_norm",
                      init_base_rng=init_base_rng,
                      across_axes={"embedding": config.embedding_dim},
                      epsilon=config.layernorm_epsilon,
                      dtype=config.parameter_dtype,
                  ),
                  build_gpt_neox_attention(
                      f"{name}/attention", init_base_rng, config
                  ),
              ]),
              pz.nn.Sequential([
                  pz.nn.LayerNorm.from_config(
                      name=f"{name}/pre_ffw_norm",
                      init_base_rng=init_base_rng,
                      across_axes={"embedding": config.embedding_dim},
                      epsilon=config.layernorm_epsilon,
                      dtype=config.parameter_dtype,
                  ),
                  build_gpt_neox_feedforward(
                      f"{name}/mlp", init_base_rng, config
                  ),
              ]),
          ])
      ],
  )


def build_gpt_neox_transformer(
    config: GPTNeoXTransformerConfig,
    init_base_rng: jax.Array | None = None,
    name: str = "transformer",
) -> model_parts.TransformerLM:
  """Builds a Llama-like transformer model from a configuration.

  Args:
    config: The configuration of the model.
    init_base_rng: Base RNG for initializing the parameters.
    name: Name for the top-level model, used as a prefix for all parameters.

  Returns:
    A full transformer model.
  """

  # Embedding table is shared between first and last layers.
  emb_table = pz.nn.EmbeddingTable.from_config(
      name=f"{name}/embedder",
      init_base_rng=init_base_rng,
      vocab_size=config.vocab_size,
      embedding_axes={"embedding": config.embedding_dim},
      dtype=config.parameter_dtype,
  )
  sublayers = []
  sublayers.append(pz.nn.EmbeddingLookup(emb_table))
  if config.activation_dtype != config.parameter_dtype:
    sublayers.append(pz.nn.CastToDType(config.activation_dtype))

  if config.use_layer_stack:
    sublayers.append(
        pz.nn.LayerStack.from_sublayer_builder(
            builder=build_gpt_neox_block,
            stack_axis="blocks",
            stack_axis_size=config.num_decoder_blocks,
            init_base_rng=init_base_rng,
            builder_kwargs=dict(name=f"{name}/blocks", config=config),
        )
    )
  else:
    for i in range(config.num_decoder_blocks):
      sublayers.append(
          build_gpt_neox_block(f"{name}/block_{i}", init_base_rng, config)
      )

  sublayers.append(
      pz.nn.LayerNorm.from_config(
          name=f"{name}/final_norm",
          init_base_rng=init_base_rng,
          across_axes={"embedding": config.embedding_dim},
          dtype=config.parameter_dtype,
      )
  )

  sublayers.append(
      pz.nn.Linear.from_config(
          name=f"{name}/lm_head",
          init_base_rng=init_base_rng,
          input_axes={"embedding": config.embedding_dim},
          output_axes={"vocabulary": config.vocab_size},
      )
  )

  return model_parts.TransformerLM(
      metadata=model_parts.TransformerMetadata(
          common_head_axes={"heads": config.num_attention_heads},
          query_only_head_axes={},
          embedding_dim=config.embedding_dim,
          projection_dim=config.projection_dim,
          mlp_hidden_dim=config.mlp_hidden_dim,
          vocab_size=config.vocab_size,
          activation_dtype=config.activation_dtype,
      ),
      body=pz.nn.Sequential(sublayers),
  )


GPTNeoXForCausalLM = Any


def gpt_neox_from_huggingface_model(
    model: GPTNeoXForCausalLM,
    upcast_activations_to_float32: bool = False,
    use_layer_stack: bool = False,
) -> model_parts.TransformerLM:
  """Converts a GPT-NeoX model to a Penzai model.

  This function converts GPT-NeoX models from their HuggingFace implementations
  to Penzai.

  Note: Checkpoint conversion is only implemented for the most common set of
  hyperparameters for GPT-NeoX models, including GPT-NeoX-20B and the Pythia
  scaling suite.

  Args:
    model: The HuggingFace Llama or Mistral model.
    upcast_activations_to_float32: Whether to cast activations to float32 when
      the model runs. This allows analyzing activations at higher precision
      without consuming additional memory for parameters.
    use_layer_stack: Whether to use a layer stack for the decoder blocks.

  Returns:
    A Transformer model containing the loaded parameters.
  """
  # Checkpoint conversion assumes these configuration arguments are set:
  hf_config = model.config
  checked_config_args = dict(
      use_parallel_residual=True,
      rope_scaling=None,
      attention_bias=True,
      attention_dropout=0.0,
      hidden_dropout=0.0,
  )
  for k, v in checked_config_args.items():
    actual_value = getattr(hf_config, k)
    if actual_value != v:
      raise ValueError(
          f"Conversion of a GPTNeoXForCausalLM requires config.{k}={repr(v)},"
          f" but got {actual_value}"
      )

  param_dtype = {
      "torch.float32": jnp.float32,
      "torch.bfloat16": jnp.bfloat16,
  }[str(model.dtype)]
  if upcast_activations_to_float32:
    activation_dtype = jnp.float32
  else:
    activation_dtype = param_dtype

  projection_dim = hf_config.hidden_size // hf_config.num_attention_heads
  converted_activation_fn = {
      "gelu": "gelu_exact",
      "gelu_new": "gelu_approx",
      "relu": "relu",
      "silu": "silu",
  }[hf_config.hidden_act]
  pz_config = GPTNeoXTransformerConfig(
      num_attention_heads=hf_config.num_attention_heads,
      embedding_dim=hf_config.hidden_size,
      projection_dim=projection_dim,
      mlp_hidden_dim=hf_config.intermediate_size,
      num_decoder_blocks=hf_config.num_hidden_layers,
      vocab_size=hf_config.vocab_size,
      activation_fn=converted_activation_fn,
      rope_wavelength=hf_config.rotary_emb_base,
      rope_subset_size=int(hf_config.rotary_pct * projection_dim),
      layernorm_epsilon=hf_config.layer_norm_eps,
      parameter_dtype=param_dtype,
      activation_dtype=activation_dtype,
      use_layer_stack=use_layer_stack,
  )
  model_def = build_gpt_neox_transformer(
      pz_config, init_base_rng=None, name="transformer"
  )

  state_dict = model.state_dict()
  converted = {k: jax.dlpack.from_dlpack(v) for k, v in state_dict.items()}

  parameter_mapping = {
      "embedder.embeddings": pz.nx.NamedArray.wrap(
          converted["gpt_neox.embed_in.weight"]
      ).tag("vocabulary", "embedding"),
      "final_norm/scale.weights": pz.nx.NamedArray.wrap(
          converted["gpt_neox.final_layer_norm.weight"]
      ).tag("embedding"),
      "final_norm/shift.bias": pz.nx.NamedArray.wrap(
          converted["gpt_neox.final_layer_norm.bias"]
      ).tag("embedding"),
      "lm_head.weights": pz.nx.NamedArray.wrap(
          converted["embed_out.weight"]
      ).tag("vocabulary", "embedding"),
  }

  all_block_params = []

  for i in range(pz_config.num_decoder_blocks):
    cur_block_params = {}
    all_block_params.append(cur_block_params)

    cur_block_params["pre_attention_norm/scale.weights"] = (
        pz.nx.NamedArray.wrap(
            converted[f"gpt_neox.layers.{i}.input_layernorm.weight"]
        ).tag("embedding")
    )
    cur_block_params["pre_attention_norm/shift.bias"] = pz.nx.NamedArray.wrap(
        converted[f"gpt_neox.layers.{i}.input_layernorm.bias"]
    ).tag("embedding")
    cur_block_params["pre_ffw_norm/scale.weights"] = pz.nx.NamedArray.wrap(
        converted[f"gpt_neox.layers.{i}.post_attention_layernorm.weight"]
    ).tag("embedding")
    cur_block_params["pre_ffw_norm/shift.bias"] = pz.nx.NamedArray.wrap(
        converted[f"gpt_neox.layers.{i}.post_attention_layernorm.bias"]
    ).tag("embedding")
    cur_block_params["mlp/in/Linear.weights"] = pz.nx.NamedArray.wrap(
        converted[f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight"]
    ).tag("neurons", "embedding")
    cur_block_params["mlp/in/AddBias.bias"] = pz.nx.NamedArray.wrap(
        converted[f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias"]
    ).tag("neurons")
    cur_block_params["mlp/out/Linear.weights"] = pz.nx.NamedArray.wrap(
        converted[f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight"]
    ).tag("embedding", "neurons")
    cur_block_params["mlp/out/AddBias.bias"] = pz.nx.NamedArray.wrap(
        converted[f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias"]
    ).tag("embedding")

    qkv_weights = (
        pz.nx.NamedArray.wrap(
            converted[f"gpt_neox.layers.{i}.attention.query_key_value.weight"]
        )
        .reshape((
            pz_config.num_attention_heads,
            3,
            pz_config.projection_dim,
            pz_config.embedding_dim,
        ))
        .tag("heads", "qkv", "projection", "embedding")
    )
    qkv_biases = (
        pz.nx.NamedArray.wrap(
            converted[f"gpt_neox.layers.{i}.attention.query_key_value.bias"]
        )
        .reshape((
            pz_config.num_attention_heads,
            3,
            pz_config.projection_dim,
        ))
        .tag("heads", "qkv", "projection")
    )

    cur_block_params["attention/query/Linear.weights"] = qkv_weights[{"qkv": 0}]
    cur_block_params["attention/key/Linear.weights"] = qkv_weights[{"qkv": 1}]
    cur_block_params["attention/value/Linear.weights"] = qkv_weights[{"qkv": 2}]
    cur_block_params["attention/query/AddBias.bias"] = qkv_biases[{"qkv": 0}]
    cur_block_params["attention/key/AddBias.bias"] = qkv_biases[{"qkv": 1}]
    cur_block_params["attention/value/AddBias.bias"] = qkv_biases[{"qkv": 2}]

    cur_block_params["attention/output/Linear.weights"] = (
        pz.nx.NamedArray.wrap(
            converted[f"gpt_neox.layers.{i}.attention.dense.weight"]
        )
        .reshape((
            pz_config.embedding_dim,
            pz_config.num_attention_heads,
            pz_config.projection_dim,
        ))
        .tag("embedding", "heads", "projection")
    )
    cur_block_params["attention/output/AddBias.bias"] = pz.nx.NamedArray.wrap(
        converted[f"gpt_neox.layers.{i}.attention.dense.bias"]
    ).tag("embedding")

  if use_layer_stack:
    for key in all_block_params[0].keys():
      vals = [
          all_block_params[i][key] for i in range(pz_config.num_decoder_blocks)
      ]
      parameter_mapping[f"blocks/{key}"] = pz.nx.stack(vals, "blocks")
  else:
    for i in range(pz_config.num_decoder_blocks):
      for key, value in all_block_params[i].items():
        parameter_mapping[f"block_{i}/{key}"] = value

  # Create parameter objects for each parameter.
  model = pz.bind_variables(
      model_def,
      [
          pz.Parameter(value=v, label=f"transformer/{k}")
          for k, v in parameter_mapping.items()
      ],
  )
  pz.nn.assert_no_parameter_slots(model)
  return model
