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

import dataclasses
import itertools
from typing import Any

import jax
import jax.numpy as jnp
from penzai.experimental.v2 import pz
from penzai.experimental.v2.models.transformer import model_parts


@dataclasses.dataclass
class GemmaTransformerConfig:
  """Common configuration parameters for the Gemma transformer architecture.

  These are held in a single configuration object to simplify argument passing
  during construction of the model.

  Attributes:
    num_heads: The number of attention heads to use.
    embedding_dim: Dimension of the embedding vectors and residual stream.
    projection_dim: Dimension of the query, key, and value projections. Usually
      ``embedding_dim // num_heads``.
    single_kv_head: Whether a single key head and value head should be shared
      across all query heads.
    mlp_hidden_dim: Dimensionality of the hidden layer of the MLP blocks in each
      layer (the "neurons" axis).
    num_decoder_blocks: Number of transformer decoder blocks in the model.
    vocab_size: Number of tokens in the vocabulary.
    parameter_dtype: Floating dtype to use for all parameters.
    activation_dtype: Floating dtype to use for activations and KV cache tables.
  """

  num_heads: int
  embedding_dim: int
  projection_dim: int
  single_kv_head: bool
  mlp_hidden_dim: int
  num_decoder_blocks: int
  vocab_size: int
  parameter_dtype: jax.typing.DTypeLike
  activation_dtype: jax.typing.DTypeLike


def build_gemma_feedforward(
    name: str, init_base_rng: jax.Array | None, config: GemmaTransformerConfig
) -> model_parts.TransformerFeedForward:
  """Creates a Gemma feedforward block.

  Gemma's feedforward layer uses GELU-based gated linear units (GEGLU), as
  proposed by Shazeer (2020). We represent this computation as a composition
  of simpler Penzai primitives, to enable patching and post-processing of the
  various internal activations.

  Args:
    name: Name of the feedforward block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the Gemma model.

  Returns:
    An instance of TransformerFeedForward containing the GELU MLP blocks.
  """
  return model_parts.TransformerFeedForward([
      pz.nn.BranchAndMultiplyTogether(
          branches=[
              pz.nn.NamedGroup(
                  "gate",
                  [
                      pz.nn.Linear.from_config(
                          name=f"{name}/gating_linear",
                          init_base_rng=init_base_rng,
                          input_axes={"embedding": config.embedding_dim},
                          output_axes={"neurons": config.mlp_hidden_dim},
                          dtype=config.parameter_dtype,
                      ),
                      pz.nn.Elementwise(jax.nn.gelu),
                  ],
              ),
              pz.nn.Linear.from_config(
                  name=f"{name}/value_linear",
                  init_base_rng=init_base_rng,
                  input_axes={"embedding": config.embedding_dim},
                  output_axes={"neurons": config.mlp_hidden_dim},
                  dtype=config.parameter_dtype,
              ),
          ]
      ),
      pz.nn.Linear.from_config(
          name=f"{name}/out_linear",
          init_base_rng=init_base_rng,
          input_axes={"neurons": config.mlp_hidden_dim},
          output_axes={"embedding": config.embedding_dim},
          dtype=config.parameter_dtype,
      ),
  ])


def build_gemma_attention(
    name: str, init_base_rng: jax.Array | None, config: GemmaTransformerConfig
) -> pz.nn.Attention:
  """Builds a GemmaAttention block from a configuration.

  Args:
    name: Name of the attention block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the Gemma model.

  Returns:
    An Attention block.
  """
  num_heads = config.num_heads
  embedding_dim = config.embedding_dim
  projection_dim = config.projection_dim
  single_kv_head = config.single_kv_head

  if single_kv_head:
    common_head_axes = {}
    common_head_einsum = {}
    query_only_head_axes = {"query_heads": num_heads}
    query_only_head_einsum = {"query_heads": "h"}
  else:
    common_head_axes = {"heads": num_heads}
    common_head_einsum = {"heads": "h"}
    query_only_head_axes = {}
    query_only_head_einsum = {}

  return pz.nn.Attention(
      input_to_query=pz.nn.Sequential([
          pz.nn.Linear.from_config(
              name=f"{name}/query",
              init_base_rng=init_base_rng,
              input_axes={"embedding": embedding_dim},
              output_axes={
                  **common_head_axes,
                  **query_only_head_axes,
                  "projection": projection_dim,
              },
              dtype=config.parameter_dtype,
          ),
          pz.nn.ApplyRoPE(
              positions_input_name="token_positions",
              embedding_axis="projection",
              max_wavelength=10_000,
          ),
          pz.nn.ConstantRescale(
              by=jnp.array(projection_dim**-0.5, dtype=config.activation_dtype)
          ),
      ]),
      input_to_key=pz.nn.Sequential([
          pz.nn.Linear.from_config(
              name=f"{name}/key",
              init_base_rng=init_base_rng,
              input_axes={"embedding": embedding_dim},
              output_axes={**common_head_axes, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
          pz.nn.ApplyRoPE(
              positions_input_name="token_positions",
              embedding_axis="projection",
              max_wavelength=10_000,
          ),
      ]),
      input_to_value=pz.nn.Sequential([
          pz.nn.Linear.from_config(
              name=f"{name}/value",
              init_base_rng=init_base_rng,
              input_axes={"embedding": embedding_dim},
              output_axes={**common_head_axes, "projection": projection_dim},
              dtype=config.parameter_dtype,
          ),
      ]),
      query_key_to_attn=pz.nn.Sequential([
          pz.nn.NamedEinsum(
              (
                  {
                      "seq": "tq",
                      **common_head_einsum,
                      **query_only_head_einsum,
                      "projection": "p",
                  },
                  {
                      "seq": "tkv",
                      **common_head_einsum,
                      "projection": "p",
                  },
              ),
              {
                  "seq": "tq",
                  **common_head_einsum,
                  **query_only_head_einsum,
                  "kv_seq": "tkv",
              },
          ),
          pz.nn.ApplyAttentionMask(
              mask_input_name="attn_mask",
              masked_out_value=jnp.array(
                  -2.3819763e38, dtype=config.activation_dtype
              ),
          ),
          pz.nn.Softmax("kv_seq"),
      ]),
      attn_value_to_output=pz.nn.Sequential([
          pz.nn.NamedEinsum(
              (
                  {
                      "seq": "tq",
                      **common_head_einsum,
                      **query_only_head_einsum,
                      "kv_seq": "tkv",
                  },
                  {
                      "seq": "tkv",
                      **common_head_einsum,
                      "projection": "p",
                  },
              ),
              {
                  "seq": "tq",
                  **common_head_einsum,
                  **query_only_head_einsum,
                  "projection": "p",
              },
          ),
          pz.nn.Linear.from_config(
              name=f"{name}/output",
              init_base_rng=init_base_rng,
              input_axes={
                  **common_head_axes,
                  **query_only_head_axes,
                  "projection": projection_dim,
              },
              output_axes={"embedding": embedding_dim},
              dtype=config.parameter_dtype,
          ),
      ]),
  )


def build_gemma_block(
    name: str, init_base_rng: jax.Array | None, config: GemmaTransformerConfig
) -> model_parts.TransformerBlock:
  """Builds a Gemma transformer block from a configuration.

  Args:
    name: Name of the block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the Gemma model.

  Returns:
    A full transformer block.
  """
  return model_parts.TransformerBlock(
      sublayers=[
          pz.nn.Residual(
              pz.nn.Sequential([
                  pz.nn.RMSLayerNorm.from_config(
                      name=f"{name}/pre_attention_norm",
                      init_base_rng=init_base_rng,
                      across_axes={"embedding": config.embedding_dim},
                      dtype=config.parameter_dtype,
                  ),
                  build_gemma_attention(
                      f"{name}/attention", init_base_rng, config
                  ),
              ])
          ),
          pz.nn.Residual(
              pz.nn.Sequential([
                  pz.nn.RMSLayerNorm.from_config(
                      name=f"{name}/pre_ffw_norm",
                      init_base_rng=init_base_rng,
                      across_axes={"embedding": config.embedding_dim},
                      dtype=config.parameter_dtype,
                  ),
                  build_gemma_feedforward(f"{name}/mlp", init_base_rng, config),
              ])
          ),
      ],
  )


def build_gemma_transformer(
    config: GemmaTransformerConfig,
    init_base_rng: jax.Array | None = None,
    name: str = "transformer",
) -> model_parts.Transformer:
  """Builds a Gemma transformer model from a configuration.

  Args:
    config: The configuration of the Gemma model.
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
  sublayers.append(
      pz.nn.ConstantRescale(
          by=jnp.sqrt(config.embedding_dim).astype(config.activation_dtype)
      )
  )

  for i in range(config.num_decoder_blocks):
    sublayers.append(
        build_gemma_block(f"{name}/block_{i}", init_base_rng, config)
    )

  sublayers.extend([
      pz.nn.RMSLayerNorm.from_config(
          name=f"{name}/final_norm",
          init_base_rng=init_base_rng,
          across_axes={"embedding": config.embedding_dim},
          dtype=config.parameter_dtype,
      ),
      pz.nn.EmbeddingDecode(emb_table),
  ])

  if config.single_kv_head:
    common_head_axes = {}
    query_only_head_axes = {"query_heads": config.num_heads}
  else:
    common_head_axes = {"heads": config.num_heads}
    query_only_head_axes = {}

  return model_parts.Transformer(
      metadata=model_parts.TransformerMetadata(
          common_head_axes=common_head_axes,
          query_only_head_axes=query_only_head_axes,
          embedding_dim=config.embedding_dim,
          projection_dim=config.projection_dim,
          mlp_hidden_dim=config.mlp_hidden_dim,
          vocab_size=config.vocab_size,
          activation_dtype=config.activation_dtype,
      ),
      body=pz.nn.Sequential(sublayers),
  )


def gemma_from_pretrained_checkpoint(
    ckpt_params: dict[str, Any],
    upcast_activations_to_float32: bool = False,
) -> model_parts.Transformer:
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
      the model runs. This is useful for doing interpretability research at
      higher precision without consuming additional memory.

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

  config = GemmaTransformerConfig(
      num_heads=num_heads,
      embedding_dim=embed_dim,
      projection_dim=proj_dim,
      single_kv_head=single_kv_head,
      mlp_hidden_dim=hidden_dim,
      num_decoder_blocks=num_layers,
      vocab_size=vocab_size,
      parameter_dtype=attn_0_einsum_param.dtype,
      activation_dtype=activation_dtype,
  )
  model_def = build_gemma_transformer(
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
  for i in range(config.num_decoder_blocks):
    parameter_mapping.update({
        f"block_{i}/pre_attention_norm/scale.weights": pz.nx.NamedArray.wrap(
            1 + params[f"layer_{i}/pre_attention_norm"]["scale"]
        ).tag("embedding"),
        f"block_{i}/pre_ffw_norm/scale.weights": pz.nx.NamedArray.wrap(
            1 + params[f"layer_{i}/pre_ffw_norm"]["scale"]
        ).tag("embedding"),
        f"block_{i}/mlp/gating_linear.weights": pz.nx.NamedArray.wrap(
            params[f"layer_{i}/mlp/gating_einsum"]["w"][0]
        ).tag("embedding", "neurons"),
        f"block_{i}/mlp/value_linear.weights": pz.nx.NamedArray.wrap(
            params[f"layer_{i}/mlp/gating_einsum"]["w"][1]
        ).tag("embedding", "neurons"),
        f"block_{i}/mlp/out_linear.weights": pz.nx.NamedArray.wrap(
            params[f"layer_{i}/mlp/linear"]["w"]
        ).tag("neurons", "embedding"),
    })
    if config.single_kv_head:
      parameter_mapping.update({
          f"block_{i}/attention/query.weights": pz.nx.NamedArray.wrap(
              params[f"layer_{i}/attn/q_einsum"]["w"]
          ).tag("query_heads", "embedding", "projection"),
          f"block_{i}/attention/key.weights": pz.nx.NamedArray.wrap(
              params[f"layer_{i}/attn/kv_einsum"]["w"][0].squeeze(0)
          ).tag("embedding", "projection"),
          f"block_{i}/attention/value.weights": pz.nx.NamedArray.wrap(
              params[f"layer_{i}/attn/kv_einsum"]["w"][1].squeeze(0)
          ).tag("embedding", "projection"),
          f"block_{i}/attention/output.weights": pz.nx.NamedArray.wrap(
              params[f"layer_{i}/attn/attn_vec_einsum"]["w"]
          ).tag("query_heads", "projection", "embedding"),
      })
    else:
      parameter_mapping.update({
          f"block_{i}/attention/query.weights": pz.nx.NamedArray.wrap(
              params[f"layer_{i}/attn/qkv_einsum"]["w"][0]
          ).tag("heads", "embedding", "projection"),
          f"block_{i}/attention/key.weights": pz.nx.NamedArray.wrap(
              params[f"layer_{i}/attn/qkv_einsum"]["w"][1]
          ).tag("heads", "embedding", "projection"),
          f"block_{i}/attention/value.weights": pz.nx.NamedArray.wrap(
              params[f"layer_{i}/attn/qkv_einsum"]["w"][2]
          ).tag("heads", "embedding", "projection"),
          f"block_{i}/attention/output.weights": pz.nx.NamedArray.wrap(
              params[f"layer_{i}/attn/attn_vec_einsum"]["w"]
          ).tag("heads", "projection", "embedding"),
      })
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
