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

"""A common transformer family used by Llama, Mistral, Gemma, and other models.

This module implements a transformer variant with:

- GLU-based MLPs (SwiGLU or GeGLU) as introduced by Shazeer (2020),
- Optional multi-query (Shazeer, 2019) or grouped-query (Ainslie et al. 2023)
  attention,
- Rotary positional embeddings (Su et al., 2021),
- RMSNorm normalization (Zhang & Sennrich, 2019),
- No biases in any dense kernels or layer norms.

This family includes many popular open-weights models, including Llama, Mistral,
Gemma, and Reka. It is also similar to the PaLM model architecture (but without
"parallel" feedforward and attention blocks).
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, Literal

import jax
import jax.numpy as jnp
from penzai import pz
from penzai.models.transformer import model_parts


@dataclasses.dataclass(frozen=True)
class AttentionTypeGlobalCausal:
  """Marker for a global attention block."""


@dataclasses.dataclass(frozen=True)
class AttentionTypeSlidingWindowCausal:
  """Marker for a local sliding-window attention block.

  Attributes:
    window_size: Size of the sliding window.
  """

  window_size: int


AttentionType = AttentionTypeGlobalCausal | AttentionTypeSlidingWindowCausal


@dataclasses.dataclass(kw_only=True)
class LlamalikeTransformerConfig:
  """Common configuration parameters for a "llama-like" transformer.

  These are held in a single configuration object to simplify argument passing
  during construction of the model.

  Attributes:
    num_kv_heads: The number of key-value attention heads or head groups.
    query_head_multiplier: The number of query heads for each KV head.
    embedding_dim: Dimension of the embedding vectors and residual stream.
    projection_dim: Dimension of the query, key, and value projections. Usually
      ``embedding_dim // num_heads``.
    mlp_hidden_dim: Dimensionality of the hidden layer of the MLP blocks in each
      layer (the "neurons" axis).
    num_decoder_blocks: Number of transformer decoder blocks in the model.
    vocab_size: Number of tokens in the vocabulary.
    mlp_variant: Gated linear unit variant for MLPs.
    tie_embedder_and_logits: Whether to tie the weights of the input token
      embedding and output logit layers. If True, also scales down input token
      embeddings by sqrt(embedding_dim). (This is used by Gemma.)
    rope_wavelength: Wavelength for RoPE layers.
    rms_norm_eps: Epsilon for RMSNorm layers.
    attention_type: A single attention type or sequence of per-layer attention
      types. If a sequence, its length should evenly divide the number of
      decoder blocks, and will be repeated to match the number of blocks.
    parameter_dtype: Floating dtype to use for all parameters.
    activation_dtype: Floating dtype to use for activations and KV cache tables.
    use_layer_stack: Whether to stack the blocks together using a LayerStack.
  """

  num_kv_heads: int
  query_head_multiplier: int
  embedding_dim: int
  projection_dim: int
  mlp_hidden_dim: int
  num_decoder_blocks: int
  vocab_size: int
  mlp_variant: Literal["geglu_approx", "swiglu"]
  tie_embedder_and_logits: bool
  rope_wavelength: float = 10_000
  rms_norm_eps: float = 1e-6
  attention_type: AttentionType | Sequence[AttentionType] = (
      AttentionTypeGlobalCausal()
  )
  parameter_dtype: jax.typing.DTypeLike = jnp.float32
  activation_dtype: jax.typing.DTypeLike = jnp.float32
  use_layer_stack: bool = False


def build_llamalike_feedforward(
    name: str,
    init_base_rng: jax.Array | None,
    config: LlamalikeTransformerConfig,
) -> model_parts.TransformerFeedForward:
  """Creates a feedforward block.

  This family of models use gated linear units, as proposed by Shazeer (2020).
  We represent this computation as a composition of simpler Penzai primitives,
  to enable patching and post-processing of the various internal activations.

  Args:
    name: Name of the feedforward block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.

  Returns:
    An instance of TransformerFeedForward containing the GELU MLP blocks.
  """
  if config.mlp_variant == "geglu_approx":
    # Approximate is already the default in JAX, but we specify it explicitly
    # because defaults differ between JAX and PyTorch.
    act_fn = functools.partial(jax.nn.gelu, approximate=True)
  elif config.mlp_variant == "swiglu":
    act_fn = jax.nn.silu
  else:
    raise ValueError(f"Unsupported MLP variant {config.mlp_variant}")

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
                      pz.nn.Elementwise(act_fn),
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


def _head_info(config: LlamalikeTransformerConfig):
  """Computes query, key, and value head axes and einsum names."""
  if config.query_head_multiplier == 1:
    common_head_axes = {"heads": config.num_kv_heads}
    qkv_einsum = {"heads": "h"}
    query_only_head_axes = {}
    q_einsum = {}
  elif config.num_kv_heads == 1:
    common_head_axes = {}
    qkv_einsum = {}
    query_only_head_axes = {"query_heads": config.query_head_multiplier}
    q_einsum = {"query_heads": "h"}
  else:
    common_head_axes = {"head_groups": config.num_kv_heads}
    qkv_einsum = {"head_groups": "hg"}
    query_only_head_axes = {"query_heads": config.query_head_multiplier}
    q_einsum = {"query_heads": "hq"}
  return (common_head_axes, qkv_einsum, query_only_head_axes, q_einsum)


def build_llamalike_attention(
    name: str,
    init_base_rng: jax.Array | None,
    config: LlamalikeTransformerConfig,
    block_index: int | None = None,
) -> pz.nn.Attention:
  """Builds an attention block from a configuration.

  Args:
    name: Name of the attention block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.
    block_index: The index of the transformer block in the list of blocks. Can
      be None if the attention type doesn't depend on the block index.

  Returns:
    An Attention block.
  """
  embedding_dim = config.embedding_dim
  projection_dim = config.projection_dim

  common_head_axes, qkv_einsum, query_only_head_axes, q_einsum = _head_info(
      config
  )

  # As used in https://github.com/google-deepmind/gemma.
  # (This exact value is probably not important.)
  masked_out_value = jnp.array(-2.3819763e38, dtype=config.activation_dtype)

  if isinstance(config.attention_type, AttentionType):
    attention_type = config.attention_type
  else:
    if block_index is None:
      raise ValueError(
          "block_index must be specified if attention_type is a sequence."
      )
    attention_type = config.attention_type[
        block_index % len(config.attention_type)
    ]

  if isinstance(attention_type, AttentionTypeSlidingWindowCausal):
    attn_masker = pz.nn.ApplyCausalSlidingWindowAttentionMask(
        sliding_window_size=attention_type.window_size,
        masked_out_value=masked_out_value,
    )
  elif isinstance(attention_type, AttentionTypeGlobalCausal):
    attn_masker = pz.nn.ApplyCausalAttentionMask(
        masked_out_value=masked_out_value,
    )
  else:
    raise ValueError(f"Unsupported attention type {attention_type}")

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
              max_wavelength=config.rope_wavelength,
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
              max_wavelength=config.rope_wavelength,
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
                  {"seq": "tq", **qkv_einsum, **q_einsum, "projection": "p"},
                  {"seq": "tkv", **qkv_einsum, "projection": "p"},
              ),
              {"seq": "tq", **qkv_einsum, **q_einsum, "kv_seq": "tkv"},
          ),
          attn_masker,
          pz.nn.Softmax("kv_seq"),
      ]),
      attn_value_to_output=pz.nn.Sequential([
          pz.nn.NamedEinsum(
              (
                  {"seq": "tq", **qkv_einsum, **q_einsum, "kv_seq": "tkv"},
                  {"seq": "tkv", **qkv_einsum, "projection": "p"},
              ),
              {"seq": "tq", **qkv_einsum, **q_einsum, "projection": "p"},
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


def build_llamalike_block(
    name: str,
    init_base_rng: jax.Array | None,
    config: LlamalikeTransformerConfig,
    block_index: int | None = None,
) -> model_parts.TransformerBlock:
  """Builds a transformer block from a configuration.

  Args:
    name: Name of the block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.
    block_index: The index of the transformer block in the list of blocks. Can
      be None if the attention type doesn't depend on the block index.

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
                      epsilon=config.rms_norm_eps,
                  ),
                  build_llamalike_attention(
                      f"{name}/attention",
                      init_base_rng,
                      config,
                      block_index=block_index,
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
                      epsilon=config.rms_norm_eps,
                  ),
                  build_llamalike_feedforward(
                      f"{name}/mlp", init_base_rng, config
                  ),
              ])
          ),
      ],
  )


def build_llamalike_transformer(
    config: LlamalikeTransformerConfig,
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

  if config.tie_embedder_and_logits:
    sublayers.append(
        pz.nn.ConstantRescale(
            by=jnp.sqrt(config.embedding_dim).astype(config.activation_dtype)
        )
    )

  if config.use_layer_stack:
    if not isinstance(config.attention_type, AttentionType):
      raise ValueError(
          "Layer stack does not currently support per-layer attention types."
      )
    sublayers.append(
        pz.nn.LayerStack.from_sublayer_builder(
            builder=build_llamalike_block,
            stack_axis="blocks",
            stack_axis_size=config.num_decoder_blocks,
            init_base_rng=init_base_rng,
            builder_kwargs=dict(name=f"{name}/blocks", config=config),
        )
    )
  else:
    if not isinstance(config.attention_type, AttentionType):
      if config.num_decoder_blocks % len(config.attention_type) != 0:
        raise ValueError(
            "Per-layer attention types must have a length that divides the"
            " number of blocks."
        )
    for block_index in range(config.num_decoder_blocks):
      sublayers.append(
          build_llamalike_block(
              f"{name}/block_{block_index}", init_base_rng, config, block_index
          )
      )

  sublayers.append(
      pz.nn.RMSLayerNorm.from_config(
          name=f"{name}/final_norm",
          init_base_rng=init_base_rng,
          across_axes={"embedding": config.embedding_dim},
          dtype=config.parameter_dtype,
          epsilon=config.rms_norm_eps,
      )
  )

  if config.tie_embedder_and_logits:
    sublayers.append(pz.nn.EmbeddingDecode(emb_table))
  else:
    sublayers.append(
        pz.nn.Linear.from_config(
            name=f"{name}/lm_head",
            init_base_rng=init_base_rng,
            input_axes={"embedding": config.embedding_dim},
            output_axes={"vocabulary": config.vocab_size},
        )
    )

  common_head_axes, _, query_only_head_axes, _ = _head_info(config)
  return model_parts.TransformerLM(
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


def llamalike_from_huggingface_model(
    model: Any,
    upcast_activations_to_float32: bool = False,
    use_layer_stack: bool = False,
) -> model_parts.TransformerLM:
  """Converts a "llama-like" HuggingFace model to a Penzai model.

  This function converts Llama-like models from their HuggingFace
  implementations to Penzai. It does not do any checks and blindly assumes
  that the architecture follows the defaults from the Llama model family.
  You may want to use the model-specific wrappers in `variants.llama` or
  `variants.mistral` instead.

  Args:
    model: The HuggingFace model, which is assumed to be similar to the Llama or
      Mistral architectures. (Not all configuration arguments are checked, so
      this may end up producing different behavior if given an incompatible
      configuration.)
    upcast_activations_to_float32: Whether to cast activations to float32 when
      the model runs. This allows analyzing activations at higher precision
      without consuming additional memory for parameters.
    use_layer_stack: Whether to use a layer stack for the decoder blocks.

  Returns:
    A Transformer model containing the loaded parameters, assuming a Llama-like
    architecture.
  """
  hf_config = model.config
  num_kv_heads = hf_config.num_key_value_heads
  query_head_multiplier = hf_config.num_attention_heads // num_kv_heads
  assert num_kv_heads * query_head_multiplier == hf_config.num_attention_heads

  sliding_window_size = getattr(hf_config, "sliding_window", None)
  if sliding_window_size is None:
    attention_type = AttentionTypeGlobalCausal()
  else:
    attention_type = AttentionTypeSlidingWindowCausal(sliding_window_size)

  param_dtype = {
      "torch.float32": jnp.float32,
      "torch.bfloat16": jnp.bfloat16,
  }[str(model.dtype)]
  if upcast_activations_to_float32:
    activation_dtype = jnp.float32
  else:
    activation_dtype = param_dtype

  pz_config = LlamalikeTransformerConfig(
      num_kv_heads=num_kv_heads,
      query_head_multiplier=query_head_multiplier,
      embedding_dim=hf_config.hidden_size,
      projection_dim=hf_config.hidden_size // hf_config.num_attention_heads,
      mlp_hidden_dim=hf_config.intermediate_size,
      num_decoder_blocks=hf_config.num_hidden_layers,
      vocab_size=hf_config.vocab_size,
      mlp_variant="swiglu",
      rope_wavelength=10_000,
      tie_embedder_and_logits=False,
      attention_type=attention_type,
      rms_norm_eps=hf_config.rms_norm_eps,
      parameter_dtype=param_dtype,
      activation_dtype=activation_dtype,
      use_layer_stack=use_layer_stack,
  )
  model_def = build_llamalike_transformer(
      pz_config, init_base_rng=None, name="transformer"
  )

  state_dict = model.state_dict()
  converted = {k: jax.dlpack.from_dlpack(v) for k, v in state_dict.items()}

  parameter_mapping = {
      "embedder.embeddings": pz.nx.NamedArray.wrap(
          converted["model.embed_tokens.weight"]
      ).tag("vocabulary", "embedding"),
      "final_norm/scale.weights": pz.nx.NamedArray.wrap(
          converted["model.norm.weight"]
      ).tag("embedding"),
      "lm_head.weights": pz.nx.NamedArray.wrap(converted["lm_head.weight"]).tag(
          "vocabulary", "embedding"
      ),
  }

  def fix_qkvo(which, arr):
    arr = pz.nx.wrap(arr)
    if which == "q":
      if pz_config.query_head_multiplier == 1:
        return arr.reshape((
            pz_config.num_kv_heads,
            pz_config.projection_dim,
            pz_config.embedding_dim,
        )).tag("heads", "projection", "embedding")
      elif pz_config.num_kv_heads == 1:
        return arr.reshape((
            pz_config.query_head_multiplier,
            pz_config.projection_dim,
            pz_config.embedding_dim,
        )).tag("query_heads", "projection", "embedding")
      else:
        return arr.reshape((
            pz_config.num_kv_heads,
            pz_config.query_head_multiplier,
            pz_config.projection_dim,
            pz_config.embedding_dim,
        )).tag("head_groups", "query_heads", "projection", "embedding")
    elif which == "k" or which == "v":
      if pz_config.query_head_multiplier == 1:
        return arr.reshape((
            pz_config.num_kv_heads,
            pz_config.projection_dim,
            pz_config.embedding_dim,
        )).tag("heads", "projection", "embedding")
      elif pz_config.num_kv_heads == 1:
        return arr.reshape((
            pz_config.projection_dim,
            pz_config.embedding_dim,
        )).tag("projection", "embedding")
      else:
        return arr.reshape((
            pz_config.num_kv_heads,
            pz_config.projection_dim,
            pz_config.embedding_dim,
        )).tag("head_groups", "projection", "embedding")
    elif which == "o":
      if pz_config.query_head_multiplier == 1:
        return arr.reshape((
            pz_config.embedding_dim,
            pz_config.num_kv_heads,
            pz_config.projection_dim,
        )).tag("embedding", "heads", "projection")
      elif pz_config.num_kv_heads == 1:
        return arr.reshape((
            pz_config.embedding_dim,
            pz_config.query_head_multiplier,
            pz_config.projection_dim,
        )).tag("embedding", "query_heads", "projection")
      else:
        return arr.reshape((
            pz_config.embedding_dim,
            pz_config.num_kv_heads,
            pz_config.query_head_multiplier,
            pz_config.projection_dim,
        )).tag("embedding", "head_groups", "query_heads", "projection")
    else:
      raise NotImplementedError(which)

  all_block_params = []

  for i in range(pz_config.num_decoder_blocks):
    cur_block_params = {}
    all_block_params.append(cur_block_params)

    cur_block_params["pre_attention_norm/scale.weights"] = (
        pz.nx.NamedArray.wrap(
            converted[f"model.layers.{i}.input_layernorm.weight"]
        ).tag("embedding")
    )
    cur_block_params["pre_ffw_norm/scale.weights"] = pz.nx.NamedArray.wrap(
        converted[f"model.layers.{i}.post_attention_layernorm.weight"]
    ).tag("embedding")
    cur_block_params["mlp/gating_linear.weights"] = pz.nx.NamedArray.wrap(
        converted[f"model.layers.{i}.mlp.gate_proj.weight"]
    ).tag("neurons", "embedding")
    cur_block_params["mlp/value_linear.weights"] = pz.nx.NamedArray.wrap(
        converted[f"model.layers.{i}.mlp.up_proj.weight"]
    ).tag("neurons", "embedding")
    cur_block_params["mlp/out_linear.weights"] = pz.nx.NamedArray.wrap(
        converted[f"model.layers.{i}.mlp.down_proj.weight"]
    ).tag("embedding", "neurons")

    cur_block_params["attention/query.weights"] = fix_qkvo(
        "q", converted[f"model.layers.{i}.self_attn.q_proj.weight"]
    )
    cur_block_params["attention/key.weights"] = fix_qkvo(
        "k", converted[f"model.layers.{i}.self_attn.k_proj.weight"]
    )
    cur_block_params["attention/value.weights"] = fix_qkvo(
        "v", converted[f"model.layers.{i}.self_attn.v_proj.weight"]
    )
    cur_block_params["attention/output.weights"] = fix_qkvo(
        "o", converted[f"model.layers.{i}.self_attn.o_proj.weight"]
    )

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
