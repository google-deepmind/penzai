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
from penzai.nn import mixture_of_experts
from penzai.nn import parameters


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

  This config encompasses the parameters for the Llama, Mistral, and Gemma
  model families.

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
    rope_wavelength: Wavelength for global RoPE layers (and for local RoPE
      layers if local_rope_wavelength is not set).
    rms_norm_eps: Epsilon for RMSNorm layers.
    attention_type: A single attention type or sequence of per-layer attention
      types. If a sequence, its length should evenly divide the number of
      decoder blocks, and will be repeated to match the number of blocks.
    use_post_attn_norm: Whether to add a normalization layer after the attention
      block.
    use_post_ffw_norm: Whether to add a normalization layer after the
      feedforward block.
    final_logit_softcap: If not None, used as the tanh soft cap for the final
      transformer logits.
    attn_logits_soft_cap: If not None, used as the tanh soft cap for the
      attention logits.
    query_scaling_factor: Scaling factor for the query vectors. If "default",
      defaults to 1 / sqrt(projection_dim).
    parameter_dtype: Floating dtype to use for all parameters.
    activation_dtype: Floating dtype to use for activations and KV cache tables.
    use_layer_stack: Whether to stack the blocks together using a LayerStack.
    use_qk_norm: Whether to use QK normalization.
    use_value_norm: Whether to use value normalization (RMSNorm on value
      projections, used by Gemma 4).
    value_norm_with_scale: Whether the value normalization includes a learnable
      scale parameter. When False (Gemma 4), value normalization is a pure
      RMS standardization with no parameters. Only meaningful when
      ``use_value_norm`` is True.
    use_skip_scale: Whether to multiply each transformer block's output by a
      learnable scalar (used by Gemma 4).
    scale_plus_one_at_load: How RMSNorm scales are stored in the source
      checkpoint. If True (Gemma 1/2/3), the checkpoint stores
      ``actual_scale - 1`` and the loader adds 1 before binding. If False
      (Gemma 4), the checkpoint stores the actual scale value and is loaded
      as-is. The runtime ``RMSLayerNorm`` always applies ``x * scale``.
    global_projection_dim: Projection dimension for global attention layers.
      If None, uses the same projection_dim as local layers.
    global_num_kv_heads: Number of KV heads for global attention layers.
      If None, uses the same num_kv_heads as local layers.
    global_rope_proportion: Fraction of head dimensions that receive RoPE in
      global attention layers. If None, all dimensions receive RoPE (as for
      local layers). Used by Gemma 4 with partial_rotary_factor=0.25.
    global_scale_factor: Scale factor for the global RoPE layers (scale factor
      for the local RoPE layers is set as 1.0 by default).
    local_rope_wavelength: Wavelength for the local RoPE layers. If None, local
      RoPE layers will use the same wavelength as global RoPE layers
      (config.rope_wavelength).
    k_eq_v_global: Whether global attention layers share key and value
      projections (K=V). Used by Gemma 4 31B and 26B-A4B.
    per_layer_input_dim: Dimension of per-layer embeddings (PLE). If None,
      per-layer embeddings are not used. Used by Gemma 4 E2B and E4B.
    num_kv_shared_layers: Number of trailing transformer blocks that share
      key/value projections with earlier blocks. Shared layers reuse the
      K/V activations from earlier layers of the same attention type,
      cycling through the non-shared layers. Used by Gemma 4 E2B and E4B.
    num_experts: Number of experts for Mixture of Experts layers. If None,
      standard dense feedforward is used. Used by Gemma 4 26B-A4B.
    num_selected_experts: Number of experts selected per token (top-k).
      Required when ``num_experts`` is set.
    expert_hidden_dim: Hidden dimension for per-expert MLPs. Required when
      ``num_experts`` is set. The dense shared branch uses ``mlp_hidden_dim``.
  """

  num_kv_heads: int
  query_head_multiplier: int
  embedding_dim: int
  projection_dim: int
  mlp_hidden_dim: int
  num_decoder_blocks: int
  vocab_size: int
  mlp_variant: Literal["geglu_exact", "geglu_approx", "swiglu"]
  tie_embedder_and_logits: bool
  rope_wavelength: float = 10_000
  rms_norm_eps: float = 1e-6
  attention_type: AttentionType | Sequence[AttentionType] = (
      AttentionTypeGlobalCausal()
  )
  use_post_attn_norm: bool = False
  use_post_ffw_norm: bool = False
  final_logit_softcap: float | None = None
  attn_logits_soft_cap: float | None = None
  query_scaling_factor: float | Literal["default"] = "default"
  parameter_dtype: jax.typing.DTypeLike = jnp.float32
  activation_dtype: jax.typing.DTypeLike = jnp.float32
  use_layer_stack: bool = False
  use_qk_norm: bool = False
  use_value_norm: bool = False
  value_norm_with_scale: bool = True
  use_skip_scale: bool = False
  scale_plus_one_at_load: bool = True
  global_projection_dim: int | None = None
  global_num_kv_heads: int | None = None
  global_rope_proportion: float | None = None
  global_scale_factor: float | None = None
  local_rope_wavelength: float | None = None
  k_eq_v_global: bool = False
  per_layer_input_dim: int | None = None
  num_kv_shared_layers: int = 0
  num_experts: int | None = None
  num_selected_experts: int | None = None
  expert_hidden_dim: int | None = None


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
  if config.mlp_variant == "geglu_exact":
    act_fn = functools.partial(jax.nn.gelu, approximate=False)
  elif config.mlp_variant == "geglu_approx":
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


def build_moe_feedforward(
    name: str,
    init_base_rng: jax.Array | None,
    config: LlamalikeTransformerConfig,
) -> model_parts.TransformerMoEFeedForward:
  """Creates a dual-branch MoE + dense feedforward block.

  Builds the Gemma 4 MoE architecture: a dense shared MLP branch and a MoE
  branch are run in parallel, their outputs summed, then a final norm is
  applied.

  Args:
    name: Name of the feedforward block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model. Must have ``num_experts``,
      ``num_selected_experts``, and ``expert_hidden_dim`` set.

  Returns:
    An instance of TransformerMoEFeedForward.
  """
  assert config.num_experts is not None
  assert config.num_selected_experts is not None
  assert config.expert_hidden_dim is not None

  embedding_dim = config.embedding_dim
  num_experts = config.num_experts
  num_selected = config.num_selected_experts
  expert_dim = config.expert_hidden_dim

  # Build the MoE layer.
  moe_layer = mixture_of_experts.MixtureOfExperts(
      input_to_routing=mixture_of_experts.MoETopKRouter(
          router_norm=pz.nn.RMSStandardize(
              across="embedding", epsilon=config.rms_norm_eps
          ),
          router_scale=parameters.make_parameter(
              f"{name}/mlp/router.scale",
              init_base_rng,
              lambda rng: pz.nx.wrap(
                  jnp.ones(embedding_dim, dtype=config.parameter_dtype)
              ).tag("embedding"),
          ),
          router_logits=pz.nn.Linear.from_config(
              name=f"{name}/mlp/router_logits",
              init_base_rng=init_base_rng,
              input_axes={"embedding": embedding_dim},
              output_axes={"experts": num_experts},
              dtype=config.parameter_dtype,
          ),
          per_expert_scale=parameters.make_parameter(
              f"{name}/mlp/per_expert.scale",
              init_base_rng,
              lambda rng: pz.nx.wrap(
                  jnp.ones(num_experts, dtype=config.parameter_dtype)
              ).tag("experts"),
          ),
          num_selected_experts=num_selected,
          embedding_dim=embedding_dim,
      ),
      input_and_routing_to_output=mixture_of_experts.MoEGatedExpertComputation(
          gating_weights=parameters.make_parameter(
              f"{name}/mlp/gating.weights",
              init_base_rng,
              lambda rng: pz.nx.wrap(
                  jax.random.normal(
                      rng,
                      (num_experts, expert_dim, embedding_dim),
                      dtype=config.parameter_dtype,
                  )
                  * 0.01
              ).tag("experts", "neurons", "embedding"),
          ),
          value_weights=parameters.make_parameter(
              f"{name}/mlp/value.weights",
              init_base_rng,
              lambda rng: pz.nx.wrap(
                  jax.random.normal(
                      rng,
                      (num_experts, expert_dim, embedding_dim),
                      dtype=config.parameter_dtype,
                  )
                  * 0.01
              ).tag("experts", "neurons", "embedding"),
          ),
          out_weights=parameters.make_parameter(
              f"{name}/mlp/out.weights",
              init_base_rng,
              lambda rng: pz.nx.wrap(
                  jax.random.normal(
                      rng,
                      (num_experts, embedding_dim, expert_dim),
                      dtype=config.parameter_dtype,
                  )
                  * 0.01
              ).tag("experts", "embedding", "neurons"),
          ),
          num_selected_experts=num_selected,
      ),
  )

  # Dual-branch structure: dense + MoE, summed, then final norm.
  return model_parts.TransformerMoEFeedForward([
      pz.nn.BranchAndAddTogether(
          branches=[
              pz.nn.NamedGroup(
                  "dense_branch",
                  [
                      pz.nn.RMSLayerNorm.from_config(
                          name=f"{name}/pre_ffw_norm",
                          init_base_rng=init_base_rng,
                          across_axes={"embedding": embedding_dim},
                          dtype=config.parameter_dtype,
                          epsilon=config.rms_norm_eps,
                      ),
                      build_llamalike_feedforward(
                          f"{name}/mlp2", init_base_rng, config
                      ),
                      pz.nn.RMSLayerNorm.from_config(
                          name=f"{name}/post_ffw1_norm",
                          init_base_rng=init_base_rng,
                          across_axes={"embedding": embedding_dim},
                          dtype=config.parameter_dtype,
                          epsilon=config.rms_norm_eps,
                      ),
                  ],
              ),
              pz.nn.NamedGroup(
                  "moe_branch",
                  [
                      pz.nn.RMSLayerNorm.from_config(
                          name=f"{name}/pre_ffw2_norm",
                          init_base_rng=init_base_rng,
                          across_axes={"embedding": embedding_dim},
                          dtype=config.parameter_dtype,
                          epsilon=config.rms_norm_eps,
                      ),
                      moe_layer,
                      pz.nn.RMSLayerNorm.from_config(
                          name=f"{name}/post_ffw2_norm",
                          init_base_rng=init_base_rng,
                          across_axes={"embedding": embedding_dim},
                          dtype=config.parameter_dtype,
                          epsilon=config.rms_norm_eps,
                      ),
                  ],
              ),
          ]
      ),
      pz.nn.RMSLayerNorm.from_config(
          name=f"{name}/post_ffw_norm",
          init_base_rng=init_base_rng,
          across_axes={"embedding": embedding_dim},
          dtype=config.parameter_dtype,
          epsilon=config.rms_norm_eps,
      ),
  ])


def _head_info(num_kv_heads: int, query_head_multiplier: int):
  """Computes query, key, and value head axes and einsum names."""
  if query_head_multiplier == 1:
    common_head_axes = {"heads": num_kv_heads}
    qkv_einsum = {"heads": "h"}
    query_only_head_axes = {}
    q_einsum = {}
  elif num_kv_heads == 1:
    common_head_axes = {}
    qkv_einsum = {}
    query_only_head_axes = {"query_heads": query_head_multiplier}
    q_einsum = {"query_heads": "h"}
  else:
    common_head_axes = {"head_groups": num_kv_heads}
    qkv_einsum = {"head_groups": "hg"}
    query_only_head_axes = {"query_heads": query_head_multiplier}
    q_einsum = {"query_heads": "hq"}
  return (common_head_axes, qkv_einsum, query_only_head_axes, q_einsum)


# ---------------------------------------------------------------------------
# Activation sharing helpers (for KV cache sharing between layers)
# ---------------------------------------------------------------------------


@pz.pytree_dataclass
class _StoreActivation(pz.nn.Layer):
  """Runs an inner layer, stores its output in a variable, and returns it.

  Used to wrap K/V pipelines in source attention layers so that shared layers
  can later read the computed key/value activations.
  """

  inner: pz.nn.Layer
  store: pz.StateVariable

  def __call__(self, x, **side_inputs):
    result = self.inner(x, **side_inputs)
    self.store.value = result
    return result


@pz.pytree_dataclass
class _ReadStoredActivation(pz.nn.Layer):
  """Returns a previously stored activation, ignoring its input.

  Used in shared attention layers to read K/V activations computed by a source
  layer.

  Attributes:
    store: StateVariable holding the stored activation (shared with the
      corresponding ``_StoreActivation``).
    output_axes: Expected named axes of the output. Stored as metadata so that
      downstream code (e.g. sampling mode) can infer per-layer cache shapes
      without needing the inner Linear.
  """

  store: pz.StateVariable
  output_axes: dict[str, int] = dataclasses.field(
      default_factory=dict, metadata={"pytree_node": False}
  )

  def __call__(self, x, **side_inputs):
    return self.store.value


# ---------------------------------------------------------------------------
# Per-Layer Embedding (PLE) layers
# ---------------------------------------------------------------------------


@pz.pytree_dataclass
class PerLayerEmbeddingPrecompute(pz.nn.Layer):
  """Precomputes per-layer embedding vectors from initial embeddings and tokens.

  This layer sits after the embedding lookup in models with Per-Layer
  Embeddings (PLE), as used by Gemma 4 E2B and E4B. It computes combined
  PLE vectors for all layers and stores them in a ``StateVariable`` for later
  retrieval by ``PerLayerEmbeddingInject`` layers in each block.

  The computation follows the Gemma 4 PLE formula:

  1. **Token-identity**: look up token IDs in the per-layer embedding table,
     scaled by ``sqrt(per_layer_input_dim)``.
  2. **Context-aware**: project the initial embedding through a linear, scaled
     by ``1/sqrt(embedding_dim)``, then RMS-normalize.
  3. **Combine**: ``(context + tokens) / sqrt(2)``.

  Attributes:
    per_layer_table: Embedding table with named axes
      ``{"vocabulary": V, "layers": L, "ple_input": D}``.
    model_projection: Linear from ``{"embedding": E}`` to
      ``{"layers": L, "ple_input": D}``.
    projection_norm: RMSLayerNorm across ``{"ple_input": D}``.
    ple_store: StateVariable that receives the combined PLE vectors.
    num_layers: Number of transformer layers.
    per_layer_input_dim: Dimension of per-layer embeddings.
    embedding_dim: Dimension of the residual stream.
    token_ids_input_name: Side input key for token IDs.
  """

  per_layer_table: parameters.ParameterLike
  model_projection: pz.nn.Layer
  projection_norm: pz.nn.Layer
  ple_store: pz.StateVariable
  num_layers: int = dataclasses.field(metadata={"pytree_node": False})
  per_layer_input_dim: int = dataclasses.field(metadata={"pytree_node": False})
  embedding_dim: int = dataclasses.field(metadata={"pytree_node": False})
  token_ids_input_name: str = dataclasses.field(
      default="token_ids", metadata={"pytree_node": False}
  )

  def __call__(self, embedding, **side_inputs):
    token_ids = side_inputs[self.token_ids_input_name]

    # Token-identity PLE: look up per-layer embeddings by token ID.
    # per_layer_table.value has named axes {"vocabulary": V, "layers": L,
    # "ple_input": D}. NamedArray dictionary indexing handles the lookup.
    ple_tokens = self.per_layer_table.value[{"vocabulary": token_ids}]
    scale = jnp.sqrt(
        jnp.array(self.per_layer_input_dim, dtype=embedding.dtype)
    )
    ple_tokens = ple_tokens * scale

    # Context-aware PLE: project the initial embedding.
    ple_context = self.model_projection(embedding, **side_inputs)
    inv_sqrt_dim = jnp.array(
        1.0 / jnp.sqrt(float(self.embedding_dim)), dtype=embedding.dtype
    )
    ple_context = ple_context * inv_sqrt_dim
    ple_context = self.projection_norm(ple_context, **side_inputs)

    # Combine and store for PerLayerEmbeddingInject layers to read.
    inv_sqrt_2 = jnp.array(1.0 / jnp.sqrt(2.0), dtype=embedding.dtype)
    self.ple_store.value = (ple_context + ple_tokens) * inv_sqrt_2

    return embedding  # pass through unchanged


@pz.pytree_dataclass
class PerLayerEmbeddingInject(pz.nn.Layer):
  """Injects per-layer embedding information into the residual stream.

  Reads precomputed PLE vectors from a shared ``StateVariable``, gates them
  with the current hidden state, projects back to embedding dimension,
  normalizes, and adds to the residual stream.

  Attributes:
    ple_store: StateVariable holding precomputed PLE vectors (shared with
      ``PerLayerEmbeddingPrecompute``).
    layer_index: Which layer slice to read from the PLE vectors.
    gate: Linear from ``{"embedding": E}`` to ``{"ple_input": D}``.
    projection: Linear from ``{"ple_input": D}`` to ``{"embedding": E}``.
    post_norm: RMSLayerNorm across ``{"embedding": E}``.
  """

  ple_store: pz.StateVariable
  layer_index: int = dataclasses.field(metadata={"pytree_node": False})
  gate: pz.nn.Layer
  projection: pz.nn.Layer
  post_norm: pz.nn.Layer

  def __call__(self, hidden_state, **side_inputs):
    # Read this layer's PLE vector.
    ple = self.ple_store.value[{"layers": self.layer_index}]
    # Gate with the current hidden state, then project and normalize.
    gated = pz.nx.nmap(jax.nn.gelu)(self.gate(hidden_state, **side_inputs)) * ple
    projected = self.projection(gated, **side_inputs)
    normed = self.post_norm(projected, **side_inputs)
    return hidden_state + normed


def build_llamalike_attention(
    name: str,
    init_base_rng: jax.Array | None,
    config: LlamalikeTransformerConfig,
    block_index: int | None = None,
    kv_source_stores: tuple[pz.StateVariable, pz.StateVariable] | None = None,
    kv_shared_stores: tuple[pz.StateVariable, pz.StateVariable] | None = None,
) -> pz.nn.Attention:
  """Builds an attention block from a configuration.

  Args:
    name: Name of the attention block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.
    block_index: The index of the transformer block in the list of blocks. Can
      be None if the attention type doesn't depend on the block index.
    kv_source_stores: If this is a KV source layer (for KV cache sharing),
      a tuple of ``(key_store, value_store)`` StateVariables. The K/V
      activations will be stored here after computation, for later retrieval
      by shared layers.
    kv_shared_stores: If this is a KV shared layer, a tuple of
      ``(key_store, value_store)`` StateVariables pointing to the source
      layer's stores. K/V activations are read from here instead of being
      computed.

  Returns:
    An Attention block.
  """
  embedding_dim = config.embedding_dim

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

  # Resolve per-layer effective dimensions based on attention type.
  is_global = isinstance(attention_type, AttentionTypeGlobalCausal)
  if is_global and config.global_projection_dim is not None:
    projection_dim = config.global_projection_dim
  else:
    projection_dim = config.projection_dim

  if is_global and config.global_num_kv_heads is not None:
    effective_num_kv_heads = config.global_num_kv_heads
  else:
    effective_num_kv_heads = config.num_kv_heads

  total_query_heads = config.num_kv_heads * config.query_head_multiplier
  effective_query_head_multiplier = total_query_heads // effective_num_kv_heads

  common_head_axes, qkv_einsum, query_only_head_axes, q_einsum = _head_info(
      effective_num_kv_heads, effective_query_head_multiplier
  )

  if config.query_scaling_factor == "default":
    query_scaling_factor = projection_dim**-0.5
  else:
    query_scaling_factor = config.query_scaling_factor

  # Determine RoPE wavelength, scale factor, and masker for this layer type.
  if isinstance(attention_type, AttentionTypeSlidingWindowCausal):
    attn_masker = pz.nn.ApplyCausalSlidingWindowAttentionMask(
        sliding_window_size=attention_type.window_size,
        masked_out_value=masked_out_value,
    )
    if config.local_rope_wavelength is not None:
      wavelength = config.local_rope_wavelength
    else:
      wavelength = config.rope_wavelength
    scale_factor = 1.0
  elif isinstance(attention_type, AttentionTypeGlobalCausal):
    attn_masker = pz.nn.ApplyCausalAttentionMask(
        masked_out_value=masked_out_value,
    )
    wavelength = config.rope_wavelength
    if config.global_scale_factor is not None:
      scale_factor = config.global_scale_factor
    else:
      scale_factor = 1.0
  else:
    raise ValueError(f"Unsupported attention type {attention_type}")

  # Decide whether to use full or partial RoPE for this layer.
  # Gemma 4 partial RoPE follows a zero-padded-frequency convention (rotate
  # all split-half pairs but with timescale=inf for the last
  # ``(1-rope_proportion)*head_dim/2`` pairs, which yields identity). This is
  # different from the prefix-slice convention of ``ApplyRoPEToSubset``.
  layer_rope_proportion = (
      config.global_rope_proportion if is_global else None
  )
  if layer_rope_proportion is None:
    layer_rope_proportion = 1.0

  def _make_rope_layer():
    return pz.nn.ApplyRoPE(
        positions_input_name="token_positions",
        embedding_axis="projection",
        max_wavelength=wavelength,
        scale_factor=scale_factor,
        rope_proportion=layer_rope_proportion,
    )

  # Build query-key attention scoring sublayers.
  query_key_to_attn_sublayers = [
      pz.nn.NamedEinsum(
          (
              {"seq": "tq", **qkv_einsum, **q_einsum, "projection": "p"},
              {"seq": "tkv", **qkv_einsum, "projection": "p"},
          ),
          {"seq": "tq", **qkv_einsum, **q_einsum, "kv_seq": "tkv"},
      ),
  ]
  if config.attn_logits_soft_cap is not None:
    query_key_to_attn_sublayers.append(
        pz.nn.TanhSoftCap(
            soft_cap=jnp.array(
                config.attn_logits_soft_cap, dtype=config.activation_dtype
            )
        )
    )
  query_key_to_attn_sublayers.extend([
      attn_masker,
      pz.nn.Softmax("kv_seq"),
  ])

  # Build query sublayers.
  input_to_query_sublayers = [
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
  ]
  if config.use_qk_norm:
    input_to_query_sublayers.append(
        pz.nn.RMSLayerNorm.from_config(
            name=f"{name}/query_norm",
            init_base_rng=init_base_rng,
            across_axes={"projection": projection_dim},
            dtype=config.parameter_dtype,
            epsilon=config.rms_norm_eps,
        ),
    )
  input_to_query_sublayers.extend([
      _make_rope_layer(),
      pz.nn.ConstantRescale(
          by=jnp.array(query_scaling_factor, dtype=config.activation_dtype)
      ),
  ])

  # Build key and value sublayers.
  kv_output_axes = {**common_head_axes, "projection": projection_dim}

  if kv_shared_stores is not None:
    # KV shared layer: read K/V activations from a source layer's stores.
    input_to_key = _ReadStoredActivation(
        store=kv_shared_stores[0], output_axes=kv_output_axes
    )
    input_to_value = _ReadStoredActivation(
        store=kv_shared_stores[1], output_axes=kv_output_axes
    )
  else:
    # Build K/V projections normally.
    k_eq_v = is_global and config.k_eq_v_global

    kv_linear = pz.nn.Linear.from_config(
        name=f"{name}/key",
        init_base_rng=init_base_rng,
        input_axes={"embedding": embedding_dim},
        output_axes=kv_output_axes,
        dtype=config.parameter_dtype,
    )

    input_to_key_sublayers = [kv_linear]
    if config.use_qk_norm:
      input_to_key_sublayers.append(
          pz.nn.RMSLayerNorm.from_config(
              name=f"{name}/key_norm",
              init_base_rng=init_base_rng,
              across_axes={"projection": projection_dim},
              dtype=config.parameter_dtype,
              epsilon=config.rms_norm_eps,
          ),
      )
    input_to_key_sublayers.append(_make_rope_layer())

    if k_eq_v:
      # K=V sharing: reuse the same Linear for both key and value,
      # following the embedding-tying pattern.
      input_to_value_sublayers = [kv_linear]
    else:
      input_to_value_sublayers = [
          pz.nn.Linear.from_config(
              name=f"{name}/value",
              init_base_rng=init_base_rng,
              input_axes={"embedding": embedding_dim},
              output_axes=kv_output_axes,
              dtype=config.parameter_dtype,
          ),
      ]
    if config.use_value_norm:
      if config.value_norm_with_scale:
        input_to_value_sublayers.append(
            pz.nn.RMSLayerNorm.from_config(
                name=f"{name}/value_norm",
                init_base_rng=init_base_rng,
                across_axes={"projection": projection_dim},
                dtype=config.parameter_dtype,
                epsilon=config.rms_norm_eps,
            ),
        )
      else:
        input_to_value_sublayers.append(
            pz.nn.RMSStandardize(
                across=("projection",),
                epsilon=jnp.asarray(
                    config.rms_norm_eps, dtype=config.parameter_dtype
                ),
            ),
        )

    key_seq = pz.nn.Sequential(input_to_key_sublayers)
    value_seq = pz.nn.Sequential(input_to_value_sublayers)

    if kv_source_stores is not None:
      # KV source layer: wrap K/V pipelines to store activations.
      input_to_key = _StoreActivation(
          inner=key_seq, store=kv_source_stores[0]
      )
      input_to_value = _StoreActivation(
          inner=value_seq, store=kv_source_stores[1]
      )
    else:
      input_to_key = key_seq
      input_to_value = value_seq

  return pz.nn.Attention(
      input_to_query=pz.nn.Sequential(input_to_query_sublayers),
      input_to_key=input_to_key,
      input_to_value=input_to_value,
      query_key_to_attn=pz.nn.Sequential(query_key_to_attn_sublayers),
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
    kv_source_stores: tuple[pz.StateVariable, pz.StateVariable] | None = None,
    kv_shared_stores: tuple[pz.StateVariable, pz.StateVariable] | None = None,
    ple_store: pz.StateVariable | None = None,
) -> model_parts.TransformerBlock:
  """Builds a transformer block from a configuration.

  Args:
    name: Name of the block.
    init_base_rng: Base RNG for initializing the parameters.
    config: The configuration of the model.
    block_index: The index of the transformer block in the list of blocks. Can
      be None if the attention type doesn't depend on the block index.
    kv_source_stores: If this is a KV source layer, ``(key_store, value_store)``
      StateVariables to store K/V activations.
    kv_shared_stores: If this is a KV shared layer, ``(key_store, value_store)``
      StateVariables to read source K/V from.
    ple_store: StateVariable holding precomputed PLE vectors. If provided, a
      ``PerLayerEmbeddingInject`` layer is appended to this block.

  Returns:
    A full transformer block.
  """
  attn_sequence = [
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
          kv_source_stores=kv_source_stores,
          kv_shared_stores=kv_shared_stores,
      ),
  ]
  if config.use_post_attn_norm:
    attn_sequence.append(
        pz.nn.RMSLayerNorm.from_config(
            name=f"{name}/post_attention_norm",
            init_base_rng=init_base_rng,
            across_axes={"embedding": config.embedding_dim},
            dtype=config.parameter_dtype,
            epsilon=config.rms_norm_eps,
        )
    )
  if config.num_experts is not None:
    # MoE feedforward: dual-branch (dense + MoE) with its own norms.
    ffw_layer = build_moe_feedforward(name, init_base_rng, config)
  else:
    ffw_sequence = [
        pz.nn.RMSLayerNorm.from_config(
            name=f"{name}/pre_ffw_norm",
            init_base_rng=init_base_rng,
            across_axes={"embedding": config.embedding_dim},
            dtype=config.parameter_dtype,
            epsilon=config.rms_norm_eps,
        ),
        build_llamalike_feedforward(f"{name}/mlp", init_base_rng, config),
    ]
    if config.use_post_ffw_norm:
      ffw_sequence.append(
          pz.nn.RMSLayerNorm.from_config(
              name=f"{name}/post_ffw_norm",
              init_base_rng=init_base_rng,
              across_axes={"embedding": config.embedding_dim},
              dtype=config.parameter_dtype,
              epsilon=config.rms_norm_eps,
          )
      )
    ffw_layer = pz.nn.Sequential(ffw_sequence)
  attn_delta = pz.nn.Sequential(attn_sequence)
  block_sublayers = [
      pz.nn.Residual(attn_delta),
      pz.nn.Residual(ffw_layer),
  ]

  # Add per-layer embedding injection if enabled.
  if ple_store is not None and config.per_layer_input_dim is not None:
    block_sublayers.append(
        PerLayerEmbeddingInject(
            ple_store=ple_store,
            layer_index=block_index,
            gate=pz.nn.Linear.from_config(
                name=f"{name}/per_layer_input_gate",
                init_base_rng=init_base_rng,
                input_axes={"embedding": config.embedding_dim},
                output_axes={"ple_input": config.per_layer_input_dim},
                dtype=config.parameter_dtype,
            ),
            projection=pz.nn.Linear.from_config(
                name=f"{name}/per_layer_projection",
                init_base_rng=init_base_rng,
                input_axes={"ple_input": config.per_layer_input_dim},
                output_axes={"embedding": config.embedding_dim},
                dtype=config.parameter_dtype,
            ),
            post_norm=pz.nn.RMSLayerNorm.from_config(
                name=f"{name}/post_per_layer_input_norm",
                init_base_rng=init_base_rng,
                across_axes={"embedding": config.embedding_dim},
                dtype=config.parameter_dtype,
                epsilon=config.rms_norm_eps,
            ),
        )
    )

  # Trailing learnable scalar that scales the entire block output (Gemma 4).
  if config.use_skip_scale:
    block_sublayers.append(
        pz.nn.LearnableRescale(
            scale=parameters.make_parameter(
                f"{name}/skip.scale",
                init_base_rng,
                initializer=lambda _rng: pz.nx.wrap(jnp.array(1.0)),
            ),
        )
    )

  return model_parts.TransformerBlock(sublayers=block_sublayers)


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

  # Set up per-layer embeddings (PLE) if enabled.
  ple_store = None
  per_layer_input_dim = config.per_layer_input_dim
  if per_layer_input_dim is not None:
    ple_store = pz.StateVariable(value=None, label=f"{name}/ple_store")
    sublayers.append(
        PerLayerEmbeddingPrecompute(
            per_layer_table=parameters.make_parameter(
                f"{name}/embedder/per_layer_embeddings",
                init_base_rng,
                lambda rng: pz.nx.wrap(
                    jax.random.normal(
                        rng,
                        (
                            config.vocab_size,
                            config.num_decoder_blocks,
                            per_layer_input_dim,
                        ),
                        dtype=config.parameter_dtype,
                    )
                    * 0.01
                ).tag("vocabulary", "layers", "ple_input"),
            ),
            model_projection=pz.nn.Linear.from_config(
                name=f"{name}/embedder/per_layer_model_projection",
                init_base_rng=init_base_rng,
                input_axes={"embedding": config.embedding_dim},
                output_axes={
                    "layers": config.num_decoder_blocks,
                    "ple_input": per_layer_input_dim,
                },
                dtype=config.parameter_dtype,
            ),
            projection_norm=pz.nn.RMSLayerNorm.from_config(
                name=f"{name}/embedder/per_layer_projection_norm",
                init_base_rng=init_base_rng,
                across_axes={"ple_input": per_layer_input_dim},
                dtype=config.parameter_dtype,
                epsilon=config.rms_norm_eps,
            ),
            ple_store=ple_store,
            num_layers=config.num_decoder_blocks,
            per_layer_input_dim=per_layer_input_dim,
            embedding_dim=config.embedding_dim,
        )
    )

  # Set up KV cache sharing if enabled.
  kv_stores: dict[int, tuple[pz.StateVariable, pz.StateVariable]] = {}
  kv_source_map: dict[int, int] = {}
  if config.num_kv_shared_layers > 0:
    num_non_shared = config.num_decoder_blocks - config.num_kv_shared_layers
    # Resolve per-layer attention types.
    if isinstance(config.attention_type, AttentionType):
      attn_types = [config.attention_type] * config.num_decoder_blocks
    else:
      attn_types = [
          config.attention_type[i % len(config.attention_type)]
          for i in range(config.num_decoder_blocks)
      ]
    # Create KV stores for non-shared (source) layers.
    for i in range(num_non_shared):
      kv_stores[i] = (
          pz.StateVariable(value=None, label=f"{name}/kv_key_{i}"),
          pz.StateVariable(value=None, label=f"{name}/kv_value_{i}"),
      )
    # Map shared layers to the LAST non-shared layer of the same attention
    # type. This matches the Gemma 4 reference (gm/nn/gemma4/_config.py:
    # ``create_kv_cache_sharing_patterns``), where every shared global layer
    # reuses K/V from the last non-shared global layer, and likewise for
    # sliding layers.
    last_source_by_type: dict[str, int] = {}
    for i in range(num_non_shared):
      last_source_by_type[type(attn_types[i]).__name__] = i
    for j in range(num_non_shared, config.num_decoder_blocks):
      type_key = type(attn_types[j]).__name__
      if type_key not in last_source_by_type:
        raise ValueError(
            f"KV-shared layer {j} has attention type {type_key} but no"
            " non-shared source layer of that type exists."
        )
      kv_source_map[j] = last_source_by_type[type_key]

  if config.use_layer_stack:
    if not isinstance(config.attention_type, AttentionType):
      raise ValueError(
          "Layer stack does not currently support per-layer attention types."
      )
    if config.num_kv_shared_layers > 0:
      raise ValueError(
          "Layer stack does not currently support KV cache sharing."
      )
    sublayers.append(
        pz.nn.LayerStack.from_sublayer_builder(
            builder=build_llamalike_block,
            stack_axis="blocks",
            stack_axis_size=config.num_decoder_blocks,
            init_base_rng=init_base_rng,
            builder_kwargs=dict(
                name=f"{name}/blocks", config=config, ple_store=ple_store
            ),
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
      # Determine KV sharing role for this block.
      block_kv_source = None
      block_kv_shared = None
      if block_index in kv_stores:
        block_kv_source = kv_stores[block_index]
      elif block_index in kv_source_map:
        block_kv_shared = kv_stores[kv_source_map[block_index]]

      sublayers.append(
          build_llamalike_block(
              f"{name}/block_{block_index}",
              init_base_rng,
              config,
              block_index,
              kv_source_stores=block_kv_source,
              kv_shared_stores=block_kv_shared,
              ple_store=ple_store,
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

  if config.final_logit_softcap:
    sublayers.append(
        pz.nn.TanhSoftCap(
            soft_cap=jnp.array(
                config.final_logit_softcap, dtype=config.activation_dtype
            )
        )
    )

  common_head_axes, _, query_only_head_axes, _ = _head_info(
      config.num_kv_heads, config.query_head_multiplier
  )
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

  # Map HuggingFace hidden_act to Penzai mlp_variant
  hidden_act_to_mlp_variant = {
      "silu": "swiglu",
      "gelu": "geglu_exact",
      "gelu_new": "geglu_approx",
  }
  mlp_variant = hidden_act_to_mlp_variant[hf_config.hidden_act]

  pz_config = LlamalikeTransformerConfig(
      num_kv_heads=num_kv_heads,
      query_head_multiplier=query_head_multiplier,
      embedding_dim=hf_config.hidden_size,
      projection_dim=hf_config.hidden_size // hf_config.num_attention_heads,
      mlp_hidden_dim=hf_config.intermediate_size,
      num_decoder_blocks=hf_config.num_hidden_layers,
      vocab_size=hf_config.vocab_size,
      mlp_variant=mlp_variant,
      rope_wavelength=hf_config.rope_theta,
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
