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

"""Core layers for the Gemma model architecture.

See the Gemma technical report at
https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf
and the accompanying reference implementation at
https://github.com/google-deepmind/gemma.

All of the layers and models in this file use the following axis naming
convention:

* "seq" is the temporal axis of the token sequence, i.e. the axis along which
  the prompt tokens are laid out. In an attention matrix, it specifically
  refers to the query token(s) (the ones we are currently processing).

* "embedding" is the axis for embedding vectors and the residual stream.

* "projection" is the axis for query, key, and value head projection vectors,
  i.e. the axis where query-key dot products happen, and for which the values of
  attention heads are retrieved.

* "heads" is the axis that ranges across the different attention heads. Note
  that depending on the configuration, the key and value computations may not
  have this axis, because they are shared across heads.

* "kv_seq" is a temporary copy of the "seq" axis that represents the position
  of the keys and values in an attention matrix.

* "neurons" is the axis for the neurons in the MLP blocks, which have an
  activation function (GEGLU) applied elementwise and therefore have a
  priveleged basis.

Additionally, they use the following effect tags:

* "token_positions" is the name of the side input that provides the position of
  each token for the purposes of positional embeddings.

* "attn_mask" is the name of the side input that provides the attention mask
  for each attention layer.

* Where applicable, "cache_end_index" is the name of the side input that
  identifies the current length of the key/value cache state. This determines
  where the new keys and values are inserted into the cache.

* Where applicable, "kv_cache" is the name of the local state category that
  contains all key/value caches.

Note that the top-level `GemmaTransformer` and `GemmaKVCachingTransformer`
classes will handle these effects for you in most cases, so this is most
relevant if you plan to initialize parts of the transformer without using these
top-level classes.

The KV caching logic is defined in the separate module
`penzai.deprecated.v1.example_models.gemma.sampling_mode`.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import jax
import jax.numpy as jnp
from penzai.deprecated.v1 import pz


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


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class GemmaFeedForward(pz.nn.Sequential):
  """Implementation of the feed-forward block in Gemma."""

  @classmethod
  def from_config(
      cls,
      embedding_dim: int,
      hidden_dim: int,
      dtype: jax.typing.DTypeLike = jnp.float32,
  ) -> GemmaFeedForward:
    """Constructs an uninitialized Gemma feed-forward layer.

    Gemma's feedforward layer uses GELU-based gated linear units (GEGLU), as
    proposed by Shazeer (2020). We represent this computation as a composition
    of simpler Penzai primitives, to enable patching and post-processing of the
    various internal activations.

    We assume that the input embedding axis is called "embedding", and the
    neurons axis is called "neurons". Other axes will be treated as batch
    dimensions and vectorized over.

    Args:
      embedding_dim: The dimensionality of the input and output embeddings.
      hidden_dim: The dimensionality of the hidden layer.
      dtype: The data type to use for the parameters.

    Returns:
      An instance of GemmaFeedForward containing uninitialized parameters of
      the appropriate shapes and dtypes.
    """
    return cls([
        pz.nn.BranchAndMultiplyTogether(
            branches=[
                pz.nn.NamedGroup(
                    "gate",
                    [
                        pz.nn.add_parameter_prefix(
                            "gating_linear",
                            pz.nn.Linear.from_config(
                                input_axes={"embedding": embedding_dim},
                                output_axes={"neurons": hidden_dim},
                                dtype=dtype,
                            ),
                        ),
                        pz.nn.Elementwise(jax.nn.gelu),
                    ],
                ),
                pz.nn.add_parameter_prefix(
                    "value_linear",
                    pz.nn.Linear.from_config(
                        input_axes={"embedding": embedding_dim},
                        output_axes={"neurons": hidden_dim},
                        dtype=dtype,
                    ),
                ),
            ]
        ),
        pz.nn.add_parameter_prefix(
            "out_linear",
            pz.nn.Linear.from_config(
                input_axes={"neurons": hidden_dim},
                output_axes={"embedding": embedding_dim},
                dtype=dtype,
            ),
        ),
    ])


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class GemmaAttention(pz.nn.Attention):
  """Gemma-specific configuration of the self-attention layer.

  ``GemmaAttention`` has the same runtime behavior as the base `pz.nn.Attention`
  combinator, but adds a classmethod that constructs the layer according to
  the Gemma architecture.
  """

  @classmethod
  def from_config(cls, config: GemmaTransformerConfig) -> GemmaAttention:
    """Builds a GemmaAttention block from a configuration.

    Args:
      config: The configuration of the Gemma model.

    Returns:
      A GemmaAttention block with uninitialized parameters.
    """
    num_heads = config.num_heads
    embedding_dim = config.embedding_dim
    projection_dim = config.projection_dim
    single_kv_head = config.single_kv_head

    if single_kv_head:
      kv_output_axes = {"projection": projection_dim}
      kv_einsum_heads = {}
    else:
      kv_output_axes = {"heads": num_heads, "projection": projection_dim}
      kv_einsum_heads = {"heads": "h"}

    return cls(
        input_to_query=pz.nn.Sequential([
            pz.nn.add_parameter_prefix(
                "query",
                pz.nn.Linear.from_config(
                    input_axes={"embedding": embedding_dim},
                    output_axes={
                        "heads": num_heads,
                        "projection": projection_dim,
                    },
                    dtype=config.parameter_dtype,
                ),
            ),
            pz.nn.ApplyRoPE.from_config(
                positions_tag="token_positions",
                embedding_axis="projection",
            ),
            pz.nn.ConstantRescale(
                by=jnp.array(
                    projection_dim**-0.5, dtype=config.activation_dtype
                )
            ),
        ]),
        input_to_key=pz.nn.Sequential([
            pz.nn.add_parameter_prefix(
                "key",
                pz.nn.Linear.from_config(
                    input_axes={"embedding": embedding_dim},
                    output_axes=kv_output_axes,
                    dtype=config.parameter_dtype,
                ),
            ),
            pz.nn.ApplyRoPE.from_config(
                positions_tag="token_positions",
                embedding_axis="projection",
            ),
        ]),
        input_to_value=pz.nn.Sequential([
            pz.nn.add_parameter_prefix(
                "value",
                pz.nn.Linear.from_config(
                    input_axes={"embedding": embedding_dim},
                    output_axes=kv_output_axes,
                    dtype=config.parameter_dtype,
                ),
            ),
        ]),
        query_key_to_attn=pz.nn.Sequential([
            pz.nn.NamedEinsum(
                (
                    {"seq": "tq", "heads": "h", "projection": "p"},
                    {"seq": "tkv", **kv_einsum_heads, "projection": "p"},
                ),
                {"seq": "tq", "heads": "h", "kv_seq": "tkv"},
            ),
            pz.nn.ApplyAttentionMask.from_config(
                mask_tag="attn_mask",
                masked_out_value=jnp.array(
                    -2.3819763e38, dtype=config.activation_dtype
                ),
            ),
            pz.nn.Softmax("kv_seq"),
        ]),
        attn_value_to_output=pz.nn.Sequential([
            pz.nn.NamedEinsum(
                (
                    {"seq": "tq", "heads": "h", "kv_seq": "tkv"},
                    {"seq": "tkv", **kv_einsum_heads, "projection": "p"},
                ),
                {"seq": "tq", "heads": "h", "projection": "p"},
            ),
            pz.nn.add_parameter_prefix(
                "output",
                pz.nn.Linear.from_config(
                    input_axes={
                        "heads": num_heads,
                        "projection": projection_dim,
                    },
                    output_axes={"embedding": embedding_dim},
                    dtype=config.parameter_dtype,
                ),
            ),
        ]),
    )


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class GemmaTransformerBlock(pz.nn.Sequential):
  """Main decoder block for the Gemma transformer architecture.

  ``GemmaTransformerBlock`` is a tagged alias of `pz.nn.Sequential`, which means
  it just runs its sublayers in sequence. However, it has its own type to make
  it easier to identify with selectors, and also can be constructed from a
  `GemmaTransformerConfig`.
  """

  @classmethod
  def from_config(cls, config: GemmaTransformerConfig) -> GemmaTransformerBlock:
    """Builds a ``GemmaTransformerBlock`` from a configuration.

    Args:
      config: The configuration of the Gemma model.

    Returns:
      A ``GemmaTransformerBlock`` with uninitialized parameters.
    """
    return cls(
        sublayers=[
            pz.nn.Residual(
                pz.nn.Sequential([
                    pz.nn.add_parameter_prefix(
                        "pre_attention_norm",
                        pz.nn.RMSLayerNorm.from_config(
                            {"embedding": config.embedding_dim},
                            dtype=config.parameter_dtype,
                        ),
                    ),
                    pz.nn.add_parameter_prefix(
                        "attn", GemmaAttention.from_config(config)
                    ),
                ])
            ),
            pz.nn.Residual(
                pz.nn.Sequential([
                    pz.nn.add_parameter_prefix(
                        "pre_ffw_norm",
                        pz.nn.RMSLayerNorm.from_config(
                            {"embedding": config.embedding_dim},
                            dtype=config.parameter_dtype,
                        ),
                    ),
                    pz.nn.add_parameter_prefix(
                        "mlp",
                        GemmaFeedForward.from_config(
                            embedding_dim=config.embedding_dim,
                            hidden_dim=config.mlp_hidden_dim,
                            dtype=config.parameter_dtype,
                        ),
                    ),
                ])
            ),
        ],
    )


@pz.pytree_dataclass
class GemmaInputs(pz.Struct):
  """Input structure for `GemmaTransformer`.

  Attributes:
    tokens: Sequence of tokens, as an integer named array with a "seq" axis and
      possibly batch axes. Usually starts with the beginning-of-sequence token.
    positions: Sequence of token positions, as an integer named array with a
      "seq" axis and possibly batch axes. Usually starts from 0 and increments
      along the "seq" axis, but can be different to support e.g. example
      packing.
    attention_mask: Boolean attention mask with "seq" and "kv_seq" axes of the
      same length, and possibly batch axes. Usually a causal mask, but can be
      different to support e.g. example packing or dropping out inputs.
  """

  tokens: pz.nx.NamedArray
  positions: pz.nx.NamedArray
  attention_mask: pz.nx.NamedArray

  @classmethod
  def from_basic_segments(cls, tokens: pz.nx.NamedArray) -> GemmaInputs:
    """Constructs a simple input structure for a batch of single segments.

    This can be used to process inputs that do not need advanced position or
    attention mask handling, and which just consist of ordinary sequences that
    are not packed together or padded. It augments the tokens with a standard
    position array and causal attention mask, as expected by the Gemma model.

    Args:
      tokens: Sequence of tokens, as an integer named array with a "seq" axis
        and possibly batch axes, which starts with the beginning-of-sequence
        token. Each 1d vector along the "seq" axis should represent an unpadded
        sequence.

    Returns:
      A full input structure containing the provided tokens, along with a simple
      incrementing position array and a causal mask.
    """
    seq = tokens.named_shape["seq"]
    # Query tokens can attend to keys/values if the query position is larger.
    attention_mask = pz.nx.arange("seq", seq) >= pz.nx.arange("kv_seq", seq)
    return cls(
        tokens=tokens,
        positions=pz.nx.arange("seq", seq),
        attention_mask=attention_mask,
    )


@pz.pytree_dataclass
class GemmaTransformer(pz.Layer):
  """Top-level Gemma transformer decoder, encapsulating all internal effects.

  This class represents the full Gemma model, and can be loaded from the
  official Gemma checkpoints.

  Attributes:
    config: The configuration for the transformer. Although not directly used
      when the model is called, this can be useful for re-building the model or
      converting it to autoregressive sampling mode.
    body: The implementation of the transformer. Usually a side-input effect
      handler wrapping the main sequence of transformer blocks, but may be
      modified after the model is loaded due to patching.
  """

  config: GemmaTransformerConfig = dataclasses.field(
      metadata={"pytree_node": False}
  )
  body: pz.LayerLike

  @pz.checked_layer_call
  def __call__(self, inputs: GemmaInputs) -> pz.nx.NamedArray:
    """Scores log-probabilities for the given inputs.

    Args:
      inputs: Structure of input arguments, containing tokens, segment
        positions, and an attention mask.

    Returns:
      The final matrix of logits from the embedding decoding layer, which
      (in the normal configuration) will have axes "seq" and "vocabulary".
    """
    return self.body((inputs.tokens, inputs.positions, inputs.attention_mask))

  def input_structure(self) -> pz.chk.StructureAnnotation:
    return GemmaInputs(
        tokens=pz.chk.Wildcard("tokens"),
        positions=pz.chk.Wildcard("positions"),
        attention_mask=pz.chk.Wildcard("attention mask"),
    )

  def output_structure(self) -> pz.chk.StructureAnnotation:
    return pz.chk.Wildcard("unnormalized logits")

  @classmethod
  def from_config(cls, config: GemmaTransformerConfig) -> GemmaTransformer:
    """Constructs an uninitialized transformer with the Gemma architecture.

    Args:
      config: The configuration of the Gemma model.

    Returns:
      A ``GemmaTransformer`` with uninitialized parameters. All side input
      effects will have already been appropriately handled.
    """
    # Embedding table is shared between first and last layers.
    emb_table = pz.nn.mark_shareable(
        pz.nn.add_parameter_prefix(
            "embedder",
            pz.nn.EmbeddingTable.from_config(
                vocab_size=config.vocab_size,
                embedding_axes={"embedding": config.embedding_dim},
                dtype=config.parameter_dtype,
            ),
        )
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
          pz.nn.add_parameter_prefix(
              f"block_{i}", GemmaTransformerBlock.from_config(config)
          )
      )
    sublayers.extend([
        pz.nn.add_parameter_prefix(
            "final_norm",
            pz.nn.RMSLayerNorm.from_config(
                across_axes={"embedding": config.embedding_dim},
                dtype=config.parameter_dtype,
            ),
        ),
        pz.nn.EmbeddingDecode(emb_table),
    ])
    # Handle shared parameters and side inputs.
    return GemmaTransformer(
        config=config,
        body=pz.de.WithSideInputsFromInputTuple.handling(
            pz.nn.attach_shared_parameters(pz.nn.Sequential(sublayers)),
            tags=["token_positions", "attn_mask"],
        ),
    )

  @classmethod
  def from_pretrained(
      cls,
      ckpt_params: dict[str, Any],
      upcast_activations_to_float32: bool = False,
  ) -> GemmaTransformer:
    """Constructs a ``GemmaTransformer`` from the official Gemma Flax checkpoint.

    The parameters of the loaded ``GemmaTransformer`` will be close to those in
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
      A GemmaTransformer model containing the loaded parameters.
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
    model_def = cls.from_config(config)
    parameter_mapping = {
        "embedder.embeddings": pz.nx.NamedArray.wrap(
            params["embedder"]["input_embedding"]
        ).tag("vocabulary", "embedding"),
        "final_norm.scale.weights": pz.nx.NamedArray.wrap(
            1 + params["final_norm"]["scale"]
        ).tag("embedding"),
    }
    for i in range(config.num_decoder_blocks):
      parameter_mapping.update({
          f"block_{i}.pre_attention_norm.scale.weights": pz.nx.NamedArray.wrap(
              1 + params[f"layer_{i}/pre_attention_norm"]["scale"]
          ).tag("embedding"),
          f"block_{i}.pre_ffw_norm.scale.weights": pz.nx.NamedArray.wrap(
              1 + params[f"layer_{i}/pre_ffw_norm"]["scale"]
          ).tag("embedding"),
          f"block_{i}.mlp.gating_linear.weights": pz.nx.NamedArray.wrap(
              params[f"layer_{i}/mlp/gating_einsum"]["w"][0]
          ).tag("embedding", "neurons"),
          f"block_{i}.mlp.value_linear.weights": pz.nx.NamedArray.wrap(
              params[f"layer_{i}/mlp/gating_einsum"]["w"][1]
          ).tag("embedding", "neurons"),
          f"block_{i}.mlp.out_linear.weights": pz.nx.NamedArray.wrap(
              params[f"layer_{i}/mlp/linear"]["w"]
          ).tag("neurons", "embedding"),
          f"block_{i}.attn.output.weights": pz.nx.NamedArray.wrap(
              params[f"layer_{i}/attn/attn_vec_einsum"]["w"]
          ).tag("heads", "projection", "embedding"),
      })
      if config.single_kv_head:
        parameter_mapping.update({
            f"block_{i}.attn.query.weights": pz.nx.NamedArray.wrap(
                params[f"layer_{i}/attn/q_einsum"]["w"]
            ).tag("heads", "embedding", "projection"),
            f"block_{i}.attn.key.weights": pz.nx.NamedArray.wrap(
                params[f"layer_{i}/attn/kv_einsum"]["w"][0].squeeze(0)
            ).tag("embedding", "projection"),
            f"block_{i}.attn.value.weights": pz.nx.NamedArray.wrap(
                params[f"layer_{i}/attn/kv_einsum"]["w"][1].squeeze(0)
            ).tag("embedding", "projection"),
        })
      else:
        parameter_mapping.update({
            f"block_{i}.attn.query.weights": pz.nx.NamedArray.wrap(
                params[f"layer_{i}/attn/qkv_einsum"]["w"][0]
            ).tag("heads", "embedding", "projection"),
            f"block_{i}.attn.key.weights": pz.nx.NamedArray.wrap(
                params[f"layer_{i}/attn/qkv_einsum"]["w"][1]
            ).tag("heads", "embedding", "projection"),
            f"block_{i}.attn.value.weights": pz.nx.NamedArray.wrap(
                params[f"layer_{i}/attn/qkv_einsum"]["w"][2]
            ).tag("heads", "embedding", "projection"),
        })
    return (
        model_def.select()
        .at_instances_of(pz.nn.UninitializedParameter)
        .apply(
            lambda param: param.initialize_with_value(
                parameter_mapping[param.name],
                strict_dtype=False,
            )
        )
    )
