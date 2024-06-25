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

"""Core components of a Transformer model.

Specific instantiations of the Transformer model will use the following axis
naming conventions:

* "seq" is the temporal axis of the token sequence, i.e. the axis along which
  the prompt tokens are laid out. In an attention matrix, it specifically
  refers to the query token(s) (the ones we are currently processing).

* "embedding" is the axis for embedding vectors and the residual stream.

* "projection" is the axis for query, key, and value head projection vectors,
  i.e. the axis where query-key dot products happen, and for which the values of
  attention heads are retrieved.

* "heads", "head_groups", and "query_heads" are axes for attention heads,
  depending on whether full multi-head, multi-query, or grouped-query attention
  are used.

  * In full multi-head attention, the "heads" axis appears in queries, keys,
    and values.
  * In multi-query attention, the "query_heads" axis appears in queries, and
    keys and values do not have a heads axis.
  * In grouped-query attention, the "head_groups" axis appears in queries, keys,
    and values, and the "query_heads" axis appears in queries only.

* "kv_seq" is a temporary copy of the "seq" axis that represents the position
  of the keys and values in an attention matrix.

* "neurons" is the axis for the neurons in the MLP blocks, which have an
  activation function (GEGLU) applied elementwise and therefore have a
  priveleged basis.

Additionally, they use the following side input names:

* "token_positions" is the name of the side input that provides the position of
  each token for the purposes of positional embeddings.

* "attn_mask" is the name of the side input that provides the attention mask
  for each attention layer.

* Where applicable, "cache_end_index" is the name of the side input that
  identifies the current length of the key/value cache state. This determines
  where the new keys and values are inserted into the cache. The top-level
  `KVCachingTransformer` class will usually handle this for you.

The KV caching logic is defined in the separate module `sampling_mode`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
from penzai.experimental.v2 import pz


@dataclasses.dataclass
class TransformerMetadata:
  """Common axis sizes and other information for transformer models.

  These values are kept on the main transformer object to simplify model
  transformations that depend on axis sizes or dtypes, by making it possible
  to infer the shape of intermediate activations in advance.

  Attributes:
    common_head_axes: A map of axis names to sizes for head axes that are common
      to queries, keys, and values.
    query_only_head_axes: A map of axis names to sizes for head axes that are
      only used for queries.
    embedding_dim: Dimension of the embedding vectors and residual stream.
    projection_dim: Dimension of the query, key, and value projections.
    mlp_hidden_dim: Dimensionality of the hidden layer of the MLP blocks in each
      layer (the "neurons" axis).
    vocab_size: Number of tokens in the vocabulary.
    activation_dtype: Floating dtype to use for activations and KV cache tables.
  """

  common_head_axes: dict[str, int]
  query_only_head_axes: dict[str, int]
  embedding_dim: int
  projection_dim: int
  mlp_hidden_dim: int
  vocab_size: int
  activation_dtype: jax.typing.DTypeLike


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class TransformerFeedForward(pz.nn.Sequential):
  """Informatively-named Sequential subclass for feedforward/MLP layers."""


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class TransformerBlock(pz.nn.Sequential):
  """Informatively-named Sequential subclass for the main transformer blocks."""


@pz.pytree_dataclass
class Transformer(pz.nn.Layer):
  """Top-level transformer decoder wrapper.

  This class is a simple wrapper that holds configuration data and runs safety
  checks.

  Attributes:
    body: The implementation of the transformer.
    metadata: The axis size and dtype info for the transformer.
  """

  body: pz.nn.Layer
  metadata: TransformerMetadata = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def __call__(
      self,
      tokens: pz.nx.NamedArray,
      token_positions: pz.nx.NamedArray,
      attn_mask: pz.nx.NamedArray,
      **extra_side_inputs,
  ) -> pz.nx.NamedArray:
    """Scores log-probabilities for the given inputs.

    For simple sequences, the ``token_positions`` and ``attn_mask`` arguments
    can be computed via `simple_causal_side_inputs`.

    Args:
      tokens: Array of token IDs, as an integer named array with a "seq" axis
        and possibly batch axes. Usually starts with the beginning-of-sequence
        token.
      token_positions: Sequence of token positions, as an integer named array
        with a "seq" axis and possibly batch axes. Usually starts from 0 and
        increments along the "seq" axis, but can be different to support e.g.
        example packing.
      attn_mask: Boolean attention mask with "seq" and "kv_seq" axes of the same
        length, and possibly batch axes. Usually a causal mask, but can be
        different to support e.g. example packing or dropping out inputs.
      **extra_side_inputs: Other side inputs, which will be forwarded to the
        body.

    Returns:
      The final matrix of logits from the embedding decoding layer, which
      (in the normal configuration) will have axes "seq" and "vocabulary".
    """

    return self.body(
        tokens,
        token_positions=token_positions,
        attn_mask=attn_mask,
        **extra_side_inputs,
    )

  def simple_causal_side_inputs(
      self, tokens: pz.nx.NamedArray
  ) -> dict[str, Any]:
    """Builds a side-input dictionary for a batch of single segments.

    This can be used to process inputs that do not need advanced position or
    attention mask handling, and which just consist of ordinary sequences that
    are not packed together or padded.

    Args:
      tokens: Sequence of tokens, as an integer named array with a "seq" axis
        and possibly batch axes, which starts with the beginning-of-sequence
        token. Each 1d vector along the "seq" axis should represent an unpadded
        sequence.

    Returns:
      A dictionary with key "positions" mapping to a simple incrementing
      position array and key "attn_mask" mapping to a causal mask, suitable for
      passing as side (keyword) arguments to this model.
    """
    seq = tokens.named_shape["seq"]
    # Query tokens can attend to keys/values if the query position is larger.
    return {
        "token_positions": pz.nx.arange("seq", seq),
        "attn_mask": pz.nx.arange("seq", seq) >= pz.nx.arange("kv_seq", seq),
    }
