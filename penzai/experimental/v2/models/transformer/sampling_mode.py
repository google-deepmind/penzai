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

"""Sampling-mode adapters for Transformer models.

This file includes the kv-cache sampling mode of the base Transformer model.
This mode is intended to be hot-swapped for the main Transformer implementation:
you should generally start by loading a `model_parts.Transformer` and then
converting it to a `KVCachingTransformer` using
`KVCachingTransformer.from_uncached`.

The layers defined here follow the same conventions documented in the module
docstring for `model_parts`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from penzai.experimental.v2 import pz
from penzai.experimental.v2.models.transformer import model_parts


@pz.pytree_dataclass
class KVCachingTransformer(pz.nn.Layer):
  """Top-level transformer in (stateful) cached autoregressive sampling mode.

  This class represents the sampling mode of the model, and manages the sampling
  state. It is designed to be loaded from an existing `Transformer`. If you want
  to load this from the pretrained checkpoint, first load a `Transformer`, then
  call `KVCachingTransformer.from_uncached`.

  Attributes:
    body: The implementation of the transformer. Usually a nested set of state
      and side-effect handlers wrapping the main sequence of transformer blocks,
      but may be modified after the model is loaded due to patching.
    cache_end_index: A variable containing the current end index of the caches.
    previous_tokens: A variable containing all previously-seen tokens.
    metadata: The configuration for the transformer.
    cache_len: The length of the internal key-value caches.
    batch_axes: The batch axes of the internal key-value caches.
    pad_id: Token ID that indicates padding.
  """

  body: pz.nn.Layer
  cache_end_index: pz.StateVariable[int]
  previous_tokens: pz.StateVariable[pz.nx.NamedArray]
  metadata: model_parts.TransformerMetadata = dataclasses.field(
      metadata={"pytree_node": False}
  )
  cache_len: int = dataclasses.field(metadata={"pytree_node": False})
  batch_axes: dict[str, int] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  pad_id: int = dataclasses.field(metadata={"pytree_node": False})

  def __call__(
      self, tokens: pz.nx.NamedArray, **extra_side_inputs: dict[Any, Any]
  ) -> pz.nx.NamedArray:
    """Processes a new subsequence of tokens and adds them to the K/V cache.

    When called, the internal variables tracking the key-value cache will be
    updated with the new state.

    Args:
      tokens: Array of token IDs, as an integer named array with a "seq" axis
        and possibly batch axes. The batch axes must match the `batch_axes`
        attribute. Padding tokens are ignored.
      **extra_side_inputs: Extra side inputs, which will be forwarded on to the
        body. The "token_positions", "attn_mask", and "cache_end_index" inputs
        will be added automatically and do not need to be provided.

    Returns:
      Matrix of logits from the embedding decoding layer, which (in the
      normal configuration) will have axes "seq" and "vocabulary".
    """
    pz.chk.check_structure(
        tokens,
        pz.chk.ArraySpec(
            named_shape={**self.batch_axes, "seq": pz.chk.var("S")}
        ),
    )

    # Update the written tokens.
    updated_tokens = pz.nx.nmap(jax.lax.dynamic_update_slice)(
        self.previous_tokens.value.untag("seq"),
        tokens.untag("seq"),
        (self.cache_end_index.value,),
    ).tag("seq")
    # Query position and key position are both based on the number of preceding
    # padding tokens (after updating the written tokens). Compute by summing
    # up non-padding tokens, then slicing the part we are updating.
    kv_nonpad_mask = updated_tokens != self.pad_id
    kv_nonpad_so_far_inclusive = pz.nx.nmap(jnp.cumsum)(
        kv_nonpad_mask.untag("seq"), dtype=jnp.int32
    ).tag("seq")
    kv_nonpad_so_far_exclusive = (
        kv_nonpad_so_far_inclusive - kv_nonpad_mask.astype(jnp.int32)
    )
    query_positions = pz.nx.nmap(jax.lax.dynamic_slice)(
        kv_nonpad_so_far_exclusive.untag("seq"),
        (self.cache_end_index.value,),
        (tokens.named_shape["seq"],),
    ).tag("seq")
    key_value_positions = kv_nonpad_so_far_exclusive.untag("seq").tag("kv_seq")
    # Tokens can attend to any kv-token position that they are after, as long as
    # it was NOT padding.
    attention_mask = (
        (query_positions >= key_value_positions)
        & (tokens != self.pad_id)
        & kv_nonpad_mask.untag("seq").tag("kv_seq")
    )
    # Run the model.
    outs = self.body(
        tokens,
        token_positions=query_positions,
        attn_mask=attention_mask,
        cache_end_index=self.cache_end_index.value,
        **extra_side_inputs,
    )
    # Update the state variables.
    self.previous_tokens.value = updated_tokens
    self.cache_end_index.value = (
        self.cache_end_index.value + tokens.named_shape["seq"]
    )

    return outs

  @classmethod
  def from_uncached(
      cls,
      uncached: model_parts.Transformer,
      cache_len: int,
      batch_axes: dict[str, int],
      pad_id: int = 0,
      variable_name_prefix: str = "sampler",
  ) -> KVCachingTransformer:
    """Transforms a `Transformer` into cached sampling mode.

    This constructor hot-swaps all `pz.nn.Attention` layers in the
    original model to enable key-value caching, then installs new handlers to
    update their states appropriately. Note that any modifications to the
    uncached model will persist in the decoding mode.

    Args:
      uncached: The original `Transformer` model.
      cache_len: Maximum sequence length for the key/value caches.
      batch_axes: Names and sizes for the batch axes that will be used for
        sampling. Required for initializing the key/value caches.
      pad_id: ID for the padding token.
      variable_name_prefix: Prefix for cached sampling variable names.

    Returns:
      A KVCachingTransformer.
    """
    cached_axes = {
        **batch_axes,
        **uncached.metadata.common_head_axes,
        "projection": uncached.metadata.projection_dim,
    }
    caching_body = (
        pz.select(uncached.body)
        .at_instances_of(pz.nn.Attention)
        .apply_with_selected_index(
            lambda ix, attn: pz.nn.KVCachingAttention.from_uncached(
                attn,
                cache_len=cache_len,
                cached_axes=cached_axes,
                cache_dtype=uncached.metadata.activation_dtype,
                sequence_axis="seq",
                cache_end_index_key="cache_end_index",
                cache_label=f"{variable_name_prefix}/cache_{ix}",
            )
        )
    )
    return cls(
        metadata=uncached.metadata,
        cache_len=cache_len,
        batch_axes=batch_axes,
        pad_id=pad_id,
        previous_tokens=pz.StateVariable(
            value=pz.nx.zeros(
                {**batch_axes, "seq": cache_len}, dtype=jnp.int32
            ),
            label=f"{variable_name_prefix}/previous_tokens",
        ),
        cache_end_index=pz.StateVariable(
            value=0,
            label=f"{variable_name_prefix}/cache_end_index",
        ),
        body=caching_body,
    )
