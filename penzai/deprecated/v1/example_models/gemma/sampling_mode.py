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

"""Sampling-mode adapters for the Gemma model.

This file includes the kv-cache sampling variant of the Gemma model. This
variant is intended to be hot-swapped for the main Gemma variant: you should
generally start by loading a `model_core.GemmaTransformer` and then
converting it to a `GemmaKVCachingTransformer` using
`GemmaKVCachingTransformer.from_uncached`.

The layers defined here follow the same conventions documented in the module
docstring for `model_core`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from penzai.deprecated.v1 import pz
from penzai.deprecated.v1.example_models.gemma import model_core


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class GemmaKVCachingAttention(pz.nn.KVCachingAttention):
  """Gemma-specific configuration of the key-value-caching attention layer.

  `GemmaKVCachingAttention` has the same runtime behavior as the base
  `pz.nn.KVCachingAttention` layer, but is specialized to the conventions of the
  Gemma model.
  """

  @classmethod
  def from_uncached(
      cls,
      original: model_core.GemmaAttention,
      cache_len: int,
      cached_axes: dict[str, int],
      cache_dtype: jax.typing.DTypeLike = jnp.float32,
  ) -> GemmaKVCachingAttention:
    """Builds a caching attention from an uncached attention.

    Args:
      original: The original attention layer that this block should replace.
      cache_len: Length of the cache; used to populate the initial state.
      cached_axes: Axis names and sizes for all other axes of the key and value
        arrays (e.g. for batch, heads, and the projected embeddings). These are
        used to initialize the cache.
      cache_dtype: Dtype for the data to store in the cache. Should match the
        dtype of the key and value arrays.

    Returns:
      A `GemmaKVCachingAttention` instance that behaves like the original
      `Attention` layer, but updates key-value caches iteratively, using new
      side input and state effect requests.
    """
    return super().from_uncached(
        original=original,
        sequence_axis="seq",
        cache_len=cache_len,
        cached_axes=cached_axes,
        cache_end_index_tag="cache_end_index",
        state_category="kv_cache",
        cache_dtype=cache_dtype,
    )


@pz.pytree_dataclass
class GemmaKVCachingState(pz.Struct):
  """Sampling state for the key-value-caching Gemma variant.

  You should not usually need to construct this on your own. Instead, it will
  be returned by `GemmaKVCachingTransformer.from_uncached` and updated by
  `GemmaKVCachingTransformer.__call__`.

  Attributes:
    cache_len: The length of the key-value caches along the "seq" axis.
    batch_axes: Axis names and sizes for the batch axes in the key-value caches.
    kv_caches: A dictionary of key-value caches extracted from the model.
    cache_end_index: The current end index of the KV caches, used as the offset
      at which new keys and values will be inserted.
  """

  cache_len: int = dataclasses.field(metadata={"pytree_node": False})
  batch_axes: dict[str, int] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  kv_caches: dict[str, Any]
  cache_end_index: int | jax.Array


@pz.pytree_dataclass
class GemmaKVCachingInputs(pz.Struct):
  """Input structure for the `GemmaKVCachingTransformer`.

  Attributes:
    tokens: Subsequence of the current tokens we are processing, as an integer
      named array with a "seq" axis and possibly batch axes. When pre-filling,
      this can be the length of the prompt. When sampling, the "seq" instance
      will usually have length 1.
    positions: Sequence of current token positions, as an integer named array
      with a "seq" axis and possibly batch axes, of the same sequence length as
      `tokens`. Should usually increase with each call to the transformer.
    attention_mask: Boolean attention mask with "seq" and "kv_seq" axes and
      possibly batch axes. The "seq" axis should match `tokens` and `positions`,
      and the "kv_seq" axis should match the `cache_len` of the
      `sampling_state`. Usually a slice of the causal mask.
    sampling_state: Current sampling state, containing key-value caches.
  """

  tokens: pz.nx.NamedArray
  positions: pz.nx.NamedArray
  attention_mask: pz.nx.NamedArray
  sampling_state: GemmaKVCachingState

  @classmethod
  def from_basic_subsegments(
      cls, tokens: pz.nx.NamedArray, sampling_state: GemmaKVCachingState
  ) -> GemmaKVCachingInputs:
    """Constructs a simple input structure for a batch of unpadded samples.

    This can be used to process inputs that do not need advanced position or
    attention mask handling, and which just consist of ordinary sequences that
    are not packed together or padded. It augments the tokens with a standard
    position array and causal attention mask, adjusted by the current cache
    offset.

    Args:
      tokens: Subsequence of tokens, as an integer named array with a "seq" axis
        and possibly batch axes. When pre-filling, the "seq" axis can be the
        length of the prompt. When sampling, the "seq" instance will usually
        have length 1.
      sampling_state: Current sampling state, containing key-value caches.

    Returns:
      A full input structure containing the provided tokens, along with a simple
      incrementing position array and a causal mask, offset by the current
      sampling state.
    """
    seq = tokens.named_shape["seq"]
    offset = sampling_state.cache_end_index
    positions = pz.nx.arange("seq", seq) + offset
    # Query tokens can attend to keys/values if the query position is larger,
    # taking into account the cache offset.
    attention_mask = positions >= pz.nx.arange(
        "kv_seq", sampling_state.cache_len
    )
    return cls(
        tokens=tokens,
        positions=positions,
        attention_mask=attention_mask,
        sampling_state=sampling_state,
    )


@pz.pytree_dataclass
class GemmaKVCachingTransformer(pz.Layer):
  """Top-level Gemma transformer in cached autoregressive sampling mode.

  This class represents the sampling mode of the Gemma model, and is designed
  to be loaded from an existing `GemmaTransformer`. If you want to load this
  from the pretrained checkpoint, first load a `GemmaTransformer`, then call
  `GemmaKVCachingTransformer.from_uncached`.

  Attributes:
    config: The configuration for the transformer.
    body: The implementation of the transformer. Usually a nested set of state
      and side-effect handlers wrapping the main sequence of transformer blocks,
      but may be modified after the model is loaded due to patching.
  """

  config: model_core.GemmaTransformerConfig = dataclasses.field(
      metadata={"pytree_node": False}
  )
  body: pz.LayerLike

  @pz.checked_layer_call
  def __call__(
      self, inputs: GemmaKVCachingInputs
  ) -> tuple[pz.nx.NamedArray, GemmaKVCachingState]:
    """Processes a new subsequence of tokens and adds them to the K/V cache.

    Args:
      inputs: Structure of input arguments, containing tokens, segment
        positions, an attention mask, and the current sampling state.

    Returns:
      A tuple ``(outputs, new_sampling_state)``, whre ``outputs`` is the final
      matrix of logits from the embedding decoding layer, which (in the normal
      configuration) will have axes "seq" and "vocabulary", and
      ``new_sampling_state`` is the updated sampling state with the updated
      key-value caches.
    """
    outs, kv_caches = self.body((
        (
            (inputs.tokens, inputs.positions, inputs.attention_mask),
            inputs.sampling_state.cache_end_index,
        ),
        inputs.sampling_state.kv_caches,
    ))
    return outs, GemmaKVCachingState(
        cache_len=inputs.sampling_state.cache_len,
        batch_axes=inputs.sampling_state.batch_axes,
        kv_caches=kv_caches,
        cache_end_index=(
            inputs.sampling_state.cache_end_index
            + inputs.tokens.named_shape["seq"]
        ),
    )

  def input_structure(self) -> pz.chk.StructureAnnotation:
    return GemmaKVCachingInputs(
        tokens=pz.chk.Wildcard("tokens"),
        positions=pz.chk.Wildcard("positions"),
        attention_mask=pz.chk.Wildcard("attention mask"),
        sampling_state=pz.chk.Wildcard("previous GemmaKVCachingState"),
    )

  def output_structure(self) -> pz.chk.StructureAnnotation:
    return (
        pz.chk.Wildcard("unnormalized logits"),
        pz.chk.Wildcard("updated GemmaKVCachingState"),
    )

  @classmethod
  def from_uncached(
      cls,
      uncached: model_core.GemmaTransformer,
      cache_len: int,
      batch_axes: dict[str, int],
  ) -> tuple[GemmaKVCachingTransformer, GemmaKVCachingState]:
    """Transforms a `GemmaTransformer` into cached sampling mode.

    This constructor hot-swaps all `model_core.GemmaAttention` layers in the
    original model to enable key-value caching, then installs new handlers to
    update their states appropriately. Note that any modifications to the
    uncached model will persist in the decoding mode.

    Args:
      uncached: The original `GemmaTransformer` model.
      cache_len: Maximum sequence length for the key/value caches.
      batch_axes: Names and sizes for the batch axes that will be used for
        sampling. Required for initializing the key/value caches.

    Returns:
      Tuple ``(sampler_model, initial_sampling_state)``, where ``sampler_model``
      is a `GemmaKVCachingTransformer`, and ``initial_sampling_state`` holds the
      initial empty key/value caches.
    """
    cached_axes = {
        **batch_axes,
        "projection": uncached.config.projection_dim,
    }
    if not uncached.config.single_kv_head:
      cached_axes["heads"] = uncached.config.num_heads
    caching_body = (
        pz.select(uncached.body)
        .at_instances_of(model_core.GemmaAttention)
        .apply(
            lambda attn: GemmaKVCachingAttention.from_uncached(
                attn,
                cache_len=cache_len,
                cached_axes=cached_axes,
                cache_dtype=uncached.config.activation_dtype,
            )
        )
    )
    handled_body, initial_state = pz.de.handle_local_states(
        pz.de.WithSideInputsFromInputTuple.handling(
            caching_body, tags=["cache_end_index"]
        ),
        category="kv_cache",
    )
    inference_model = cls(config=uncached.config, body=handled_body)
    sampling_state = GemmaKVCachingState(
        cache_len=cache_len,
        batch_axes=batch_axes,
        kv_caches=initial_state,
        cache_end_index=0,
    )
    return inference_model, sampling_state
