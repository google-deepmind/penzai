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

"""Base attention dataflow combinators.

This module contains basic primitives for attention operations in Transformer
neural networks. These primitives are intentionally as simple as possible,
and do not include the actual initialization logic or attention weight
computation. Instead, they abstract away the core dataflow patterns across
training and kv-cache inference modes.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from penzai.core import named_axes
from penzai.core import struct
from penzai.deprecated.v1.core import layer as layer_base
from penzai.deprecated.v1.data_effects import local_state
from penzai.deprecated.v1.data_effects import side_input


@struct.pytree_dataclass
class ApplyAttentionMask(layer_base.Layer):
  """Applies an attention mask to its input logit array.

  This layer retrieves a causal attention mask from its side input, and uses it
  to mask its argument. Masked out values are replaced with the
  ``masked_out_value`` attribute, which is usually a large (but finite) negative
  value.

  Attributes:
    mask: A side input that provides the attention mask to apply to the input
      attention scores. This side input should be provided as a boolean array
      that is broadcastable with the input.
  """

  mask: side_input.SideInputEffect[named_axes.NamedArray]
  masked_out_value: jax.typing.ArrayLike

  @classmethod
  def from_config(
      cls,
      mask_tag: Any,
      masked_out_value: jax.typing.ArrayLike = -2.3819763e38,
  ) -> ApplyAttentionMask:
    """Creates an ``ApplyAttentionMask`` layer from a tag and a mask value.

    Args:
      mask_tag: Side input tag for the mask side input. This should be used to
        identify the sdie inputs that correspond to the same attention mask
        throughout the model.
      masked_out_value: The value to replace masked out values with. This is
        usually a large (but finite) negative value, so that it maps to a
        negligible attention weight in a numerically stable way.

    Returns:
      A new ``ApplyAttentionMask`` layer with the given configuration.
    """
    return cls(
        mask=side_input.SideInputRequest(tag=mask_tag),
        masked_out_value=masked_out_value,
    )

  def __call__(self, x: named_axes.NamedArray) -> named_axes.NamedArray:
    """Applies the attention mask to the input array.

    Args:
      x: The input array to mask. Usually the matrix of query-key dot products.

    Returns:
      An adjusted matrix of logits, where any value where the mask is False has
      been replaced with the `masked_out_value` argument.
    """
    return named_axes.nmap(jnp.where)(self.mask.ask(), x, self.masked_out_value)


@struct.pytree_dataclass
class Attention(layer_base.Layer):
  """A basic attention combinator.

  An attention layer contains five subcomputations, for computing queries, keys,
  and values, combining queries and keys into attention weights, and combining
  attention weights and values into an output. This class abstracts away the
  dataflow patterns common to all attention layers, and leaves the details of
  the actual computations to the sublayers.

  Attributes:
    input_to_query: A layer that maps the input to an array of queries.
    input_to_key: A layer that maps the input to an array of keys.
    input_to_value: A layer that maps the input to an array of values.
    query_key_to_attn: A layer that maps a tuple of (queries, keys) to attention
      weights.
    attn_value_to_output: A layer that maps a a tuple of (attention weights,
      values) to a final output.
  """

  input_to_query: layer_base.LayerLike
  input_to_key: layer_base.LayerLike
  input_to_value: layer_base.LayerLike
  query_key_to_attn: layer_base.LayerLike
  attn_value_to_output: layer_base.LayerLike

  def __call__(self, x: named_axes.NamedArray) -> named_axes.NamedArray:
    """Runs the attention computation.

    Args:
      x: The input to the computation, which will be mapped to queries, keys,
        and values by the sublayers.

    Returns:
      The final output of the ``attn_value_to_output`` sublayer.
    """
    query = self.input_to_query(x)
    key = self.input_to_key(x)
    value = self.input_to_value(x)
    attn = self.query_key_to_attn((query, key))
    output = self.attn_value_to_output((attn, value))
    return output


@struct.pytree_dataclass
class KVCachingAttention(layer_base.Layer):
  """Key/value caching variant of `Attention`.

  ``KVCachingAttention`` is a drop-in replacement for `Attention`, but adds
  key/value caching logic using Penzai's effect system. This means that a model
  initially configured for training can be quickly adapted to do inference
  without making the training logic more complicated.

  Attributes:
    input_to_query: A layer that maps the input to an array of queries, usually
      taken from the original `Attention` layer.
    input_to_key: A layer that maps the input to an array of keys, usually taken
      from the original `Attention` layer. The output of this layer will
      additionally be stored in the stateful key/value cache.
    input_to_value: A layer that maps the input to an array of values, usually
      taken from the original `Attention` layer. The output of this layer will
      additionally be stored in the stateful key/value cache.
    query_key_to_attn: A layer that maps a tuple of ``(queries, keys)`` to
      attention weights, usually taken from the original `Attention` layer. The
      key input will contain the full key cache, rather than the slice produced
      for the current token.
    attn_value_to_output: A layer that maps a a tuple of ``(attention weights,
      values)`` to a final output, usually taken from the original `Attention`
      layer. The value input will contain the full value cache, rather than the
      slice produced for the current token.
    sequence_axis: The axis along which to do key/value caching. Should be an
      axis name that appears in the output of the ``input_to_key`` and
      ``input_to_value`` sublayers.
    kv_cache_end_index: A side input that identifies the current dynamic size of
      the key/value caches, i.e. the number of elements that have been populated
      with entries. Should be populated by a scalar integer array.
    kv_cache: A state effect variable that stores a tuple of key and value
      caches. This will be initialized when this layer is constructed, and will
      be updated as it runs.
  """

  input_to_query: layer_base.LayerLike
  input_to_key: layer_base.LayerLike
  input_to_value: layer_base.LayerLike
  query_key_to_attn: layer_base.LayerLike
  attn_value_to_output: layer_base.LayerLike

  sequence_axis: str = dataclasses.field(metadata={"pytree_node": False})
  kv_cache_end_index: side_input.SideInputEffect[jax.Array]
  kv_cache: local_state.LocalStateEffect[
      tuple[named_axes.NamedArray, named_axes.NamedArray]
  ]

  def __call__(self, x: named_axes.NamedArray) -> named_axes.NamedArray:
    """Runs the caching attention computation and update the K/V cache state.

    When called, ``self.kv_cache_end_index`` should be filled with
    a scalar integer identifying the current size of the cache (before inserting
    this token), and ``self.kv_cache`` should be a `LocalState` that contains
    the current state.

    Args:
      x: The input to the computation, which will be mapped to queries, keys,
        and values by the sublayers.

    Returns:
      The final output of the ``attn_value_to_output`` sublayer.
    """

    # Retrieve effectful inputs.
    kvc_end_index = self.kv_cache_end_index.ask()
    key_cache, value_cache = self.kv_cache.get()

    # Compute queries, keys, and values as normal.
    query = self.input_to_query(x)
    key = self.input_to_key(x)
    value = self.input_to_value(x)

    # Update the KV caches.
    new_key_cache = named_axes.nmap(jax.lax.dynamic_update_slice)(
        key_cache.untag(self.sequence_axis),
        key.untag(self.sequence_axis),
        (kvc_end_index,),
    ).tag(self.sequence_axis)
    new_value_cache = named_axes.nmap(jax.lax.dynamic_update_slice)(
        value_cache.untag(self.sequence_axis),
        value.untag(self.sequence_axis),
        (kvc_end_index,),
    ).tag(self.sequence_axis)
    self.kv_cache.set((new_key_cache, new_value_cache))

    # Run the rest on the updated KV caches.
    attn = self.query_key_to_attn((query, new_key_cache))
    output = self.attn_value_to_output((attn, new_value_cache))
    return output

  @classmethod
  def from_uncached(
      cls,
      original: Attention,
      sequence_axis: str,
      cache_len: int,
      cached_axes: dict[str, int],
      cache_end_index_tag: side_input.Tag,
      state_category: local_state.Category,
      cache_dtype: jax.typing.DTypeLike = jnp.float32,
  ) -> KVCachingAttention:
    """Builds a caching attention from an uncached attention.

    Args:
      original: The original attention layer that this block should replace.
      sequence_axis: The axis along which keys and values should be cached.
        Should be present in the output of the ``input_to_key`` and
        ``input_to_value`` sublayers.
      cache_len: Length of the cache; used to populate the initial state.
      cached_axes: Axis names and sizes for all other axes of the key and value
        arrays (e.g. for batch, heads, and the projected embeddings). These are
        used to initialize the cache.
      cache_end_index_tag: Side input tag for the cache position side input.
        This should be used to identify the side inputs that should receive the
        cache position information, and should (usually) be provided to the
        `pz.de.WithSideInputsFromInputTuple` handler that actually provides this
        side input.
      state_category: Category for the local state. This should be used to
        identify the state variables that correspond to key-value caches in the
        model, and should (usually) be provided to the
        `pz.de.handle_local_states` call that functionalizes the state effect.
      cache_dtype: Dtype for the data to store in the cache. Should match the
        dtype of the key and value arrays.

    Returns:
      A ``KVCachingAttention`` instance that behaves like the original
      `Attention` layer, but updates key-value caches iteratively, using new
      side input and state effect requests.
    """

    def kv_cache_initializer():
      empty_cache = named_axes.zeros(
          {**cached_axes, sequence_axis: cache_len},
          dtype=cache_dtype,
      )
      return (empty_cache, empty_cache)

    return cls(
        input_to_query=original.input_to_query,
        input_to_key=original.input_to_key,
        input_to_value=original.input_to_value,
        query_key_to_attn=original.query_key_to_attn,
        attn_value_to_output=original.attn_value_to_output,
        sequence_axis=sequence_axis,
        kv_cache_end_index=side_input.SideInputRequest(cache_end_index_tag),
        kv_cache=local_state.InitialLocalStateRequest(
            kv_cache_initializer, category=state_category
        ),
    )
