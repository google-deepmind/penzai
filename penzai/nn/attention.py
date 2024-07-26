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
from typing import Any, Hashable

import jax
import jax.numpy as jnp
from penzai.core import named_axes
from penzai.core import struct
from penzai.core import variables
from penzai.nn import layer as layer_base
from penzai.nn import layer_stack


@struct.pytree_dataclass
class ApplyExplicitAttentionMask(layer_base.Layer):
  """Applies an explicit attention mask to its input logit array.

  This layer retrieves an attention mask from its side input, and uses it
  to mask its main argument. Masked out values are replaced with the
  ``masked_out_value`` attribute, which is usually a large (but finite) negative
  value.

  Attributes:
    mask_input_name: Key in the side input dictionary to use to identify the
      attention mask.
    masked_out_value: The value to substitute for masked-out locations.
  """

  mask_input_name: str = dataclasses.field(metadata={"pytree_node": False})
  masked_out_value: jax.typing.ArrayLike

  def __call__(
      self, x: named_axes.NamedArray, **side_inputs: Any
  ) -> named_axes.NamedArray:
    """Applies the attention mask to the input array.

    Args:
      x: The input array to mask. Usually the matrix of query-key dot products.
      **side_inputs: Side inputs. Must include ``mask_input_name``.

    Returns:
      An adjusted matrix of logits, where any value where the mask is False has
      been replaced with the `masked_out_value` argument.
    """
    mask = side_inputs[self.mask_input_name]
    return named_axes.nmap(jnp.where)(mask, x, self.masked_out_value)


@struct.pytree_dataclass
class ApplyCausalAttentionMask(layer_base.Layer):
  """Builds and applies a causal attention mask based on token positions.

  This layer retrieves the token positions from its side input, and uses them
  to build a causal attention mask. Masked out values are replaced with the
  ``masked_out_value`` attribute, which is usually a large (but finite) negative
  value.

  Attributes:
    masked_out_value: The value to substitute for masked-out locations.
    query_positions_input_name: Key in the side input dictionary to use to
      identify the query token positions, which should be an integer array with
      the `seq_axis` axis.
    kv_positions_input_name: Key in the side input dictionary to use to identify
      the key/value token positions, which should be an integer array the
      `seq_axis` axis. (This axis will be renamed to match `kv_seq_axis`.)
    seq_axis: Name of the sequence axis, which should be present in both the
      query and key/value token position side inputs.
    kv_seq_axis: Name of the key/value sequence axis, which represents the keys
      and values in the input logits array.
  """

  masked_out_value: jax.typing.ArrayLike
  query_positions_input_name: str = dataclasses.field(
      default="token_positions", metadata={"pytree_node": False}
  )
  kv_positions_input_name: str = dataclasses.field(
      default="token_positions", metadata={"pytree_node": False}
  )
  seq_axis: str = dataclasses.field(
      default="seq", metadata={"pytree_node": False}
  )
  kv_seq_axis: str = dataclasses.field(
      default="kv_seq", metadata={"pytree_node": False}
  )

  def __call__(
      self, x: named_axes.NamedArray, **side_inputs: Any
  ) -> named_axes.NamedArray:
    """Applies the attention mask to the input array.

    Args:
      x: The input array to mask. Usually the matrix of query-key dot products.
      **side_inputs: Side inputs. Must include ``query_positions_input_name``
        and ``kv_positions_input_name``.

    Returns:
      An adjusted matrix of logits, where any value where the mask is False has
      been replaced with the `masked_out_value` argument.
    """
    query_positions = side_inputs[self.query_positions_input_name]
    kv_positions = (
        side_inputs[self.kv_positions_input_name]
        .untag(self.seq_axis)
        .tag(self.kv_seq_axis)
    )
    mask = (query_positions >= kv_positions) & (kv_positions >= 0)
    return named_axes.nmap(jnp.where)(mask, x, self.masked_out_value)


@struct.pytree_dataclass
class ApplyCausalSlidingWindowAttentionMask(layer_base.Layer):
  """Builds and applies a sliding-window attention mask based on token positions.

  This layer retrieves the token positions from its side input, and uses them
  to build a causal sliding-window attention mask, where values at a distance of
  `window_size` or further away from the current token are masked out. Masked
  out values are replaced with the ``masked_out_value`` attribute, which is
  usually a large (but finite) negative value.

  Attributes:
    masked_out_value: The value to substitute for masked-out locations.
    sliding_window_size: The size of the sliding window.
    query_positions_input_name: Key in the side input dictionary to use to
      identify the query token positions, which should be an integer array with
      the `seq_axis` axis.
    kv_positions_input_name: Key in the side input dictionary to use to identify
      the key/value token positions, which should be an integer array the
      `seq_axis` axis. (This axis will be renamed to match `kv_seq_axis`.)
    seq_axis: Name of the sequence axis, which should be present in both the
      query and key/value token position side inputs.
    kv_seq_axis: Name of the key/value sequence axis, which represents the keys
      and values in the input logits array.
  """

  masked_out_value: jax.typing.ArrayLike
  sliding_window_size: int | jax.typing.ArrayLike
  query_positions_input_name: str = dataclasses.field(
      default="token_positions", metadata={"pytree_node": False}
  )
  kv_positions_input_name: str = dataclasses.field(
      default="token_positions", metadata={"pytree_node": False}
  )
  seq_axis: str = dataclasses.field(
      default="seq", metadata={"pytree_node": False}
  )
  kv_seq_axis: str = dataclasses.field(
      default="kv_seq", metadata={"pytree_node": False}
  )

  def __call__(
      self, x: named_axes.NamedArray, **side_inputs: Any
  ) -> named_axes.NamedArray:
    """Applies the attention mask to the input array.

    Args:
      x: The input array to mask. Usually the matrix of query-key dot products.
      **side_inputs: Side inputs. Must include ``query_positions_input_name``
        and ``kv_positions_input_name``.

    Returns:
      An adjusted matrix of logits, where any value where the mask is False has
      been replaced with the `masked_out_value` argument.
    """
    query_positions = side_inputs[self.query_positions_input_name]
    kv_positions = (
        side_inputs[self.kv_positions_input_name]
        .untag(self.seq_axis)
        .tag(self.kv_seq_axis)
    )
    mask = (
        (query_positions >= kv_positions)
        & (query_positions - kv_positions < self.sliding_window_size)
        & (kv_positions >= 0)
    )
    return named_axes.nmap(jnp.where)(mask, x, self.masked_out_value)


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

  input_to_query: layer_base.Layer
  input_to_key: layer_base.Layer
  input_to_value: layer_base.Layer
  query_key_to_attn: layer_base.Layer
  attn_value_to_output: layer_base.Layer

  def __call__(
      self, x: named_axes.NamedArray, **side_inputs: Any
  ) -> named_axes.NamedArray:
    """Runs the attention computation.

    Args:
      x: The input to the computation, which will be mapped to queries, keys,
        and values by the sublayers.
      **side_inputs: Side inputs for all sublayers.

    Returns:
      The final output of the ``attn_value_to_output`` sublayer.
    """
    query = self.input_to_query(x, **side_inputs)
    key = self.input_to_key(x, **side_inputs)
    value = self.input_to_value(x, **side_inputs)
    attn = self.query_key_to_attn((query, key), **side_inputs)
    output = self.attn_value_to_output((attn, value), **side_inputs)
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
    kv_cache_end_index_key: The key for the side input that identifies the
      current dynamic size of the key/value caches, i.e. the number of elements
      that have been populated with entries. The corresponding side input should
      be a scalar integer array.
    kv_cache: A state effect variable that stores a tuple of key and value
      caches. This will be initialized when this layer is constructed, and will
      be updated as it runs.
  """

  input_to_query: layer_base.Layer
  input_to_key: layer_base.Layer
  input_to_value: layer_base.Layer
  query_key_to_attn: layer_base.Layer
  attn_value_to_output: layer_base.Layer

  sequence_axis: str = dataclasses.field(metadata={"pytree_node": False})
  kv_cache_end_index_key: Hashable = dataclasses.field(
      metadata={"pytree_node": False}
  )
  kv_cache: variables.StateVariable[
      tuple[named_axes.NamedArray, named_axes.NamedArray]
  ]

  def __call__(
      self, x: named_axes.NamedArray, **side_inputs: Any
  ) -> named_axes.NamedArray:
    """Runs the caching attention computation and update the K/V cache state.

    When called, ``self.kv_cache_end_index`` should be filled with
    a scalar integer identifying the current size of the cache (before inserting
    this token), and ``self.kv_cache`` should be a `LocalState` that contains
    the current state.

    Args:
      x: The input to the computation, which will be mapped to queries, keys,
        and values by the sublayers.
      **side_inputs: Side inputs for all sublayers. Should contain the key-value
        cache end index at the key indicated by this layer's
        `kv_cache_end_index_key` attribute.

    Returns:
      The final output of the ``attn_value_to_output`` sublayer.
    """

    # Retrieve effectful inputs.
    kvc_end_index = side_inputs[self.kv_cache_end_index_key]
    key_cache, value_cache = self.kv_cache.value

    # Compute queries, keys, and values as normal.
    query = self.input_to_query(x, **side_inputs)
    key = self.input_to_key(x, **side_inputs)
    value = self.input_to_value(x, **side_inputs)

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
    self.kv_cache.value = (new_key_cache, new_value_cache)

    # Run the rest on the updated KV caches.
    attn = self.query_key_to_attn((query, new_key_cache), **side_inputs)
    output = self.attn_value_to_output((attn, new_value_cache), **side_inputs)
    return output

  @classmethod
  def from_uncached(
      cls,
      original: Attention,
      sequence_axis: str,
      cache_len: int,
      cached_axes: dict[str, int],
      cache_end_index_key: Hashable,
      cache_dtype: jax.typing.DTypeLike = jnp.float32,
      cache_label: variables.VariableLabel | None = None,
      layerstack_axes: dict[named_axes.AxisName, int] | None = None,
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
      cache_end_index_key: Key to use for the cache position side input.
      cache_dtype: Dtype for the data to store in the cache. Should match the
        dtype of the key and value arrays.
      cache_label: Optional label for the KV cache variable.
      layerstack_axes: Stacked axes that are used inside a LayerStack
        combinator. Usually inferred from `pz.nn.layerstack_axes_from_keypath`.

    Returns:
      A ``KVCachingAttention`` instance that behaves like the original
      `Attention` layer, but updates key-value caches iteratively, using new
      side input and state effect requests.
    """
    if layerstack_axes:
      cache_metadata = {
          "layerstack_axes": {
              ax: layer_stack.LayerStackVarBehavior.PER_LAYER
              for ax in layerstack_axes.keys()
          }
      }
    else:
      cache_metadata = {}
      layerstack_axes = {}
    empty_cache = named_axes.zeros(
        {**cached_axes, **layerstack_axes, sequence_axis: cache_len},
        dtype=cache_dtype,
    )

    return cls(
        input_to_query=original.input_to_query,
        input_to_key=original.input_to_key,
        input_to_value=original.input_to_value,
        query_key_to_attn=original.query_key_to_attn,
        attn_value_to_output=original.attn_value_to_output,
        sequence_axis=sequence_axis,
        kv_cache_end_index_key=cache_end_index_key,
        kv_cache=variables.StateVariable(
            label=cache_label,
            value=(empty_cache, empty_cache),
            metadata=cache_metadata,
        ),
    )
