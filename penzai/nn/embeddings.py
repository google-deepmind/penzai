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

"""Structs for handling tokens."""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing  # pylint: disable=unused-import

from penzai.core import named_axes
from penzai.core import struct
from penzai.core import syntactic_sugar
from penzai.nn import layer as layer_base
from penzai.nn import linear_and_affine
from penzai.nn import parameters

_slice = syntactic_sugar.slice


@struct.pytree_dataclass
class EmbeddingTable(struct.Struct):
  """A table of embedding vectors for a vocabulary of tokens.

  ``EmbeddingTable`` owns the embedding parameters used when either encoding a
  token to an embedding or decoding an embedding vector to a distribution over
  tokens. It does not directly provide callable methods, and should be wrapped
  in a `EmbeddingLookup` or `EmbeddingDecode` layer before being inserted into a
  model. This is to allow the same initialization logic to be shared between the
  two methods, and to simplify parameter sharing when tying the embeddings
  between the first and last layers of a language model.

  If you wish to set up weight tying between the encoding and decoding steps,
  you should initialize a single embedding table, and then pass that same
  table to the `EmbeddingLookup` and `EmbeddingDecode` layers.

  Attributes:
    embeddings: The embedding parameters. One axis corresponds to the
      vocabulary, and all other axes will be considered part of the embedding.
      (Usually, there will only be one other axis.)
    vocabulary_axis: The name of the axis that corresponds to the vocabulary.
      This axis will be indexed into when performing embedding lookups.
  """

  embeddings: parameters.ParameterLike[named_axes.NamedArray]
  vocabulary_axis: str = dataclasses.field(metadata={"pytree_node": False})

  @classmethod
  def from_config(
      cls,
      name: str,
      init_base_rng: jax.Array | None,
      vocab_size: int,
      embedding_axes: dict[str, int],
      vocabulary_axis: str = "vocabulary",
      initializer: linear_and_affine.LinearOperatorWeightInitializer = (
          functools.partial(
              linear_and_affine.variance_scaling_initializer,
              scale=1.0,
              mode="fan_out",
              distribution="normal",
          )
      ),
      dtype: np.typing.DTypeLike = np.float32,
  ) -> EmbeddingTable:
    """Constructs an ``EmbeddingTable``.

    Args:
      name: The name of the layer.
      init_base_rng: The base RNG to use for initializing model parameters.
      vocab_size: The size of the vocabulary.
      embedding_axes: A dictionary mapping embedding axis names to their sizes.
        Will usually be a single-element dictionary of the form ``{"embedding":
        embedding_size}``.
      vocabulary_axis: The name of the axis that corresponds to the vocabulary.
        This axis will be indexed into when performing embedding lookups. Must
        not appear in ``embedding_axes``.
      initializer: A weight initializer that will be used to initialize the
        parameters. For the purposes of initialization, the "input axes" are of
        dimension 1, and the "output axes" are the ``embedding_axes``; the
        vocabulary axis is treated as a parallel axis. Defaults to fan-out
        normalization over the embedding axes.
      dtype: The data type of the embedding parameters.

    Returns:
      An EmbeddingTable with of the given shape.
    """
    if vocabulary_axis in embedding_axes:
      raise ValueError(
          f"`vocabulary_axis` {vocabulary_axis} should not appear in"
          f"`embedding_axes` {embedding_axes}"
      )

    return cls(
        embeddings=parameters.make_parameter(
            f"{name}.embeddings",
            init_base_rng,
            initializer,
            input_axes={},
            output_axes=embedding_axes,
            parallel_axes={vocabulary_axis: vocab_size},
            convolution_spatial_axes={},
            dtype=dtype,
        ),
        vocabulary_axis=vocabulary_axis,
    )


@struct.pytree_dataclass
class EmbeddingLookup(layer_base.Layer):
  """Looks up token IDs in an embedding table.

  This layer can be used to look up token embeddings in an embedding table. It
  is usually the first layer in a language model.

  This lookup layer does not include any rescaling of the embeddings. If you
  would like to scale the embeddings after retrieval, consider adding a
  `pz.nn.ConstantRescale` layer after the ``EmbeddingLookup``.

  Attributes:
    table: The embedding table to look up embeddings in.
  """

  table: EmbeddingTable

  def __call__(
      self, token_index: named_axes.NamedArray, **_unused_side_inputs
  ) -> named_axes.NamedArray:
    """Retrieves tokens from the embedding table.

    Args:
      token_index: A named array of token indices. Axis names of this array must
        be disjoint from the axis names in the embedding table.
      **_unused_side_inputs: Side inputs (unused).

    Returns:
      A named array of embeddings, which includes all named axes of the input
      along with the (non-vocabulary) named axes of the embedding table.
    """
    return self.table.embeddings.value[
        {self.table.vocabulary_axis: token_index}
    ]


@struct.pytree_dataclass
class EmbeddingDecode(layer_base.Layer):
  """Uses an embedding table to map embeddings back to token scores.

  This layer can be used to map a model's output embedding to the logits for
  a distribution over output tokens. It is usually the last layer in a language
  model.

  The primary purpose of this layer is to simplify sharing weights between
  the embedding lookup and decode layers. It functions similarly to a Linear
  layer, but retrieves its parameter from an `EmbeddingTable`.

  Attributes:
    table: The embedding table to look up embeddings in.
  """

  table: EmbeddingTable

  def __call__(
      self, out_embeddings: named_axes.NamedArray, **_unused_side_inputs
  ) -> named_axes.NamedArray:
    """Retrieves tokens from the embedding table.

    Args:
      out_embeddings: The output embeddings that should be mapped to token
        logits. Should be a named array that includes the same axes as the
        embedding table, except for the vocabulary axis, and may also include
        additional batch axes.
      **_unused_side_inputs: Side inputs (unused).

    Returns:
      A named array of logits, which includes all batch axes of the input
      along with the vocabulary axis of the embedding table.
    """
    contracting_axes = [
        name
        for name in self.table.embeddings.value.named_shape.keys()
        if name != self.table.vocabulary_axis
    ]
    return linear_and_affine.contract(
        contracting_axes, out_embeddings, self.table.embeddings.value
    )


@struct.pytree_dataclass
class ApplyRoPE(layer_base.Layer):
  """Adjusts input embeddings using rotary position embeddings (RoPE).

  Rotary position embeddings, proposed by Su et al. (2021), incorporate relative
  position information into the queries and keys of attention layers by applying
  periodic rotations to the elements of the query and key projections.

  The ``ApplyRoPE`` layer can be inserted into an attention computation
  immediately before computing query-key dot products in order to add rotational
  position information to them.

  See https://arxiv.org/abs/2104.09864.

  Attributes:
    embedding_axis: The axis name of the input that contains the embedding
      vector (e.g. "embedding" or "projection").
    max_wavelength: The maximum wavelength of the periodic positional
      embeddings.
    positions_input_name: Key for the side input that provides the position of
      each token in the sequence. This side input should be provided as an
      integer array that is broadcastable with the input, and which does NOT
      include the embedding axis.
  """

  embedding_axis: str = dataclasses.field(metadata={"pytree_node": False})
  max_wavelength: int = dataclasses.field(metadata={"pytree_node": False})
  positions_input_name: str = dataclasses.field(metadata={"pytree_node": False})

  def _apply_1d(self, input_slice: jax.Array, position: jax.Array) -> jax.Array:
    """Apply RoPE to a one-dimensional JAX array."""
    assert input_slice.ndim == 1
    assert position.ndim == 0
    # Infer `head_dim` from the input shape
    [head_dim] = input_slice.shape
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = self.max_wavelength**fraction
    # Since we're assuming `timescale` is a vector and `position` is a scalar,
    # we don't need any axis alignment.
    sinusoid_inp = position / timescale
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    first_half, second_half = jnp.split(input_slice, 2)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    return jnp.concatenate([first_part, second_part])

  def __call__(
      self, inputs: named_axes.NamedArray, **side_inputs
  ) -> named_axes.NamedArray:
    positions = side_inputs[self.positions_input_name]

    if self.embedding_axis in positions.named_shape:
      raise ValueError(
          f"Embedding axis {self.embedding_axis} should not already be part of"
          " the positions side input."
      )

    if positions.positional_shape:
      raise ValueError("Positions side input must have no positional axes.")
    if not all(axis in inputs.named_shape for axis in positions.named_shape):
      raise ValueError(
          "Every axis of the positions should appear in the main input.\n"
          f"Positions has: {positions.named_shape}\n"
          f"Main input has: {inputs.named_shape}\n"
      )

    # Unbind the embedding axis from the inputs, producing a 1-D
    # positional view.
    inputs_view = inputs.untag(self.embedding_axis)
    # Run the logic over our 1D view using `pz.nmap`, which vectorizes a
    # function over all named axes:
    out = named_axes.nmap(self._apply_1d)(inputs_view, positions)
    # Finally, re-bind the embedding axis.
    out_named = out.tag(self.embedding_axis)
    return out_named.astype(inputs.dtype)


@struct.pytree_dataclass
class ApplyRoPEToSubset(layer_base.Layer):
  """Adjusts a subset of embeddings using rotary position embeddings (RoPE).

  This is like `ApplyRoPE`, but only applies to a subset of dimensions, for
  compatibility with the GPT-NeoX configuration and similar models.

  Attributes:
    embedding_axis: The axis name of the input that contains the embedding
      vector (e.g. "embedding" or "projection").
    max_wavelength: The maximum wavelength of the periodic positional
      embeddings.
    rope_subset_size: Size of the prefix of the embedding axis that we should
      apply rotary embeddings to. The suffix will be left unchanged.
    positions_input_name: Key for the side input that provides the position of
      each token in the sequence. This side input should be provided as an
      integer array that is broadcastable with the input, and which does NOT
      include the embedding axis.
  """

  embedding_axis: str = dataclasses.field(metadata={"pytree_node": False})
  max_wavelength: int = dataclasses.field(metadata={"pytree_node": False})
  rope_subset_size: int = dataclasses.field(metadata={"pytree_node": False})
  positions_input_name: str = dataclasses.field(metadata={"pytree_node": False})

  def __call__(
      self, inputs: named_axes.NamedArray, **side_inputs
  ) -> named_axes.NamedArray:
    rotary_input = inputs[
        {self.embedding_axis: _slice[: self.rope_subset_size]}
    ]
    passthrough_input = inputs[
        {self.embedding_axis: _slice[self.rope_subset_size :]}
    ]

    rotator = ApplyRoPE(
        embedding_axis=self.embedding_axis,
        max_wavelength=self.max_wavelength,
        positions_input_name=self.positions_input_name,
    )
    rotated_result = rotator(rotary_input, **side_inputs)
    return named_axes.concatenate(
        [rotated_result, passthrough_input], self.embedding_axis
    )
