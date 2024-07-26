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

"""Helpers to visualize tokens, token probabilities, and token sequences."""

from __future__ import annotations

import itertools
from typing import Protocol, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from penzai import pz
from treescope import figures

# pylint: disable=invalid-name


class SentencePieceProcessorLike(Protocol):
  """Protocol defining the methods we need from a tokenizer.

  Sentencepiece tokenizers conform to this interface, but anything else
  that implements it will also work.
  """

  def GetPieceSize(self) -> int:
    """Returns the number of tokens in the vocabulary."""
    ...

  def IdToPiece(self, id: int) -> str:  # pylint: disable=redefined-builtin
    """Decodes a token ID to a string."""
    ...

  def IsControl(self, id: int) -> bool:  # pylint: disable=redefined-builtin
    """Identifies whether a token is a control token."""
    ...


def _fixup_tokens_and_name_order(
    token_arrays: list[pz.nx.NamedArrayBase | jax.Array | np.ndarray],
    axis_name_order: Sequence[str] | None = None,
) -> tuple[list[pz.nx.NamedArrayView], Sequence[str]]:
  """Helper function to preprocess token arrays and axis names."""
  namified_arrays = []
  common_shape = {}
  for token_array in token_arrays:
    if not isinstance(token_array, pz.nx.NamedArrayBase):
      token_array = pz.nx.wrap(
          jax.device_put(
              token_array, device=jax.local_devices(backend="cpu")[0]
          )
      )
    if token_array.positional_shape:
      token_array = token_array.tag(
          *(f"axis{i}" for i in range(len(token_array.positional_shape)))
      )
    token_array = jax.device_get(token_array)
    common_shape.update(token_array.named_shape)
    namified_arrays.append(token_array.as_namedarrayview())

  out_arrays = []
  shaped_dummy = pz.nx.zeros(common_shape)
  for token_array in namified_arrays:
    # Use nmap as an easy way to broadcast.
    out_arrays.append(pz.nx.nmap(lambda x, y: x)(token_array, shaped_dummy))

  if axis_name_order is None:
    base_order = list(out_arrays[0].named_shape.keys())
    if "batch" in base_order:
      base_order.remove("batch")
      base_order.insert(0, "batch")
    if "seq" in base_order:
      base_order.remove("seq")
      base_order.append("seq")
    axis_name_order = base_order

  axis_name_order = list(axis_name_order)

  if not out_arrays[0].named_shape:
    out_arrays = [token_array[{"seq": None}] for token_array in out_arrays]
    axis_name_order.append("seq")

  return out_arrays, axis_name_order


def show_token_array(
    tokens: pz.nx.NamedArrayBase | jax.Array | np.ndarray,
    vocab: SentencePieceProcessorLike,
    axis_name_order: Sequence[str] | None = None,
):
  """Renders an array of tokens.

  Args:
    tokens: An array of token IDs.
    vocab: Vocabulary, usually a SentencePiece tokenizer.
    axis_name_order: Optional ordering of the axis names. The last axis name
      will be laid out horizontally. Inferred if not given.

  Returns:
    A visualization of the token array.
  """
  [tokens], axis_name_order = _fixup_tokens_and_name_order(
      [tokens], axis_name_order
  )

  parts = []
  index_iters = []
  for name in axis_name_order[:-1]:
    index_iters.append([(name, i) for i in range(tokens.named_shape[name])])
  for outer in itertools.product(*index_iters):
    ixs = dict(outer)
    if ixs:
      parts.append(pz.ts.styled(f"\n{repr(ixs)}:", "font-style: italic"))
    subparts = []
    for i in range(tokens.named_shape[axis_name_order[-1]]):
      tok = int(tokens[{**ixs, axis_name_order[-1]: i}])
      if vocab.IsControl(tok):
        piece = pz.ts.with_color(vocab.IdToPiece(tok), "gray")
      else:
        piece = repr(vocab.IdToPiece(tok))
        if (piece.startswith("'") or piece.startswith('"')) and (
            piece.endswith("'") or piece.endswith('"')
        ):
          piece = piece[1:-1]
        else:
          piece = pz.ts.with_color(piece, "orange")
      subparts.append(
          pz.ts.inline(
              pz.ts.integer_digitbox(tok, label=f"{tok}"),
              " ",
              piece,
              wrap=False,
          )
      )
      subparts.append(" ")

    # Add an indentation level, but allow sequences to wrap in the indented
    # block.
    parts.append(figures.indented(pz.ts.inline(*subparts, wrap=True)))

  return pz.ts.inline(*parts)


def show_token_scores(
    tokens: pz.nx.NamedArrayBase | jax.Array | np.ndarray,
    scores: pz.nx.NamedArrayBase | jax.Array | np.ndarray,
    vocab: SentencePieceProcessorLike,
    axis_name_order: Sequence[str] | None = None,
    vmax: float | None = None,
):
  """Renders an array of token scores, but with the scored tokens as words.

  Args:
    tokens: An array of token IDs.
    scores: An array of token scores of the same shape as `tokens`.
    vocab: Vocabulary, usually a SentencePiece tokenizer.
    axis_name_order: Optional ordering of the axis names. The last axis name
      will be laid out horizontally. Inferred if not given.
    vmax: Value at which to truncate the colormap.

  Returns:
    A visualization of the token array.
  """
  [tokens, scores], axis_name_order = _fixup_tokens_and_name_order(
      [tokens, scores], axis_name_order
  )
  if vmax is None:
    root_mean_square = jnp.sqrt(jnp.mean(jnp.square(scores.data_array)))
    vmax = 3 * root_mean_square
  parts = []
  index_iters = []
  for name in axis_name_order[:-1]:
    index_iters.append([(name, i) for i in range(tokens.named_shape[name])])
  for outer in itertools.product(*index_iters):
    ixs = dict(outer)
    if ixs:
      parts.append(pz.ts.styled(f"\n{repr(ixs)}:", "font-style: italic"))
    else:
      # Make space for hover tooltips.
      parts.append("\n")
    subparts = []
    for i in range(tokens.named_shape[axis_name_order[-1]]):
      tok = int(tokens[{**ixs, axis_name_order[-1]: i}])
      score = float(scores[{**ixs, axis_name_order[-1]: i}])
      subparts.append(
          pz.ts.text_on_color(text=vocab.IdToPiece(tok), value=score, vmax=vmax)
      )
      subparts.append("")

    # Add an indentation level, but allow sequences to wrap in the indented
    # block.
    parts.append(figures.indented(pz.ts.inline(*subparts, wrap=True)))

  return pz.ts.inline(*parts)
