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

"""A simple decoding loop for the Gemma model.

This can be used to sample from Gemma in decoding mode, and can also be used
as a starting point for more sophisticated sampling algorithms.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from penzai.deprecated.v1 import pz
from penzai.deprecated.v1.example_models.gemma import sampling_mode


@pz.pytree_dataclass
class SamplingState(pz.Struct):
  """State that manages the set decoded tokens during sampling.

  The purpose of this class is to keep the outputs, inputs, and key-value caches
  of the transformer in sync during decoding, even in the presence of padding
  tokens. This makes it possible to sample from batches of prompts which have
  different lengths.

  Padding tokens are treated as "invisible" to the model; they will be skipped
  over and masked out from the model's queries if they appear in the middle of
  the sequence.

  Attributes:
    kv_cache_state: The state of the KV caches.
    previous_tokens: An array of all of the outputs that have been written so
      far. This is also used to identify the effective sequence position of the
      next token by counting the non-padding tokens.
    pad_id: Token ID that indicates padding.
  """

  kv_cache_state: sampling_mode.GemmaKVCachingState
  previous_tokens: pz.nx.NamedArray
  pad_id: int


def prefill(
    model: sampling_mode.GemmaKVCachingTransformer,
    initial_cache_state: sampling_mode.GemmaKVCachingState,
    prompt: pz.nx.NamedArray,
    pad_id: int,
) -> tuple[pz.nx.NamedArray, SamplingState]:
  """Prefills the key-value caches based on a prompt.

  Args:
    model: The converted model we are running inference with.
    initial_cache_state: The initial cache state created while converting the
      model.
    prompt: A named array of prompt tokens. Must have a "seq" axis along which
      the tokens for each batch element are arranged. Should usually start with
      the beginning of sequence token. This function assumes (but does not
      check) that all non-padding tokens preceed all padding tokens.
    pad_id: Token ID that corresponds to padding.

  Returns:
    A tuple ``(next_log_probs, sampling_state)``, where ``next_log_probs`` are
    the log-probabilities of the next token to sample, and ``sampling_state`` is
    a state that can be passed to future sampling calls.
  """
  not_padding_mask = prompt != pad_id

  # Since padding tokens are at the end, the offset of the last non-padding
  # token is one before the number of non-padding tokens.
  last_non_padding_index = (
      not_padding_mask.astype(jnp.int32).untag("seq").sum() - 1
  )

  # Token positions are just offsets along the token axis, since we're assuming
  # that padding only appears at the end.
  query_positions = pz.nx.arange("seq", prompt.named_shape["seq"])
  # Tokens can attend to any kv-token position that they are after.
  key_value_positions = pz.nx.arange("kv_seq", initial_cache_state.cache_len)
  attention_mask = query_positions >= key_value_positions
  # Run the model:
  out_logits, new_cache_state = model(
      sampling_mode.GemmaKVCachingInputs(
          tokens=prompt,
          positions=query_positions,
          attention_mask=attention_mask,
          sampling_state=initial_cache_state,
      )
  )
  # Extract the log probs from the last non-padding index, since that determines
  # the probs for the next sampled token.
  final_token_log_probs = pz.nx.nmap(jax.nn.log_softmax)(
      out_logits[{"seq": last_non_padding_index}].untag("vocabulary")
  ).tag("vocabulary")
  # Pad out the prompt to form our initial state matrix.
  padded_prompt = pz.nx.nmap(jnp.pad)(
      prompt.untag("seq"),
      [(0, initial_cache_state.cache_len - prompt.named_shape["seq"])],
      mode="constant",
      constant_values=pad_id,
  ).tag("seq")

  return final_token_log_probs, SamplingState(
      kv_cache_state=new_cache_state,
      previous_tokens=padded_prompt,
      pad_id=pad_id,
  )


def advance_one_token(
    model: sampling_mode.GemmaKVCachingTransformer,
    state: SamplingState,
    next_token: jax.Array,
) -> tuple[pz.nx.NamedArray, SamplingState]:
  """Advances a sampling state by one token.

  This can be used to feed new sampled tokens one-at-a-time through the model,
  producing new log-probs that can be used to sample new tokens.

  Args:
    model: The converted model we are running inference with.
    state: The current sampling state.
    next_token: The next token to feed. Should not have a "seq" axis.

  Returns:
    A tuple ``(next_log_probs, sampling_state)``, where ``next_log_probs`` are
    the log-probabilities of the next token to sample, and ``sampling_state`` is
    a state that can be passed to future sampling calls.
  """
  # Update the written tokens.
  update_t = state.kv_cache_state.cache_end_index
  updated_tokens = pz.nx.nmap(lambda s, v: s.at[update_t].set(v))(
      state.previous_tokens.untag("seq"), next_token
  ).tag("seq")
  assert dict(updated_tokens.named_shape) == dict(
      state.previous_tokens.named_shape
  )
  # Query position and key position are both based on the number of preceding
  # padding tokens (after updating the written tokens).
  nonpad_mask = updated_tokens != state.pad_id
  nonpad_so_far_inclusive = pz.nx.nmap(jnp.cumsum)(
      nonpad_mask.untag("seq"), dtype=jnp.int32
  ).tag("seq")
  nonpad_so_far_exclusive = nonpad_so_far_inclusive - nonpad_mask.astype(
      jnp.int32
  )
  query_positions = nonpad_so_far_inclusive[{"seq": -1}]
  key_value_positions = nonpad_so_far_exclusive.untag("seq").tag("kv_seq")
  # Tokens can attend to any kv-token position that they are after, as long as
  # it was NOT padding.
  attention_mask = pz.nx.nmap(jnp.logical_and)(
      query_positions >= key_value_positions,
      nonpad_mask.untag("seq").tag("kv_seq"),
  )
  # Run and update just like before, but add a tokens axis:
  out_logits, new_cache_state = model(
      sampling_mode.GemmaKVCachingInputs(
          tokens=next_token[{"seq": None}],
          positions=query_positions[{"seq": None}],
          attention_mask=attention_mask[{"seq": None}],
          sampling_state=state.kv_cache_state,
      )
  )
  # Extract the log probs from this token.
  final_token_log_probs = pz.nx.nmap(jax.nn.log_softmax)(
      out_logits.untag("seq").squeeze(0).untag("vocabulary")
  ).tag("vocabulary")
  return final_token_log_probs, SamplingState(
      kv_cache_state=new_cache_state,
      previous_tokens=updated_tokens,
      pad_id=state.pad_id,
  )


def temperature_sample_pyloop(
    model: sampling_mode.GemmaKVCachingTransformer,
    initial_cache_state: sampling_mode.GemmaKVCachingState,
    prompt: pz.nx.NamedArray,
    rng: jax.Array,
    pad_id: int,
    temperature: float = 1.0,
    max_sampling_steps: int | None = None,
) -> pz.nx.NamedArray:
  """Runs temperature sampling in a Python for loop.

  Args:
    model: The converted model we are running inference with.
    initial_cache_state: The initial cache state created while converting the
      model.
    prompt: A named array of prompt tokens. Must have a "seq" axis along which
      the tokens for each batch element are arranged. Should usually start with
      the beginning of sequence token. This function assumes (but does not
      check) that all non-padding tokens preceed all padding tokens.
    rng: JAX PRNGKey to use for sampling.
    pad_id: Token ID that corresponds to padding.
    temperature: Temperature to sample at.
    max_sampling_steps: Maximum number of sampling steps to run. If None,
      samples until filling up the key-value cache.

  Returns:
    A named array of continuations of the prompt.
  """
  if max_sampling_steps is None:
    max_sampling_steps = initial_cache_state.cache_len
  next_log_probs, sampling_state = prefill(
      model, initial_cache_state, prompt, pad_id
  )
  sampled_count = 0
  initial_cache_index = sampling_state.kv_cache_state.cache_end_index
  while True:
    kvcstate = sampling_state.kv_cache_state
    rng, key = jax.random.split(rng)
    # Split a key across named axes:
    batched_keys = pz.nx.random_split(key, kvcstate.batch_axes)
    next_token = pz.nx.nmap(jax.random.categorical)(
        batched_keys, next_log_probs.untag("vocabulary") / temperature
    )
    sampled_count += 1
    # Are we done?
    if (
        kvcstate.cache_end_index >= kvcstate.cache_len
        or sampled_count >= max_sampling_steps
    ):
      break
    next_log_probs, sampling_state = advance_one_token(
        model, sampling_state, next_token
    )
  # Add the last token we sampled (which doesn't need to be run through the
  # model).
  start = initial_cache_index
  end = sampling_state.kv_cache_state.cache_end_index
  final_written = pz.nx.concatenate(
      [
          sampling_state.previous_tokens[{"seq": pz.slice[start:end]}],
          next_token[{"seq": None}],
      ],
      "seq",
  )
  return final_written
