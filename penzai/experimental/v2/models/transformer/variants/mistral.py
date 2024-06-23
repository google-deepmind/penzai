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

"""Mistral architecture transformer variant."""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
from penzai.experimental.v2 import pz
from penzai.experimental.v2.models.transformer import model_parts
from penzai.experimental.v2.models.transformer.variants import llamalike_common


MistralForCausalLM = Any


def mistral_from_huggingface_model(
    model: MistralForCausalLM,
    upcast_activations_to_float32: bool = False,
    use_layer_stack: bool = False,
) -> model_parts.Transformer:
  """Converts a HuggingFace Mistral model to a Penzai model.

  This function converts Mistral models from their HuggingFace
  implementations to Penzai. (Other models with the same architecture may also
  be supported if they use the same configuration, but this has not been
  tested.)

  Note: Mistral models use sliding window attention. Penzai does not compute
  this attention mask automatically, so you will have to manually adjust the
  mask if using an input longer than Mistral's sliding window. (Additionally,
  the KV-cache logic does not currently support sliding-window attention.)

  Args:
    model: The HuggingFace Mistral model.
    upcast_activations_to_float32: Whether to cast activations to float32 when
      the model runs. This allows analyzing activations at higher precision
      without consuming additional memory for parameters.
    use_layer_stack: Whether to use a layer stack for the decoder blocks.

  Returns:
    A Transformer model containing the loaded parameters.
  """
  if type(model).__name__ != "MistralForCausalLM":
    raise ValueError(
        "mistral_from_huggingface_model should be called with a"
        f" MistralForCausalLM instance, but got {type(model).__name__}."
    )
  # Checkpoint conversion assumes these configuration arguments are set:
  hf_config = model.config
  checked_config_args = dict(
      hidden_act="silu",
      rms_norm_eps=1e-6,
      tie_word_embeddings=False,
      attention_dropout=0.0,
  )
  for k, v in checked_config_args.items():
    actual_value = getattr(hf_config, k)
    if actual_value != v:
      raise ValueError(
          f"Conversion of a LlamaForCausalLM requires config.{k}={repr(v)}, but"
          f" got {actual_value}"
      )

  if model.config.sliding_window is not None:
    warnings.warn(
        "Sliding window attention is not directly supported in Penzai. Training"
        " or scoring mode should work with a manually-constructed attention"
        " mask, but decoding may not work correctly in key-value cache"
        " decoding mode!"
    )

  return llamalike_common.llamalike_from_huggingface_model(
      model,
      upcast_activations_to_float32=upcast_activations_to_float32,
      use_layer_stack=use_layer_stack,
  )
