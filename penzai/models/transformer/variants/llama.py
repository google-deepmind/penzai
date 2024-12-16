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

"""Llama architecture transformer variant."""

from __future__ import annotations

from typing import Any

from penzai.models.transformer import model_parts
from penzai.models.transformer.variants import llamalike_common


LlamaForCausalLM = Any


def llama_from_huggingface_model(
    model: LlamaForCausalLM,
    upcast_activations_to_float32: bool = False,
    use_layer_stack: bool = False,
) -> model_parts.TransformerLM:
  """Converts a HuggingFace Llama model to a Penzai model.

  This function converts Llama models from their HuggingFace
  implementations to Penzai. (Other models with the same architecture may also
  be supported if they use the same configuration, but this has not been
  tested.)

  Args:
    model: The HuggingFace Llama model.
    upcast_activations_to_float32: Whether to cast activations to float32 when
      the model runs. This allows analyzing activations at higher precision
      without consuming additional memory for parameters.
    use_layer_stack: Whether to use a layer stack for the decoder blocks.

  Returns:
    A Transformer model containing the loaded parameters.
  """
  try:
    import transformers  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise RuntimeError("HuggingFace transformers is not available") from exc

  if type(model) is not transformers.LlamaForCausalLM:  # pylint: disable=unidiomatic-typecheck
    raise ValueError(
        "llama_from_huggingface_model should be called with a"
        f" LlamaForCausalLM instance, but got {type(model).__name__}."
    )

  # Check any modified configuration arguments against the base config to make
  # sure we support newer architecture features. (Assumes that new features are
  # added in a backwards-compatible way and do not change the defaults for the
  # configuration class.)
  hf_config_attributes = model.config.to_dict()
  reference_attributes = transformers.LlamaConfig().to_dict()
  handled_or_ignored_attributes = {
      # Handled during conversion:
      "hidden_size",
      "intermediate_size",
      "num_attention_heads",
      "num_hidden_layers",
      "num_key_value_heads",
      "rms_norm_eps",
      "rope_theta",
      "vocab_size",
      # Ignored by conversion:
      "max_position_embeddings",
      "torch_dtype",
      "architectures",
      "bos_token_id",
      "eos_token_id",
      "_attn_implementation_autoset",
      "head_dim",
  }
  bad_attributes = {}
  for k, v in hf_config_attributes.items():
    if k in handled_or_ignored_attributes or (
        k in reference_attributes and v == reference_attributes[k]
    ):
      pass
    else:
      bad_attributes[k] = v
  if bad_attributes:
    raise ValueError(
        "Conversion of a LlamaForCausalLM does not support these configuration"
        f" attributes: {repr(bad_attributes)}"
    )

  return llamalike_common.llamalike_from_huggingface_model(
      model,
      upcast_activations_to_float32=upcast_activations_to_float32,
      use_layer_stack=use_layer_stack,
  )
