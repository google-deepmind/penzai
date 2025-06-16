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

"""The Gemma architecture transformer variant.

Supports all the Gemma 1, Gemma 2 and Gemma 3 architectures. Based on the
Flax reference implementation at https://github.com/google-deepmind/gemma.

See the Gemma technical reports for more information:

* Gemma 1: https://arxiv.org/abs/2403.08295
* Gemma 2: https://arxiv.org/abs/2408.00118
* Gemma 3: https://arxiv.org/abs/2503.19786
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
from penzai import pz
from penzai.models.transformer import model_parts
from penzai.models.transformer.variants import llamalike_common


def _make_attention_layers_types(
    pattern: tuple[llamalike_common.AttentionType, ...],
    *,
    num_layers: int,
) -> tuple[llamalike_common.AttentionType, ...]:
  """Returns the list of attention types for every layers."""

  pattern_size = len(pattern)
  out = pattern * (num_layers // pattern_size)
  if num_layers % pattern_size != 0:
    out += pattern[: num_layers % pattern_size]
  return tuple(out)


_GEMMA_PRESETS = {
    "gemma_2b": dict(
        num_decoder_blocks=18,
        vocab_size=256_128,
        num_kv_heads=1,
        query_head_multiplier=8,
        embedding_dim=2048,
        projection_dim=256,
        mlp_hidden_dim=16_384,
    ),
    "gemma_7b": dict(
        num_decoder_blocks=28,
        vocab_size=256_128,
        num_kv_heads=16,
        query_head_multiplier=1,
        embedding_dim=3072,
        projection_dim=256,
        mlp_hidden_dim=24_576,
    ),
    "gemma2_2b": dict(
        num_decoder_blocks=26,
        vocab_size=256_128,
        num_kv_heads=4,
        query_head_multiplier=2,
        embedding_dim=2304,
        projection_dim=256,
        mlp_hidden_dim=9216,
        attention_type=(
            llamalike_common.AttentionTypeSlidingWindowCausal(4096),
            llamalike_common.AttentionTypeGlobalCausal(),
        ),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcap=30.0,
        attn_logits_soft_cap=50.0,
    ),
    "gemma2_9b": dict(
        num_decoder_blocks=42,
        vocab_size=256_128,
        num_kv_heads=8,
        query_head_multiplier=2,
        embedding_dim=3584,
        projection_dim=256,
        mlp_hidden_dim=14_336,
        attention_type=(
            llamalike_common.AttentionTypeSlidingWindowCausal(4096),
            llamalike_common.AttentionTypeGlobalCausal(),
        ),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcap=30.0,
        attn_logits_soft_cap=50.0,
    ),
    "gemma2_27b": dict(
        num_decoder_blocks=46,
        vocab_size=256_128,
        num_kv_heads=16,
        query_head_multiplier=2,
        embedding_dim=4608,
        projection_dim=128,
        mlp_hidden_dim=36_864,
        # query scaling factor: 1/sqrt(embedding_dim / num_query_heads)
        query_scaling_factor=(4608 // 32) ** -0.5,
        attention_type=(
            llamalike_common.AttentionTypeSlidingWindowCausal(4096),
            llamalike_common.AttentionTypeGlobalCausal(),
        ),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcap=30.0,
        attn_logits_soft_cap=50.0,
    ),
    "gemma3_1b": dict(
        num_decoder_blocks=26,
        vocab_size=262_144,
        num_kv_heads=1,
        query_head_multiplier=4,
        embedding_dim=1152,
        projection_dim=256,
        mlp_hidden_dim=6 * 1152,
        attention_type=_make_attention_layers_types(
            pattern=(llamalike_common.AttentionTypeSlidingWindowCausal(512),)
            * 5
            + (llamalike_common.AttentionTypeGlobalCausal(),),
            num_layers=26,
        ),
        use_qk_norm=True,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        rope_wavelength=1_000_000,
        local_rope_wavelength=10_000,
    ),
    "gemma3_4b": dict(
        num_decoder_blocks=34,
        vocab_size=262_144,
        num_kv_heads=4,
        query_head_multiplier=2,
        embedding_dim=2560,
        projection_dim=256,
        mlp_hidden_dim=2560 * 8 // 2,
        attention_type=_make_attention_layers_types(
            pattern=(llamalike_common.AttentionTypeSlidingWindowCausal(1024),)
            * 5
            + (llamalike_common.AttentionTypeGlobalCausal(),),
            num_layers=34,
        ),
        use_qk_norm=True,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        global_scale_factor=8.0,
        rope_wavelength=1_000_000,
        local_rope_wavelength=10_000,
    ),
    "gemma3_12b": dict(
        num_decoder_blocks=48,
        vocab_size=262_144,
        num_kv_heads=8,
        query_head_multiplier=2,
        embedding_dim=30 * 128,
        projection_dim=256,
        mlp_hidden_dim=8 * 30 * 128 // 2,
        attention_type=_make_attention_layers_types(
            pattern=(llamalike_common.AttentionTypeSlidingWindowCausal(1024),)
            * 5
            + (llamalike_common.AttentionTypeGlobalCausal(),),
            num_layers=48,
        ),
        use_qk_norm=True,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        global_scale_factor=8.0,
        rope_wavelength=1_000_000,
        local_rope_wavelength=10_000,
    ),
    "gemma3_27b": dict(
        num_decoder_blocks=62,
        vocab_size=262_144,
        num_kv_heads=16,
        query_head_multiplier=2,
        embedding_dim=5376,
        projection_dim=128,
        mlp_hidden_dim=5376 * 8 // 2,
        # query scaling factor: 1/sqrt(embedding_dim / num_query_heads)
        query_scaling_factor=(5376 // 32) ** -0.5,
        attention_type=_make_attention_layers_types(
            pattern=(llamalike_common.AttentionTypeSlidingWindowCausal(1024),)
            * 5
            + (llamalike_common.AttentionTypeGlobalCausal(),),
            num_layers=62,
        ),
        use_qk_norm=True,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        global_scale_factor=8.0,
        rope_wavelength=1_000_000,
        local_rope_wavelength=10_000,
    ),
}
_NEEDS_GATING_TRANSPOSE = {
    "gemma_2b": False,
    "gemma_7b": False,
    "gemma2_2b": False,
    "gemma2_9b": True,
    "gemma2_27b": True,
    "gemma3_1b": True,
    "gemma3_4b": True,
    "gemma3_12b": True,
    "gemma3_27b": True,
}


def gemma_from_pretrained_checkpoint(
    ckpt_params: dict[str, Any],
    upcast_activations_to_float32: bool = False,
    use_layer_stack: bool = False,
    preset_name: Literal[
        "gemma_2b",
        "gemma_7b",
        "gemma2_2b",
        "gemma2_9b",
        "gemma2_27b",
        "gemma3_1b",
        "gemma3_4b",
        "gemma3_12b",
        "gemma3_27b",
        "auto",
    ] = "auto",
) -> model_parts.TransformerLM:
  """Builds a Gemma model from a pretrained checkpoint.

  The parameters of the loaded ``Transformer`` will be close to those in
  the original checkpoint with a few modifications:

  * Query, key, and value heads are stored in three separate matrices instead
    of being stored either as a single matrix (qkv_einsum) or as two (q_einsum
    and kv_einsum).

  * `RMSLayerNorm` weights have their values increased by one, instead of
    adding one at call time.

  * Axes of parameters are identified by name instead of by position.

  Args:
    ckpt_params: Nested dictionary of weights from the Gemma checkpoint.
    upcast_activations_to_float32: Whether to cast activations to float32 when
      the model runs. This allows analyzing activations at higher precision
      without consuming additional memory for parameters.
    use_layer_stack: Whether to use a layer stack for the decoder blocks.
    preset_name: Preset name, used to determine model config. If "auto", uses
      the number of layers and whether the model needs qk norm in the checkpoint
      to determine the configuration.

  Returns:
    A Transformer model containing the loaded parameters.
  """
  params = {k.removeprefix("transformer/"): v for k, v in ckpt_params.items()}

  if preset_name == "auto":
    num_layers = 0
    while f"layer_{num_layers}/mlp/linear" in params:
      num_layers += 1
    if (
        "layer_0/attn/_query_norm" in params
        and "layer_0/attn/_key_norm" in params
    ):
      qk_norm = True
    else:
      qk_norm = False
    is_match = False
    for gemma_preset_name, kwargs in _GEMMA_PRESETS.items():
      if kwargs["num_decoder_blocks"] == num_layers:
        if qk_norm and "use_qk_norm" in kwargs:
          if kwargs["use_qk_norm"]:
            is_match = True
            preset_name = gemma_preset_name
            break
        if (not qk_norm) and ("use_qk_norm" not in kwargs):
          is_match = True
          preset_name = gemma_preset_name
          break
    if not is_match:
      raise ValueError(
          f"Could not determine preset for model with {num_layers} layers and"
          f" qk norm {qk_norm}."
      )

  preset_kwargs = _GEMMA_PRESETS[preset_name]
  preset_needs_gating_transpose = _NEEDS_GATING_TRANSPOSE[preset_name]

  parameter_dtype = params["layer_0/attn/attn_vec_einsum"]["w"].dtype

  if upcast_activations_to_float32:
    activation_dtype = jnp.float32
  else:
    activation_dtype = parameter_dtype

  config = llamalike_common.LlamalikeTransformerConfig(
      **preset_kwargs,
      parameter_dtype=parameter_dtype,
      mlp_variant="geglu_approx",
      tie_embedder_and_logits=True,
      activation_dtype=activation_dtype,
      use_layer_stack=use_layer_stack,
  )
  model_def = llamalike_common.build_llamalike_transformer(
      config, init_base_rng=None, name="transformer"
  )
  parameter_mapping = {
      "embedder.embeddings": pz.nx.NamedArray.wrap(
          params["embedder"]["input_embedding"]
      ).tag("vocabulary", "embedding"),
      "final_norm/scale.weights": pz.nx.NamedArray.wrap(
          1 + params["final_norm"]["scale"]
      ).tag("embedding"),
  }

  all_block_params = []

  for i in range(config.num_decoder_blocks):
    cur_block_params = {}
    all_block_params.append(cur_block_params)

    cur_block_params["pre_attention_norm/scale.weights"] = (
        pz.nx.NamedArray.wrap(
            1 + params[f"layer_{i}/pre_attention_norm"]["scale"]
        ).tag("embedding")
    )
    # Add qk norm if needed
    if config.use_qk_norm:
      cur_block_params["attention/query_norm/scale.weights"] = (
          pz.nx.NamedArray.wrap(
              1 + params[f"layer_{i}/attn/_query_norm"]["scale"]
          ).tag("projection")
      )
      cur_block_params["attention/key_norm/scale.weights"] = (
          pz.nx.NamedArray.wrap(
              1 + params[f"layer_{i}/attn/_key_norm"]["scale"]
          ).tag("projection")
      )

    if config.use_post_attn_norm:
      cur_block_params["post_attention_norm/scale.weights"] = (
          pz.nx.NamedArray.wrap(
              1 + params[f"layer_{i}/post_attention_norm"]["scale"]
          ).tag("embedding")
      )

    cur_block_params["pre_ffw_norm/scale.weights"] = pz.nx.NamedArray.wrap(
        1 + params[f"layer_{i}/pre_ffw_norm"]["scale"]
    ).tag("embedding")
    if config.use_post_ffw_norm:
      cur_block_params["post_ffw_norm/scale.weights"] = pz.nx.NamedArray.wrap(
          1 + params[f"layer_{i}/post_ffw_norm"]["scale"]
      ).tag("embedding")

    gating_einsum_w = params[f"layer_{i}/mlp/gating_einsum"]["w"]
    if preset_needs_gating_transpose:
      gating_einsum_w = gating_einsum_w.transpose((0, 2, 1))
    cur_block_params["mlp/gating_linear.weights"] = pz.nx.NamedArray.wrap(
        gating_einsum_w[0]
    ).tag("embedding", "neurons")
    cur_block_params["mlp/value_linear.weights"] = pz.nx.NamedArray.wrap(
        gating_einsum_w[1]
    ).tag("embedding", "neurons")

    cur_block_params["mlp/out_linear.weights"] = pz.nx.NamedArray.wrap(
        params[f"layer_{i}/mlp/linear"]["w"]
    ).tag("neurons", "embedding")

    if config.num_kv_heads == 1:
      cur_block_params["attention/query.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/q_einsum"]["w"]
      ).tag("query_heads", "embedding", "projection")
      cur_block_params["attention/key.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/kv_einsum"]["w"][0].squeeze(0)
      ).tag("embedding", "projection")
      cur_block_params["attention/value.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/kv_einsum"]["w"][1].squeeze(0)
      ).tag("embedding", "projection")
      cur_block_params["attention/output.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/attn_vec_einsum"]["w"]
      ).tag("query_heads", "projection", "embedding")
    elif config.query_head_multiplier == 1:
      cur_block_params["attention/query.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/qkv_einsum"]["w"][0]
      ).tag("heads", "embedding", "projection")
      cur_block_params["attention/key.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/qkv_einsum"]["w"][1]
      ).tag("heads", "embedding", "projection")
      cur_block_params["attention/value.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/qkv_einsum"]["w"][2]
      ).tag("heads", "embedding", "projection")
      cur_block_params["attention/output.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/attn_vec_einsum"]["w"]
      ).tag("heads", "projection", "embedding")
    else:
      # Grouped query attention: split attention heads into groups.
      cur_block_params["attention/key.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/kv_einsum"]["w"][0]
      ).tag("head_groups", "embedding", "projection")
      cur_block_params["attention/value.weights"] = pz.nx.NamedArray.wrap(
          params[f"layer_{i}/attn/kv_einsum"]["w"][1]
      ).tag("head_groups", "embedding", "projection")

      q_weights = params[f"layer_{i}/attn/q_einsum"]["w"]
      out_weights = params[f"layer_{i}/attn/attn_vec_einsum"]["w"]
      cur_block_params["attention/query.weights"] = pz.nx.NamedArray.wrap(
          q_weights.reshape((
              config.num_kv_heads,
              config.query_head_multiplier,
              config.embedding_dim,
              config.projection_dim,
          ))
      ).tag("head_groups", "query_heads", "embedding", "projection")
      cur_block_params["attention/output.weights"] = pz.nx.NamedArray.wrap(
          out_weights.reshape((
              config.num_kv_heads,
              config.query_head_multiplier,
              config.projection_dim,
              config.embedding_dim,
          ))
      ).tag("head_groups", "query_heads", "projection", "embedding")

  if use_layer_stack:
    for key in all_block_params[0].keys():
      vals = [
          all_block_params[i][key] for i in range(config.num_decoder_blocks)
      ]
      parameter_mapping[f"blocks/{key}"] = pz.nx.stack(vals, "blocks")
  else:
    for i in range(config.num_decoder_blocks):
      for key, value in all_block_params[i].items():
        parameter_mapping[f"block_{i}/{key}"] = value

  # Create parameter objects for each parameter, and bind them to the model's
  # slots.
  model = pz.bind_variables(
      model_def,
      [
          pz.Parameter(value=v, label=f"transformer/{k}")
          for k, v in parameter_mapping.items()
      ],
  )
  pz.nn.assert_no_parameter_slots(model)
  return model
