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

"""Mixture of Experts (MoE) dataflow combinators and computation layers.

This module implements the core building blocks for Mixture of Experts models,
following the same design philosophy as ``Attention``: a high-level dataflow
combinator (``MixtureOfExperts``) orchestrates named sub-components that can
be individually inspected and modified via ``pz.select``.

The module provides:

* ``MixtureOfExperts``: Dataflow combinator that connects routing and expert
  computation.
* ``MoETopKRouter``: Routes tokens to top-k experts via learned gating.
* ``MoEGatedExpertComputation``: Computes gated (GeGLU) expert outputs and
  combines them with routing weights.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from penzai.core import named_axes
from penzai.core import struct
from penzai.nn import layer as layer_base
from penzai.nn import parameters


@struct.pytree_dataclass
class MixtureOfExperts(layer_base.Layer):
  """A mixture-of-experts dataflow combinator.

  Routes input tokens to a subset of experts via a learned router, runs the
  selected experts, and combines their outputs with routing weights. This
  layer abstracts away the core MoE dataflow pattern, leaving the details
  of routing and expert computation to the sublayers.

  Follows the same design philosophy as ``Attention``: named sub-components
  that can be individually inspected and modified via ``pz.select``.

  Attributes:
    input_to_routing: Maps the input to a tuple of
      ``(routing_weights, expert_indices)``. ``routing_weights`` should have
      a ``"selected_experts"`` named axis, and ``expert_indices`` should be
      integer indices into the experts dimension.
    input_and_routing_to_output: Maps a tuple of
      ``(input, routing_weights, expert_indices)`` to the combined expert
      output.
  """

  input_to_routing: layer_base.Layer
  input_and_routing_to_output: layer_base.Layer

  def __call__(
      self, x: named_axes.NamedArray, **side_inputs: Any
  ) -> named_axes.NamedArray:
    """Runs the MoE computation.

    Args:
      x: The input to the computation, which will be routed to experts.
      **side_inputs: Side inputs for all sublayers.

    Returns:
      The combined output from the selected experts.
    """
    routing_weights, expert_indices = self.input_to_routing(x, **side_inputs)
    output = self.input_and_routing_to_output(
        (x, routing_weights, expert_indices), **side_inputs
    )
    return output

  def treescope_color(self):
    return "oklch(0.785 0.103 310 / 1.0)"


@struct.pytree_dataclass
class MoETopKRouter(layer_base.Layer):
  """Routes tokens to top-k experts via learned gating.

  Applies RMS standardization (without learned scale), multiplies by a
  learned per-dimension scale (with ``1/sqrt(embedding_dim)`` factor),
  projects to per-expert logits, takes softmax, selects top-k experts,
  renormalises the weights, and applies per-expert scale factors.

  Attributes:
    router_norm: RMS standardization layer (no learned scale).
    router_scale: Learned per-dimension scaling applied after normalisation.
    router_logits: Linear projection from embedding to expert logits.
    per_expert_scale: Learned per-expert output scaling factor.
    num_selected_experts: Number of experts selected per token (k).
    embedding_dim: Embedding dimension (for the ``1/sqrt(d)`` factor).
  """

  router_norm: layer_base.Layer
  router_scale: parameters.ParameterLike
  router_logits: layer_base.Layer
  per_expert_scale: parameters.ParameterLike
  num_selected_experts: int = dataclasses.field(
      metadata={"pytree_node": False}
  )
  embedding_dim: int = dataclasses.field(metadata={"pytree_node": False})

  def __call__(
      self, x: named_axes.NamedArray, **side_inputs: Any
  ) -> tuple[named_axes.NamedArray, named_axes.NamedArray]:
    """Computes routing weights and expert indices for each token.

    Args:
      x: Input activations with an ``"embedding"`` named axis.
      **side_inputs: Side inputs (forwarded to sublayers).

    Returns:
      A tuple of ``(routing_weights, expert_indices)``, each with a
      ``"selected_experts"`` named axis.
    """
    normed = self.router_norm(x, **side_inputs)
    inv_sqrt_dim = jnp.array(
        1.0 / jnp.sqrt(float(self.embedding_dim)), dtype=x.dtype
    )
    scaled = normed * self.router_scale.value * inv_sqrt_dim
    logits = self.router_logits(scaled, **side_inputs)

    k = self.num_selected_experts
    routing_weights, expert_indices = named_axes.nmap(
        functools.partial(_topk_route, k=k)
    )(logits.untag("experts"))

    # Apply per-expert scale to the routing weights.
    selected_scales = self.per_expert_scale.value[{"experts": expert_indices}]
    routing_weights = routing_weights * selected_scales

    return routing_weights.tag("selected_experts"), expert_indices.tag(
        "selected_experts"
    )

  def treescope_color(self):
    return "oklch(0.785 0.103 310 / 1.0)"


def _topk_route(
    logits: jax.Array, k: int
) -> tuple[jax.Array, jax.Array]:
  """Softmax → top-k → renormalize (pure function for use with nmap)."""
  probs = jax.nn.softmax(logits)
  top_weights, top_indices = jax.lax.top_k(probs, k)
  top_weights = top_weights / jnp.sum(top_weights)
  return top_weights, top_indices


@struct.pytree_dataclass
class MoEGatedExpertComputation(layer_base.Layer):
  """Computes gated expert outputs and combines them with routing weights.

  For each token, gathers the weights of its selected experts, runs a
  GeGLU MLP per expert, and sums the results weighted by routing weights.

  Expert weights are stored as batched ``NamedArray`` parameters with an
  ``"experts"`` axis. The initial implementation uses gather-based dispatch,
  which is simple and portable.

  Attributes:
    gating_weights: Expert gating parameter with named axes
      ``{"experts": E, "neurons": N, "embedding": D}``.
    value_weights: Expert value parameter with named axes
      ``{"experts": E, "neurons": N, "embedding": D}``.
    out_weights: Expert output parameter with named axes
      ``{"experts": E, "embedding": D, "neurons": N}``.
    num_selected_experts: Number of selected experts per token (must match
      the ``"selected_experts"`` axis size in routing inputs).
  """

  gating_weights: parameters.ParameterLike
  value_weights: parameters.ParameterLike
  out_weights: parameters.ParameterLike
  num_selected_experts: int = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def __call__(
      self,
      args_tuple: tuple[
          named_axes.NamedArray, named_axes.NamedArray, named_axes.NamedArray
      ],
      **side_inputs: Any,
  ) -> named_axes.NamedArray:
    """Computes the weighted sum of selected expert outputs.

    Args:
      args_tuple: A tuple of ``(input, routing_weights, expert_indices)``.
      **side_inputs: Side inputs (unused).

    Returns:
      Combined expert output with the same named axes as the input
      (minus ``"selected_experts"``).
    """
    x, routing_weights, expert_indices = args_tuple

    output = named_axes.nmap(
        functools.partial(
            _expert_dispatch,
            num_selected=self.num_selected_experts,
        )
    )(
        x.untag("embedding"),
        routing_weights.untag("selected_experts"),
        expert_indices.untag("selected_experts"),
        self.gating_weights.value.untag("experts", "neurons", "embedding"),
        self.value_weights.value.untag("experts", "neurons", "embedding"),
        self.out_weights.value.untag("experts", "embedding", "neurons"),
    ).tag("embedding")
    return output

  def treescope_color(self):
    return "oklch(0.785 0.103 310 / 1.0)"


def _expert_dispatch(
    x: jax.Array,
    routing_weights: jax.Array,
    expert_indices: jax.Array,
    gating_w: jax.Array,
    value_w: jax.Array,
    out_w: jax.Array,
    num_selected: int,
) -> jax.Array:
  """Per-token gather-based expert dispatch (pure function for nmap).

  Args:
    x: Input vector, shape ``(embedding_dim,)``.
    routing_weights: Routing weights, shape ``(num_selected,)``.
    expert_indices: Expert indices, shape ``(num_selected,)``.
    gating_w: All expert gating weights, shape
      ``(num_experts, neurons, embedding_dim)``.
    value_w: All expert value weights, shape
      ``(num_experts, neurons, embedding_dim)``.
    out_w: All expert output weights, shape
      ``(num_experts, embedding_dim, neurons)``.
    num_selected: Number of selected experts (loop bound).

  Returns:
    Combined expert output, shape ``(embedding_dim,)``.
  """
  result = jnp.zeros_like(x)
  for k in range(num_selected):
    idx = expert_indices[k]
    w = routing_weights[k]

    g = gating_w[idx]  # (neurons, embedding)
    v = value_w[idx]  # (neurons, embedding)
    o = out_w[idx]  # (embedding, neurons)

    gate_out = jax.nn.gelu(g @ x, approximate=True)  # (neurons,)
    val_out = v @ x  # (neurons,)
    expert_out = o @ (gate_out * val_out)  # (embedding,)

    result = result + w * expert_out
  return result
