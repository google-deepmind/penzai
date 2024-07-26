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

"""Helper classes for rewiring, ablating, and intervening on model activations.

These helpers are intended to be inserted into a model to enable analysis of the
causal impact of different model components. For instance, they can be used to
ablate attention heads, to implement activation patching, or to linearize parts
of a model for easier comparisons.

For an example of how to use these components, see the
:doc:`induction heads tutorial notebook </notebooks/induction_heads>`.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from penzai import pz


@pz.pytree_dataclass
class KnockOutAttentionHeads(pz.nn.Layer):
  """Layer that redirects masked-out heads to attend to the ``<BOS>`` token.

  This layer can be inserted into a tramsformer model's attention layer
  immediately after the softmax operation, in order to ablate a subset of the
  attention heads. It assumes that a reasonable "default" behavior for the
  head is to attend to the ``<BOS>`` token, which is common for many attention
  heads. (This ablation may be less effective for heads that never attend to
  ``BOS``.)

  Attributes:
    head_mask: NamedArray with 1s for heads we want to keep, and 0s for heads
      that should be rewritten to point to ``BOS``. Values between 0 and 1 will
      smoothly interpolate between them.
  """

  head_mask: pz.nx.NamedArray

  def __call__(
      self, attn_weights: pz.nx.NamedArray, **_unused_side_inputs
  ) -> pz.nx.NamedArray:
    knocked_out_attn = pz.nx.wrap(
        jnp.zeros(
            [attn_weights.named_shape["kv_seq"]],
            attn_weights.dtype,
        )
        .at[0]
        .set(1.0)
    ).tag("kv_seq")
    return knocked_out_attn + self.head_mask * (attn_weights - knocked_out_attn)


@dataclasses.dataclass(frozen=True)
class From:
  """A connection between two parallel computations.

  This class identifies a source "world" and combines it with a weight value
  or array. This is used to indicate how to rewire connections using
  `RewireComputationPaths`.

  Attributes:
    source: The parallel world name this connection reads from.
    weight: The weight of this connection. Should either be a scalar float, or a
      `pz.nx.NamedArray` with a scalar positional shape. If a NamedArray, the
      named axes will indicate different connection behavior for each named axis
      (e.g. if `weight` has a "heads" axis, a different weight will be used for
      each head of the input activation).
  """

  source: str
  weight: float | pz.nx.NamedArray = 1.0


@pz.pytree_dataclass
class RewireComputationPaths(pz.nn.Layer):
  """Rewires computation across parallel model runs along a worlds axis.

  This layer can be used to implement sophisticated ablation, activation
  patching, and path patching analyses. It assumes its input activation has
  a particular "worlds" axis, which indicates a minibatch of examples that
  represent counterfactual variants of the input or of the model. It then
  re-writes the activations by coping activations between the worlds according
  to its weights.

  This layer is intended to be directly inserted into the model at the point
  where you want to "bridge" the parallel worlds. For instance, if you want to
  patch the value activations from an attention block, you can insert this
  inside the `input_to_value` sublayer of the attention block, and configure it
  to copy the desired part of the input. If you want to freeze the attention
  patterns of a set of blocks, you can insert this after the attention softmax,
  and configure it to copy from the "original" world instead of the current
  world. You can also use a length-zero tuple to indicate that the value should
  be entirely dropped and zeroed out; this can be useful for e.g. disabling
  writes to the residual stream of a transformer.

  One useful pattern you can use is to insert a `RewireComputationPaths` block
  into a `LinearizeAndAdjust` block's ``linearize_around`` attribute. This will
  allow you to linearize a nonlinear operation around a single world's input,
  but evaluate the linear approximation around each world individually.

  Another useful pattern is to either rewire or zero out the contributions of
  layers to the residual stream, or rewire the inputs of those layers when they
  read from the residual stream. This can allow you to measure the residual
  vectors flowing from one layer to another, or to measure the direct
  contribution of a layer to the final output logits ignoring the reads or
  writes from other layers.

  Note that this layer is designed to be used for "batched rewiring": all of the
  different input conditions are run through the model in a single batched
  forward pass. For instance, you might have a *clean* world where nothing is
  ablated and all rewiring blocks read back from the *clean* world (a no-op)
  and a *corrupted* world where some activations are rewired to be copied from
  the *clean* world, where these worlds map to indices 0 and 1 along a "worlds"
  axis of the input. This is a declarative alternative to running multiple
  forward passes and saving/restoring activations from a cache. The batched
  rewiring version is easier to express in Penzai due to being a stateless
  function, and may reduce memory and compute overhead from saving many small
  forward passes. It can also be compiled into a single JIT-ted computation
  using `jax.jit` (which is even easier if you use
  `penzai.toolshed.jit_wrapper`).

  Attributes:
    worlds_axis: Axis name of the "worlds" axis (often just "worlds"). Should be
      an axis name that is NOT already used by the model.
    world_ordering: A tuple of world names. We assume the input will have a
      worlds axis of the same length as this tuple.
    taking: A dictionary that maps destination world names to a source or
      sources that those worlds should read from. The keys should exactly match
      ``world_ordering`` and represent each world we are outputting to. The
      values should be instances of `From` or tuples of instances of `From`,
      determining where the value for each world should be taken from. If the
      ``taking`` key and `From` source are the same, and the weight is 1, this
      represents a no-op. A common pattern is to have clean or unablated worlds
      read from themselves, but ablated or corrupted worlds take from the clean
      worlds.
  """

  worlds_axis: str = dataclasses.field(metadata={"pytree_node": False})
  world_ordering: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  taking: dict[str, From | tuple[From, ...]] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def path_matrix(self) -> pz.nx.NamedArray:
    """Builds a matrix that maps "from" indices to their "to" indices."""
    result = [[0 for _ in self.world_ordering] for _ in self.world_ordering]
    assert len(self.taking) == len(self.world_ordering)
    assert set(self.taking.keys()) == set(self.world_ordering)
    for dest, connections in self.taking.items():
      if isinstance(connections, From):
        connections = (connections,)
      for connection in connections:
        from_ix = self.world_ordering.index(connection.source)
        to_ix = self.world_ordering.index(dest)
        result[to_ix][from_ix] += connection.weight
    # Allows the weights to be named arrays by `nmap`ing the array constructor.
    # This works as long as each weight had an empty positional shape.
    return pz.nx.nmap(jnp.array)(result)

  def __call__(
      self, inputs: pz.nx.NamedArray, **_unused_side_inputs
  ) -> pz.nx.NamedArray:
    mat = self.path_matrix().astype(inputs.dtype)
    rewired = pz.nx.nmap(jnp.dot)(mat, inputs.untag(self.worlds_axis))
    return rewired.tag(self.worlds_axis)


@pz.pytree_dataclass
class LinearizeAndAdjust(pz.nn.Layer):
  """Linearizes and evaluates a model around two adjusted inputs.

  This layer splits its input into two paths, and allows each path to be
  adjusted independently. Then, these two inputs are used to construct and
  evaluate a first-order approximation of the target layer: the first adjusted
  input is used as the linearization point, and the second adjusted input is
  used as the point of evaluation.

  If ``linearize_around`` and ``evaluate_at`` are the same, this will behave
  the same as an ordinary sequence of operations, since evaluating a linear
  function at the linearization point is the same as evaluating the target
  function normally.
  """

  linearize_around: pz.nn.Layer
  evaluate_at: pz.nn.Layer
  target: pz.nn.Layer

  def __call__(self, inputs, **side_inputs):
    primal_point = self.linearize_around(inputs, **side_inputs)
    eval_point = self.evaluate_at(inputs, **side_inputs)
    primal_leaves, primal_structure = jax.tree.flatten(
        primal_point, is_leaf=pz.nx.is_namedarray
    )
    eval_leaves, eval_structure = jax.tree.flatten(
        eval_point, is_leaf=pz.nx.is_namedarray
    )
    assert primal_structure == eval_structure

    # f(eval) ~= f(primal) + (eval-primal) f'(primal)
    primal_leaves_bc = []
    tangent_leaves = []
    for primal, eval_leaf in zip(primal_leaves, eval_leaves):
      if not isinstance(primal, pz.nx.NamedArrayBase):
        primal = pz.nx.wrap(primal)
      if not isinstance(eval_leaf, pz.nx.NamedArrayBase):
        eval_leaf = pz.nx.wrap(eval_leaf)
      dtype = jnp.result_type(primal, eval_leaf)
      tangent = eval_leaf - primal
      # Make sure named array structures exactly match.
      primal = primal.broadcast_like(eval_leaf).astype(dtype)
      tangent = tangent.order_like(primal).astype(dtype)
      primal_leaves_bc.append(primal)
      tangent_leaves.append(tangent)

    def go(x):
      return pz.freeze_variables(self.target)(
          x, **pz.freeze_variables(side_inputs)
      )

    primal_out, tangent_out = jax.jvp(
        go,
        (primal_structure.unflatten(primal_leaves_bc),),
        (primal_structure.unflatten(tangent_leaves),),
    )
    return jax.tree_util.tree_map(
        lambda p_out, t_out: p_out + t_out,
        primal_out,
        tangent_out,
        is_leaf=pz.nx.is_namedarray,
    )
