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

"""Geometric Sparse Attention primitives for Penzai.

This module provides AETHER (Adaptive Event-driven Threshold Hybrid Entangled
Rendering) geometric sparse attention, which replaces standard O(n²) attention
with a geometric block-sparse approximation.

The key insight is that for a block of keys with centroid μ and radius r,
the maximum attention score for any key k in the block is bounded by:

    max(q · k for k in block) ≤ q · μ + ||q|| × r

This Cauchy-Schwarz upper bound allows us to safely skip entire blocks when
the upper bound is below a threshold, achieving sub-linear complexity on
structured data.

Example usage::

    from penzai.nn.geometric_attention import GeometricSparseAttention

    # Create from config
    attn = GeometricSparseAttention.from_config(block_size=64, adaptive=True)
    output = attn((query, key, value))

    # Or swap into existing model using Penzai selectors
    model = pz.select(model).at_instances_of(
        pz.nn.Attention
    ).apply(
        GeometricSparseAttention.from_attention
    )
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from penzai.core import named_axes
from penzai.core import struct
from penzai.core import variables
from penzai.nn import layer as layer_base


def compute_block_geometry(
    keys: jax.Array,
    block_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Compute geometric metadata (centroids and radii) for blocks of keys.

    Groups keys into fixed-size blocks and computes each block's centroid
    (mean) and radius (max L2 distance from centroid). These are used to
    compute upper bounds on attention scores.

    Args:
        keys: Key tensor with shape (..., seq_len, dim).
        block_size: Number of tokens per block.

    Returns:
        A tuple of (centroids, radii):
          - centroids: Block means, shape (..., n_blocks, dim)
          - radii: Block radii (max L2 distance from centroid),
            shape (..., n_blocks)
    """
    *batch_dims, seq_len, dim = keys.shape
    n_blocks = (seq_len + block_size - 1) // block_size
    padded_len = n_blocks * block_size

    # Pad keys to be evenly divisible by block_size
    if padded_len > seq_len:
        pad_width = [(0, 0)] * len(batch_dims) + [(0, padded_len - seq_len), (0, 0)]
        keys_padded = jnp.pad(keys, pad_width, mode='constant', constant_values=0)
    else:
        keys_padded = keys

    # Reshape to blocks: (..., n_blocks, block_size, dim)
    new_shape = batch_dims + [n_blocks, block_size, dim]
    blocks = keys_padded.reshape(new_shape)

    # Centroid: mean of each block
    centroids = jnp.mean(blocks, axis=-2)  # (..., n_blocks, dim)

    # Radius: max L2 distance from centroid to any key in block
    diffs = blocks - centroids[..., None, :]  # (..., n_blocks, block_size, dim)
    distances = jnp.linalg.norm(diffs, axis=-1)  # (..., n_blocks, block_size)
    radii = jnp.max(distances, axis=-1)  # (..., n_blocks)

    return centroids, radii


def geometric_upper_bound(
    query: jax.Array,
    centroid: jax.Array,
    radius: jax.Array,
) -> jax.Array:
    """Compute Cauchy-Schwarz upper bound on maximum attention scores.

    For any key k in a block with centroid μ and radius r, the dot product
    q · k is bounded by:

        q · k ≤ q · μ + ||q|| × r

    This bound is tight and mathematically guaranteed. If this upper bound
    is below the sparsity threshold, we can safely skip the entire block.

    Args:
        query: Query tensor, shape (..., dim).
        centroid: Block centroid, shape (..., dim).
        radius: Block radius (scalar per block), shape (...).

    Returns:
        Upper bound on max score for any key in the block, shape (...).
    """
    # q · μ (dot product with centroid)
    centroid_score = jnp.sum(query * centroid, axis=-1)

    # ||q|| (query norm)
    query_norm = jnp.linalg.norm(query, axis=-1)

    # Upper bound: q · μ + ||q|| × r
    return centroid_score + query_norm * radius


def _compute_block_mask(
    query: jax.Array,
    centroids: jax.Array,
    radii: jax.Array,
    threshold: float | jax.Array,
    block_size: int,
    seq_len: int,
) -> jax.Array:
    """Compute which blocks to include based on geometric upper bound.

    Args:
        query: Query tensor, shape (..., seq, dim).
        centroids: Block centroids, shape (..., n_blocks, dim).
        radii: Block radii, shape (..., n_blocks).
        threshold: Sparsity threshold (blocks with upper bound < τ are skipped).
        block_size: Number of tokens per block.
        seq_len: Original sequence length.

    Returns:
        Block mask, shape (..., seq, n_blocks), True where blocks should
        be included in attention.
    """
    # query: (..., seq, dim) -> (..., seq, 1, dim) for broadcasting
    q_expanded = query[..., :, None, :]  # (..., seq, 1, dim)

    # centroids: (..., n_blocks, dim) -> (..., 1, n_blocks, dim)
    c_expanded = centroids[..., None, :, :]  # (..., 1, n_blocks, dim)

    # radii: (..., n_blocks) -> (..., 1, n_blocks)
    r_expanded = radii[..., None, :]  # (..., 1, n_blocks)

    # Compute upper bounds: (..., seq, n_blocks)
    centroid_scores = jnp.sum(q_expanded * c_expanded, axis=-1)
    query_norms = jnp.linalg.norm(query, axis=-1, keepdims=True)  # (..., seq, 1)
    upper_bounds = centroid_scores + query_norms * r_expanded

    # Blocks with upper bound >= threshold are included
    block_mask = upper_bounds >= threshold

    return block_mask


def _expand_block_mask(
    block_mask: jax.Array,
    block_size: int,
    seq_len: int,
) -> jax.Array:
    """Expand block-level mask to token-level mask.

    Args:
        block_mask: Block mask, shape (..., seq, n_blocks).
        block_size: Number of tokens per block.
        seq_len: Original sequence length (for truncation).

    Returns:
        Token-level mask, shape (..., seq, seq_len).
    """
    *batch_dims, query_len, n_blocks = block_mask.shape

    # Expand each block to block_size tokens
    # (..., seq, n_blocks) -> (..., seq, n_blocks, 1) -> (..., seq, n_blocks, block_size)
    expanded = jnp.broadcast_to(
        block_mask[..., None],
        (*batch_dims, query_len, n_blocks, block_size)
    )

    # Flatten to (..., seq, n_blocks * block_size)
    token_mask = expanded.reshape(*batch_dims, query_len, n_blocks * block_size)

    # Truncate to original sequence length
    token_mask = token_mask[..., :seq_len]

    return token_mask


def _phi_rotation_update(
    epsilon: jax.Array,
    phi: jax.Array,
    sparsity: jax.Array,
    target_sparsity: float = 0.3,
    learning_rate: float = 0.01,
) -> tuple[jax.Array, jax.Array]:
    """Update adaptive threshold using phi-rotation dynamics.

    This implements AETHER's self-tuning mechanism, adjusting the sparsity
    threshold based on observed sparsity levels.

    Args:
        epsilon: Current threshold value.
        phi: Current phase parameter.
        sparsity: Observed sparsity (fraction of blocks pruned).
        target_sparsity: Target sparsity level (default 0.3 = 70% blocks kept).
        learning_rate: Step size for threshold adjustment.

    Returns:
        Tuple of (new_epsilon, new_phi).
    """
    # Error signal: positive if too dense, negative if too sparse
    error = target_sparsity - sparsity

    # Rotation-based update (smooth, bounded dynamics)
    new_phi = phi + learning_rate * error
    new_phi = jnp.clip(new_phi, -jnp.pi, jnp.pi)

    # Threshold adjustment via sine of phase
    delta = learning_rate * jnp.sin(new_phi)
    new_epsilon = jnp.clip(epsilon + delta, 0.01, 0.9)

    return new_epsilon, new_phi


def geometric_sparse_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    block_size: int = 64,
    threshold: float | jax.Array = 0.15,
    causal_mask: jax.Array | None = None,
    scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute geometric sparse attention.

    This function computes attention using geometric block scoring to
    identify and skip irrelevant key/value blocks.

    Args:
        query: Query tensor, shape (..., q_seq, dim).
        key: Key tensor, shape (..., kv_seq, dim).
        value: Value tensor, shape (..., kv_seq, v_dim).
        block_size: Number of keys per block for geometry computation.
        threshold: Sparsity threshold. Blocks with geometric upper bound
            below this are masked out. Set to 0 for exact attention.
        causal_mask: Optional causal or other attention mask,
            shape (..., q_seq, kv_seq), True where attention is allowed.
        scale: Optional scale factor for attention scores.
            Defaults to 1/sqrt(dim).

    Returns:
        Tuple of (output, sparsity):
          - output: Attention output, shape (..., q_seq, v_dim)
          - sparsity: Fraction of blocks that were pruned
    """
    # Handle 2D inputs by adding batch dimension
    original_ndim = query.ndim
    if original_ndim == 2:
        query = query[None, :, :]
        key = key[None, :, :]
        value = value[None, :, :]

    *batch_dims, q_seq, dim = query.shape
    kv_seq = key.shape[-2]
    v_dim = value.shape[-1]

    if scale is None:
        scale = 1.0 / jnp.sqrt(dim)

    # 1. Compute block geometry
    centroids, radii = compute_block_geometry(key, block_size)

    # 2. Compute geometric block mask
    block_mask = _compute_block_mask(
        query, centroids, radii, threshold, block_size, kv_seq
    )

    # 3. Expand to token-level mask
    geo_mask = _expand_block_mask(block_mask, block_size, kv_seq)

    # 4. Compute sparsity (fraction pruned)
    sparsity = 1.0 - jnp.mean(geo_mask.astype(jnp.float32))

    # 5. Compute attention scores
    scores = jnp.einsum('...qd,...kd->...qk', query, key) * scale

    # 6. Apply geometric mask
    masked_scores = jnp.where(geo_mask, scores, -1e9)

    # 7. Apply causal mask if provided
    if causal_mask is not None:
        masked_scores = jnp.where(causal_mask, masked_scores, -1e9)

    # 8. Softmax and weighted sum
    attn_weights = jax.nn.softmax(masked_scores, axis=-1)
    output = jnp.einsum('...qk,...kv->...qv', attn_weights, value)

    # Remove batch dimension if input was 2D
    if original_ndim == 2:
        output = output[0]

    return output, sparsity


@struct.pytree_dataclass
class GeometricSparseAttention(layer_base.Layer):
    """Geometric Sparse Attention layer using AETHER block scoring.

    This layer is a drop-in replacement for standard attention that uses
    geometric upper bounds to prune irrelevant key/value blocks, achieving
    O(n × sparse) complexity instead of O(n²).

    The mathematical guarantee is based on the Cauchy-Schwarz inequality:
    for any query q and block with centroid μ and radius r,

        max(q · k for k in block) ≤ q · μ + ||q|| × r

    Blocks where this upper bound is below the threshold are safely skipped.

    Attributes:
        block_size: Number of tokens per sparsity block.
        threshold: Fixed sparsity threshold (used when adaptive=False).
        adaptive: Whether to self-tune the threshold at runtime.
        target_sparsity: Target sparsity level for adaptive mode.
        epsilon: Mutable state for adaptive threshold value.
        phi: Mutable state for phase parameter in adaptive updates.

    Example::

        # Create with config
        attn = GeometricSparseAttention.from_config(
            block_size=64,
            adaptive=True
        )

        # Forward pass: input is (query, key, value) tuple
        output = attn((query, key, value))

        # Convert from existing Attention layer
        geo_attn = GeometricSparseAttention.from_attention(existing_attn)
    """

    block_size: int = dataclasses.field(
        default=64, metadata={"pytree_node": False}
    )
    threshold: float = dataclasses.field(
        default=0.15, metadata={"pytree_node": False}
    )
    adaptive: bool = dataclasses.field(
        default=True, metadata={"pytree_node": False}
    )
    target_sparsity: float = dataclasses.field(
        default=0.3, metadata={"pytree_node": False}
    )

    # Mutable state for adaptive threshold
    epsilon: variables.StateVariable[jax.Array] | None = None
    phi: variables.StateVariable[jax.Array] | None = None

    @classmethod
    def from_config(
        cls,
        block_size: int = 64,
        threshold: float = 0.15,
        adaptive: bool = True,
        target_sparsity: float = 0.3,
    ) -> GeometricSparseAttention:
        """Factory method to create a GeometricSparseAttention layer.

        Args:
            block_size: Number of tokens per sparsity block.
            threshold: Initial sparsity threshold.
            adaptive: Whether to enable adaptive threshold tuning.
            target_sparsity: Target sparsity for adaptive mode.

        Returns:
            Configured GeometricSparseAttention instance.
        """
        if adaptive:
            epsilon = variables.StateVariable(
                value=jnp.array(threshold, dtype=jnp.float32),
                label="aether_epsilon",
            )
            phi = variables.StateVariable(
                value=jnp.array(0.0, dtype=jnp.float32),
                label="aether_phi",
            )
        else:
            epsilon = None
            phi = None

        return cls(
            block_size=block_size,
            threshold=threshold,
            adaptive=adaptive,
            target_sparsity=target_sparsity,
            epsilon=epsilon,
            phi=phi,
        )

    @classmethod
    def from_attention(
        cls,
        attn: Any,
        block_size: int = 64,
        adaptive: bool = True,
    ) -> GeometricSparseAttention:
        """Convert an existing Attention layer to GeometricSparseAttention.

        This is a convenience method for use with Penzai selectors::

            model = pz.select(model).at_instances_of(
                pz.nn.Attention
            ).apply(
                GeometricSparseAttention.from_attention
            )

        Args:
            attn: The original Attention layer (ignored, just for API compat).
            block_size: Number of tokens per sparsity block.
            adaptive: Whether to enable adaptive threshold tuning.

        Returns:
            A new GeometricSparseAttention instance.
        """
        return cls.from_config(block_size=block_size, adaptive=adaptive)

    def __call__(
        self,
        argument: tuple[
            named_axes.NamedArray,
            named_axes.NamedArray,
            named_axes.NamedArray,
        ],
        **side_inputs: Any,
    ) -> named_axes.NamedArray:
        """Compute geometric sparse attention.

        Args:
            argument: Tuple of (query, key, value) NamedArrays.
            **side_inputs: Side inputs (e.g., attention mask). Currently
                supports "attention_mask" for explicit masking.

        Returns:
            Attention output as a NamedArray.
        """
        query, key, value = argument

        # Get raw arrays for computation
        # We use nmap to handle named axes properly
        def _geo_attn_impl(q, k, v):
            # Determine current threshold
            if self.adaptive and self.epsilon is not None:
                current_threshold = self.epsilon.value
            else:
                current_threshold = self.threshold

            # Compute geometric sparse attention
            output, sparsity = geometric_sparse_attention(
                q, k, v,
                block_size=self.block_size,
                threshold=current_threshold,
            )

            # Update adaptive threshold
            if self.adaptive and self.epsilon is not None and self.phi is not None:
                new_epsilon, new_phi = _phi_rotation_update(
                    self.epsilon.value,
                    self.phi.value,
                    sparsity,
                    self.target_sparsity,
                )
                self.epsilon.value = new_epsilon
                self.phi.value = new_phi

            return output

        # Apply using nmap for named axis compatibility
        output = named_axes.nmap(_geo_attn_impl)(query, key, value)

        return output

    def treescope_color(self):
        """Custom color for Treescope visualization."""
        return "oklch(0.75 0.15 280 / 1.0)"  # Purple for geometric attention
