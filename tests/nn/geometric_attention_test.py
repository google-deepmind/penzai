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

"""Tests for geometric sparse attention."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from penzai import pz
from penzai.nn.geometric_attention import (
    GeometricSparseAttention,
    compute_block_geometry,
    geometric_upper_bound,
    geometric_sparse_attention,
)


def _standard_attention(query, key, value, scale=None):
    """Reference implementation of standard dense attention."""
    dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / jnp.sqrt(dim)
    scores = jnp.einsum('...qd,...kd->...qk', query, key) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum('...qk,...kv->...qv', weights, value)


class ComputeBlockGeometryTest(absltest.TestCase):

    def test_centroid_computation(self):
        """Verify centroids are correctly computed as block means."""
        # Create simple keys where blocks have known means
        keys = jnp.array([
            [1.0, 0.0],  # Block 0
            [3.0, 0.0],  # Block 0
            [5.0, 2.0],  # Block 1
            [5.0, 4.0],  # Block 1
        ])
        centroids, radii = compute_block_geometry(keys, block_size=2)

        expected_centroids = jnp.array([
            [2.0, 0.0],  # Mean of [1,0] and [3,0]
            [5.0, 3.0],  # Mean of [5,2] and [5,4]
        ])
        np.testing.assert_allclose(centroids, expected_centroids, rtol=1e-5)

    def test_radius_computation(self):
        """Verify radii are max L2 distance from centroid."""
        keys = jnp.array([
            [0.0, 0.0],
            [2.0, 0.0],  # Distance 1.0 from centroid [1,0]
        ])
        centroids, radii = compute_block_geometry(keys, block_size=2)

        # Centroid = [1, 0], both keys are distance 1.0 from it
        expected_radius = 1.0
        np.testing.assert_allclose(radii[0], expected_radius, rtol=1e-5)

    def test_batch_dimensions(self):
        """Verify block geometry works with batch dimensions."""
        batch_size = 3
        seq_len = 8
        dim = 4
        block_size = 2

        keys = jax.random.normal(jax.random.key(0), (batch_size, seq_len, dim))
        centroids, radii = compute_block_geometry(keys, block_size=block_size)

        expected_n_blocks = seq_len // block_size
        self.assertEqual(centroids.shape, (batch_size, expected_n_blocks, dim))
        self.assertEqual(radii.shape, (batch_size, expected_n_blocks))


class GeometricUpperBoundTest(absltest.TestCase):

    def test_upper_bound_never_violated(self):
        """Verify Cauchy-Schwarz bound is never violated."""
        key = jax.random.key(42)
        for _ in range(20):
            key, q_key, b_key = jax.random.split(key, 3)

            # Random query
            query = jax.random.normal(q_key, (16,))

            # Random block of keys
            block = jax.random.normal(b_key, (64, 16))
            centroid = jnp.mean(block, axis=0)
            diffs = block - centroid
            radius = jnp.max(jnp.linalg.norm(diffs, axis=-1))

            # Compute upper bound
            upper_bound = geometric_upper_bound(query, centroid, radius)

            # Compute actual max score
            actual_scores = jnp.dot(block, query)
            actual_max = jnp.max(actual_scores)

            # Upper bound must never be less than actual max
            self.assertGreaterEqual(
                float(upper_bound) + 1e-5,  # Small tolerance
                float(actual_max),
                "Cauchy-Schwarz upper bound was violated!"
            )


class GeometricSparseAttentionFunctionTest(absltest.TestCase):

    def test_exact_match_with_zero_threshold(self):
        """With threshold=0, geometric attention matches dense attention."""
        key = jax.random.key(123)
        q_key, k_key, v_key = jax.random.split(key, 3)

        query = jax.random.normal(q_key, (4, 32, 64))
        keys = jax.random.normal(k_key, (4, 128, 64))
        values = jax.random.normal(v_key, (4, 128, 64))

        # Standard attention
        expected = _standard_attention(query, keys, values)

        # Geometric attention with threshold=0 (no pruning)
        actual, sparsity = geometric_sparse_attention(
            query, keys, values,
            block_size=16,
            threshold=0.0,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        self.assertEqual(sparsity, 0.0)  # No blocks pruned

    def test_sparsity_on_clustered_data(self):
        """Verify sparsity is achieved on clustered data."""
        # Create highly clustered keys - attention should be sparse
        key = jax.random.key(456)

        # Queries in cluster 0 region
        query = jnp.ones((1, 4, 16)) * 10.0

        # Keys: cluster 0 (high values) and cluster 1 (low values)
        cluster_0 = jnp.ones((1, 64, 16)) * 10.0  # Similar to queries
        cluster_1 = jnp.ones((1, 64, 16)) * -10.0  # Very different

        keys = jnp.concatenate([cluster_0, cluster_1], axis=1)
        values = jnp.ones_like(keys)

        _, sparsity = geometric_sparse_attention(
            query, keys, values,
            block_size=16,
            threshold=0.5,
        )

        # Should prune some blocks (cluster 1 is far from queries)
        self.assertGreater(float(sparsity), 0.1)


class GeometricSparseAttentionLayerTest(absltest.TestCase):

    def test_from_config_creates_state(self):
        """Verify from_config initializes adaptive state correctly."""
        attn = GeometricSparseAttention.from_config(
            block_size=32,
            threshold=0.2,
            adaptive=True,
        )

        self.assertEqual(attn.block_size, 32)
        self.assertTrue(attn.adaptive)
        self.assertIsNotNone(attn.epsilon)
        self.assertIsNotNone(attn.phi)
        np.testing.assert_allclose(attn.epsilon.value, 0.2, rtol=1e-5)
        np.testing.assert_allclose(attn.phi.value, 0.0, rtol=1e-5)

    def test_from_config_no_state_when_not_adaptive(self):
        """Verify non-adaptive mode has no state variables."""
        attn = GeometricSparseAttention.from_config(
            block_size=64,
            adaptive=False,
        )

        self.assertFalse(attn.adaptive)
        self.assertIsNone(attn.epsilon)
        self.assertIsNone(attn.phi)

    def test_layer_is_pytree(self):
        """Verify the layer is a valid JAX pytree."""
        attn = GeometricSparseAttention.from_config(adaptive=True)

        # Flatten and unflatten should work
        leaves, treedef = jax.tree_util.tree_flatten(attn)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        self.assertEqual(reconstructed.block_size, attn.block_size)

    def test_jit_compatibility(self):
        """Verify the layer works under jax.jit."""
        attn = GeometricSparseAttention.from_config(
            block_size=16,
            adaptive=False,  # Non-adaptive for pure JIT
        )

        @jax.jit
        def forward(q, k, v):
            return attn((q, k, v))

        key = jax.random.key(789)
        q = jax.random.normal(key, (2, 16, 32))
        k = jax.random.normal(key, (2, 64, 32))
        v = jax.random.normal(key, (2, 64, 32))

        # Should compile and run without error
        output = forward(q, k, v)
        # Handle NamedArrayView output
        if hasattr(output, 'data_array'):
            out_shape = output.data_array.shape
        else:
            out_shape = output.shape
        self.assertEqual(out_shape, (2, 16, 32))

    def test_vmap_compatibility(self):
        """Verify the layer vectorizes correctly."""
        attn = GeometricSparseAttention.from_config(
            block_size=8,
            adaptive=False,
        )

        def forward(qkv):
            q, k, v = qkv
            return attn((q, k, v))

        key = jax.random.key(101)
        batch_size = 4
        q = jax.random.normal(key, (batch_size, 8, 16))
        k = jax.random.normal(key, (batch_size, 32, 16))
        v = jax.random.normal(key, (batch_size, 32, 16))

        # vmap over the batch dimension
        batched_forward = jax.vmap(forward)
        outputs = batched_forward((q, k, v))

        # Handle NamedArrayView output
        if hasattr(outputs, 'data_array'):
            out_shape = outputs.data_array.shape
        else:
            out_shape = outputs.shape
        self.assertEqual(out_shape, (batch_size, 8, 16))

    def test_treescope_color(self):
        """Verify custom treescope color is defined."""
        attn = GeometricSparseAttention.from_config()
        color = attn.treescope_color()
        self.assertIsInstance(color, str)
        self.assertIn("oklch", color)


class AdaptiveThresholdTest(absltest.TestCase):

    def test_adaptive_updates_state(self):
        """Verify adaptive mode updates epsilon and phi."""
        attn = GeometricSparseAttention.from_config(
            block_size=16,
            threshold=0.5,
            adaptive=True,
        )

        initial_epsilon = float(attn.epsilon.value)
        initial_phi = float(attn.phi.value)

        # Run forward pass
        key = jax.random.key(999)
        q = jax.random.normal(key, (8, 32))
        k = jax.random.normal(key, (64, 32))
        v = jax.random.normal(key, (64, 32))

        _ = attn((q, k, v))

        # State should have changed
        new_epsilon = float(attn.epsilon.value)
        new_phi = float(attn.phi.value)

        # At least one should change (unless sparsity exactly matches target)
        self.assertTrue(
            new_epsilon != initial_epsilon or new_phi != initial_phi,
            "Adaptive state should update after forward pass"
        )


if __name__ == "__main__":
    absltest.main()
