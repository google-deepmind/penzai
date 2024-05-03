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

"""Tests for automatic nmap wrapper."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from penzai import pz
from penzai.toolshed import auto_nmap


class AutoNmapTest(absltest.TestCase):

  def test_auto_nmap(self):
    njax = auto_nmap.wrap_module(jax)
    njnp = auto_nmap.wrap_module(jnp)

    with self.subTest("array"):
      arr = njnp.array([1, 2, 3])
      self.assertIsInstance(arr, pz.nx.NamedArrayBase)
      self.assertEqual(arr.positional_shape, (3,))
      self.assertEqual(dict(arr.named_shape), {})

    with self.subTest("linspace"):
      arr = njnp.linspace(-0.5, 0.5, 10).tag("bar")
      self.assertIsInstance(arr, pz.nx.NamedArrayBase)
      self.assertEqual(arr.positional_shape, ())
      self.assertEqual(dict(arr.named_shape), {"bar": 10})

    with self.subTest("uniform"):
      arr = njax.random.uniform(jax.random.key(0), (5, 10)).tag("baz", "qux")
      self.assertEqual(dict(arr.named_shape), {"baz": 5, "qux": 10})

    with self.subTest("top_k"):
      arr = njnp.array([[1, 2, 3], [5, 6, 4]]).tag("foo", "bar")
      topk_vals, topk_ixs = njax.lax.top_k(arr.untag("bar"), k=2)

      chex.assert_trees_all_equal(
          topk_vals.canonicalize(),
          (
              pz.nx.wrap([[3, 2], [6, 5]])
              .tag("foo", "bar")
              .untag("bar")
              .canonicalize()
          ),
      )
      chex.assert_trees_all_equal(
          topk_ixs.canonicalize(),
          (
              pz.nx.wrap([[2, 1], [1, 0]])
              .tag("foo", "bar")
              .untag("bar")
              .canonicalize()
          ),
      )


if __name__ == "__main__":
  absltest.main()
