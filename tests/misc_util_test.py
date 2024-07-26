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

"""Tests for miscellaneous self-contained utility objects."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from penzai import pz
from penzai.core import tree_util


class TreeUtilTest(absltest.TestCase):
  """Tests for core/tree_util.py."""

  def test_tree_flatten_exactly_one_level_nested_pytree(self):
    result = tree_util.tree_flatten_exactly_one_level(
        {"a": [1, 2], "b": [3, 4]}
    )
    self.assertIsNotNone(result)
    keys_and_subtrees, treedef = result
    self.assertEqual(
        keys_and_subtrees,
        [
            (jax.tree_util.DictKey("a"), [1, 2]),
            (jax.tree_util.DictKey("b"), [3, 4]),
        ],
    )
    self.assertEqual(treedef.unflatten(["x", "y"]), {"a": "x", "b": "y"})

  def test_tree_flatten_exactly_one_level_leaf(self):
    self.assertIsNone(tree_util.tree_flatten_exactly_one_level(42))


class RandomStreamTest(absltest.TestCase):

  def test_random_stream_lifecycle(self):
    root_key = jax.random.PRNGKey(0)
    stream = pz.RandomStream(root_key)
    self.assertEqual(stream.state, "pending")

    with self.subTest("pending"):
      with self.assertRaisesRegex(
          ValueError, "Cannot use a random stream that is not yet active"
      ):
        _ = stream.next_key()

    with self.subTest("active"):
      with stream:
        self.assertEqual(stream.state, "active")
        key_1 = stream.next_key()
        self.assertTrue(jnp.array_equal(key_1, jax.random.fold_in(root_key, 0)))
        key_2 = stream.next_key()
        self.assertTrue(jnp.array_equal(key_2, jax.random.fold_in(root_key, 1)))

    with self.subTest("expired"):
      with self.assertRaisesRegex(
          ValueError, "Cannot use a random stream that has expired"
      ):
        _ = stream.next_key()

  def test_random_stream_mark_active(self):
    root_key = jax.random.PRNGKey(0)
    stream = pz.RandomStream(root_key)
    stream.unsafe_mark_active()
    self.assertEqual(stream.state, "active")
    key = stream.next_key()
    self.assertTrue(jnp.array_equal(key, jax.random.fold_in(root_key, 0)))


class SliceLikeTest(absltest.TestCase):

  def test_slice(self):
    indexer = pz.slice[1, 2::10, :3, "foo"]
    self.assertEqual(indexer, (1, slice(2, None, 10), slice(None, 3), "foo"))


if __name__ == "__main__":
  absltest.main()
