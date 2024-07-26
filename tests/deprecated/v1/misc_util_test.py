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
from penzai.deprecated.v1 import pz


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


if __name__ == "__main__":
  absltest.main()
