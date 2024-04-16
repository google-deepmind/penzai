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

"""Tests for model rewiring utilities."""

from absl.testing import absltest
import chex
import jax.numpy as jnp
from penzai import pz
from penzai.toolshed import model_rewiring


class ModelRewiringTest(absltest.TestCase):

  def test_knock_out_heads(self):
    layer = model_rewiring.KnockOutAttentionHeads(
        head_mask=pz.nx.wrap(jnp.array([1, 0])).tag("worlds")
    )
    in_arg = pz.nx.wrap(
        jnp.array([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0]],
        ])
    ).tag("heads", "seq", "kv_seq")
    result = layer(in_arg)
    chex.assert_trees_all_equal(
        result.canonicalize(),
        (
            pz.nx.wrap(
                jnp.array([
                    [
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        [[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0]],
                    ],
                    [
                        [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                        [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                    ],
                ])
            )
            .tag("worlds", "heads", "seq", "kv_seq")
            .canonicalize()
        ),
    )

  def test_rewire_computation_paths(self):
    layer = model_rewiring.RewireComputationPaths(
        worlds_axis="worlds",
        world_ordering=("a", "b", "c", "d", "e"),
        taking={
            "a": model_rewiring.From("a"),
            "b": model_rewiring.From("b", weight=0.5),
            "c": (),
            "d": model_rewiring.From("a"),
            "e": (
                model_rewiring.From(
                    "a", pz.nx.wrap(jnp.array([1, 0])).tag("switch")
                ),
                model_rewiring.From(
                    "b", pz.nx.wrap(jnp.array([0, 1])).tag("switch")
                ),
            ),
        },
    )
    in_arg = pz.nx.wrap(
        jnp.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ],
            jnp.float32,
        )
    ).tag("worlds", "features")
    result = layer(in_arg)
    chex.assert_trees_all_equal(
        result.canonicalize(),
        (
            pz.nx.wrap(
                jnp.array(
                    [
                        [
                            [1, 2, 3, 4],
                            [5 * 0.5, 6 * 0.5, 7 * 0.5, 8 * 0.5],
                            [0, 0, 0, 0],
                            [1, 2, 3, 4],
                            [1, 2, 3, 4],
                        ],
                        [
                            [1, 2, 3, 4],
                            [5 * 0.5, 6 * 0.5, 7 * 0.5, 8 * 0.5],
                            [0, 0, 0, 0],
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                        ],
                    ],
                    jnp.float32,
                )
            )
            .tag("switch", "worlds", "features")
            .canonicalize()
        ),
    )

  def test_linearize_and_adjust(self):
    def target(stuff):
      return {
          "a": stuff["a"] ** 2,
          "b": stuff["b"] ** 3,
      }

    layer = model_rewiring.LinearizeAndAdjust(
        linearize_around=lambda x: {
            "a": x,
            "b": pz.nx.wrap(jnp.array([1.0, 2.0, 3.0])).tag("foo"),
        },
        evaluate_at=lambda x: {
            "a": x + 1.0,
            "b": pz.nx.wrap(jnp.array([4.0, 5.0, 6.0])).tag("bar"),
        },
        target=target,
    )
    result = layer(10.0)
    chex.assert_trees_all_equal(
        result["a"], pz.nx.wrap(10.0**2 + 2 * 10.0 * 1.0)
    )
    chex.assert_trees_all_equal(
        result["b"].canonicalize(),
        (
            pz.nx.wrap(
                jnp.array([
                    [
                        1**3 + 3 * 1**2 * (4 - 1),
                        1**3 + 3 * 1**2 * (5 - 1),
                        1**3 + 3 * 1**2 * (6 - 1),
                    ],
                    [
                        2**3 + 3 * 2**2 * (4 - 2),
                        2**3 + 3 * 2**2 * (5 - 2),
                        2**3 + 3 * 2**2 * (6 - 2),
                    ],
                    [
                        3**3 + 3 * 3**2 * (4 - 3),
                        3**3 + 3 * 3**2 * (5 - 3),
                        3**3 + 3 * 3**2 * (6 - 3),
                    ],
                ], jnp.float32)
            )
            .tag("foo", "bar")
            .canonicalize()
        ),
    )


if __name__ == "__main__":
  absltest.main()
