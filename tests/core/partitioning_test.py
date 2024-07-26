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

"""Tests for partitioning utilities."""

from absl.testing import absltest
from penzai import pz


class PartitioningTest(absltest.TestCase):

  def test_partition_by_selection(self):
    my_object = {
        "a": {"a1": 1, "a2": 2},
        "b": {"b1": 3, "b2": 4},
        "c": {"c1": 5, "c2": 6},
    }
    selection = pz.select(my_object).at(
        lambda root: (root["a"], root["b"]["b1"]), multiple=True
    )

    with self.subTest("at_selected"):
      left, right = selection.partition()
      self.assertEqual(
          left,
          {
              "a": {"a1": 1, "a2": 2},
              "b": {"b1": 3, "b2": pz.NotInThisPartition()},
              "c": pz.NotInThisPartition(),
          },
      )
      self.assertEqual(
          right,
          {
              "a": pz.NotInThisPartition(),
              "b": {"b1": pz.NotInThisPartition(), "b2": 4},
              "c": {"c1": 5, "c2": 6},
          },
      )

    with self.subTest("at_leaves"):
      left, right = selection.partition(at_leaves=True)
      self.assertEqual(
          left,
          {
              "a": {"a1": 1, "a2": 2},
              "b": {"b1": 3, "b2": pz.NotInThisPartition()},
              "c": {
                  "c1": pz.NotInThisPartition(),
                  "c2": pz.NotInThisPartition(),
              },
          },
      )
      self.assertEqual(
          right,
          {
              "a": {
                  "a1": pz.NotInThisPartition(),
                  "a2": pz.NotInThisPartition(),
              },
              "b": {"b1": pz.NotInThisPartition(), "b2": 4},
              "c": {"c1": 5, "c2": 6},
          },
      )

  def test_combine_same_structure(self):
    left = {
        "a": {
            "a1": pz.NotInThisPartition(),
            "a2": pz.NotInThisPartition(),
        },
        "b": {"b1": pz.NotInThisPartition(), "b2": 4},
        "c": {"c1": 5, "c2": 6},
    }
    right = {
        "a": {"a1": 1, "a2": 2},
        "b": {"b1": 3, "b2": pz.NotInThisPartition()},
        "c": {
            "c1": pz.NotInThisPartition(),
            "c2": pz.NotInThisPartition(),
        },
    }
    self.assertEqual(
        pz.combine(left, right),
        {
            "a": {"a1": 1, "a2": 2},
            "b": {"b1": 3, "b2": 4},
            "c": {"c1": 5, "c2": 6},
        },
    )

  def test_combine_compatible_but_not_prefix(self):
    left = {
        "a": pz.NotInThisPartition(),
        "b": {"b1": pz.NotInThisPartition(), "b2": 4},
        "c": {"c1": 5, "c2": 6},
    }
    right = {
        "a": {"a1": 1, "a2": 2},
        "b": {"b1": 3, "b2": pz.NotInThisPartition()},
        "c": pz.NotInThisPartition(),
    }
    self.assertEqual(
        pz.combine(left, right),
        {
            "a": {"a1": 1, "a2": 2},
            "b": {"b1": 3, "b2": 4},
            "c": {"c1": 5, "c2": 6},
        },
    )

  def test_combine_incomplete(self):
    left = {
        "a": pz.NotInThisPartition(),
        "b": {"b1": pz.NotInThisPartition(), "b2": 4},
        "c": {"c1": pz.NotInThisPartition(), "c2": 6},
        "d": pz.NotInThisPartition(),
    }
    right = {
        "a": {"a1": 1, "a2": pz.NotInThisPartition()},
        "b": {"b1": 3, "b2": pz.NotInThisPartition()},
        "c": pz.NotInThisPartition(),
        "d": pz.NotInThisPartition(),
    }
    self.assertEqual(
        pz.combine(left, right),
        {
            "a": {"a1": 1, "a2": pz.NotInThisPartition()},
            "b": {"b1": 3, "b2": 4},
            "c": {"c1": pz.NotInThisPartition(), "c2": 6},
            "d": pz.NotInThisPartition(),
        },
    )

  def test_combine_multi(self):
    a = {
        "a": 1,
        "b": pz.NotInThisPartition(),
        "c": pz.NotInThisPartition(),
    }
    b = {
        "a": pz.NotInThisPartition(),
        "b": 2,
        "c": pz.NotInThisPartition(),
    }
    c = {
        "a": pz.NotInThisPartition(),
        "b": pz.NotInThisPartition(),
        "c": 3,
    }
    self.assertEqual(
        pz.combine(a, b, c),
        {"a": 1, "b": 2, "c": 3},
    )


if __name__ == "__main__":
  absltest.main()
