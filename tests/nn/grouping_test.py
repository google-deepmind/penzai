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

"""Tests for grouping layers and associated utilities."""

import dataclasses
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from penzai import pz


@pz.pytree_dataclass
class TaggedStruct(pz.nn.Identity):
  """A simple identity layer with a comment tag, for assertions."""

  comment: str = dataclasses.field(metadata={"pytree_node": False})


class GroupingTest(parameterized.TestCase):

  @parameterized.parameters("sequential", "named_group")
  def test_group_call_runs_children(self, which):
    layer_1 = mock.MagicMock(return_value=2)
    layer_2 = mock.MagicMock(return_value="foo")
    layer_3 = mock.MagicMock(return_value=(1, 2, 3))

    if which == "named_group":
      my_group = pz.nn.NamedGroup("some_name", [layer_1, layer_2, layer_3])
    else:
      my_group = pz.nn.Sequential([layer_1, layer_2, layer_3])

    result = my_group(1, some_side_input="bar")
    self.assertEqual(result, (1, 2, 3))
    layer_1.assert_called_once_with(1, some_side_input="bar")
    layer_2.assert_called_once_with(2, some_side_input="bar")
    layer_3.assert_called_once_with("foo", some_side_input="bar")

  def test_inline_anonymous_sequentials(self):

    @pz.pytree_dataclass(has_implicitly_inherited_fields=True)
    class MySequentialSubclass(pz.nn.Sequential):
      pass

    original_model = pz.nn.Sequential([
        TaggedStruct("1"),
        pz.nn.Sequential([
            TaggedStruct("2"),
            pz.nn.Sequential([
                TaggedStruct("3"),
            ]),
            pz.nn.NamedGroup(
                "foo",
                [
                    TaggedStruct("4"),
                    pz.nn.Sequential([
                        TaggedStruct("5"),
                        TaggedStruct("6"),
                    ]),
                ],
            ),
            TaggedStruct("7"),
            MySequentialSubclass([
                TaggedStruct("8"),
                TaggedStruct("9"),
            ]),
            TaggedStruct("10"),
        ]),
        TaggedStruct("11"),
    ])

    expected_inlined_model = pz.nn.Sequential(
        sublayers=[
            TaggedStruct("1"),
            TaggedStruct("2"),
            TaggedStruct("3"),
            pz.nn.NamedGroup(
                name="foo",
                sublayers=[
                    TaggedStruct("4"),
                    TaggedStruct("5"),
                    TaggedStruct("6"),
                ],
            ),
            TaggedStruct("7"),
            MySequentialSubclass(
                sublayers=[
                    TaggedStruct("8"),
                    TaggedStruct("9"),
                ]
            ),
            TaggedStruct("10"),
            TaggedStruct("11"),
        ]
    )

    self.assertEqual(
        pz.nn.inline_anonymous_sequentials(original_model),
        expected_inlined_model,
    )


if __name__ == "__main__":
  absltest.main()
