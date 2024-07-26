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

"""Tests for side output effect."""

from typing import Any

from absl.testing import absltest
import jax
from penzai.deprecated.v1 import pz


@pz.pytree_dataclass
class MySideOutputWritingLayer(pz.Layer):
  int_writer: pz.de.SideOutputEffect[int] = pz.de.SideOutputRequest(
      "channel_one"
  )
  tuple_writer: pz.de.SideOutputEffect[tuple[bool, bool]] = (
      pz.de.SideOutputRequest("channel_two")
  )

  @pz.checked_layer_call
  def __call__(self, argument: Any, /) -> Any:
    assert argument == {"foo": None}
    self.int_writer.tell(123)
    self.tuple_writer.tell((True, False))
    self.int_writer.tell(456)
    return {"bar": None}

  def input_structure(self):
    return {"foo": None}

  def output_structure(self):
    return {"bar": None}


class SideInputTest(absltest.TestCase):

  def test_side_outputs(self):
    # Wrap in a sequential just to give some nesting
    layer = pz.nn.Sequential([MySideOutputWritingLayer()])
    wrapped = pz.de.CollectingSideOutputs.handling(layer)
    result, logs = wrapped({"foo": None})
    self.assertEqual(result, {"bar": None})
    keypath_strings = [
        (
            jax.tree_util.keystr(sideoutval.keypath),
            sideoutval.tag,
            sideoutval.value,
        )
        for sideoutval in logs
    ]
    self.assertEqual(
        keypath_strings,
        [
            (".sublayers[0].int_writer", "channel_one", 123),
            (".sublayers[0].tuple_writer", "channel_two", (True, False)),
            (".sublayers[0].int_writer", "channel_one", 456),
        ],
    )

  def test_side_output_collect_subset(self):
    # Wrap in a sequential just to give some nesting.
    layer = pz.nn.Sequential([MySideOutputWritingLayer()])
    # Collect only channel 1. Channel 2 will stay unhandled, but unhandled
    # side outputs can still be used (they just don't produce anything).
    wrapped = pz.de.CollectingSideOutputs.handling(layer, tag="channel_one")
    self.assertEqual(
        wrapped.body.sublayers[0],
        MySideOutputWritingLayer(
            int_writer=pz.de.HandledSideOutputRef(
                handler_id=wrapped.handler_id, tag="channel_one"
            ),
            tuple_writer=pz.de.SideOutputRequest(tag="channel_two"),
        ),
    )
    result, logs = wrapped({"foo": None})
    self.assertEqual(result, {"bar": None})
    keypath_strings = [
        (
            jax.tree_util.keystr(sideoutval.keypath),
            sideoutval.tag,
            sideoutval.value,
        )
        for sideoutval in logs
    ]
    self.assertEqual(
        keypath_strings,
        [
            (".sublayers[0].int_writer", "channel_one", 123),
            (".sublayers[0].int_writer", "channel_one", 456),
        ],
    )


if __name__ == "__main__":
  absltest.main()
