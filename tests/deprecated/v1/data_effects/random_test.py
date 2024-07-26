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

"""Tests for randomness effect."""

from typing import Any

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from penzai.deprecated.v1 import pz


@pz.pytree_dataclass
class MyEffectfulLayer(pz.Layer):
  random_source: pz.de.RandomEffect = pz.de.RandomRequest()

  @pz.checked_layer_call
  def __call__(self, argument: Any, /) -> Any:
    assert argument == {"foo": None}
    return (
        jax.random.uniform(self.random_source.next_key()),
        jax.random.uniform(self.random_source.next_key()),
    )

  def input_structure(self):
    return {"foo": None}

  def output_structure(self):
    return (
        pz.chk.ArraySpec(shape=(), dtype=jnp.float32),
        pz.chk.ArraySpec(shape=(), dtype=jnp.float32),
    )


@pz.pytree_dataclass
class IdentityWithExtraPayload(pz.nn.Identity):
  payload: Any


class RandomEffectTest(absltest.TestCase):

  def test_random_basic(self):
    wrapped = pz.de.WithRandomKeyFromArg.handling(MyEffectfulLayer())
    outs = wrapped(({"foo": None}, jax.random.key(42)))
    self.assertNotEqual(outs[0], outs[1])

  def test_random_predicate(self):
    layer = IdentityWithExtraPayload({
        "a": pz.de.RandomRequest(),
        "b": pz.de.TaggedRandomRequest("b"),
        "c": pz.de.TaggedRandomRequest("c"),
        "d": pz.de.RandomRequest(),
    })
    with self.subTest("default"):
      wrapped = pz.de.WithRandomKeyFromArg.handling(layer)
      self.assertEqual(
          wrapped.body.payload,
          {
              "a": pz.de.HandledRandomRef(handler_id=wrapped.handler_id),
              "b": pz.de.TaggedRandomRequest(tag="b"),
              "c": pz.de.TaggedRandomRequest(tag="c"),
              "d": pz.de.HandledRandomRef(handler_id=wrapped.handler_id),
          },
      )

    with self.subTest("explicit"):

      def predicate(req):
        return isinstance(req, pz.de.TaggedRandomRequest) and req.tag == "b"

      wrapped = pz.de.WithRandomKeyFromArg.handling(
          layer, hole_predicate=predicate
      )
      self.assertEqual(
          wrapped.body.payload,
          {
              "a": pz.de.RandomRequest(),
              "b": pz.de.HandledRandomRef(handler_id=wrapped.handler_id),
              "c": pz.de.TaggedRandomRequest(tag="c"),
              "d": pz.de.RandomRequest(),
          },
      )

  def test_random_with_state(self):
    wrapped = pz.de.WithStatefulRandomKey.handling(
        MyEffectfulLayer(), initial_key=jax.random.key(42)
    )
    wrapped_handled, state_dict = pz.de.handle_local_states(wrapped, "random")
    self.assertEqual(
        state_dict, {"WithStatefulRandomKey.random_state": jax.random.key(42)}
    )
    outs, new_state = wrapped_handled(({"foo": None}, state_dict))
    self.assertNotEqual(outs[0], outs[1])
    chex.assert_trees_all_equal_shapes_and_dtypes(new_state, state_dict)
    self.assertNotEqual(
        new_state["WithStatefulRandomKey.random_state"],
        state_dict["WithStatefulRandomKey.random_state"],
    )

  def test_random_frozen(self):
    wrapped = pz.de.WithFrozenRandomState.handling(
        MyEffectfulLayer(), random_key=jax.random.key(42), starting_offset=1
    )
    outs = wrapped({"foo": None})
    self.assertNotEqual(outs[0], outs[1])


if __name__ == "__main__":
  absltest.main()
