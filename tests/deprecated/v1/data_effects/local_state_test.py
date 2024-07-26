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

"""Tests for mutable local state effect."""

import re
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from penzai.deprecated.v1 import pz


@pz.pytree_dataclass
class MyStatefulLayer(pz.Layer):
  some_named_state: dict[str, dict[str, pz.de.LocalStateEffect[jax.Array]]]
  some_unnamed_state: dict[str, dict[str, pz.de.LocalStateEffect[jax.Array]]]

  @classmethod
  def build(cls):
    request_1 = pz.de.InitialLocalStateRequest(
        name="mystate",
        state_initializer=lambda: jnp.zeros((), dtype=jnp.int32),
        category="test_state",
    )
    request_2 = pz.de.InitialLocalStateRequest(
        state_initializer=lambda: jnp.ones((), dtype=jnp.int32),
        category="test_state",
    )
    return pz.nn.add_parameter_prefix(
        "some.prefix",
        cls(
            some_named_state={"foo": {"bar": request_1}},
            some_unnamed_state={"baz": {"qux": request_2}},
        ),
    )

  @pz.checked_layer_call
  def __call__(self, argument: Any, /) -> Any:
    delta = argument["delta"]
    old_1 = self.some_named_state["foo"]["bar"].get()
    old_2 = self.some_unnamed_state["baz"]["qux"].get()
    self.some_named_state["foo"]["bar"].set(old_1 + delta[0])
    self.some_unnamed_state["baz"]["qux"].set(old_2 + delta[1])
    new_1 = self.some_named_state["foo"]["bar"].get()
    new_2 = self.some_unnamed_state["baz"]["qux"].get()
    return {"old": jnp.stack([old_1, old_2]), "new": jnp.stack([new_1, new_2])}

  def input_structure(self):
    return {"delta": pz.chk.ArraySpec(shape=(2,), dtype=jnp.int32)}

  def output_structure(self):
    return {
        "old": pz.chk.ArraySpec(shape=(2,), dtype=jnp.int32),
        "new": pz.chk.ArraySpec(shape=(2,), dtype=jnp.int32),
    }


@pz.pytree_dataclass
class IdentityWithExtraPayload(pz.nn.Identity):
  payload: Any


@pz.pytree_dataclass
class IncrementEverythingLayer(pz.Layer):
  duplicates: list[pz.de.LocalStateEffect[jax.Array]]

  def __call__(self, argument: Any, /):
    for state in self.duplicates:
      state.set(state.get() + 1)


class LocalStateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="lazy", lazy=True),
      dict(testcase_name="eager", lazy=False),
  )
  def test_state_workflow(self, lazy):
    # Create a model with initial states.
    my_model = pz.nn.Sequential([MyStatefulLayer.build()])
    # Handle it
    wrapped, initial_state = pz.de.handle_local_states(
        my_model, "test_state", lazy=lazy
    )
    if lazy:
      self.assertTrue(callable(initial_state))
      initial_state = initial_state()

    # Check the initial state
    with self.subTest("initial_state"):
      chex.assert_trees_all_equal(
          initial_state,
          {
              "some.prefix.mystate": jnp.array(0, dtype=jnp.int32),
              (
                  "Sequential.sublayers[0]/MyStatefulLayer"
                  ".some_unnamed_state['baz']['qux']"
              ): jnp.array(1, dtype=jnp.int32),
          },
      )

    # Run it on the initial state
    out, new_state = wrapped(({"delta": jnp.array([10, 20])}, initial_state))

    with self.subTest("output"):
      chex.assert_trees_all_equal(
          out,
          {
              "old": jnp.array([0, 1], dtype=jnp.int32),
              "new": jnp.array([10, 21], dtype=jnp.int32),
          },
      )

    with self.subTest("new_state"):
      chex.assert_trees_all_equal(
          new_state,
          {
              "some.prefix.mystate": jnp.array(10, dtype=jnp.int32),
              (
                  "Sequential.sublayers[0]/MyStatefulLayer"
                  ".some_unnamed_state['baz']['qux']"
              ): jnp.array(21, dtype=jnp.int32),
          },
      )

    # Freeze it back into the model.
    frozen = pz.de.freeze_local_states(wrapped, new_state)

    with self.subTest("frozen_model"):
      self.assertIsInstance(frozen, pz.nn.Sequential)
      self.assertIsInstance(frozen.sublayers[0], MyStatefulLayer)
      self.assertEqual(
          frozen.sublayers[0].some_named_state["foo"]["bar"],
          pz.de.FrozenLocalStateRequest(
              state=jnp.array(10, dtype=jnp.int32),
              name="some.prefix.mystate",
              category="test_state",
          ),
      )
      self.assertEqual(
          frozen.sublayers[0].some_unnamed_state["baz"]["qux"],
          pz.de.FrozenLocalStateRequest(
              state=jnp.array(21, dtype=jnp.int32),
              category="test_state",
          ),
      )

    # Unfreeze it again, check that it's the same
    rethawed_wrapped, rethawed_state = pz.de.handle_local_states(
        frozen, "test_state", lazy=False, handler_id=wrapped.handler_id
    )
    with self.subTest("rethawed_model"):
      chex.assert_trees_all_equal(rethawed_wrapped, wrapped)
    with self.subTest("rethawed_state"):
      chex.assert_trees_all_equal(rethawed_state, new_state)

  def test_category_predicate(self):
    # Create a model with initial states.
    my_model = IdentityWithExtraPayload({
        "foo": pz.de.InitialLocalStateRequest(
            name="mystate", state_initializer=lambda: 0.0, category="a"
        ),
        "bar": pz.de.InitialLocalStateRequest(
            state_initializer=lambda: 0.0, category="b"
        ),
        "baz": pz.de.InitialLocalStateRequest(
            state_initializer=lambda: 0.0, category="c"
        ),
    })
    _, initial_state = pz.de.handle_local_states(
        my_model, category_predicate=lambda st: st in ("a", "c")
    )
    self.assertEqual(
        initial_state,
        {
            "IdentityWithExtraPayload.payload['baz']": 0.0,
            "mystate": 0.0,
        },
    )

  def test_state_sharing_initial(self):
    state_one = pz.de.InitialLocalStateRequest(
        name="foo", state_initializer=lambda: 0.0, category="test_state"
    )
    state_two = pz.de.InitialLocalStateRequest(
        name="bar", state_initializer=lambda: 0.0, category="test_state"
    )
    my_model = IncrementEverythingLayer(
        [state_one, state_two, state_one, state_two, state_one]
    )

    with self.subTest("forbidden"):
      with self.assertRaisesRegex(
          ValueError,
          re.escape("Detected two local states with the same explicit name"),
      ):
        _ = pz.de.handle_local_states(my_model, "test_state")

    for mode in ["allowed", "unsafe"]:
      with self.subTest(mode):
        wrapped, initial_state = pz.de.handle_local_states(
            my_model,
            "test_state",
            state_sharing=mode,
        )
        _, new_state = wrapped((None, initial_state))
        chex.assert_trees_all_equal(new_state, {"foo": 3.0, "bar": 2.0})

  @parameterized.named_parameters(
      dict(testcase_name="diff_initializers", freeze=False),
      dict(testcase_name="diff_frozen", freeze=True),
  )
  def test_state_sharing_incoherent(self, freeze):
    if freeze:
      mk_one = lambda: pz.de.FrozenLocalStateRequest(
          state=0.0, name="foo", category="test_state"
      )
      mk_two = lambda: pz.de.FrozenLocalStateRequest(
          state=0.0, name="bar", category="test_state"
      )
    else:
      mk_one = lambda: pz.de.InitialLocalStateRequest(
          name="foo", state_initializer=lambda: 0.0, category="test_state"
      )
      mk_two = lambda: pz.de.InitialLocalStateRequest(
          name="bar", state_initializer=lambda: 0.0, category="test_state"
      )
    my_model = IncrementEverythingLayer(
        [mk_one(), mk_two(), mk_one(), mk_two(), mk_one()]
    )

    for mode in ["forbidden", "allowed"]:
      with self.assertRaisesRegex(
          ValueError,
          re.escape("Detected two local states with the same explicit name"),
      ):
        _ = pz.de.handle_local_states(
            my_model, "test_state", state_sharing=mode
        )

    with self.subTest("unsafe"):
      wrapped, initial_state = pz.de.handle_local_states(
          my_model,
          "test_state",
          state_sharing="unsafe",
      )
      _, new_state = wrapped((None, initial_state))
      chex.assert_trees_all_equal(new_state, {"foo": 3.0, "bar": 2.0})

  def test_state_sharing_explicit(self):
    state_one = pz.de.FrozenLocalStateRequest(
        state=0.0, name="foo", category="test_state"
    )
    state_two = pz.de.FrozenLocalStateRequest(
        state=0.0, name="bar", category="test_state"
    )
    state_one_ref = pz.de.SharedLocalStateRequest(
        name="foo", category="test_state"
    )
    state_two_ref = pz.de.SharedLocalStateRequest(
        name="bar", category="test_state"
    )
    my_model = IncrementEverythingLayer(
        [state_one, state_two, state_one_ref, state_two_ref, state_one_ref]
    )

    with self.subTest("forbidden"):
      with self.assertRaisesRegex(
          ValueError,
          re.escape(
              "SharedLocalStateRequest must only be used when state_sharing is"
              " set to 'allowed' or 'unsafe'"
          ),
      ):
        _ = pz.de.handle_local_states(my_model, "test_state")

    for mode in ["allowed", "unsafe"]:
      with self.subTest(mode):
        wrapped, initial_state = pz.de.handle_local_states(
            my_model,
            "test_state",
            state_sharing=mode,
        )
        _, new_state = wrapped((None, initial_state))
        chex.assert_trees_all_equal(new_state, {"foo": 3.0, "bar": 2.0})

        refrozen = pz.de.freeze_local_states(wrapped, initial_state)
        chex.assert_trees_all_equal(my_model, refrozen)

  def test_hoist_shared_state_requests(self):
    shared_initial = pz.de.InitialLocalStateRequest(
        state_initializer=lambda: 10, name="foo", category="a"
    )
    hoistable = [
        [
            shared_initial,
            shared_initial,
            pz.de.FrozenLocalStateRequest(state=20, name="bar", category="b"),
            pz.de.SharedLocalStateRequest(name="bar", category="b"),
            pz.de.SharedLocalStateRequest(name="foo", category="a"),
        ],
        [
            pz.de.SharedLocalStateRequest(name="foo", category="a"),
            pz.de.SharedLocalStateRequest(name="bar", category="b"),
            pz.de.SharedLocalStateRequest(name="foo", category="a"),
            pz.de.SharedLocalStateRequest(name="bar", category="b"),
            pz.de.SharedLocalStateRequest(name="foo", category="a"),
            pz.de.SharedLocalStateRequest(name="bar", category="b"),
        ],
    ]
    hoisted, bindings = pz.de.hoist_shared_state_requests(hoistable)
    re_embedded = pz.de.embed_shared_state_requests(hoisted[1], bindings)
    self.assertEqual(
        re_embedded,
        [
            shared_initial,
            pz.de.FrozenLocalStateRequest(state=20, name="bar", category="b"),
            pz.de.SharedLocalStateRequest(name="foo", category="a"),
            pz.de.SharedLocalStateRequest(name="bar", category="b"),
            pz.de.SharedLocalStateRequest(name="foo", category="a"),
            pz.de.SharedLocalStateRequest(name="bar", category="b"),
        ],
    )


if __name__ == "__main__":
  absltest.main()
