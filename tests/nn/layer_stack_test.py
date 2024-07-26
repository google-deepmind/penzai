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

"""Tests for linear layers."""

from typing import Any
from absl.testing import absltest
import chex
import jax
from penzai import pz


@pz.pytree_dataclass
class MyTestLayer(pz.nn.Layer):
  params: tuple[pz.Parameter[pz.nx.NamedArray], ...]
  expected_param_shapes: tuple[dict[str, int], ...] | None
  states: tuple[pz.StateVariable[pz.nx.NamedArray], ...]
  stuff: Any

  def __call__(self, arg, /, shared_counter, **side_inputs):
    if self.expected_param_shapes is not None:
      for param, expected in zip(self.params, self.expected_param_shapes):
        assert param.value.named_shape == expected

    for state in self.states:
      state.value += 1

    shared_counter.value += 1
    return arg + 1


class LayerStackTest(absltest.TestCase):

  def test_stack_call(self):
    the_layer = pz.nn.LayerStack(
        stacked_sublayers=MyTestLayer(
            params=(
                pz.Parameter(
                    value=pz.nx.zeros({"stack": 3, "foo": 4}), label="param1"
                ),
                pz.Parameter(value=pz.nx.zeros({"bar": 5}), label="param2"),
            ),
            expected_param_shapes=({"foo": 4}, {"bar": 5}),
            states=(
                pz.StateVariable(
                    value=0,
                    label="state1",
                    metadata={
                        "layerstack_axes": {
                            "stack": pz.nn.LayerStackVarBehavior.SHARED
                        }
                    },
                ),
                pz.StateVariable(
                    value=pz.nx.zeros({"foo": 3}),
                    label="state2",
                    metadata={
                        "layerstack_axes": {
                            "stack": pz.nn.LayerStackVarBehavior.SHARED
                        }
                    },
                ),
                pz.StateVariable(
                    value=pz.nx.arange("stack", 3),
                    label="state3",
                    metadata={
                        "layerstack_axes": {
                            "stack": pz.nn.LayerStackVarBehavior.PER_LAYER
                        }
                    },
                ),
            ),
            stuff={
                "value": 100.0,
                "named": pz.nx.arange("bar", 3),
                "named_stacked": pz.nx.arange("stack", 3),
            },
        ),
        stack_axis="stack",
        stack_axis_size=3,
    )
    counter = pz.StateVariable(0)
    result = the_layer(10, shared_counter=counter)

    self.assertEqual(result, 13.0)
    self.assertEqual(counter.value, 3)
    chex.assert_trees_all_equal(
        the_layer.stacked_sublayers.states[0].value, 3.0
    )
    chex.assert_trees_all_equal(
        the_layer.stacked_sublayers.states[1].value.canonicalize(),
        pz.nx.full({"foo": 3}, fill_value=3.0).canonicalize(),
    )
    chex.assert_trees_all_equal(
        the_layer.stacked_sublayers.states[2].value.canonicalize(),
        (pz.nx.arange("stack", 3) + 1).canonicalize(),
    )

  def test_layer_stack_build(self):
    def builder(init_base_rng, some_value):
      return MyTestLayer(
          params=(
              pz.nn.make_parameter(
                  "foo",
                  init_base_rng,
                  lambda k: pz.nx.wrap(
                      jax.random.uniform(k, shape=(4,)), "vals"
                  ),
              ),
          ),
          expected_param_shapes=None,
          states=(pz.StateVariable(value=pz.nx.zeros({}), label="varstate"),),
          stuff={"value": some_value, "named_value": pz.nx.arange("bar", 4)},
      )

    layer = pz.nn.LayerStack.from_sublayer_builder(
        builder=builder,
        stack_axis="stack",
        stack_axis_size=3,
        init_base_rng=jax.random.key(1),
        builder_kwargs={"some_value": 100.0},
    )
    self.assertEqual(
        layer.stacked_sublayers.params[0].value.named_shape,
        {"stack": 3, "vals": 4},
    )
    self.assertEqual(
        layer.stacked_sublayers.states[0].value.named_shape, {"stack": 3}
    )
    self.assertEqual(layer.stacked_sublayers.stuff["value"], 100.0)
    chex.assert_trees_all_equal(
        layer.stacked_sublayers.stuff["named_value"].canonicalize(),
        (pz.nx.arange("bar", 4) * pz.nx.ones({"stack": 3})).canonicalize(),
    )

    slot_layer = pz.nn.LayerStack.from_sublayer_builder(
        builder=builder,
        stack_axis="stack",
        stack_axis_size=3,
        init_base_rng=None,
        builder_kwargs={"some_value": 100.0},
    )

    unbound_layer, layer_vars = pz.unbind_variables(layer)
    unbound_slot_layer, slot_layer_vars = pz.unbind_variables(slot_layer)

    chex.assert_trees_all_equal(unbound_layer, unbound_slot_layer)

    slot_layer_vars_dict = {var.label: var for var in slot_layer_vars}
    for var in layer_vars:
      if isinstance(var, pz.Parameter):
        self.assertNotIn(var.label, slot_layer_vars_dict)
      else:
        other_var = slot_layer_vars_dict[var.label]
        chex.assert_trees_all_equal(var.value, other_var.value)
        self.assertEqual(var.metadata, other_var.metadata)


if __name__ == "__main__":
  absltest.main()
