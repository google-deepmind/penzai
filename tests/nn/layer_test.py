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

"""Tests for layer helpers."""

from absl.testing import absltest
from penzai import pz


class LayerTest(absltest.TestCase):

  def test_layer_variable_call(self):
    @pz.pytree_dataclass
    class MyLayer(pz.nn.Layer):
      counter: pz.StateVariable[int]

      def __call__(self, x, scale, **_side_inputs):
        self.counter.value += x * scale
        return x

    @pz.pytree_dataclass
    class MutabilityChecker(pz.nn.Layer):
      to_check: pz.StateVariable[int] | pz.StateVariableValue[int]

      def __call__(self, x, scale, **_side_inputs):
        if isinstance(self.to_check, pz.StateVariable):
          return (x, {"mutable": self.to_check.value})
        else:
          return (x, {"frozen": self.to_check.value})

    layer_1 = MyLayer(counter=pz.StateVariable(value=0, label="counter_1"))
    layer_2 = MyLayer(counter=pz.StateVariable(value=100, label="counter_2"))
    mut_check = MutabilityChecker(
        to_check=pz.StateVariable(value=999, label="flag")
    )
    my_model = pz.nn.Sequential([layer_1, layer_2, mut_check])

    result = my_model(5, scale=2)
    self.assertEqual(result, (5, {"mutable": 999}))
    self.assertEqual(layer_1.counter.value, 10)
    self.assertEqual(layer_2.counter.value, 110)
    result = my_model(2, scale=1)
    self.assertEqual(result, (2, {"mutable": 999}))
    self.assertEqual(layer_1.counter.value, 12)
    self.assertEqual(layer_2.counter.value, 112)

    unbound_model, unbound_vars = pz.unbind_variables(
        my_model, lambda var: var.label != "flag"
    )
    self.assertEqual(
        unbound_model,
        pz.nn.Sequential([
            MyLayer(counter=pz.StateVariableSlot("counter_1")),
            MyLayer(counter=pz.StateVariableSlot("counter_2")),
            mut_check,
        ]),
    )
    self.assertEqual(unbound_vars, (layer_1.counter, layer_2.counter))

    frozen_values = (layer_1.counter.freeze(), layer_2.counter.freeze())
    result, new_frozen = unbound_model.stateless_call(
        frozen_values, 100, scale=3
    )
    self.assertEqual(result, (100, {"frozen": 999}))
    self.assertEqual(
        new_frozen,
        (
            pz.StateVariableValue(value=312, label="counter_1"),
            pz.StateVariableValue(value=412, label="counter_2"),
        ),
    )
    self.assertEqual(layer_1.counter.value, 12)


if __name__ == "__main__":
  absltest.main()
