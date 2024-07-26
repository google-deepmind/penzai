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

"""Tests for saving intermediates and their shapes."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from penzai import pz
from penzai.models import simple_mlp
from penzai.toolshed import save_intermediates


class SaveIntermediatesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='values_ungrouped',
          shapes_only=False,
          intermediates_group=None,
      ),
      dict(
          testcase_name='values_grouped',
          shapes_only=False,
          intermediates_group='intermediates',
      ),
      dict(
          testcase_name='shapes_only_ungrouped',
          shapes_only=True,
          intermediates_group=None,
      ),
      dict(
          testcase_name='shapes_only_grouped',
          shapes_only=True,
          intermediates_group='intermediates',
      ),
  )
  def test_saving_all_intermediates(self, shapes_only, intermediates_group):
    mlp = simple_mlp.MLP.from_config(
        name='mlp',
        init_base_rng=jax.random.key(10),
        feature_sizes=[8, 16, 64, 32],
    )
    saving_mlp = save_intermediates.saving_all_intermediates(
        mlp, intermediates_group=intermediates_group, shapes_only=shapes_only
    )

    _, variables = pz.unbind_state_vars(saving_mlp)
    self.assertLen(variables, 12)

    for var in variables:
      if intermediates_group is None:
        self.assertIsInstance(var.label, pz.AutoStateVarLabel)
      else:
        self.assertIsInstance(var.label, pz.ScopedStateVarLabel)
        self.assertEqual(var.label.group, intermediates_group)
      self.assertIsNone(var.value)

    result = saving_mlp(pz.nx.zeros({'batch': 3, 'features': 8}))

    self.assertEqual(result.named_shape, {'batch': 3, 'features': 32})

    var_shapes = [var.value.named_shape for var in variables]
    self.assertEqual(
        var_shapes,
        [
            {'batch': 3, 'features': 8},
            {'batch': 3, 'features_out': 16},
            {'batch': 3, 'features': 16},
            {'batch': 3, 'features': 16},
            {'batch': 3, 'features': 16},
            {'batch': 3, 'features_out': 64},
            {'batch': 3, 'features': 64},
            {'batch': 3, 'features': 64},
            {'batch': 3, 'features': 64},
            {'batch': 3, 'features_out': 32},
            {'batch': 3, 'features': 32},
            {'batch': 3, 'features': 32},
        ],
    )
    for var in variables:
      if shapes_only:
        self.assertIsInstance(var.value, pz.chk.ArraySpec)
      else:
        self.assertIsInstance(var.value, pz.nx.NamedArray)


if __name__ == '__main__':
  absltest.main()
