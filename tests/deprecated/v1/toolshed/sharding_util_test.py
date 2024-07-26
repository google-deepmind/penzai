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

"""Tests for sharding utilities."""

import collections

from absl.testing import absltest
import chex
import jax
import numpy as np
from penzai.deprecated.v1 import pz
from penzai.deprecated.v1.toolshed import sharding_util

P = jax.sharding.PartitionSpec


class ShardingUtilTest(absltest.TestCase):

  def test_name_to_name_sharding(self):
    chex.set_n_cpu_devices(8)
    some_array_tree = {
        'one': pz.nx.ones({'aaaa': 4, 'bbbb': 8, 'cccc': 6}),
        'two': pz.nx.ones({'aaaa': 8}),
        'three': pz.nx.ones({'bbbb': 8, 'dddd': 12}),
    }
    explicit_mesh = jax.sharding.Mesh(
        devices=np.array(jax.devices()).reshape((4, 2)),
        axis_names=('mesh_axis_0', 'mesh_axis_1'),
    )
    implicit_mesh = jax.sharding.Mesh(
        devices=np.array(jax.devices()).reshape((4, 2)),
        axis_names=('bbbb', 'aaaa'),
    )

    with self.subTest('name_to_name_explicit'):
      shardings = sharding_util.name_to_name_sharding(
          some_array_tree,
          explicit_mesh,
          axis_name_to_mesh_name={
              'aaaa': 'mesh_axis_1',
              'bbbb': 'mesh_axis_0',
          },
      )
      chex.assert_trees_all_equal(
          shardings,
          {
              'one': pz.nx.NamedArray(
                  named_axes=collections.OrderedDict(
                      {'aaaa': 4, 'bbbb': 8, 'cccc': 6}
                  ),
                  data_array=jax.sharding.NamedSharding(
                      explicit_mesh, P('mesh_axis_1', 'mesh_axis_0', None)
                  ),
              ),
              'two': pz.nx.NamedArray(
                  named_axes=collections.OrderedDict({'aaaa': 8}),
                  data_array=jax.sharding.NamedSharding(
                      explicit_mesh, P('mesh_axis_1')
                  ),
              ),
              'three': pz.nx.NamedArray(
                  named_axes=collections.OrderedDict({'bbbb': 8, 'dddd': 12}),
                  data_array=jax.sharding.NamedSharding(
                      explicit_mesh, P('mesh_axis_0', None)
                  ),
              ),
          },
      )

    with self.subTest('name_to_name_implicit'):
      shardings = sharding_util.name_to_name_sharding(
          some_array_tree, implicit_mesh
      )
      chex.assert_trees_all_equal(
          shardings,
          {
              'one': pz.nx.NamedArray(
                  named_axes=collections.OrderedDict(
                      {'aaaa': 4, 'bbbb': 8, 'cccc': 6}
                  ),
                  data_array=jax.sharding.NamedSharding(
                      implicit_mesh, P('aaaa', 'bbbb', None)
                  ),
              ),
              'two': pz.nx.NamedArray(
                  named_axes=collections.OrderedDict({'aaaa': 8}),
                  data_array=jax.sharding.NamedSharding(
                      implicit_mesh, P('aaaa')
                  ),
              ),
              'three': pz.nx.NamedArray(
                  named_axes=collections.OrderedDict({'bbbb': 8, 'dddd': 12}),
                  data_array=jax.sharding.NamedSharding(
                      implicit_mesh, P('bbbb', None)
                  ),
              ),
          },
      )

    with self.subTest('name_to_name_multiaxis'):
      shardings = sharding_util.name_to_name_sharding(
          some_array_tree,
          explicit_mesh,
          axis_name_to_mesh_name={'bbbb': ('mesh_axis_1', 'mesh_axis_0')},
      )
      chex.assert_trees_all_equal(
          shardings,
          {
              'one': pz.nx.NamedArray(
                  named_axes=collections.OrderedDict(
                      {'aaaa': 4, 'bbbb': 8, 'cccc': 6}
                  ),
                  data_array=jax.sharding.NamedSharding(
                      explicit_mesh,
                      P(None, ('mesh_axis_1', 'mesh_axis_0'), None),
                  ),
              ),
              'two': pz.nx.NamedArray(
                  named_axes=collections.OrderedDict({'aaaa': 8}),
                  data_array=jax.sharding.NamedSharding(explicit_mesh, P(None)),
              ),
              'three': pz.nx.NamedArray(
                  named_axes=collections.OrderedDict({'bbbb': 8, 'dddd': 12}),
                  data_array=jax.sharding.NamedSharding(
                      explicit_mesh,
                      P(('mesh_axis_1', 'mesh_axis_0'), None),
                  ),
              ),
          },
      )

    with self.subTest('name_to_name_as_view'):
      shardings = sharding_util.name_to_name_sharding(
          {k: v.as_namedarrayview() for k, v in some_array_tree.items()},
          explicit_mesh,
          axis_name_to_mesh_name={
              'aaaa': 'mesh_axis_1',
              'bbbb': 'mesh_axis_0',
          },
      )
      chex.assert_trees_all_equal(
          shardings,
          {
              'one': pz.nx.NamedArrayView(
                  data_shape=(4, 8, 6),
                  data_axis_for_logical_axis=(),
                  data_axis_for_name={'aaaa': 0, 'bbbb': 1, 'cccc': 2},
                  data_array=jax.sharding.NamedSharding(
                      explicit_mesh, P('mesh_axis_1', 'mesh_axis_0', None)
                  ),
              ),
              'two': pz.nx.NamedArrayView(
                  data_shape=(8,),
                  data_axis_for_logical_axis=(),
                  data_axis_for_name={'aaaa': 0},
                  data_array=jax.sharding.NamedSharding(
                      explicit_mesh, P('mesh_axis_1')
                  ),
              ),
              'three': pz.nx.NamedArrayView(
                  data_shape=(8, 12),
                  data_axis_for_logical_axis=(),
                  data_axis_for_name={'bbbb': 0, 'dddd': 1},
                  data_array=jax.sharding.NamedSharding(
                      explicit_mesh, P('mesh_axis_0', None)
                  ),
              ),
          },
      )

    with self.subTest('name_to_name_struct'):
      shardings = sharding_util.name_to_name_sharding(
          some_array_tree,
          explicit_mesh,
          axis_name_to_mesh_name={
              'aaaa': 'mesh_axis_1',
              'bbbb': 'mesh_axis_0',
          },
          as_shape_dtype_struct=True,
      )
      chex.assert_trees_all_equal(
          shardings,
          {
              'one': pz.nx.NamedArray(
                  named_axes=collections.OrderedDict(
                      {'aaaa': 4, 'bbbb': 8, 'cccc': 6}
                  ),
                  data_array=jax.ShapeDtypeStruct(
                      shape=(4, 8, 6),
                      dtype=np.dtype('float32'),
                      named_shape={},
                      sharding=jax.sharding.NamedSharding(
                          explicit_mesh, P('mesh_axis_1', 'mesh_axis_0', None)
                      ),
                  ),
              ),
              'two': pz.nx.NamedArray(
                  named_axes=collections.OrderedDict({'aaaa': 8}),
                  data_array=jax.ShapeDtypeStruct(
                      shape=(8,),
                      dtype=np.dtype('float32'),
                      named_shape={},
                      sharding=jax.sharding.NamedSharding(
                          explicit_mesh, P('mesh_axis_1')
                      ),
                  ),
              ),
              'three': pz.nx.NamedArray(
                  named_axes=collections.OrderedDict({'bbbb': 8, 'dddd': 12}),
                  data_array=jax.ShapeDtypeStruct(
                      shape=(8, 12),
                      dtype=np.dtype('float32'),
                      named_shape={},
                      sharding=jax.sharding.NamedSharding(
                          explicit_mesh, P('mesh_axis_0', None)
                      ),
                  ),
              ),
          },
      )

    with self.subTest('name_to_name_struct_as_view'):
      shardings = sharding_util.name_to_name_sharding(
          {k: v.as_namedarrayview() for k, v in some_array_tree.items()},
          explicit_mesh,
          axis_name_to_mesh_name={
              'aaaa': 'mesh_axis_1',
              'bbbb': 'mesh_axis_0',
          },
          as_shape_dtype_struct=True,
      )
      chex.assert_trees_all_equal(
          shardings,
          {
              'one': pz.nx.NamedArrayView(
                  data_shape=(4, 8, 6),
                  data_axis_for_logical_axis=(),
                  data_axis_for_name={'aaaa': 0, 'bbbb': 1, 'cccc': 2},
                  data_array=jax.ShapeDtypeStruct(
                      shape=(4, 8, 6),
                      dtype=np.dtype('float32'),
                      named_shape={},
                      sharding=jax.sharding.NamedSharding(
                          explicit_mesh, P('mesh_axis_1', 'mesh_axis_0', None)
                      ),
                  ),
              ),
              'two': pz.nx.NamedArrayView(
                  data_shape=(8,),
                  data_axis_for_logical_axis=(),
                  data_axis_for_name={'aaaa': 0},
                  data_array=jax.ShapeDtypeStruct(
                      shape=(8,),
                      dtype=np.dtype('float32'),
                      named_shape={},
                      sharding=jax.sharding.NamedSharding(
                          explicit_mesh, P('mesh_axis_1')
                      ),
                  ),
              ),
              'three': pz.nx.NamedArrayView(
                  data_shape=(8, 12),
                  data_axis_for_logical_axis=(),
                  data_axis_for_name={'bbbb': 0, 'dddd': 1},
                  data_array=jax.ShapeDtypeStruct(
                      shape=(8, 12),
                      dtype=np.dtype('float32'),
                      named_shape={},
                      sharding=jax.sharding.NamedSharding(
                          explicit_mesh, P('mesh_axis_0', None)
                      ),
                  ),
              ),
          },
      )

  def test_initialize_parameters_sharded(self):
    chex.set_n_cpu_devices(8)
    some_param_tree = {
        'one': pz.nn.UninitializedParameter(
            lambda key: pz.nx.full(
                {'aaaa': 4, 'bbbb': 8, 'cccc': 6}, jax.random.uniform(key)
            ),
            'p_one',
        ),
        'two': pz.nn.UninitializedParameter(
            lambda key: pz.nx.full({'aaaa': 8}, jax.random.uniform(key)),
            'p_two',
        ),
        'three': pz.nn.UninitializedParameter(
            lambda key: pz.nx.full(
                {'bbbb': 8, 'dddd': 12}, jax.random.uniform(key)
            ),
            'p_three',
        ),
    }
    explicit_mesh = jax.sharding.Mesh(
        devices=np.array(jax.devices()).reshape((4, 2)),
        axis_names=('mesh_axis_0', 'mesh_axis_1'),
    )

    axis_name_to_mesh_name = {
        'aaaa': 'mesh_axis_1',
        'bbbb': 'mesh_axis_0',
    }
    sharded_params = sharding_util.initialize_parameters_sharded(
        some_param_tree,
        jax.random.key(1),
        explicit_mesh,
        axis_name_to_mesh_name=axis_name_to_mesh_name,
    )
    for k, param in sharded_params.items():
      with self.subTest(f'check_{k}'):
        self.assertIsInstance(param, pz.nn.Parameter)
        param.value.check_valid()
        self.assertEqual(
            param.value.data_array.sharding,
            sharding_util.name_to_name_sharding(
                param.value, explicit_mesh, axis_name_to_mesh_name
            ).data_array,
        )


if __name__ == '__main__':
  absltest.main()
