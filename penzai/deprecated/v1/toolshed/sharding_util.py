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

"""Utilities for working with sharded arrays and parameters in Penzai."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
from penzai.deprecated.v1 import pz
from penzai.toolshed import sharding_util as sharding_util_v2

PyTreeOfArrays = Any
PyTreeOfNamedArrays = Any
PyTreeOfShardings = Any

name_to_name_sharding = sharding_util_v2.name_to_name_sharding
name_to_name_device_put = sharding_util_v2.name_to_name_device_put


def initialize_parameters_sharded(
    model: Any,
    prng_key: jax.Array,
    mesh: jax.sharding.Mesh,
    axis_name_to_mesh_name: dict[str, str | tuple[str, ...]] | None = None,
) -> Any:
  """Initializes the parameters of a model, sharded according to a mesh.

  Args:
    model: A model whose parameters we should initialize. Should usually contain
      `pz.nn.UninitializedParameter` instances.
    prng_key: Key to use to initialize parameters.
    mesh: The ``Mesh`` to shard the tree to.
    axis_name_to_mesh_name: A mapping from array axis names to mesh axis names.
      If an axis name is not present, that axis will not be sharded. If a mesh
      axis name is a tuple, the corresponding axis will be sharded to multiple
      mesh axes. If this dictionary is not provided, it will be inferred as an
      "identity" mapping, where each axis is sharded to a mesh axis with the
      same name (if present).

  Returns:
    An initialized version of `model`, with all uninitialized parameters
    replaced with initialized parameters, sharded according to the arguments.
  """
  uninit_params = (
      pz.select(model)
      .at_instances_of(pz.nn.UninitializedParameter)
      .get_sequence()
  )
  out_shapes = jax.eval_shape(
      pz.nn.initialize_parameters, uninit_params, prng_key
  )
  out_shardings = name_to_name_sharding(
      out_shapes, mesh, axis_name_to_mesh_name
  )
  init_fn = jax.jit(pz.nn.initialize_parameters, out_shardings=out_shardings)
  initialized_params = init_fn(uninit_params, prng_key)
  return (
      pz.select(model)
      .at_instances_of(pz.nn.UninitializedParameter)
      .set_sequence(initialized_params)
  )


@pz.pytree_dataclass
class ConstrainSharding(pz.Layer):
  """A layer that constrains the sharding of a tree of arrays.

  Attributes:
    sharding: A PyTree of shardings. The PyTree structure must match the
      structure of the tree of arrays that will be passed to this layer.
  """

  sharding: PyTreeOfShardings = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def __call__(self, tree: PyTreeOfArrays) -> PyTreeOfArrays:
    return jax.lax.with_sharding_constraint(tree, self.sharding)


@pz.pytree_dataclass
class ConstrainShardingByName(pz.Layer):
  """A layer that constrains the sharding of a tree of NamedArrays by name.

  Attributes:
    mesh: The ``Mesh`` to shard the tree to.
    axis_name_to_mesh_name: A mapping from array axis names to mesh axis names.
      If an axis name is not present, that axis will not be sharded. If a mesh
      axis name is a tuple, the corresponding axis will be sharded to multiple
      mesh axes. If this dictionary is not provided, it will be inferred as an
      "identity" mapping, where each axis is sharded to a mesh axis with the
      same name (if present).
  """

  mesh: jax.sharding.Mesh = dataclasses.field(metadata={"pytree_node": False})
  axis_name_to_mesh_name: dict[str, str | tuple[str, ...]] | None = (
      dataclasses.field(default=None, metadata={"pytree_node": False})
  )

  def __call__(self, tree: PyTreeOfNamedArrays) -> PyTreeOfNamedArrays:
    return jax.lax.with_sharding_constraint(
        tree,
        name_to_name_sharding(tree, self.mesh, self.axis_name_to_mesh_name),
    )
