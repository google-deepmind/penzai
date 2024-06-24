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
from penzai.experimental.v2 import pz
from penzai.toolshed import sharding_util as sharding_util_v1

PyTreeOfArrays = Any
PyTreeOfNamedArrays = Any
PyTreeOfShardings = Any

name_to_name_sharding = sharding_util_v1.name_to_name_sharding
name_to_name_device_put = sharding_util_v1.name_to_name_device_put


@pz.pytree_dataclass
class ConstrainSharding(pz.nn.Layer):
  """A layer that constrains the sharding of a tree of arrays.

  Note: Defined using the experimental v2 API, but compatible with both v1 and
  v2 APIs (due to not having any parameters or state variables).

  Attributes:
    sharding: A PyTree of shardings. The PyTree structure must match the
      structure of the tree of arrays that will be passed to this layer.
  """

  sharding: PyTreeOfShardings = dataclasses.field(
      metadata={"pytree_node": False}
  )

  def __call__(
      self, tree: PyTreeOfArrays, **_unused_side_inputs
  ) -> PyTreeOfArrays:
    return jax.lax.with_sharding_constraint(tree, self.sharding)


@pz.pytree_dataclass
class ConstrainShardingByName(pz.nn.Layer):
  """A layer that constrains the sharding of a tree of NamedArrays by name.

  Note: Defined using the experimental v2 API, but compatible with both v1 and
  v2 APIs (due to not having any parameters or state variables).

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

  def __call__(
      self, tree: PyTreeOfNamedArrays, **_unused_side_inputs
  ) -> PyTreeOfNamedArrays:
    return jax.lax.with_sharding_constraint(
        tree,
        name_to_name_sharding(tree, self.mesh, self.axis_name_to_mesh_name),
    )
