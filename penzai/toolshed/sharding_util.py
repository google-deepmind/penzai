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
from penzai import pz

PyTreeOfArrays = Any
PyTreeOfNamedArrays = Any
PyTreeOfShardings = Any


def name_to_name_sharding(
    tree: PyTreeOfNamedArrays,
    mesh: jax.sharding.Mesh,
    axis_name_to_mesh_name: (
        dict[pz.nx.AxisName, str | tuple[str, ...]] | None
    ) = None,
    ignore_unnamed_arrays: bool = False,
    as_shape_dtype_struct: bool = False,
) -> PyTreeOfShardings:
  """Shards a tree of `pz.nx.NamedArray` objects based on their axis names.

  Args:
    tree: A PyTree of `pz.nx.NamedArray` instances, with the same structure as
      the tree you want to shard. It is OK for the NamedArray instances to have
      invalid or missing data arrays; the data is not used.
    mesh: The `jax.sharding.Mesh` to shard the tree to.
    axis_name_to_mesh_name: A mapping from array axis names to mesh axis names.
      If an axis name is not present, that axis will not be sharded. If a mesh
      axis name is a tuple, the corresponding axis will be sharded to multiple
      mesh axes. If this dictionary is not provided, it will be inferred as an
      "identity" mapping, where each axis is sharded to a mesh axis with the
      same name (if present).
    ignore_unnamed_arrays: Whether to ignore non-NamedArray leaves. If True, any
      leaf that is not a NamedArray will be given a ``None`` sharding, usually
      indicating that JAX should infer a sharding. If False, a ``ValueError``
      will be raised if any leaf is not a NamedArray.
    as_shape_dtype_struct: If True, instead of directly returning a PyTree of
      ``NamedSharding``, return a PyTree of ``jax.ShapeDTypeStruct`` where the
      ``.sharding`` attribute is the ``NamedSharding``. This can be useful for
      building inputs to `orbax.checkpoint`, for instance.

  Returns:
    A PyTree with the same structure as the input tree, but with all
    `pz.nx.NamedArray` instances replaced with versions that have
    ``jax.sharding.NamedSharding`` leaves in place of their actual data arrays.
    This is suitable for passing as the ``in_shardings`` or ``out_shardings``
    for `jax.jit`, or as the sharding for `jax.device_put`.
  """
  if axis_name_to_mesh_name is None:
    axis_name_to_mesh_name = {name: name for name in mesh.axis_names}

  def sharding_for_leaf(leaf):
    if isinstance(leaf, pz.nx.NamedArray):
      pspec_elts = []
      for axis_name in leaf.named_axes.keys():
        if axis_name in axis_name_to_mesh_name:
          pspec_elts.append(axis_name_to_mesh_name[axis_name])
        else:
          pspec_elts.append(None)
      data_array = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(*pspec_elts)
      )
      if as_shape_dtype_struct:
        data_array = jax.ShapeDtypeStruct(
            shape=leaf.data_array.shape,
            dtype=leaf.data_array.dtype,
            sharding=data_array,
        )
      return dataclasses.replace(leaf, data_array=data_array)
    elif isinstance(leaf, pz.nx.NamedArrayView):
      pspec_elts = [None for _ in leaf.data_shape]
      for axis_name, data_axis in leaf.data_axis_for_name.items():
        if axis_name in axis_name_to_mesh_name:
          pspec_elts[data_axis] = axis_name_to_mesh_name[axis_name]
      data_array = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(*pspec_elts)
      )
      if as_shape_dtype_struct:
        data_array = jax.ShapeDtypeStruct(
            shape=leaf.data_array.shape,
            dtype=leaf.data_array.dtype,
            sharding=data_array,
        )
      return dataclasses.replace(leaf, data_array=data_array)
    elif not ignore_unnamed_arrays:
      raise ValueError(
          f"Cannot infer a name-based sharding for non-NamedArray leaf {leaf}."
          " If this leaf should be given a `None` sharding, set"
          " `ignore_unnamed_arrays=True`."
      )
    elif as_shape_dtype_struct:
      return jax.ShapeDtypeStruct(
          shape=leaf.shape, dtype=leaf.dtype, sharding=None
      )
    else:
      return None

  return jax.tree_util.tree_map(
      sharding_for_leaf, tree, is_leaf=pz.nx.is_namedarray
  )


def name_to_name_device_put(
    tree: PyTreeOfNamedArrays,
    mesh: jax.sharding.Mesh,
    axis_name_to_mesh_name: dict[str, str | tuple[str, ...]] | None = None,
) -> PyTreeOfNamedArrays:
  """Shards a tree of `pz.nx.NamedArray` objects based on their axis names.

  Args:
    tree: A PyTree of NamedArrays.
    mesh: The Mesh to shard the tree to.
    axis_name_to_mesh_name: A mapping from array axis names to mesh axis names.
      If an axis name is not present, that axis will not be sharded. If a mesh
      axis name is a tuple, the corresponding axis will be sharded to multiple
      mesh axes. If this dictionary is not provided, it will be inferred as an
      "identity" mapping, where each axis is sharded to a mesh axis with the
      same name (if present).

  Returns:
    A PyTree with the same structure as the input tree, but with all NamedArrays
    put onto the appropriate devices according to the mesh.
  """
  return jax.device_put(
      tree, name_to_name_sharding(tree, mesh, axis_name_to_mesh_name)
  )


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
