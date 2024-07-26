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

"""Generalized linear operator layer and associated utilities."""

from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import typing
from typing import Any, Callable, Literal, Protocol, Sequence

import jax
import jax.numpy as jnp
from penzai.core import named_axes
from penzai.core import shapecheck
from penzai.core import struct
from penzai.deprecated.v1.core import layer as layer_base
from penzai.deprecated.v1.nn import grouping
from penzai.deprecated.v1.nn import parameters


LayerLike = layer_base.LayerLike
NamedArray = named_axes.NamedArray


BiasInitializer: typing.TypeAlias = Callable[
    [jax.Array, tuple[int, ...], jnp.dtype], jax.Array
]


class LinearOperatorWeightInitializer(Protocol):
  """Protocol for an initializer for a general linear `NamedArray` weight."""

  def __call__(
      self,
      key: jax.Array,
      *,
      input_axes: dict[str, int],
      output_axes: dict[str, int],
      parallel_axes: dict[str, int],
      convolution_spatial_axes: dict[str, int],
      dtype: jax.typing.DTypeLike,
  ) -> NamedArray:
    """Signature for a generalized linear operator `NamedArray` initializer.

    This signature attempts to make explicit all of the dimensions used by an
    initializer, so that it can be used to initialize general linear layers
    without making assumptions about which axes are inputs or outputs.

    Provided sets of axes must not overlap.

    Args:
      key: Random key.
      input_axes: Names and lengths for axes that the linear operator should
        contract over.
      output_axes: Names and lengths for new axes that the linear operator
        should produce.
      parallel_axes: Names and lengths for axes that should be processed in
        parallel, such as the "heads" of an attention layer. These axes may
        appear in both the input and the output, and the resulting linear
        operator will apply a different operator to each slice. (This is similar
        to a block-diagonal matrix.)
      convolution_spatial_axes: Names and lengths for axes that correspond to
        spatial dimensions of a convolution, e.g. the convolution kernel's width
        and height. (Not expressable as an einsum.)
      dtype: Desired dtype.

    Returns:
      An initialized weight.
    """
    ...


def check_unique_axis_names_for_initializer(
    input_axes: dict[str, int],
    output_axes: dict[str, int],
    parallel_axes: dict[str, int],
    convolution_spatial_axes: dict[str, int],
) -> None:
  """Checks that indices appear exactly once in the given axis specifications."""
  seen_counts = collections.Counter(
      itertools.chain(
          input_axes.keys(),
          output_axes.keys(),
          parallel_axes.keys(),
          convolution_spatial_axes.keys(),
      )
  )
  bad_names = {name for name, count in seen_counts.items() if count > 1}
  if bad_names:
    raise ValueError(
        f"Duplicate axis names during initialization: {bad_names}."
    )


def variance_scaling_initializer(
    key: jax.Array,
    *,
    scale: float,
    mode: Literal["fan_in", "fan_out", "fan_avg"],
    distribution: Literal["uniform", "normal", "truncated_normal"],
    input_axes: dict[str, int],
    output_axes: dict[str, int],
    parallel_axes: dict[str, int],
    convolution_spatial_axes: dict[str, int],
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
  """Generic variance scaling initializer."""
  check_unique_axis_names_for_initializer(
      input_axes, output_axes, parallel_axes, convolution_spatial_axes
  )

  # JAX's variance scaling initializer assumes any missing axis indices are
  # convolution spatial axes.
  shape = []
  names = []
  input_axis_indices = []
  output_axis_indices = []
  parallel_axis_indices = []

  for name, size in parallel_axes.items():
    parallel_axis_indices.append(len(shape))
    names.append(name)
    shape.append(size)

  for name, size in convolution_spatial_axes.items():
    names.append(name)
    shape.append(size)

  for name, size in input_axes.items():
    input_axis_indices.append(len(shape))
    names.append(name)
    shape.append(size)

  for name, size in output_axes.items():
    output_axis_indices.append(len(shape))
    names.append(name)
    shape.append(size)

  # Note: At the time of writing, JAX's variance scaling initializer requires
  # its input to be at least 2d, even though it's OK with passing empty tuples
  # for any of the axis specifications, so we have to add extra axes to the
  # shape to make sure this works reliably.
  array = jax.nn.initializers.variance_scaling(
      scale=scale,
      mode=mode,
      distribution=distribution,
      in_axis=input_axis_indices,
      out_axis=output_axis_indices,
      batch_axis=parallel_axis_indices,
  )(key=key, shape=tuple(shape) + (1, 1), dtype=dtype)[..., 0, 0]

  return named_axes.wrap(array).tag(*names)


xavier_uniform_initializer = functools.partial(
    variance_scaling_initializer,
    scale=1.0,
    mode="fan_avg",
    distribution="uniform",
)

xavier_normal_initializer = functools.partial(
    variance_scaling_initializer,
    scale=1.0,
    mode="fan_avg",
    distribution="normal",
)


def constant_initializer(value: float) -> LinearOperatorWeightInitializer:
  """Returns an initializer that uses a constant value."""

  def _initializer(
      key: jax.Array,
      *,
      input_axes: dict[str, int],
      output_axes: dict[str, int],
      parallel_axes: dict[str, int],
      convolution_spatial_axes: dict[str, int],
      dtype: jax.typing.DTypeLike,
  ) -> jax.Array:
    """Zeros initializer for named arrays."""
    del key
    check_unique_axis_names_for_initializer(
        input_axes, output_axes, parallel_axes, convolution_spatial_axes
    )
    return named_axes.full(
        {
            **input_axes,
            **output_axes,
            **parallel_axes,
            **convolution_spatial_axes,
        },
        value,
        dtype=dtype,
    )

  return _initializer


zero_initializer = constant_initializer(0.0)


@struct.pytree_dataclass
class RenameAxes(layer_base.Layer):
  """Convenience layer that renames axes of its input.

  Attributes:
    old: The old axis names.
    new: The new axis names.
  """

  old: str | tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  new: str | tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  @layer_base.checked_layer_call
  def __call__(self, value: NamedArray) -> NamedArray:
    old = (self.old,) if isinstance(self.old, str) else self.old
    new = (self.new,) if isinstance(self.new, str) else self.new
    return value.untag(*old).tag(*new)

  def input_structure(self):
    old = (self.old,) if isinstance(self.old, str) else self.old
    return shapecheck.ArraySpec(
        named_shape={
            **shapecheck.vars_for_axes("renaming", old),
            **shapecheck.var("others"),
        }
    )

  def output_structure(self):
    old = (self.old,) if isinstance(self.old, str) else self.old
    new = (self.new,) if isinstance(self.new, str) else self.new
    if len(old) != len(new):
      raise layer_base.MisconfiguredLayerError(
          f"Length of new names {new} does not match length of old names {old}"
      )
    oldvars = shapecheck.vars_for_axes("renaming", old)
    return shapecheck.ArraySpec(
        named_shape={
            **{newname: oldvars[oldname] for oldname, newname in zip(old, new)},
            **shapecheck.var("others"),
        }
    )


@struct.pytree_dataclass
class LinearInPlace(grouping.Sequential):
  """Container for "in-place" linear operators that preserve axis names.

  This is used when initializing `Linear` layers that have overlapping names in
  their input and output specifications. We subclass `Sequential` to make
  this layer type easier to identify and manipulate.
  """

  sublayers: list[layer_base.LayerLike]

  def treescope_color(self) -> tuple[str, str]:
    return "#eba875", "color-mix(in oklab, #eba875 25%, white)"


def contract(
    names: str | Sequence[named_axes.AxisName],
    left: NamedArray,
    right: NamedArray,
) -> NamedArray:
  """Contracts two named arrays along the given axis names.

  Args:
    names: The axis names to contract. Can also be a single string axis name.
    left: The left-hand side of the contraction, as a NamedArray with only named
      axes.
    right: The right-hand side of the contraction, as a NamedArray with only
      named axes.

  Returns:
    Result of the contraction, which will have all names present in either
    `left` or `right`, except for the names in `names` which will be contracted
    away.
  """
  names = (names,) if isinstance(names, str) else names
  return named_axes.nmap(jnp.tensordot)(
      left.untag(*names), right.untag(*names), axes=len(names)
  )


@struct.pytree_dataclass
class NamedEinsum(layer_base.Layer):
  """An Einsum operation that contracts based on axis names.

  This layer behaves like a standard einsum tensor contraction, but indexed by
  axis names instead of by position. In its full generality, it is specified
  based on mappings from each named axis to the summation index to use, e.g.
  an einsum "thp,shp->hts" could be specified as ::

    NamedEinsum(
        (
            {"tokens":"t", "heads":"h", "projection":"p"},
            {"kv_tokens":"s", "heads":"h", "projection":"p"}
        ),
        {"heads":"h", "tokens":"t", "kv_tokens":"s"}
    )

  For the common case where each axis name should have its own summation index,
  you can also omit the values and just write something like ::

    NamedEinsum(
        (
            ("tokens", "heads", "projection"),
            ("kv_tokens", "heads", "projection"),
        ),
        ("heads", "tokens", "kv_tokens"),
    )

  Additionally, arbitrary batch axes can be added as long as they are present in
  every array, and will be added to the output array.

  Attributes:
    inputs_axes: Tuple of axis name specifications for each of the inputs. Each
      specification is either a mapping from axis names to a summation index
      name, or just a tuple of axis names if the summation indices should be the
      same as the axis names.
    output_axes: Specification of axis names in the output.
  """

  input_axes: tuple[tuple[str, ...] | dict[str, str], ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  output_axes: tuple[str, ...] | dict[str, str] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  @layer_base.checked_layer_call
  def __call__(
      self, x: tuple[named_axes.NamedArray, ...]
  ) -> named_axes.NamedArray:
    """Runs the einsum operation."""
    all_sum_names = set()
    for axname_to_sumname in self.input_axes:
      if isinstance(axname_to_sumname, tuple):
        axname_to_sumname = {ax: ax for ax in axname_to_sumname}
      all_sum_names.update(axname_to_sumname.values())

    # Build an einsum pattern using the non-string specification mode:
    einsum_args = []
    for x, spec in zip(x, self.input_axes):
      if isinstance(spec, tuple):
        axname_to_sumname = {ax: ax for ax in spec}
      else:
        axname_to_sumname = spec
      local_ordering = [
          axname for axname in x.named_shape if axname in axname_to_sumname
      ]
      einsum_args.append(x.untag(*local_ordering))
      einsum_args.append(
          [axname_to_sumname[axname] for axname in local_ordering]
      )

    if isinstance(self.output_axes, tuple):
      axname_to_sumname = {ax: ax for ax in self.output_axes}
    else:
      axname_to_sumname = self.output_axes
    einsum_args.append(list(axname_to_sumname.values()))
    output = named_axes.nmap(jnp.einsum)(*einsum_args)
    return output.tag(*axname_to_sumname.keys())

  def input_structure(self) -> shapecheck.StructureAnnotation:
    structures = []
    for axname_to_sumname in self.input_axes:
      if isinstance(axname_to_sumname, tuple):
        axname_to_sumname = {ax: ax for ax in axname_to_sumname}
      structures.append(
          shapecheck.ArraySpec(
              named_shape={
                  **shapecheck.var("B"),
                  **{
                      axname: shapecheck.var("I")[sumname]
                      for axname, sumname in axname_to_sumname.items()
                  },
              },
              dtype=jnp.floating,
          )
      )
    return tuple(structures)

  def output_structure(self) -> shapecheck.StructureAnnotation:
    if isinstance(self.output_axes, tuple):
      axname_to_sumname = {ax: ax for ax in self.output_axes}
    else:
      axname_to_sumname = self.output_axes
    return shapecheck.ArraySpec(
        named_shape={
            **shapecheck.var("B"),
            **{
                axname: shapecheck.var("I")[sumname]
                for axname, sumname in axname_to_sumname.items()
            },
        },
        dtype=jnp.floating,
    )


@struct.pytree_dataclass
class Linear(layer_base.Layer):
  """A generalized linear (not affine) operator, for named arrays.

  Applies an arbitrary contraction to the input `NamedArray` and a weight
  parameter. This can be used to express an arbitrary linear operator.

  ``Linear`` layers are often (but not always) followed by `AddBias` to make an
  affine transformation.

  Attributes:
    weights: The named array holding the weights for the linear operator.
    in_axis_names: The names of the axes to contract with the input, removing
      them.
    out_axis_names: The names of the axes that should not appear in the input
      and will be inserted into the output.
  """

  weights: parameters.ParameterLike[NamedArray]
  in_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  out_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  @layer_base.checked_layer_call
  def __call__(self, in_array: NamedArray) -> NamedArray:
    """Runs the linear operator."""
    return contract(self.in_axis_names, in_array, self.weights.value)

  @classmethod
  def from_config(
      cls,
      input_axes: dict[str, int],
      output_axes: dict[str, int],
      parallel_axes: dict[str, int] | None = None,
      parallel_broadcast_axes: dict[str, int] | None = None,
      initializer: LinearOperatorWeightInitializer = xavier_uniform_initializer,
      dtype: jax.typing.DTypeLike = jnp.float32,
      rename_outputs_if_necessary: bool = True,
  ) -> Linear | LinearInPlace:
    """Constructs a ``Linear`` layer from a configuration.

    This can be used when building a new linear operator at the start of
    training. The returned operator will include `UninitializedParameter` nodes
    which should be initialized before training.

    Note: For the purposes of the initializer, the ``parallel_axes`` and
    ``parallel_broadcast_axes`` are treated in the same way, without
    participating in output-dimension variance scaling. However, after
    initialization, the ``parallel_broadcast_axes`` will be treated like extra
    output axes (and assumed not to be present in the input).

    Args:
      input_axes: Names and lengths for axes that the linear operator should
        contract over.
      output_axes: Names and lengths for new axes that the linear operator
        should produce. If any axis names overlap with ``input_axes``, the
        argument ``rename_outputs_if_necessary`` must be True.
      parallel_axes: Names and lengths for axes that should be processed in
        parallel. These axes should appear in both the input and the output, and
        the resulting linear operator will apply a different operator to each
        slice. (This is similar to a block-diagonal matrix.) Must not overlap
        with any axes named in ``input_axes`` or ``output_axes``.
      parallel_broadcast_axes: Names and lengths for axes that should be treated
        like ``parallel_axes`` but will only appear in the output. The input
        will be implicitly broadcast over these axes. Must not overlap with any
        axes named in ``input_axes``, ``output_axes`` or ``parallel_axes``.
      initializer: Function to use to initialize the weight.
      dtype: Dtype for the weight.
      rename_outputs_if_necessary: If True, and if ``output_axes`` and
        ``input_axes`` have overlapping names, avoids name conflicts by adding
        "primed" versions of the overlapping names, and returns an instance of
        `LinearInPlace` instead of a ``Linear`` layer directly.

    Returns:
      A ``Linear`` layer with uninitialized weights, or possibly a
      `LinearInPlace` layer if ``rename_outputs_if_necessary`` is True and
      ``input_axes`` overlaps with ``output_axes``.
    """
    if parallel_axes is None:
      parallel_axes = {}
    if parallel_broadcast_axes is None:
      parallel_broadcast_axes = {}
    if any(name in input_axes for name in output_axes):
      # Name overlap!
      if rename_outputs_if_necessary:
        output_axes_after_rename = {}
        original_names = []
        primed_names = []

        for old_name in output_axes.keys():
          if old_name in input_axes:
            primed_name = old_name + "_out"
            if primed_name in input_axes:
              raise ValueError(
                  f"Tried to rename {old_name} to {primed_name} to avoid a"
                  " conflict, but both names are already in input_axes. Please"
                  " rename axes manually to avoid this conflict."
              )
            original_names.append(old_name)
            primed_names.append(primed_name)
            output_axes_after_rename[primed_name] = output_axes[old_name]
          else:
            output_axes_after_rename[old_name] = output_axes[old_name]

        return LinearInPlace(
            sublayers=[
                cls.from_config(
                    input_axes=input_axes,
                    output_axes=output_axes_after_rename,
                    parallel_axes=parallel_axes,
                    parallel_broadcast_axes=parallel_broadcast_axes,
                    initializer=initializer,
                    dtype=dtype,
                    rename_outputs_if_necessary=False,
                ),
                RenameAxes(old=tuple(primed_names), new=tuple(original_names)),
            ],
        )
      else:
        raise ValueError(
            "input_axes and output_axes must not overlap if"
            " rename_outputs_if_necessary is not set; got"
            f" input_axes={input_axes}, output_axes={output_axes}."
        )

    if set(parallel_axes).intersection(
        set(input_axes).union(output_axes)
    ) or set(parallel_broadcast_axes).intersection(
        set(input_axes).union(output_axes, parallel_axes)
    ):
      raise ValueError(
          "parallel_axes and parallel_broadcast_axes must not overlap with"
          f" each other or with input/output axes; got input_axes={input_axes},"
          f" output_axes={output_axes}, parallel_axes={parallel_axes},"
          f" parallel_broadcast_axes={parallel_broadcast_axes}."
      )

    return cls(
        weights=parameters.UninitializedParameter(
            functools.partial(
                initializer,
                input_axes=input_axes,
                output_axes=output_axes,
                parallel_axes={**parallel_axes, **parallel_broadcast_axes},
                convolution_spatial_axes={},
                dtype=dtype,
            ),
            name="weights",
        ),
        in_axis_names=tuple(input_axes.keys()),
        out_axis_names=(
            tuple(output_axes.keys()) + tuple(parallel_broadcast_axes.keys())
        ),
    )

  def input_structure(self):
    known_in_axes = {
        name: size
        for name, size in self.weights.value_structure.named_shape.items()
        if name not in self.out_axis_names
    }
    return shapecheck.ArraySpec(
        named_shape={**shapecheck.var("B"), **known_in_axes},
        dtype=jnp.floating,
    )

  def output_structure(self):
    known_out_axes = {
        name: size
        for name, size in self.weights.value_structure.named_shape.items()
        if name not in self.in_axis_names
    }
    return shapecheck.ArraySpec(
        named_shape={**shapecheck.var("B"), **known_out_axes},
        dtype=jnp.floating,
    )

  def treescope_color(self) -> str:
    return "#eba875"

  @property
  def input_axes(self) -> dict[str, int]:
    """The axis names and sizes that should appear in the input only."""
    return {
        name: size
        for name, size in self.weights.value_structure.named_shape.items()
        if name in self.in_axis_names
    }

  @property
  def output_axes(self) -> dict[str, int]:
    """The axis names and sizes that will appear in the output only."""
    return {
        name: size
        for name, size in self.weights.value_structure.named_shape.items()
        if name in self.out_axis_names
    }

  @property
  def parallel_axes(self) -> dict[str, int]:
    """The axis names and sizes that should appear in both the input and output."""
    return {
        name: size
        for name, size in self.weights.value_structure.named_shape.items()
        if name not in self.in_axis_names and name not in self.out_axis_names
    }


@struct.pytree_dataclass
class AddBias(layer_base.Layer):
  """Shifts an input by a learnable offset (a bias).

  This layer uses named arrays to automatically apply across the correct
  set of dimensions.

  Attributes:
    bias: The learnable bias.
    new_axis_names: The new axes in the output that we do not expect to see in
      the input.
  """

  bias: parameters.ParameterLike[NamedArray]
  new_axis_names: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )

  @layer_base.checked_layer_call
  def __call__(self, value: NamedArray) -> NamedArray:
    """Adds a learned bias to the value."""
    # Elementwise functions broadcast automatically
    return value + self.bias.value

  @classmethod
  def from_config(
      cls,
      biased_axes: dict[str, int],
      new_output_axes: dict[str, int] | None = None,
      initializer: BiasInitializer = jax.nn.initializers.zeros,
      dtype: jax.typing.DTypeLike = jnp.float32,
  ) -> AddBias:
    """Constructs an ``AddBias`` layer from a configuration.

    Args:
      biased_axes: Names and lengths for the axes in the input that the bias
        should act over. Other axes will be broadcast over.
      new_output_axes: Names and lengths of new axes that should be introduced
        into the input.
      initializer: Function to use to initialize the weight.
      dtype: Dtype for the bias.

    Returns:
      A new ``AddBias`` layer with an uninitialized bias parameter.
    """
    if new_output_axes is None:
      new_output_axes = {}
    bias_shape = []
    bias_names = []
    for name, size in itertools.chain(
        biased_axes.items(), new_output_axes.items()
    ):
      if name in bias_names:
        raise ValueError(f"Duplicate axis name in bias: {name}")
      bias_names.append(name)
      bias_shape.append(size)

    def bias_initializer(prng_key):
      return named_axes.wrap(
          initializer(prng_key, tuple(bias_shape), dtype)
      ).tag(*bias_names)

    return cls(
        bias=parameters.UninitializedParameter(bias_initializer, name="bias"),
        new_axis_names=tuple(new_output_axes.keys()),
    )

  def input_structure(self):
    known_in_axes = {
        name: size
        for name, size in self.bias.value_structure.named_shape.items()
        if name not in self.new_axis_names
    }
    return shapecheck.ArraySpec(
        named_shape={**shapecheck.var("B"), **known_in_axes},
        dtype=jnp.floating,
    )

  def output_structure(self):
    for name in self.new_axis_names:
      if name not in self.bias.value_structure.named_shape:
        raise layer_base.MisconfiguredLayerError(
            f"Expected out axis {name} was missing in bias parameter"
        )
    return shapecheck.ArraySpec(
        named_shape={
            **self.bias.value_structure.named_shape,
            **shapecheck.var("B"),
        },
        dtype=jnp.floating,
    )

  def treescope_color(self) -> str:
    return "#65cfbc"


@struct.pytree_dataclass
class Affine(grouping.Sequential):
  """Affine layer: combination of `Linear` and `AddBias`."""

  sublayers: list[layer_base.LayerLike]

  @classmethod
  def from_config(
      cls,
      input_axes: dict[str, int],
      output_axes: dict[str, int],
      parallel_axes: dict[str, int] | None = None,
      parallel_broadcast_axes: dict[str, int] | None = None,
      linear_initializer: LinearOperatorWeightInitializer = xavier_uniform_initializer,
      bias_initializer: BiasInitializer = jax.nn.initializers.zeros,
      dtype: jax.typing.DTypeLike = jnp.float32,
      rename_outputs_if_necessary: bool = True,
  ) -> Affine:
    if parallel_axes is None:
      parallel_axes = {}
    if parallel_broadcast_axes is None:
      parallel_broadcast_axes = {}
    linear = parameters.add_parameter_prefix(
        "Linear",
        Linear.from_config(
            input_axes=input_axes,
            output_axes=output_axes,
            parallel_axes=parallel_axes,
            parallel_broadcast_axes=parallel_broadcast_axes,
            initializer=linear_initializer,
            dtype=dtype,
            rename_outputs_if_necessary=rename_outputs_if_necessary,
        ),
    )
    add_bias = parameters.add_parameter_prefix(
        "AddBias",
        AddBias.from_config(
            {
                **parallel_axes,
                **parallel_broadcast_axes,
                **output_axes,
            },
            initializer=bias_initializer,
            dtype=dtype,
        ),
    )
    if isinstance(linear, LinearInPlace):
      return cls(sublayers=[*linear.sublayers, add_bias])
    else:
      return cls(sublayers=[linear, add_bias])

  def treescope_color(self) -> tuple[str, str]:
    return "#eba875", "color-mix(in oklab, #eba875 25%, white)"


@struct.pytree_dataclass
class ConstantRescale(layer_base.Layer):
  """Applies a constant scaling factor.

  Attributes:
    by: The constant scaling factor.
  """

  by: float

  @layer_base.checked_layer_call
  def __call__(self, value: Any) -> Any:
    """Scales its input by the scaling factor."""
    return jax.tree_util.tree_map(lambda x: x * self.by, value)

  def input_structure(self):
    return shapecheck.ArraySpec(
        shape=tuple(shapecheck.var("P")),
        named_shape=dict(shapecheck.var("N")),
        dtype=jnp.floating,
    )

  def output_structure(self):
    return self.input_structure()
