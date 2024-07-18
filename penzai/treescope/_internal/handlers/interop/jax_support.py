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

"""Lazy setup logic for adding JAX support to treescope."""

from __future__ import annotations

import typing
from typing import Mapping

import numpy as np
from penzai.treescope import canonical_aliases
from penzai.treescope import context
from penzai.treescope import dtype_util
from penzai.treescope import lowering
from penzai.treescope import ndarray_adapters
from penzai.treescope import renderer
from penzai.treescope import rendering_parts
from penzai.treescope import repr_lib
from penzai.treescope import type_registries
from penzai.treescope._internal.parts import part_interface

# pylint: disable=g-import-not-at-top
try:
  import jax
except ImportError:
  assert not typing.TYPE_CHECKING
  jax = None
# pylint: enable=g-import-not-at-top


def _finite_mean_std_any(array):
  """Helper to compute mean and standard deviation only over finite elements."""
  assert jax is not None
  jnp = jax.numpy
  isfinite = jnp.isfinite(array)
  inf_to_nan = jnp.where(isfinite, array, jnp.array(jnp.nan, dtype=array.dtype))
  mean = jnp.nanmean(inf_to_nan)
  std = jnp.nanstd(inf_to_nan)
  return mean, std, jnp.any(isfinite)


def _is_subdtype(dtype, base) -> bool:
  """Safely checks for dtype subtyping."""
  assert jax is not None
  jnp = jax.numpy
  try:
    return jnp.issubdtype(dtype, base)
  except TypeError:
    return False


summarization_threshold: context.ContextualValue[Mapping[str, int | None]] = (
    context.ContextualValue(
        module=__name__,
        qualname="summarization_threshold",
        initial_value={
            "tpu": 1_000_000_000,
            "gpu": 10_000_000,
            "default": 100_000,
        },
    )
)
"""Threshold for summarization of NDArrays for each backend.

This threshold determines the largest number of elements we will
summarize with summary statistics (e.g. mean, standard deviation)
when rendering in treescope. Larger values may make it slower to
display large NDArrays.

Each key should be the name of a JAX array platform, e.g. "cpu" or
"tpu". It can also be "numpy" to refer to Numpy arrays, or "default"
to refer to any other accelerator. The value is the size of the
array at which point we avoid showing summary statistics. `None`
means no limit.

This configuration argument is intended to be set at the top level
by the user, e.g. in IPython.
"""


def safe_to_summarize(array: jax.Array) -> bool:
  """Checks if the array is safe to summarize (not a tracer and not replicated)."""
  assert jax is not None, "JAX is not available."
  if isinstance(array, jax.core.Tracer):
    return False
  if array.is_deleted():
    return False
  if not (
      getattr(array, "is_fully_addressable", False)
      or getattr(array, "is_fully_replicated", False)
  ):
    return False
  thresh_dict = summarization_threshold.get()
  [platform] = set(device.platform for device in array.devices())
  thresh = thresh_dict.get(platform)
  if thresh is None:
    thresh = thresh_dict["default"]
  return thresh is None or array.size < thresh


def _truncate_part_with_slices(
    array: jax.Array,
    mask: jax.Array,
    prefix_slices: tuple[slice, ...],
    remaining_edge_items_per_axis: tuple[int | None, ...],
) -> tuple[jax.Array, jax.Array]:
  """Helper to truncate names of an array.

  Args:
    array: An array to truncate.
    mask: Mask array, which must have the same number of dimensions as `array`,
      and whose axis sizes must be either 1 or the same as that axis of `array`
      (e.g. they are broadcast compatible).
    prefix_slices: Slices to apply to each axis of `array` and `mask`, starting
      at axis 0, which we have already computed.
    remaining_edge_items_per_axis: Number of edge items to keep for each axis,
      ignoring any axes whose slices are already computed in `prefix_slices`.

  Returns:
    Truncated array and mask, which will both be the same shape.
  """
  assert jax is not None, "JAX is not available."
  jnp = jax.numpy
  if not remaining_edge_items_per_axis:
    # Perform the base case slice.
    assert len(prefix_slices) == len(array.shape)
    truncated_array = array[prefix_slices]

    valid_mask_slices = tuple(
        slice(None) if mask.shape[i] == 1 else array_slice
        for i, array_slice in enumerate(prefix_slices)
    )
    truncated_mask = jnp.broadcast_to(
        jnp.array(mask[valid_mask_slices]), truncated_array.shape
    )
    return truncated_array, truncated_mask

  # Recursive step: extract one name, run the function on each side, and
  # concatenate.
  axis = len(prefix_slices)
  edge_items = remaining_edge_items_per_axis[0]
  if edge_items is None:
    # Don't need to slice.
    return _truncate_part_with_slices(
        array,
        mask,
        prefix_slices=prefix_slices + (slice(None),),
        remaining_edge_items_per_axis=remaining_edge_items_per_axis[1:],
    )
  else:
    assert array.shape[axis] > 2 * edge_items
    result_a, valid_a = _truncate_part_with_slices(
        array,
        mask,
        prefix_slices=prefix_slices + (slice(None, edge_items),),
        remaining_edge_items_per_axis=remaining_edge_items_per_axis[1:],
    )
    result_b, valid_b = _truncate_part_with_slices(
        array,
        mask,
        prefix_slices=prefix_slices + (slice(-edge_items, None),),
        remaining_edge_items_per_axis=remaining_edge_items_per_axis[1:],
    )
    padding_shape = list(result_a.shape)
    padding_shape[axis] = 1
    result = jnp.concatenate(
        [result_a, jnp.zeros(padding_shape, result_a.dtype), result_b],
        axis=axis,
    )
    valid = jnp.concatenate(
        [valid_a, jnp.zeros(padding_shape, valid_a.dtype), valid_b], axis=axis
    )
    return result, valid


def truncate_array_and_mask(
    array: jax.Array,
    mask: jax.Array,
    edge_items_per_axis: tuple[int | None, ...],
) -> tuple[jax.Array, jax.Array]:
  """Truncates an array along the given axis names.

  Args:
    array: Array to truncate.
    mask: Mask array, which must have the same number of dimensions as `array`,
      and whose axis sizes must be either 1 or the same as that axis of `array`
      (e.g. they are broadcast compatible).
    edge_items_per_axis: Number of edge items to keep for each axis, ignoring
      any axes whose slices are already computed in `prefix_slices`.

  Returns:
    A tuple containing a truncated version of the array along with a valid mask.
    Values taken from the original array have the valid mask as True, and there
    is one extra element in the middle with valid as False (standing in for the
    omitted elements). The return value is always fully replicated, because
    we cannot guarantee that it is evenly sharded across devices, and this
    function is usually used immediately before copying to the host.
  """
  assert jax is not None, "JAX is not available."
  sharding_kwargs = {}
  if hasattr(array, "sharding") and hasattr(
      array.sharding, "_device_assignment"
  ):
    # _truncate_part_with_slices usually returns slices that have odd
    # dimensions, which aren't divisible by most shardings. Unfortunately,
    # the XLA GSPMD partitioner sometimes still infers a sharding over one of
    # these axes, which then leads to partitioning errors in JAX whenever we
    # try to `device_get` the resulting array or call any additional operations
    # on it. To avoid this, we'd like to tell JAX to always produce an output
    # that is not sharded over any axis. Unfortunately, this is difficult
    # because JAX requires the in_shardings and out_shardings to have the same
    # devices in the same internal order, and at the time of writing JAX does
    # not provide any public API to look up the order of the devices in a
    # sharding (it allows looking up the device *set*, but not their order).
    # Whether or not this error happens seems to be somewhat nondeterministic.
    # To avoid this, we use the private property `_device_assignment` of
    # each sharding in order to figure out what device order it has, and then
    # explicitly request a fully-replicated output that is definitely safe to
    # retrieve.
    sharding_kwargs["out_shardings"] = (
        jax.sharding.GSPMDSharding.get_replicated(
            array.sharding._device_assignment  # pylint: disable=protected-access
        )
    )
  fn = jax.jit(
      _truncate_part_with_slices, static_argnums=(2, 3), **sharding_kwargs
  )
  return fn(array, mask, (), edge_items_per_axis)


def faster_array_repr(array: jax.Array) -> str:
  """Computes ``repr(array)``, only copying the rendered array elements.

  ``repr(array)`` on a very large jax Array can be slow, because it copies the
  entire array to host memory even when only a few elements are actually needed.
  We can avoid this by truncating the array on device before fetching it.

  Args:
    array: The array to summarize.

  Returns:
    A string representation of the array. May differ slightly from the ordinary
    ``repr``, but should contain the same elements.
  """
  assert jax is not None, "JAX is not available."
  jnp = jax.numpy
  if array.size < np.get_printoptions()["threshold"]:
    return repr(array)

  if array.aval is not None and array.aval.weak_type:
    dtype_str = f"dtype={array.dtype.name}, weak_type=True)"
  else:
    dtype_str = f"dtype={array.dtype.name})"

  edgeitems = np.get_printoptions()["edgeitems"]
  edge_items_per_axis = []
  for size in array.shape:
    if size > 2 * edgeitems + 1:
      edge_items_per_axis.append(edgeitems)
    else:
      edge_items_per_axis.append(None)
  array_edges, _ = truncate_array_and_mask(
      array,
      jnp.ones((1,) * array.ndim, dtype=jnp.bool_),
      edge_items_per_axis=tuple(edge_items_per_axis),
  )
  prefix = "Array("
  datastring = np.array2string(
      np.array(array_edges),
      prefix=prefix,
      suffix=",",
      separator=", ",
      threshold=0,
      edgeitems=edgeitems,
  )
  return f"{prefix}{datastring}, {dtype_str}"


def render_shape_dtype_struct(
    node: jax.ShapeDtypeStruct,
    path: str | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders jax.ShapeDtypeStruct."""
  assert jax is not None, "JAX is not available."
  if type(node) is not jax.ShapeDtypeStruct:  # pylint: disable=unidiomatic-typecheck
    return NotImplemented
  attributes = {
      "shape": node.shape,
      "dtype": node.dtype,
  }
  if node.named_shape:
    attributes["named_shape"] = node.named_shape
  if node.sharding is not None:
    attributes["sharding"] = node.sharding

  # Make sure we can correctly round-trip it. We check because ShapeDtypeStruct
  # occasionally adds new attributes for new JAX features.
  rebuilt = jax.ShapeDtypeStruct(**attributes)
  if rebuilt != node:
    return NotImplemented
  else:
    return repr_lib.render_object_constructor(
        object_type=jax.ShapeDtypeStruct,
        attributes=attributes,
        path=path,
        subtree_renderer=subtree_renderer,
        roundtrippable=True,
    )


def render_precision(
    node: jax.lax.Precision,
    path: str | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders jax.lax.Precision."""
  assert jax is not None, "JAX is not available."
  if type(node) is not jax.lax.Precision:  # pylint: disable=unidiomatic-typecheck
    return NotImplemented
  return repr_lib.render_enumlike_item(
      object_type=jax.lax.Precision,
      item_name=node.name,
      item_value=node.value,
      path=path,
      subtree_renderer=subtree_renderer,
  )


def summarize_array_data(array: jax.Array) -> str:
  """Summarized the data of a JAX array.

  Args:
    array: The array to summarize.

  Returns:
    A string summarizing the data of the array.
  """
  assert jax is not None, "JAX is not available."
  jnp = jax.numpy
  output_parts = []
  if array.is_deleted():
    output_parts.append(" - deleted!")
  elif safe_to_summarize(array):
    with jax.core.ensure_compile_time_eval():
      is_floating = _is_subdtype(array.dtype, jnp.floating)
      is_integer = _is_subdtype(array.dtype, jnp.integer)
      is_bool = _is_subdtype(array.dtype, jnp.bool_)

      if is_floating:
        mean, std, any_finite = jax.jit(_finite_mean_std_any)(array)

        if any_finite:
          output_parts.append(f" ≈{float(mean):.2} ±{float(std):.2}")
          output_parts.append(
              f" [≥{float(jnp.nanmin(array)):.2},"
              f" ≤{float(jnp.nanmax(array)):.2}]"
          )

      if is_integer:
        output_parts.append(f" [≥{jnp.min(array):_d}, ≤{jnp.max(array):_d}]")

      if is_floating or is_integer:
        ct_zero = jnp.count_nonzero(array == 0)
        if ct_zero:
          output_parts.append(f" zero:{ct_zero:_d}")

        ct_nonzero = jnp.count_nonzero(array)
        if ct_nonzero:
          output_parts.append(f" nonzero:{ct_nonzero:_d}")

      if is_floating:
        ct_nan = jnp.count_nonzero(jnp.isnan(array))
        if ct_nan:
          output_parts.append(f" nan:{ct_nan:_d}")

        ct_inf = jnp.count_nonzero(jnp.isposinf(array))
        if ct_inf:
          output_parts.append(f" inf:{ct_inf:_d}")

        ct_neginf = jnp.count_nonzero(jnp.isneginf(array))
        if ct_neginf:
          output_parts.append(f" -inf:{ct_neginf:_d}")

      if is_bool:
        ct_true = jnp.count_nonzero(array)
        if ct_true:
          output_parts.append(f" true:{ct_true:_d}")

        ct_false = jnp.count_nonzero(jnp.logical_not(array))
        if ct_false:
          output_parts.append(f" false:{ct_false:_d}")
  return "".join(output_parts)


class JAXArrayAdapter(ndarray_adapters.NDArrayAdapter[jax.Array]):
  """Array adapter for JAX arrays."""

  def get_axis_info_for_array_data(
      self, array: jax.Array
  ) -> tuple[ndarray_adapters.AxisInfo, ...]:
    assert jax is not None, "JAX is not available."
    return tuple(
        ndarray_adapters.PositionalAxisInfo(i, size)
        for i, size in enumerate(array.shape)
    )

  def get_array_data_with_truncation(
      self,
      array: jax.Array,
      mask: jax.Array | None,
      edge_items_per_axis: tuple[int | None, ...],
  ) -> tuple[jax.Array, jax.Array]:
    assert jax is not None, "JAX is not available."
    jnp = jax.numpy
    assert not isinstance(array, jax.core.Tracer)
    assert not array.is_deleted()
    if mask is not None:
      # Make sure we can broadcast the shape correctly.
      _ = jax.eval_shape(lambda: jnp.broadcast_to(mask, array.shape))
      mask = mask[(None,) * (array.ndim - mask.ndim) + (...,)]
    else:
      mask = jnp.ones((1,) * array.ndim, dtype=jnp.bool_)

    if edge_items_per_axis == (None,) * array.ndim:
      # No truncation.
      return array, jnp.broadcast_to(mask, array.shape)

    return truncate_array_and_mask(array, mask, edge_items_per_axis)

  def get_array_summary(self, array: jax.Array, fast: bool) -> str:
    output_parts = ["jax.Array "]

    output_parts.append(dtype_util.get_dtype_name(array.dtype))
    output_parts.append(repr(array.shape))
    if array.is_deleted():
      output_parts.append(" - deleted!")
    elif not fast:
      output_parts.append(summarize_array_data(array))

    return "".join(output_parts)

  def get_numpy_dtype(self, array: jax.Array) -> np.dtype | None:
    if isinstance(array.dtype, np.dtype):
      return array.dtype
    else:
      return None

  def get_sharding_info_for_array_data(
      self, array: jax.Array
  ) -> ndarray_adapters.ShardingInfo | None:
    assert jax is not None, "JAX is not available."
    if isinstance(array, jax.core.Tracer) or array.is_deleted():
      return None

    [platform] = set(device.platform for device in array.sharding.device_set)
    device_map = array.sharding.devices_indices_map(array.shape)
    return ndarray_adapters.ShardingInfo(
        shard_shape=array.sharding.shard_shape(array.shape),
        device_index_to_shard_slices={
            device.id: slices for device, slices in device_map.items()
        },
        device_type=platform.upper(),
        fully_replicated=array.is_fully_replicated,
    )

  def should_autovisualize(self, array: jax.Array) -> bool:
    assert jax is not None, "JAX is not available."
    return not isinstance(array, jax.core.Tracer) and not array.is_deleted()


def render_jax_arrays(
    node: jax.Array,
    path: str | None,
    subtree_renderer: renderer.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a JAX array."""
  assert jax is not None, "JAX is not available."
  del subtree_renderer
  assert isinstance(node, jax.Array)
  if isinstance(node, jax.core.Tracer):
    return NotImplemented

  adapter = JAXArrayAdapter()

  if node.is_deleted():
    return rendering_parts.error_color(
        rendering_parts.text(
            "<" + adapter.get_array_summary(node, fast=True) + ">"
        )
    )

  def _placeholder() -> rendering_parts.RenderableTreePart:
    return rendering_parts.fake_placeholder_foldable(
        rendering_parts.deferred_placeholder_style(
            rendering_parts.text(adapter.get_array_summary(node, fast=True))
        ),
        extra_newlines_guess=8,
    )

  def _thunk(placeholder):
    # Is this array simple enough to render without a summary?
    node_repr = faster_array_repr(node)
    if "\n" not in node_repr and "..." not in node_repr:
      rendering = rendering_parts.abbreviation_color(
          rendering_parts.text(f"<jax.{node_repr}>")
      )
    else:
      if node_repr.count("\n") <= 15:
        if isinstance(placeholder, part_interface.FoldableTreeNode):
          default_expand_state = placeholder.get_expand_state()
        else:
          assert placeholder is None
          default_expand_state = part_interface.ExpandState.WEAKLY_EXPANDED
      else:
        # Always start big NDArrays in collapsed mode to hide irrelevant detail.
        default_expand_state = rendering_parts.ExpandState.COLLAPSED

      # Render it with a summary.
      summarized = adapter.get_array_summary(node, fast=False)
      rendering = rendering_parts.build_custom_foldable_tree_node(
          label=rendering_parts.abbreviation_color(
              rendering_parts.comment_color_when_expanded(
                  rendering_parts.siblings(
                      rendering_parts.fold_condition(
                          expanded=rendering_parts.text("# "),
                          collapsed=rendering_parts.text("<"),
                      ),
                      summarized,
                      rendering_parts.fold_condition(
                          collapsed=rendering_parts.text(">")
                      ),
                  )
              )
          ),
          contents=rendering_parts.fold_condition(
              expanded=rendering_parts.indented_children(
                  [rendering_parts.text(node_repr)]
              )
          ),
          path=path,
          expand_state=default_expand_state,
      ).renderable

    return rendering

  return rendering_parts.RenderableAndLineAnnotations(
      renderable=lowering.maybe_defer_rendering(
          main_thunk=_thunk, placeholder_thunk=_placeholder
      ),
      annotations=rendering_parts.build_copy_button(path),
  )


def set_up_treescope():
  """Sets up treescope to render JAX objects."""
  if jax is None:
    raise RuntimeError(
        "Cannot set up JAX support in treescope: JAX cannot be imported."
    )
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.ShapeDtypeStruct] = (
      render_shape_dtype_struct
  )
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.lax.Precision] = (
      render_precision
  )

  # The concrete type of a JAX array is a private type that is dynamically
  # registered as a jax.Array subclass, so we need to add it to the list of
  # dynamically-checked virtual base classes.
  type_registries.VIRTUAL_BASE_CLASSES.append(jax.Array)
  type_registries.IMMUTABLE_TYPES_REGISTRY[jax.Array] = True
  type_registries.NDARRAY_ADAPTER_REGISTRY[jax.Array] = JAXArrayAdapter()
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.Array] = render_jax_arrays

  for jax_api_module in [
      jax.lax,
      jax.numpy,
      jax.scipy,
      jax.random,
      jax.nn,
      jax.custom_derivatives,
      jax,
  ]:
    canonical_aliases.populate_from_public_api(
        jax_api_module, canonical_aliases.prefix_filter("jax")
    )
