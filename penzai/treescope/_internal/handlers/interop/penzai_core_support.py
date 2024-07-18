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

"""Lazy setup logic for adding `penzai.core` support to treescope.

This is defined in a separate lazily-imported module to allow `penzai.treescope`
to render Penzai's named arrays if they are used, but not require importing
`penzai.core` if they are not used.
"""

import jax
import numpy as np
from penzai.core import named_axes
from penzai.core._treescope_handlers import named_axes_handlers
from penzai.treescope import ndarray_adapters
from penzai.treescope import type_registries
from penzai.treescope._internal.handlers.interop import jax_support


class NamedArrayAdapter(
    ndarray_adapters.NDArrayAdapter[named_axes.NamedArrayBase]
):
  """Array adapter for Penzai named arrays."""

  def get_axis_info_for_array_data(
      self, array: named_axes.NamedArrayBase
  ) -> tuple[ndarray_adapters.AxisInfo, ...]:
    array = array.as_namedarrayview()
    infos = {}
    for name, axis in array.data_axis_for_name.items():
      infos[axis] = ndarray_adapters.NamedPositionlessAxisInfo(
          axis_name=name,
          size=array.data_shape[axis],
      )
    for logical_axis, axis in enumerate(array.data_axis_for_logical_axis):
      infos[axis] = ndarray_adapters.PositionalAxisInfo(
          axis_logical_index=logical_axis,
          size=array.data_shape[axis],
      )
    return tuple(infos[i] for i in range(len(infos)))

  def get_array_data_with_truncation(
      self,
      array: named_axes.NamedArrayBase,
      mask: named_axes.NamedArrayBase | jax.Array | np.ndarray | None,
      edge_items_per_axis: tuple[int | None, ...],
  ) -> tuple[named_axes.NamedArrayBase, named_axes.NamedArrayBase]:
    array = array.as_namedarrayview()
    if mask is None:
      mask_data = None
    else:
      # Make sure mask is compatible.
      if not isinstance(mask, named_axes.NamedArrayBase):
        mask = named_axes.wrap(mask)
      bad_names = set(mask.named_shape.keys()) - set(array.named_shape.keys())
      if bad_names:
        raise ValueError(
            "Valid mask must be broadcastable to the shape of `array`, but it"
            f" had extra axis names {bad_names}"
        )

      vshape = mask.positional_shape
      ashape = array.positional_shape
      if np.broadcast_shapes(vshape, ashape) != ashape:
        raise ValueError(
            "Valid mask must be broadcastable to the shape of `array`, but its"
            f" positional shape {vshape} does not match (a suffix of) the"
            f" positional shape {ashape} of `array`"
        )

      # Insert new length-1 axes.
      new_names = set(array.named_shape.keys()) - set(mask.named_shape.keys())
      if new_names:
        mask = mask[{name: None for name in new_names}]
      new_positional_axis_count = len(ashape) - len(vshape)
      if new_positional_axis_count:
        mask = mask[(None,) * new_positional_axis_count + (...,)]

      # Possibly transpose the mask to match the main array, and extract its
      # data.
      mask_data = mask.order_like(array).data_array

    return jax_support.JAXArrayAdapter().get_array_data_with_truncation(
        array=array.data_array,
        mask=mask_data,
        edge_items_per_axis=edge_items_per_axis,
    )

  def get_array_summary(
      self, array: named_axes.NamedArrayBase, fast: bool
  ) -> str:
    summary, contained_type = (
        named_axes_handlers.named_array_and_contained_type_summary(
            array, inspect_device_data=not fast
        )
    )
    return f"{type(array).__name__} {summary} (wrapping {contained_type})"

  def get_numpy_dtype(
      self, array: named_axes.NamedArrayBase
  ) -> np.dtype | None:
    if isinstance(array.dtype, np.dtype):
      return array.dtype
    else:
      return None

  def get_sharding_info_for_array_data(
      self, array: named_axes.NamedArrayBase
  ) -> ndarray_adapters.ShardingInfo | None:
    array = array.as_namedarrayview()
    if not isinstance(array.data_array, jax.Array):
      return None
    return jax_support.JAXArrayAdapter().get_sharding_info_for_array_data(
        array.data_array
    )

  def should_autovisualize(self, array: named_axes.NamedArrayBase) -> bool:
    array = array.as_namedarrayview()
    return (
        isinstance(array.data_array, jax.Array)
        and not isinstance(array.data_array, jax.core.Tracer)
        and not array.data_array.is_deleted()
    )


def set_up_treescope():
  """Sets up treescope to render Penzai named arrays."""
  type_registries.NDARRAY_ADAPTER_REGISTRY[named_axes.NamedArrayBase] = (
      NamedArrayAdapter()
  )
