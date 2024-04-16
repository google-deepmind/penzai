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

"""Single-purpose ndarray visualizer for Python in vanilla Javascript.

Designed to visualize the contents of arbitrarily-high-dimensional NDArrays
quickly and without any dependencies, to allow them to be visualized by default
instead of requiring lots of manual effort.
"""

from __future__ import annotations

import base64
import collections
import dataclasses
import functools
import io
import itertools
import json
import os
from typing import Any, Literal, Mapping, Sequence
import uuid

import jax
import jax.numpy as jnp
import numpy as np
from penzai.core import context
from penzai.core import named_axes
from penzai.treescope import figures
from penzai.treescope import html_escaping
from penzai.treescope import ndarray_summarization
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import part_interface


def load_arrayvis_javascript() -> str:
  """Loads the contents of `arrayvis.js` from the Python package.

  Returns:
    Source code for arrayviz.
  """
  filepath = __file__
  if filepath is None:
    raise ValueError("Could not find the path to arrayviz.js!")

  # Look for the resource relative to the current module's filesystem path.
  base = filepath.removesuffix("arrayviz.py")
  load_path = os.path.join(base, "js", "arrayviz.js")

  with open(load_path, "r") as f:
    return f.read()


def _html_setup() -> (
    set[part_interface.CSSStyleRule | part_interface.JavaScriptDefn]
):
  """Builds the setup HTML that should be included in any arrayviz output cell."""
  return {
      part_interface.CSSStyleRule(html_escaping.without_repeated_whitespace("""
        .arrayviz_container {
            white-space: normal;
        }
        .arrayviz_container .info {
            font-family: monospace;
            color: #aaaaaa;
            margin-bottom: 0.25em;
            white-space: pre;
        }
        .arrayviz_container .info input[type="range"] {
            vertical-align: middle;
            filter: grayscale(1) opacity(0.5);
        }
        .arrayviz_container .info input[type="range"]:hover {
            filter: grayscale(0.5);
        }
        .arrayviz_container .info input[type="number"]:not(:focus) {
            border-radius: 3px;
        }
        .arrayviz_container .info input[type="number"]:not(:focus):not(:hover) {
            color: #777777;
            border: 1px solid #777777;
        }
        .arrayviz_container .info.sliders {
            white-space: pre;
        }
        .arrayviz_container .hovertip {
            display: none;
            position: absolute;
            background-color: white;
            border: 1px solid black;
            padding: 0.25ch;
            pointer-events: none;
            width: fit-content;
            overflow: visible;
            white-space: pre;
            z-index: 1000;
        }
        .arrayviz_container .hoverbox {
            display: none;
            position: absolute;
            box-shadow: 0 0 0 1px black, 0 0 0 2px white;
            pointer-events: none;
            z-index: 900;
        }
        .arrayviz_container .clickdata {
            white-space: pre;
        }
        .arrayviz_container .loading_message {
            color: #aaaaaa;
        }
      """)),
      part_interface.JavaScriptDefn(
          html_escaping.heuristic_strip_javascript_comments(
              load_arrayvis_javascript()
          )
      ),
  }


def _render_array_to_html(
    array_data: np.ndarray | jax.Array,
    valid_mask: np.ndarray | jax.Array,
    column_axes: Sequence[int],
    row_axes: Sequence[int],
    slider_axes: Sequence[int],
    axis_labels: list[str],
    vmin: float,
    vmax: float,
    cmap_type: Literal["continuous", "palette_index", "digitbox"],
    cmap_data: list[tuple[int, int, int]],
    info: str = "",
    formatting_instructions: list[dict[str, Any]] | None = None,
    dynamic_continous_cmap: bool = False,
    raw_min_abs: float | None = None,
    raw_max_abs: float | None = None,
) -> str:
  """Helper to render an array to HTML by passing arguments to javascript.

  Args:
    array_data: Array data to render.
    valid_mask: Mask array, of same shape as array_data, that is True for items
      we should render.
    column_axes: Axes (by index into `array_data`) to arrange as columns,
      ordered from outermost group to innermost group.
    row_axes: Axes (by index into `array_data`) to arrange as rows, ordered from
      outermost group to innermost group.
    slider_axes: Axes to bind to sliders.
    axis_labels: Labels for each axis.
    vmin: Minimum for the colormap.
    vmax: Maximum for the colormap.
    cmap_type: Type of colormap (see `render_array`)
    cmap_data: Data for the colormap, as a sequence of RGB triples.
    info: Info for the plot.
    formatting_instructions: Formatting instructions for values on mouse hover
      or click. These will be interpreted by `formatValueAndIndices` on the
      JavaScript side. Can assume each axis is named "a0", "a1", etc. when
      running in JavaScript.
    dynamic_continous_cmap: Whether to dynamically adjust the colormap during
      rendering.
    raw_min_abs: Minimum absolute value of the array, for dynamic remapping.
    raw_max_abs: Maximum absolute value of the array, for dynamic remapping.

  Returns:
    HTML source for an arrayviz rendering.
  """
  assert len(array_data.shape) == len(axis_labels)
  assert len(valid_mask.shape) == len(axis_labels)

  if formatting_instructions is None:
    formatting_instructions = [{"type": "value"}]

  # Compute strides for each axis. We refer to each axis as "a0", "a1", etc
  # across the JavaScript boundary.
  stride = 1
  strides = {}
  for i, axis_size in reversed(list(enumerate(array_data.shape))):
    strides[f"a{i}"] = stride
    stride *= axis_size

  if cmap_type == "continuous":
    converted_array_data = array_data.astype(np.float32)
    array_dtype = "float32"
  else:
    converted_array_data = array_data.astype(np.int32)
    array_dtype = "int32"

  def axis_spec_arg(i):
    return {
        "name": f"a{i}",
        "label": axis_labels[i],
        "start": 0,
        "end": array_data.shape[i],
    }

  x_axis_specs_arg = []
  for axis in column_axes:
    x_axis_specs_arg.append(axis_spec_arg(axis))

  y_axis_specs_arg = []
  for axis in row_axes:
    y_axis_specs_arg.append(axis_spec_arg(axis))

  sliced_axis_specs_arg = []
  for axis in slider_axes:
    sliced_axis_specs_arg.append(axis_spec_arg(axis))

  fresh_id = "arrayviz" + uuid.uuid4().hex
  args_json = json.dumps({
      "destinationId": fresh_id,
      "info": info,
      "arrayBase64": base64.b64encode(converted_array_data.tobytes()).decode(
          "ascii"
      ),
      "arrayDtype": array_dtype,
      "validMaskBase64": base64.b64encode(
          valid_mask.astype(np.uint8).tobytes()
      ).decode("ascii"),
      "dataStrides": strides,
      "xAxisSpecs": x_axis_specs_arg,
      "yAxisSpecs": y_axis_specs_arg,
      "slicedAxisSpecs": sliced_axis_specs_arg,
      "colormapConfig": {
          "type": cmap_type,
          "min": vmin,
          "max": vmax,
          "dynamic": dynamic_continous_cmap,
          "rawMinAbs": raw_min_abs,
          "rawMaxAbs": raw_max_abs,
          "cmapData": cmap_data,
      },
      "valueFormattingInstructions": formatting_instructions,
  })
  src = (
      f'<div id="{fresh_id}" class="arrayviz_container">'
      '<span class="loading_message">Rendering array...</span>'
      "</div>"
      '<template class="treescope_run_soon"><script>'
      f" arrayviz.buildArrayvizFigure({args_json})</script></template>"
  )
  return src


def infer_rows_and_columns(
    axis_sizes: dict[int | named_axes.AxisName, int],
    unassigned: Sequence[int | named_axes.AxisName] | None = None,
    known_rows: Sequence[int | named_axes.AxisName] = (),
    known_columns: Sequence[int | named_axes.AxisName] = (),
) -> tuple[list[int | named_axes.AxisName], list[int | named_axes.AxisName]]:
  """Infers an ordered assignment of axis indices or names to rows and columns.

  The unassigned axes are sorted by size and then assigned to rows and columns
  to try to balance the total number of elements along the row and column axes.
  Curently uses a greedy algorithm with an adjustment to try to keep columns
  longer than rows, except when there are exactly two axes and both are
  positional, in which case it lays out axis 0 as the rows and axis 1 as the
  columns.

  Args:
    axis_sizes: Mapping from axis indices or names to their axis size.
    unassigned: Sequence of unassigned axis indices or names. Inferred from the
      axis_sizes if not provided.
    known_rows: Sequence of axis indices or names that must map to rows.
    known_columns: Sequence of axis indices or names that must map to columns.

  Returns:
    Tuple (rows, columns) of assignments, which consist of `known_rows` and
    `known_columns` followed by the remaining unassigned axes in a balanced
    order.
  """
  if unassigned is None:
    unassigned = [
        key
        for key in axis_sizes.keys()
        if key not in known_rows and key not in known_columns
    ]

  if (
      not known_rows
      and not known_columns
      and len(unassigned) == 2
      and set(unassigned) == {0, 1}
  ):
    # Two-dimensional positional array. Always do rows then columns.
    return ([0], [1])

  # Sort by size descending, so that we make the most important layout decisions
  # first.
  unassigned = sorted(unassigned, key=lambda ax: -axis_sizes[ax])

  # Compute the total size every axis would have if we assigned them to the
  # same axis.
  unassigned_size = np.prod([axis_sizes[ax] for ax in unassigned])

  rows = list(known_rows)
  row_size = np.prod([axis_sizes[ax] for ax in rows])
  columns = list(known_columns)
  column_size = np.prod([axis_sizes[ax] for ax in columns])

  for ax in unassigned:
    axis_size = axis_sizes[ax]
    unassigned_size = unassigned_size // axis_size
    if row_size * axis_size > column_size * unassigned_size:
      # If we assign this to the row axis, we'll end up with a visualization
      # with more rows than columns regardless of what we do later, which can
      # waste screen space. Assign to columns instead.
      columns.append(ax)
      column_size *= axis_size
    else:
      # Assign to the row axis. We'll assign columns later.
      rows.append(ax)
      row_size *= axis_size

  # The specific ordering of axes along the rows and the columns is somewhat
  # arbitrary. Re-order each so that they have positional then named axes, and
  # so that position axes are in reverse position order, and the explicitly
  # mentioned named axes are before the unassigned ones.
  def ax_sort_key(ax: int | named_axes.AxisName):
    if isinstance(ax, int):
      return (0, -ax)
    elif ax in unassigned:
      return (2,)
    else:
      return (1,)

  return sorted(rows, key=ax_sort_key), sorted(columns, key=ax_sort_key)


@functools.partial(jax.jit, static_argnames=("around_zero", "trim_outliers"))
def _infer_vmin_vmax(
    array: jnp.Array,
    mask: jnp.Array,
    vmin: float | None,
    vmax: float | None,
    around_zero: bool,
    trim_outliers: bool,
) -> tuple[float | jax.Array, float | jax.Array]:
  """Infer reasonable lower and upper colormap bounds from an array."""
  inferring_both_bounds = vmax is None and vmin is None
  finite_mask = jnp.logical_and(jnp.isfinite(array), mask)
  if vmax is None:
    if around_zero:
      if vmin is not None:
        vmax = -vmin  # pylint: disable=invalid-unary-operand-type
      else:
        vmax = jnp.max(jnp.where(finite_mask, jnp.abs(array), 0))
    else:
      vmax = jnp.max(jnp.where(finite_mask, array, -np.inf))

  assert vmax is not None

  if vmin is None:
    if around_zero:
      vmin = -vmax  # pylint: disable=invalid-unary-operand-type
    else:
      vmin = jnp.min(jnp.where(finite_mask, array, np.inf))

  if inferring_both_bounds and trim_outliers:
    if around_zero:
      center = 0
    else:
      center = jnp.nanmean(jnp.where(finite_mask, array, np.nan))
      center = jnp.where(jnp.isfinite(center), center, 0.0)

    second_moment = jnp.nanmean(
        jnp.where(finite_mask, jnp.square(array - center), np.nan)
    )
    sigma = jnp.where(
        jnp.isfinite(second_moment), jnp.sqrt(second_moment), vmax - vmin
    )

    vmin_limit = center - 3 * sigma
    vmin = jnp.maximum(vmin, vmin_limit)
    vmax_limit = center + 3 * sigma
    vmax = jnp.minimum(vmax, vmax_limit)

  return vmin, vmax


@jax.jit
def _infer_abs_min_max(
    array: jnp.Array, mask: jnp.Array
) -> tuple[float | jax.Array, float | jax.Array]:
  """Infer smallest and largest absolute values in array."""
  finite_mask = jnp.logical_and(jnp.isfinite(array), mask)
  absmin = jnp.min(
      jnp.where(
          jnp.logical_and(finite_mask, array != 0), jnp.abs(array), np.inf
      )
  )
  absmin = jnp.where(jnp.isinf(absmin), 0.0, absmin)
  absmax = jnp.max(jnp.where(finite_mask, jnp.abs(array), -np.inf))
  absmax = jnp.where(jnp.isinf(absmax), 0.0, absmax)
  return absmin, absmax


@dataclasses.dataclass(frozen=True)
class ArrayvizRendering(figures.RendersAsRootInIPython):
  """A rendering of an array with Arrayviz.

  Attributes:
    html_src: HTML source for the rendering.
  """

  html_src: str

  def _compute_collapsed_width(self) -> int:
    return 80

  def _compute_newlines_in_expanded_parent(self) -> int:
    return 10

  def foldables_in_this_part(self) -> Sequence[part_interface.FoldableTreeNode]:
    return ()

  def _compute_tags_in_this_part(self) -> frozenset[Any]:
    return frozenset()

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    stream.write("<Arrayviz rendering>")

  def html_setup_parts(
      self, setup_context: part_interface.HtmlContextForSetup
  ) -> set[part_interface.CSSStyleRule | part_interface.JavaScriptDefn]:
    del setup_context
    return _html_setup()

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    stream.write(self.html_src)


default_sequential_colormap: context.ContextualValue[
    list[tuple[int, int, int]]
] = context.ContextualValue(
    module=__name__,
    qualname="default_sequential_colormap",
    # Matplotlib Viridis_20
    initial_value=[
        (68, 1, 84),
        (72, 20, 103),
        (72, 38, 119),
        (69, 55, 129),
        (63, 71, 136),
        (57, 85, 140),
        (50, 100, 142),
        (45, 113, 142),
        (40, 125, 142),
        (35, 138, 141),
        (31, 150, 139),
        (32, 163, 134),
        (41, 175, 127),
        (59, 187, 117),
        (86, 198, 103),
        (115, 208, 86),
        (149, 216, 64),
        (184, 222, 41),
        (221, 227, 24),
        (253, 231, 37),
    ],
)
"""Default sequential colormap.

Used by `render_array` when ``around_zero`` is False. Intended for user
customization in an interactive setting.
"""

default_diverging_colormap: context.ContextualValue[
    list[tuple[int, int, int]]
] = context.ContextualValue(
    module=__name__,
    qualname="default_diverging_colormap",
    # cmocean Balance_19_r[1:-1]
    initial_value=[
        (96, 14, 34),
        (134, 14, 41),
        (167, 36, 36),
        (186, 72, 46),
        (198, 107, 77),
        (208, 139, 115),
        (218, 171, 155),
        (228, 203, 196),
        (241, 236, 235),
        (202, 212, 216),
        (161, 190, 200),
        (117, 170, 190),
        (75, 148, 186),
        (38, 123, 186),
        (12, 94, 190),
        (41, 66, 162),
        (37, 47, 111),
    ],
)
"""Default diverging colormap.

Used by `render_array` when around_zero is True. Intended for user
customization in an interactive setting.
"""


def render_array(
    array: (
        named_axes.NamedArray
        | named_axes.NamedArrayView
        | np.ndarray
        | jax.Array
    ),
    *,
    columns: Sequence[named_axes.AxisName | int] = (),
    rows: Sequence[named_axes.AxisName | int] = (),
    sliders: Sequence[named_axes.AxisName | int] = (),
    valid_mask: (
        named_axes.NamedArray
        | named_axes.NamedArrayView
        | np.ndarray
        | jax.Array
        | None
    ) = None,
    continuous: bool | Literal["auto"] = "auto",
    around_zero: bool | Literal["auto"] = "auto",
    vmax: float | None = None,
    vmin: float | None = None,
    trim_outliers: bool = True,
    dynamic_colormap: bool | Literal["auto"] = "auto",
    colormap: list[tuple[int, int, int]] | None = None,
    truncate: bool = False,
    maximum_size: int = 10_000,
    cutoff_size_per_axis: int = 512,
    minimum_edge_items: int = 5,
    axis_item_labels: dict[named_axes.AxisName | int, list[str]] | None = None,
    value_item_labels: dict[int, str] | None = None,
    axis_labels: dict[named_axes.AxisName | int, str] | None = None,
) -> ArrayvizRendering:
  """Renders an array (positional or named) to a displayable HTML object.

  Each element of the array is rendered to a fixed-size square, with its
  position determined based on its index, and with each level of x and y axis
  represented by a "faceted" plot.

  Out-of-bounds or otherwise unusual data is rendered with an annotation:

  * "X" means a value was NaN (for continuous data) or went out-of-bounds for
    the integer palette (for discrete data).

  * "I" or "-I" means a value was infinity or negative infinity.

  * "+" or "-" means a value was finite but went outside the bounds of the
    colormap (e.g. it was larger than ``vmax`` or smaller than ``vmin``). By
    default this applies to values more than 3 standard deviations outside the
    mean.

  * Four light dots on grey means a value was masked out by ``valid_mask``, or
    truncated due to the maximum size or axis cutoff thresholds.

  By default, this method automatically chooses a color rendering strategy based
  on the arguments:

  * If an explicit colormap is provided:

    * If ``continuous`` is True, the provided colors are interpreted as color
      stops and interpolated between.

    * If ``continuous`` is False, the provided colors are interpreted as an
      indexed color palette, and each index of the palette is used to render
      the corresponding integer, starting from zero.

  * Otherwise:

    * If ``continuous`` is True:

      * If ``around_zero`` is True, uses the diverging colormap
        `default_diverging_colormap`. The initial value of this is a truncated
        version of the perceptually-uniform "Balance" colormap from cmocean,
        with blue for positive numbers and red for negative ones.

      * If ``around_zero`` is False, uses the sequential colormap
        `default_sequential_colormap`.The initial value of this is the
        perceptually-uniform "Viridis" colormap from matplotlib.

    * If ``continuous`` is False, uses a pattern-based "digitbox" rendering
      strategy to render integers up to 9,999,999 as nested squares, with one
      square per integer digit and digit colors drawn from the D3 Category20
      colormap.

  Args:
      array: The array to render.
      columns: Sequence of axis names or positional axis indices that should be
        placed on the x axis, from innermost to outermost. If not provided,
        inferred automatically.
      rows: Sequence of axis names or positional axis indices that should be
        placed on the y axis, from innermost to outermost. If not provided,
        inferred automatically.
      sliders: Sequence of axis names or positional axis indices for which we
        should show only a single slice at a time, with the index determined
        with a slider.
      valid_mask: Optionally, a boolean array with the same shape (and, if
        applicable, axis names) as `array`, which is True for the locations that
        we should actually render, and False for locations that do not have
        valid array data.
      continuous: Whether to interpret this array as numbers along the real
        line, and visualize using an interpolated colormap. If "auto", inferred
        from the dtype of `array`.
      around_zero: Whether the array data should be rendered symmetrically
        around zero using a diverging colormap, scaled based on the absolute
        magnitude of the inputs, instead of rescaled to be between the min and
        max of the data. If "auto", treated as True unless both `vmin` and
        `vmax` are set to incompatible values.
      vmax: Largest value represented in the colormap. If omitted and
        around_zero is True, inferred as ``max(abs(array))`` or as ``-vmin``. If
        omitted and around_zero is False, inferred as ``max(array)``.
      vmin: Smallest value represented in the colormap. If omitted and
        around_zero is True, inferred as ``-max(abs(array))`` or as ``-vmax``.
        If omitted and around_zero is False, inferred as ``min(array)``.
      trim_outliers: Whether to try to trim outliers when inferring ``vmin`` and
        ``vmax``. If True, clips them to 3 standard deviations away from the
        mean (or 3 sqrt-second-moments around zero) if they would otherwise
        exceed it.
      dynamic_colormap: Whether to dynamically adjust the colormap based on
        mouse hover. Requires a continuous colormap, and ``around_zero=True``.
        If "auto", will be enabled for continuous arrays if ``around_zero`` is
        True and neither ``vmin`` nor ``vmax`` are provided.
      colormap: An optional explicit colormap to use, represented as a list of
        ``(r,g,b)`` tuples, where each channel is between 0 and 255. A good
        place to get colormaps is the ``palettable`` package, e.g. you can pass
        something like ``palettable.matplotlib.Inferno_20.colors``.
      truncate: Whether or not to truncate the array to a smaller size before
        rendering.
      maximum_size: Maximum numer of elements of an array to show. Arrays larger
        than this will be truncated along one or more axes. Ignored unless
        ``truncate`` is True.
      cutoff_size_per_axis: Maximum number of elements of each individual axis
        to show without truncation. Any axis longer than this will be truncated,
        with their visual size increasing logarithmically with the true axis
        size beyond this point. Ignored unless ``truncate`` is True.
      minimum_edge_items: How many values to keep along each axis for truncated
        arrays. We may keep more than this up to the budget of maximum_size.
        Ignored unless ``truncate`` is True.
      axis_item_labels: An optional mapping from axis names/positions to a list
        of strings, of the same length as the axis length, giving a label to
        each item along that axis. For instance, this could be the token string
        corresponding to each position along a sequence axis, or the class label
        corresponding to each category across a classifier's output axis. This
        is shown in the tooltip when hovering over a pixel, and shown below the
        array when a pixel is clicked on. For convenience, names in this
        dictionary that don't match any axes in the input are simply ignored, so
        that you can pass the same labels while rendering arrays that may not
        have the same axis names.
      value_item_labels: For categorical data, an optional mapping from each
        value to a string. For instance, this could be the token value
        corresponding to each token ID in a sequence of tokens.
      axis_labels: Optional mapping from axis names / indices to the labels we
        should use for that axis. If not provided, we label the named axes with
        their names and the positional axes with "axis {i}", and also add th
        axis size.

  Returns:
    An object which can be rendered in an IPython notebook, containing the
    HTML source of an arrayviz rendering.
  """
  if axis_item_labels is None:
    axis_item_labels = {}

  if value_item_labels is None:
    value_item_labels = {}

  if axis_labels is None:
    axis_labels = {}

  # Step 1: Wrap as named arrays if needed, for consistency of the following
  # steps. But keep them on the CPU if they were numpy arrays.
  if not isinstance(array, named_axes.NamedArray | named_axes.NamedArrayView):
    if not isinstance(array, jax.Array):
      array = jax.device_put(array, jax.devices("cpu")[0])
    array = named_axes.wrap(array)

  array.check_valid()

  if valid_mask is not None:
    if not isinstance(
        valid_mask, named_axes.NamedArray | named_axes.NamedArrayView
    ):
      if not isinstance(valid_mask, jax.Array):
        valid_mask = jax.device_put(valid_mask, jax.devices("cpu")[0])
      valid_mask = named_axes.wrap(valid_mask)

    valid_mask.check_valid()

    # Make sure they are broadcast-compatible, and add length-1 axes for any
    # that are missing.
    bad_names = set(valid_mask.named_shape.keys()) - set(
        array.named_shape.keys()
    )
    if bad_names:
      raise ValueError(
          "Valid mask must be broadcastable to the shape of `array`, but it"
          f" had extra axis names {bad_names}"
      )

    vshape = valid_mask.positional_shape
    ashape = array.positional_shape
    if vshape != ashape[len(ashape) - len(vshape) :]:
      raise ValueError(
          "Valid mask must be broadcastable to the shape of `array`, but its"
          f" positional shape ({vshape}) was not a suffix of those of `array`"
          f" ({ashape})"
      )

    # Insert new axes.
    new_names = set(array.named_shape.keys()) - set(
        valid_mask.named_shape.keys()
    )
    if new_names:
      valid_mask = valid_mask[{name: None for name in new_names}]

    new_positional_axis_count = len(ashape) - len(vshape)
    if new_positional_axis_count:
      valid_mask = valid_mask[(None,) * new_positional_axis_count + (...,)]

  # Step 2: Extract a positionally-indexed array of data, and remember the
  # mapping from the original axis names and indices to their new data axes.
  # We try to avoid transposing the initial array if possible, so this won't
  # necessarily match the display order.
  # (Recall that integers are NOT valid names for a NamedArray, so there are
  # no possibilities of conflict between original axis names and indices.)
  tmp_names_for_positional_axes = [
      object() for _ in range(len(array.positional_shape))
  ]

  fully_named_array = array.tag(
      *tmp_names_for_positional_axes
  ).as_namedarrayview()
  array_data = fully_named_array.data_array

  data_axis_from_tmp_axis = {}
  tmp_axis_from_data_axis = {}
  for name, data_axis in fully_named_array.data_axis_for_name.items():
    data_axis_from_tmp_axis[name] = data_axis
    tmp_axis_from_data_axis[data_axis] = name

  data_axis_from_orig_axis = {}
  for name in array.named_shape.keys():
    data_axis_from_orig_axis[name] = data_axis_from_tmp_axis[name]
  for idx in range(len(array.positional_shape)):
    data_axis_from_orig_axis[idx] = data_axis_from_tmp_axis[
        tmp_names_for_positional_axes[idx]
    ]

  # Step 3: If the mask exists, extract its data in the same order, and add
  # length-one axes for any axes that were missing. Otherwise, create a new
  # mask array with only length-one axes.
  if valid_mask is not None:
    assert isinstance(valid_mask, named_axes.NamedArrayBase)
    fully_named_mask = (
        valid_mask.tag(*tmp_names_for_positional_axes)
        .order_as(*(tmp_axis_from_data_axis[i] for i in range(array_data.ndim)))
        .as_namedarrayview()
    )
    assert (
        fully_named_mask.data_axis_for_name
        == fully_named_array.data_axis_for_name
    )
    mask_data = fully_named_mask.data_array
  else:
    mask_data = np.ones([1] * array_data.ndim, dtype=bool)

  # Step 4: Truncate the array and valid masks if requested, and ensure that the
  # mask has the same shape as the array.
  if truncate:
    edge_items_per_axis = ndarray_summarization.infer_balanced_truncation(
        array_data.shape,
        maximum_size=maximum_size,
        cutoff_size_per_axis=cutoff_size_per_axis,
        minimum_edge_items=minimum_edge_items,
    )
    truncated_array_data, truncated_mask_data = (
        ndarray_summarization.truncate_array_and_mask(
            array=array_data,
            mask=mask_data,
            edge_items_per_axis=edge_items_per_axis,
        )
    )
  else:
    edge_items_per_axis = (None,) * array_data.ndim
    truncated_array_data = array_data
    truncated_mask_data = jnp.broadcast_to(mask_data, array_data.shape)

  # (Ensure they are fetched to the CPU to avoid device computation / sharding
  # issues)
  truncated_array_data, truncated_mask_data = jax.device_get(
      (truncated_array_data, truncated_mask_data)
  )

  skip_start_indices = [
      edge_items if edge_items is not None else size
      for edge_items, size in zip(edge_items_per_axis, array_data.shape)
  ]
  skip_end_indices = [
      size - edge_items if edge_items is not None else size
      for edge_items, size in zip(edge_items_per_axis, array_data.shape)
  ]

  # Step 5: Figure out which axes to render as rows, columns, and sliders and
  # in which order.  We start with the explicitly-requested axes, then add more
  # axes to the rows and columns until we've assigned all of them, trying to
  # balance rows and columns.

  unassigned_axes = set(array.named_shape.keys()) | set(
      range(len(array.positional_shape))
  )
  seen_axes = set()

  rows = list(rows)
  columns = list(columns)
  sliders = list(sliders)
  for axis in itertools.chain(rows, columns, sliders):
    if axis in seen_axes:
      raise ValueError(
          f"Axis {repr(axis)} appeared multiple times in rows/columns/sliders"
          " specifications. Each axis must be assigned to at most one"
          " location."
      )
    elif axis not in unassigned_axes:
      raise ValueError(
          f"Axis {repr(axis)} was assigned a location in rows/columns/sliders"
          " but was not present in the array to render."
      )
    seen_axes.add(axis)
    unassigned_axes.remove(axis)

  rows, columns = infer_rows_and_columns(
      unassigned=list(unassigned_axes),
      known_rows=rows,
      known_columns=columns,
      axis_sizes={
          **{
              orig: truncated_array_data.shape[data_axis]
              for orig, data_axis in data_axis_from_orig_axis.items()
          },
      },
  )

  # Convert the axis names into indices into our data array.
  column_data_axes = [
      data_axis_from_orig_axis[orig_axis] for orig_axis in columns
  ]
  row_data_axes = [data_axis_from_orig_axis[orig_axis] for orig_axis in rows]
  slider_data_axes = [
      data_axis_from_orig_axis[orig_axis] for orig_axis in sliders
  ]

  # Step 6: Figure out how to render the labels and indices of each axis.
  # We render indices using a small interpreted format language that can be
  # serialized to JSON and interpreted in JavaScript.
  data_axis_labels = {}
  formatting_instructions = []
  formatting_instructions.append({"type": "literal", "value": "array"})

  axis_label_instructions = []

  if array.named_shape:
    formatting_instructions.append({"type": "literal", "value": "[{"})

    for i, (name, size) in enumerate(array.named_shape.items()):
      data_axis = data_axis_from_orig_axis[name]

      if i:
        formatting_instructions.append(
            {"type": "literal", "value": f", {repr(name)}:"}
        )
      else:
        formatting_instructions.append(
            {"type": "literal", "value": f"{repr(name)}:"}
        )

      formatting_instructions.append({
          "type": "index",
          "axis": f"a{data_axis}",
          "skip_start": skip_start_indices[data_axis],
          "skip_end": skip_end_indices[data_axis],
      })

      if name in axis_labels:
        data_axis_labels[data_axis] = axis_labels[name]
      elif name in sliders:
        data_axis_labels[data_axis] = f"{str(name)}"
      else:
        data_axis_labels[data_axis] = f"{str(name)}: {size}"

      if name in axis_item_labels:
        axis_label_instructions.extend([
            {"type": "literal", "value": f"\n{str(name)} @ "},
            {
                "type": "index",
                "axis": f"a{data_axis}",
                "skip_start": skip_start_indices[data_axis],
                "skip_end": skip_end_indices[data_axis],
            },
            {"type": "literal", "value": ": "},
            {
                "type": "axis_lookup",
                "axis": f"a{data_axis}",
                "skip_start": skip_start_indices[data_axis],
                "skip_end": skip_end_indices[data_axis],
                "lookup_table": axis_item_labels[name],
            },
        ])

    formatting_instructions.append({"type": "literal", "value": "}]"})

  if array.positional_shape:
    formatting_instructions.append({"type": "literal", "value": "["})
    for orig_index, size in enumerate(array.positional_shape):
      data_axis = data_axis_from_orig_axis[orig_index]
      if orig_index:
        formatting_instructions.append({"type": "literal", "value": ", "})
      formatting_instructions.append({
          "type": "index",
          "axis": f"a{data_axis}",
          "skip_start": skip_start_indices[data_axis],
          "skip_end": skip_end_indices[data_axis],
      })

      if orig_index in axis_labels:
        data_axis_labels[data_axis] = axis_labels[orig_index]
      elif orig_index in sliders:
        data_axis_labels[data_axis] = f"axis{orig_index}"
      else:
        data_axis_labels[data_axis] = f"axis {orig_index}: {size}"

      if orig_index in axis_item_labels:
        axis_label_instructions.extend([
            {"type": "literal", "value": f"\nAxis {orig_index} @ "},
            {
                "type": "index",
                "axis": f"a{data_axis}",
                "skip_start": skip_start_indices[data_axis],
                "skip_end": skip_end_indices[data_axis],
            },
            {"type": "literal", "value": ": "},
            {
                "type": "axis_lookup",
                "axis": f"a{data_axis}",
                "skip_start": skip_start_indices[data_axis],
                "skip_end": skip_end_indices[data_axis],
                "lookup_table": axis_item_labels[orig_index],
            },
        ])

    formatting_instructions.append({"type": "literal", "value": "]"})

  formatting_instructions.append({"type": "literal", "value": "\n  = "})
  formatting_instructions.append({"type": "value"})

  # Step 7: Infer the colormap and rendering strategy.

  # Figure out whether the array is continuous.
  inferred_continuous = jnp.issubdtype(array_data.dtype, np.floating)
  if continuous == "auto":
    continuous = inferred_continuous
  elif not continuous and inferred_continuous:
    raise ValueError(
        "Cannot use continuous=False when rendering a float array; explicitly"
        " cast it to an integer array first."
    )

  if value_item_labels and not continuous:
    formatting_instructions.append({"type": "literal", "value": "  # "})
    formatting_instructions.append(
        {"type": "value_lookup", "lookup_table": value_item_labels}
    )

  formatting_instructions.extend(axis_label_instructions)

  # Figure out centering.
  definitely_not_around_zero = (
      vmin is not None and vmax is not None and vmin != -vmax  # pylint: disable=invalid-unary-operand-type
  )
  if around_zero == "auto":
    around_zero = not definitely_not_around_zero
  elif around_zero and definitely_not_around_zero:
    raise ValueError(
        "Cannot use around_zero=True while also specifying both vmin and vmax"
    )

  # Check whether we should dynamically adjust the colormap.
  if dynamic_colormap == "auto":
    dynamic_colormap = (
        continuous and around_zero and vmin is None and vmax is None
    )

  if dynamic_colormap:
    if not continuous:
      raise ValueError(
          "Cannot use dynamic_colormap with a non-continuous colormap."
      )
    if not around_zero:
      raise ValueError("Cannot use dynamic_colormap without around_zero.")

    raw_min_abs, raw_max_abs = _infer_abs_min_max(
        truncated_array_data, truncated_mask_data
    )
    raw_min_abs = float(raw_min_abs)
    raw_max_abs = float(raw_max_abs)
  else:
    raw_min_abs = None
    raw_max_abs = None

  # Infer concrete `vmin` and `vmax`.
  if continuous and (vmin is None or vmax is None):
    vmin, vmax = _infer_vmin_vmax(
        array=truncated_array_data,
        mask=truncated_mask_data,
        vmin=vmin,
        vmax=vmax,
        around_zero=around_zero,
        trim_outliers=trim_outliers,
    )
    vmin = float(vmin)
    vmax = float(vmax)

  # Figure out which colormap and rendering strategy to use.
  if colormap is None:
    if continuous:
      colormap_type = "continuous"
      if around_zero:
        colormap_data = default_diverging_colormap.get()
      else:
        colormap_data = default_sequential_colormap.get()

    else:
      colormap_type = "digitbox"
      colormap_data = []

  elif continuous:
    colormap_data = colormap
    colormap_type = "continuous"

  else:
    colormap_data = colormap
    colormap_type = "palette_index"

  # Make a title for it
  info_parts = []
  if dynamic_colormap:
    info_parts.append("Dynamic colormap (click to adjust).")
  elif continuous:
    info_parts.append(f"Linear colormap from {vmin:.6g} to {vmax:.6g}.")
  elif colormap is not None:
    info_parts.append("Indexed colors from a color list.")
  else:
    info_parts.append("Showing integer digits as nested squares.")

  info_parts.append(" Hover/click for array data.")

  # Step 8: Render it!
  html_src = _render_array_to_html(
      array_data=truncated_array_data,
      valid_mask=truncated_mask_data,
      column_axes=column_data_axes,
      row_axes=row_data_axes,
      slider_axes=slider_data_axes,
      axis_labels=[data_axis_labels[i] for i in range(array_data.ndim)],
      vmin=vmin,
      vmax=vmax,
      cmap_type=colormap_type,
      cmap_data=colormap_data,
      info="".join(info_parts),
      formatting_instructions=formatting_instructions,
      dynamic_continous_cmap=dynamic_colormap,
      raw_min_abs=raw_min_abs,
      raw_max_abs=raw_max_abs,
  )
  return ArrayvizRendering(html_src)


def _render_sharding(
    array_shape: tuple[int, ...],
    shard_shape: tuple[int, ...],
    device_indices_map: Mapping[Any, tuple[slice, ...]],
    rows: list[int | named_axes.AxisName] | None = None,
    columns: list[int | named_axes.AxisName] | None = None,
    name_to_data_axis: dict[named_axes.AxisName, int] | None = None,
    position_to_data_axis: tuple[int, ...] | None = None,
) -> ArrayvizRendering:
  """Renders the sharding of an array.

  This is a helper function for rendering array shardings. It can be used either
  to render the sharding of an actual array or of a hypothetical array of a
  given shape and sharding.

  Args:
    array_shape: Shape of the sharded array.
    shard_shape: Shape of each array shard.
    device_indices_map: Map from devices to tuples of slices into the array,
      identifying which parts of the array it corresponds to. Usually obtained
      from a JAX sharding.
    rows: Optional explicit ordering of rows in the visualization.
    columns: Optional explicit ordering of columns in the visualization.
    name_to_data_axis: Optional mapping from named axes to their axis in the
      data array.
    position_to_data_axis: Optional mapping from virtual positional axes to
      their axis in the data array.

  Returns:
    A rendering of the sharding, which re-uses the digitbox rendering mode to
    render sets of devices.
  """
  if name_to_data_axis is None and position_to_data_axis is None:
    name_to_data_axis = {}
    position_to_data_axis = {i: i for i in range(len(array_shape))}
  else:
    assert name_to_data_axis is not None
    assert position_to_data_axis is not None
  if rows is None and columns is None:
    rows, columns = infer_rows_and_columns(
        {
            name_or_pos: array_shape[data_axis]
            for name_or_pos, data_axis in itertools.chain(
                name_to_data_axis.items(), enumerate(position_to_data_axis)
            )
        },
        tuple(name_to_data_axis.keys())
        + tuple(range(len(position_to_data_axis))),
    )
  num_shards = np.prod(array_shape) // np.prod(shard_shape)
  # Compute a truncation for visualizing a single shard. Each shard will be
  # shown as a shrunken version of the actual shard dimensions, roughly
  # proportional to the shard sizes.
  mini_trunc = ndarray_summarization.infer_balanced_truncation(
      shape=array_shape,
      maximum_size=1000,
      cutoff_size_per_axis=10,
      minimum_edge_items=2,
      doubling_bonus=5,
  )
  # Build an actual matrix to represent each shard, with a size determined by
  # the inferred truncation.
  shard_mask = np.ones((), dtype=np.bool_)
  for t, sh_s, arr_s in zip(mini_trunc, shard_shape, array_shape):
    if t is None or sh_s <= 5:
      vec = np.ones((sh_s,), dtype=np.bool_)
    else:
      candidate = t // (arr_s // sh_s)
      if candidate <= 2:
        vec = np.array([True] * 2 + [False] + [True] * 2)
      else:
        vec = np.array([True] * candidate + [False] + [True] * candidate)
    shard_mask = shard_mask[..., None] * vec
  # Figure out which device is responsible for each shard.
  device_to_shard_offsets = {}
  shard_offsets_to_devices = collections.defaultdict(list)
  for device, slices in device_indices_map.items():
    shard_offsets = []
    for i, slc in enumerate(slices):
      assert slc.step is None
      if slc.start is None:
        assert slc.stop is None
        shard_offsets.append(0)
      else:
        assert slc.stop == slc.start + shard_shape[i]
        shard_offsets.append(slc.start // shard_shape[i])
    shard_offsets = tuple(shard_offsets)
    device_to_shard_offsets[device] = shard_offsets
    shard_offsets_to_devices[shard_offsets].append(device)
  # Figure out what value to show for each shard. This determines the
  # visualization color.
  shard_offset_values = {}
  shard_value_descriptions = {}
  if len(device_indices_map) <= 10 and all(
      device.id < 10 for device in device_indices_map.keys()
  ):
    # Map each device to an integer digit 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, and
    # then draw replicas as collections of base-10 digits.
    for shard_offsets, shard_devices in shard_offsets_to_devices.items():
      if len(shard_devices) >= 7:
        # All devices are in the same shard! Arrayviz only supports 7 digits at
        # a time, so draw as much as we can.
        assert num_shards == 1
        vis_value = 1234567
      else:
        acc = 0
        for i, device in enumerate(shard_devices):
          acc += 10 ** (len(shard_devices) - i - 1) * (device.id + 1)
        vis_value = acc
      shard_offset_values[shard_offsets] = vis_value
      platform = shard_devices[0].platform.upper()
      assert vis_value not in shard_value_descriptions
      shard_value_descriptions[vis_value] = (
          platform + " " + ",".join(f"{d.id}" for d in shard_devices)
      )
    render_info_message = "Colored by device index."
  elif num_shards < 10:
    # More than ten devices, less than ten shards. Give each shard its own
    # index but start at 1.
    shard_offset_values = {
        shard_offsets: i + 1
        for i, shard_offsets in enumerate(shard_offsets_to_devices.keys())
    }
    render_info_message = "With a distinct pattern for each shard."
  else:
    # A large number of devices and shards. Start at 0.
    shard_offset_values = {
        shard_offsets: i
        for i, shard_offsets in enumerate(shard_offsets_to_devices.keys())
    }
    render_info_message = "With a distinct pattern for each shard."
  # Build the sharding visualization array.
  viz_shape = tuple(
      shard_mask.shape[i] * array_shape[i] // shard_shape[i]
      for i in range(len(array_shape))
  )
  dest = np.zeros(viz_shape, dtype=np.int32)
  destmask = np.empty(viz_shape, dtype=np.int32)
  shard_labels_by_vis_pos = [
      ["????" for _ in range(viz_shape[i])] for i in range(len(viz_shape))
  ]
  for shard_offsets, value in shard_offset_values.items():
    indexers = []
    for i, offset in enumerate(shard_offsets):
      vizslc = slice(
          offset * shard_mask.shape[i],
          (offset + 1) * shard_mask.shape[i],
          None,
      )
      indexers.append(vizslc)
      label = f"{offset * shard_shape[i]}:{(offset + 1) * shard_shape[i]}"
      for j in range(viz_shape[i])[vizslc]:
        shard_labels_by_vis_pos[i][j] = label
    dest[tuple(indexers)] = np.full_like(shard_mask, value, dtype=np.int32)
    destmask[tuple(indexers)] = shard_mask
  # Create formatting instructions to show what devices are in each shard.
  axis_lookups = [
      {
          "type": "axis_lookup",
          "axis": f"a{data_axis}",
          "skip_start": viz_shape[data_axis],
          "skip_end": viz_shape[data_axis],
          "lookup_table": {
              j: str(v)
              for j, v in enumerate(shard_labels_by_vis_pos[data_axis])
          },
      }
      for data_axis in range(len(array_shape))
  ]
  data_axis_labels = {}
  formatting_instructions = []
  formatting_instructions.append({"type": "literal", "value": "array"})
  if name_to_data_axis:
    formatting_instructions.append({"type": "literal", "value": "[{"})
    for k, (name, data_axis) in enumerate(name_to_data_axis.items()):
      if k:
        formatting_instructions.append(
            {"type": "literal", "value": f", {repr(name)}:["}
        )
      else:
        formatting_instructions.append(
            {"type": "literal", "value": f"{repr(name)}:["}
        )
      formatting_instructions.append(axis_lookups[data_axis])
      formatting_instructions.append({"type": "literal", "value": "]"})
      axshards = array_shape[data_axis] // shard_shape[data_axis]
      data_axis_labels[data_axis] = (
          f"{str(name)}: {array_shape[data_axis]}/{axshards}"
      )
    formatting_instructions.append({"type": "literal", "value": "}]"})
  if position_to_data_axis:
    formatting_instructions.append({"type": "literal", "value": "["})
    for k in range(len(position_to_data_axis)):
      data_axis = position_to_data_axis[k]
      if k:
        formatting_instructions.append({"type": "literal", "value": ", "})
      formatting_instructions.append(axis_lookups[data_axis])
      axshards = array_shape[data_axis] // shard_shape[data_axis]
      data_axis_labels[data_axis] = (
          f"axis {k}: {array_shape[data_axis]}/{axshards}"
      )
    formatting_instructions.append({"type": "literal", "value": "]"})
  formatting_instructions.append({"type": "literal", "value": ":\n  "})
  formatting_instructions.append({
      "type": "value_lookup",
      "lookup_table": shard_value_descriptions,
      "ignore_invalid": True,
  })
  # Build the rendering.
  html_srcs = []
  to_data_axis = {**dict(enumerate(position_to_data_axis)), **name_to_data_axis}
  html_srcs.append(
      _render_array_to_html(
          array_data=dest,
          valid_mask=destmask,
          column_axes=[to_data_axis[c] for c in columns],
          row_axes=[to_data_axis[r] for r in rows],
          slider_axes=(),
          axis_labels=[data_axis_labels[i] for i in range(len(array_shape))],
          vmin=0,
          vmax=0,
          cmap_type="digitbox",
          cmap_data=[],
          info=render_info_message,
          formatting_instructions=formatting_instructions,
          dynamic_continous_cmap=False,
          raw_min_abs=0.0,
          raw_max_abs=0.0,
      )
  )
  html_srcs.append('<span style="font-family: monospace; white-space: pre">')
  for i, (shard_offsets, shard_devices) in enumerate(
      shard_offsets_to_devices.items()
  ):
    if i == 0:
      device = shard_devices[0]
      html_srcs.append(f"{device.platform.upper()}")
    label = ",".join(f"{d.id}" for d in shard_devices)
    subsrc = integer_digitbox(
        shard_offset_values[shard_offsets],
        label_bottom="",
    ).html_src
    html_srcs.append(f"  {subsrc} {label}")
  html_srcs.append("</span>")
  return ArrayvizRendering("".join(html_srcs))


def render_array_sharding(
    array: jax.Array | named_axes.NamedArray,
    rows: list[int | named_axes.AxisName] | None = None,
    columns: list[int | named_axes.AxisName] | None = None,
) -> ArrayvizRendering:
  """Renders the sharding of an array.

  Args:
    array: The array whose sharding we should render.
    rows: Optional explicit ordering of axes for the visualization rows.
    columns: Optional explicit ordering of axes for the visualization columns.

  Returns:
    A rendering of that array's sharding.
  """
  # Wrap as named arrays if needed, for consistency of the following steps.
  if not isinstance(array, named_axes.NamedArrayBase):
    if not isinstance(array, jax.Array):
      raise ValueError(
          "render_array_sharding can only be used on jax.Arrays and"
          " pz.nx.NamedArray / NamedArrayView."
      )
    array = named_axes.wrap(array)
  array.check_valid()
  array = array.as_namedarrayview()
  assert array.data_array.shape == array.data_shape
  if not hasattr(array.data_array, "sharding"):
    raise ValueError(
        "Provided array does not have a sharding! Is this a tracer?"
    )
  sharding = array.data_array.sharding

  return _render_sharding(
      array_shape=array.data_shape,
      shard_shape=sharding.shard_shape(array.data_shape),
      device_indices_map=sharding.devices_indices_map(array.data_shape),
      name_to_data_axis=array.data_axis_for_name,
      position_to_data_axis=array.data_axis_for_logical_axis,
      rows=rows,
      columns=columns,
  )


def render_sharded_shape(
    sharding: jax.sharding.Sharding,
    shape_or_namedarray_struct: (
        jax.ShapeDtypeStruct | named_axes.NamedArrayBase | tuple[int, ...] | Any
    ),
    rows: list[int | named_axes.AxisName] | None = None,
    columns: list[int | named_axes.AxisName] | None = None,
) -> ArrayvizRendering:
  """Renders the sharding an array would have, based on its shape.

  Args:
    sharding: A sharding to visualize.
    shape_or_namedarray_struct: Either an arbitrary object with a ``.shape``
      attribute, a tuple of integers, or a NamedArray wrapping a
      `jax.lax.ShapeDtypeStruct`.
    rows: Optional explicit ordering of axes for the visualization rows.
    columns: Optional explicit ordering of axes for the visualization columns.

  Returns:
    A rendering of the result of sharding an array with this shape using the
    given sharding.
  """
  if isinstance(shape_or_namedarray_struct, tuple):
    shape_or_namedarray_struct = jax.ShapeDtypeStruct(
        shape=shape_or_namedarray_struct, dtype=jnp.float32
    )
  elif not isinstance(shape_or_namedarray_struct, named_axes.NamedArrayBase):
    shape_or_namedarray_struct = jax.ShapeDtypeStruct(
        shape=shape_or_namedarray_struct.shape, dtype=jnp.float32
    )

  def _traced_fixup(array):
    if not isinstance(array, named_axes.NamedArrayBase):
      array = named_axes.wrap(array)
    array.check_valid()
    return array.as_namedarrayview()

  view = jax.eval_shape(_traced_fixup, shape_or_namedarray_struct)
  assert view.data_array.shape == view.data_shape
  return _render_sharding(
      array_shape=view.data_shape,
      shard_shape=sharding.shard_shape(view.data_shape),
      device_indices_map=sharding.devices_indices_map(view.data_shape),
      name_to_data_axis=view.data_axis_for_name,
      position_to_data_axis=view.data_axis_for_logical_axis,
      rows=rows,
      columns=columns,
  )


@dataclasses.dataclass(frozen=True)
class ArrayvizDigitboxRendering(ArrayvizRendering):
  """A rendering of a single digitbox with Arrayviz."""

  def _compute_collapsed_width(self) -> int:
    return 2

  def _compute_newlines_in_expanded_parent(self) -> int:
    return 1


def integer_digitbox(
    value: int,
    label_top: str = "",
    label_bottom: str | None = None,
    size: str = "1em",
) -> ArrayvizDigitboxRendering:
  """Returns a "digitbox" rendering of a single integer.

  Args:
    value: Integer value to render.
    label_top: Label to draw on top of the digitbox.
    label_bottom: Label to draw below the digitbox. If omitted, defaults to
      str(value).
    size: Size for the rendering as a CSS length. "1em" means render it at the
      current font size.

  Returns:
    A renderable object showing the digitbox rendering for this integer.
  """
  value = int(value)

  if label_bottom is None:
    label_bottom = str(value)

  fresh_id = "arrayviz" + uuid.uuid4().hex
  render_args = json.dumps({
      "value": value,
      "labelTop": label_top,
      "labelBottom": label_bottom,
  })
  size_attr = html_escaping.escape_html_attribute(size)
  src = (
      f'<span id="{fresh_id}" class="inline_digitbox"'
      f' style="font-size: {size_attr}"></span>'
      '<template class="treescope_run_soon">'
      f'<script>document.querySelector("#{fresh_id}").appendChild('
      f"arrayviz.renderOneDigitbox({render_args}));</script></template>"
  )
  return ArrayvizRendering(src)


@dataclasses.dataclass(frozen=True)
class ValueColoredTextbox(
    figures.RendersAsRootInIPython, basic_parts.DeferringToChild
):
  """A rendering of text with a colored background.

  Attributes:
    child: Child part to render.
    text_color: Color for the text.
    background_color: Color for the background, usually from a colormap.
    out_of_bounds: Whether this value was out of bounds of the colormap.
    value: Underlying float value that is being visualized. Rendered on hover.
  """

  child: part_interface.RenderableTreePart
  text_color: str
  background_color: str
  out_of_bounds: bool
  value: float

  def html_setup_parts(
      self, setup_context: part_interface.HtmlContextForSetup
  ) -> set[part_interface.CSSStyleRule | part_interface.JavaScriptDefn]:
    return (
        {
            part_interface.CSSStyleRule(
                html_escaping.without_repeated_whitespace("""
                    .arrayviz_textbox {
                        padding-left: 0.5ch;
                        padding-right: 0.5ch;
                        outline: 1px solid black;
                        position: relative;
                        display: inline-block;
                        font-family: monospace;
                        white-space: pre;
                        margin-top: 1px;
                        box-sizing: border-box;
                    }
                    .arrayviz_textbox.out_of_bounds {
                        outline: 3px double darkorange;
                    }
                    .arrayviz_textbox .value {
                        display: none;
                        position: absolute;
                        bottom: 110%;
                        left: 0;
                        overflow: visible;
                        color: black;
                        background-color: white;
                        font-size: 0.7em;
                    }
                    .arrayviz_textbox:hover .value {
                        display: block;
                    }
                    """)
            )
        }
        | self.child.html_setup_parts(setup_context)
    )

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    class_string = "arrayviz_textbox"
    if self.out_of_bounds:
      class_string += " out_of_bounds"
    bg_color = html_escaping.escape_html_attribute(self.background_color)
    text_color = html_escaping.escape_html_attribute(self.text_color)
    stream.write(
        f'<span class="{class_string}" style="background-color:{bg_color};'
        f' color:{text_color}">'
        f'<span class="value">{float(self.value):.4g}</span>'
    )
    self.child.render_to_html(
        stream,
        at_beginning_of_line=False,
        render_context=render_context,
    )
    stream.write("</span>")


def text_on_color(
    text: str,
    value: float,
    vmax: float = 1.0,
    vmin: float | None = None,
    colormap: list[tuple[int, int, int]] | None = None,
) -> ValueColoredTextbox:
  """Renders some text on colored background, similar to arrayviz coloring.

  Args:
    text: Text to display.
    value: Value to color the background with.
    vmax: Maximum value for the colormap.
    vmin: Minimum value for the colormap. Defaults to ``-vmax``.
    colormap: Explicit colormap to use. If not provided, uses the default
      diverging colormap if ``vmin`` is not provided, or the default sequential
      colormap if ``vmin`` is provided.

  Returns:
    A rendering of this word, formatted similarly to how this value would be
    formatted in an array visualization.
  """
  if colormap is None:
    if vmin is None:
      colormap = default_diverging_colormap.get()
    else:
      colormap = default_sequential_colormap.get()

  if vmin is None:
    vmin = -vmax

  if value > vmax:
    is_out_of_bounds = True
    color = colormap[-1]
  elif value < vmin:
    is_out_of_bounds = True
    color = colormap[0]
  else:
    is_out_of_bounds = False
    relative = (value - vmin) / (vmax - vmin)
    continuous_index = relative * (len(colormap) - 1)
    base_index = int(np.floor(continuous_index))
    offset = continuous_index - base_index
    if base_index == len(colormap) - 1:
      base_index -= 1
      offset = 1
    assert base_index < len(colormap) - 1
    color = [
        v0 * (1 - offset) + v1 * offset
        for v0, v1 in zip(colormap[base_index], colormap[base_index + 1])
    ]

  r, g, b = color
  # Logic matches `contrasting_style_string` in arrayviz.js
  # https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale
  intensity = 0.2126 * r + 0.7152 * g + 0.0722 * b
  if intensity > 128:
    text_color = "black"
  else:
    text_color = "white"
  return ValueColoredTextbox(
      child=basic_parts.Text(text),
      text_color=text_color,
      background_color=f"rgb({r} {g} {b})",
      out_of_bounds=is_out_of_bounds,
      value=value,
  )
