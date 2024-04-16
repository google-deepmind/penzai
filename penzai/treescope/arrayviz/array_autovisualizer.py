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

"""An automatic NDArray visualizer using arrayviz."""

import dataclasses
from typing import Any, Callable, Collection

import jax
import jax.numpy as jnp
import numpy as np
from penzai.core import named_axes
from penzai.treescope import autovisualize
from penzai.treescope import ndarray_summarization
from penzai.treescope.arrayviz import arrayviz
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import part_interface
from penzai.treescope.handlers.penzai import named_axes_handlers


def _supported_dtype(dtype):
  return (
      jnp.issubdtype(dtype, np.integer)
      or jnp.issubdtype(dtype, np.floating)
      or jnp.issubdtype(dtype, np.bool_)
  )


@dataclasses.dataclass
class ArrayAutovisualizer:
  """An automatic visualizer for arrays.

  Attributes:
    maximum_size: Maximum numer of elements of an array to show. Arrays larger
      than this will be truncated along one or more axes.
    cutoff_size_per_axis: Maximum number of elements of each individual axis to
      show without truncation. Any axis longer than this will be truncated, with
      their visual size increasing logarithmically with the true axis size
      beyond this point.
    edge_items: How many values to keep along each axis for truncated arrays.
    prefers_column: Names that should always be assigned to columns.
    prefers_row: Names that should always be assigned to rows.
    around_zero: Whether to center continous data around zero.
    force_continuous: Whether to always render integer arrays as continuous.
    include_repr_line_threshold: A threshold such that, if the `repr` of the
      array has fewer than that many lines, we will include that repr in the
      visualization. Useful for seeing small array values.
    token_lookup_fn: Optional function that looks up token IDs and adds them to
      the visualization on hover.
  """

  maximum_size: int = 4_000
  cutoff_size_per_axis: int = 128
  edge_items: int = 5
  prefers_column: Collection[str] = ()
  prefers_row: Collection[str] = ()
  around_zero: bool = True
  force_continuous: bool = False
  include_repr_line_threshold: int = 5
  token_lookup_fn: Callable[[int], str] | None = None

  def _autovisualize_namedarray(
      self,
      named_array: named_axes.NamedArrayBase,
      path: tuple[Any, ...] | None,
      label: str,
      expand_state: part_interface.ExpandState,
  ) -> part_interface.RenderableTreePart:
    """Visualizes a named array."""
    named_array = named_array.as_namedarrayview()

    # Assign axes with a preference.
    row_axes = []
    column_axes = []
    names_in_array = set(named_array.named_shape.keys())
    unassigned = set(names_in_array)

    for name in self.prefers_column:
      if name in names_in_array:
        column_axes.append(name)
        unassigned.remove(name)

    for name in self.prefers_row:
      if name in names_in_array:
        row_axes.append(name)
        unassigned.remove(name)

    # Infer remaining assignment.
    shape_after_truncation = ndarray_summarization.compute_truncated_shape(
        named_array.data_array.shape,
        ndarray_summarization.infer_balanced_truncation(
            named_array.data_array.shape,
            maximum_size=self.maximum_size,
            cutoff_size_per_axis=self.cutoff_size_per_axis,
            minimum_edge_items=self.edge_items,
        ),
    )
    row_axes, column_axes = arrayviz.infer_rows_and_columns(
        axis_sizes={
            **{
                name: shape_after_truncation[data_axis]
                for name, data_axis in named_array.data_axis_for_name.items()
            },
            **{
                i: shape_after_truncation[data_axis]
                for i, data_axis in enumerate(
                    named_array.data_axis_for_logical_axis
                )
            },
        },
        unassigned=(
            list(unassigned) + list(range(len(named_array.positional_shape)))
        ),
        known_rows=row_axes,
        known_columns=column_axes,
    )

    # Maybe infer value labels from a tokenizer.
    if (
        self.token_lookup_fn
        and not self.force_continuous
        and np.issubdtype(named_array.dtype, np.integer)
        and named_array.data_array.size < self.maximum_size
    ):
      tokens = np.unique(named_array.data_array.flatten()).tolist()
      value_item_labels = {
          token: self.token_lookup_fn(token) for token in tokens
      }
    else:
      value_item_labels = None

    array_rendering = arrayviz.render_array(
        named_array,
        columns=column_axes,
        rows=row_axes,
        truncate=True,
        maximum_size=self.maximum_size,
        cutoff_size_per_axis=self.cutoff_size_per_axis,
        minimum_edge_items=self.edge_items,
        around_zero=self.around_zero,
        continuous=True if self.force_continuous else "auto",
        value_item_labels=value_item_labels,
    )
    rendering_parts = [array_rendering]

    # Render the sharding as well.
    if (
        isinstance(named_array.data_array, jax.Array)
        and hasattr(named_array.data_array, "sharding")
        and not isinstance(
            named_array.data_array.sharding, jax.sharding.SingleDeviceSharding
        )
    ):
      sharding = named_array.data_array.sharding
      platform = next(iter(sharding.device_set)).platform
      sharding_rendering = arrayviz.render_array_sharding(
          named_array, columns=column_axes, rows=row_axes
      )
      if sharding.is_fully_replicated:
        sharding_summary_str = (
            f"Replicated across {len(sharding.device_set)}"
            f" {platform.upper()} devices"
        )
      else:
        sharding_summary_str = (
            f"Sharded across {len(sharding.device_set)}"
            f" {platform.upper()} devices"
        )
      rendering_parts.append(
          common_structures.build_custom_foldable_tree_node(
              label=common_styles.AbbreviationColor(
                  basic_parts.siblings(
                      basic_parts.Text(sharding_summary_str),
                      basic_parts.FoldCondition(
                          expanded=basic_parts.Text(":"),
                          collapsed=basic_parts.Text(" (click to expand)"),
                      ),
                  )
              ),
              contents=basic_parts.FoldCondition(
                  expanded=basic_parts.IndentedChildren([sharding_rendering]),
              ),
          )
      )

    # We render it with a path, but remove the copy path button. This will be
    # added back by the caller.
    custom_rendering = common_structures.build_custom_foldable_tree_node(
        label=common_styles.AbbreviationColor(label),
        contents=basic_parts.siblings(
            basic_parts.FoldCondition(
                expanded=basic_parts.IndentedChildren.build(rendering_parts)
            ),
            common_styles.AbbreviationColor(basic_parts.Text(">")),
        ),
        path=path,
        expand_state=expand_state,
    )
    return custom_rendering.renderable

  def __call__(
      self, value: Any, path: tuple[Any, ...] | None
  ) -> autovisualize.CustomTreescopeVisualization | None:
    """Implementation of an autovisualizer, visualizing arrays."""
    with jax.core.ensure_compile_time_eval():
      if isinstance(value, named_axes.NamedArray | named_axes.NamedArrayView):
        try:
          value.check_valid()
        except ValueError:
          return None
        if not _supported_dtype(value.dtype):
          return None

        if not isinstance(value.data_array, jax.Array):
          return None

        if (
            isinstance(value.data_array, jax.core.Tracer)
            or value.data_array.is_deleted()
        ):
          return None

        if value.data_array.size == 1:
          # Don't visualize scalars.
          return None

        def _placeholder() -> part_interface.RenderableTreePart:
          # Quick summary of the array that doesn't require device
          # computation.
          summary, contained_type = (
              named_axes_handlers.named_array_and_contained_type_summary(
                  value, inspect_device_data=False
              )
          )
          short_summary = (
              f"<{type(value).__name__} {summary} (wrapping {contained_type})>"
          )
          return common_structures.fake_placeholder_foldable(
              common_styles.DeferredPlaceholderStyle(
                  basic_parts.Text(short_summary)
              ),
              extra_newlines_guess=8,
          )

        def _thunk(placeholder) -> part_interface.RenderableTreePart:
          # Full rendering of the array.
          if isinstance(placeholder, part_interface.FoldableTreeNode):
            expand_state = placeholder.get_expand_state()
          else:
            assert placeholder is None
            expand_state = part_interface.ExpandState.WEAKLY_EXPANDED
          summary, contained_type = (
              named_axes_handlers.named_array_and_contained_type_summary(value)
          )
          label = common_styles.AbbreviationColor(
              basic_parts.Text(
                  f"<{type(value).__name__} {summary} (wrapping"
                  f" {contained_type})"
              )
          )
          named_array = value
          return self._autovisualize_namedarray(
              named_array, path, label, expand_state
          )

        return autovisualize.CustomTreescopeVisualization(
            basic_parts.RenderableAndLineAnnotations(
                renderable=foldable_impl.maybe_defer_rendering(
                    _thunk, _placeholder
                ),
                annotations=common_structures.build_copy_button(path),
            )
        )

      elif isinstance(value, (np.ndarray, jax.Array)):
        if not (
            jnp.issubdtype(value.dtype, np.integer)
            or jnp.issubdtype(value.dtype, np.floating)
        ):
          return None

        if value.size == 1:
          # Don't visualize scalars.
          return None

        if isinstance(value, np.ndarray):
          contained_type = "np.ndarray"
        elif isinstance(value, jax.Array) and not isinstance(
            value, jax.core.Tracer
        ):
          contained_type = "jax.Array"
          if value.is_deleted():
            return None
        else:
          # Unsupported type
          return None

        def _placeholder() -> part_interface.RenderableTreePart:
          # Quick summary of the array that doesn't require device
          # computation.
          dtypestr = ndarray_summarization.get_dtype_name(value.dtype)
          short_summary = (
              f"<{contained_type} {dtypestr}{repr(value.shape)} ... >"
          )
          return common_structures.fake_placeholder_foldable(
              common_styles.DeferredPlaceholderStyle(
                  basic_parts.Text(short_summary)
              ),
              extra_newlines_guess=8,
          )

        def _thunk(placeholder) -> part_interface.RenderableTreePart:
          # Full rendering of the array.
          if isinstance(placeholder, part_interface.FoldableTreeNode):
            expand_state = placeholder.get_expand_state()
          else:
            assert placeholder is None
            expand_state = part_interface.ExpandState.WEAKLY_EXPANDED
          value_repr = ndarray_summarization.faster_array_repr(value)
          if "\n" not in value_repr and "..." not in value_repr:
            if value_repr.startswith("<") and value_repr.endswith(">"):
              label = basic_parts.Text(value_repr[:-1])
            else:
              label = basic_parts.Text("<" + value_repr)
          else:
            label = basic_parts.Text(
                "<"
                + contained_type
                + " "
                + ndarray_summarization.summarize_ndarray(value)
            )
          # Convert it to a named array so we can render it.
          if isinstance(value, np.ndarray):
            to_wrap = jax.device_put(value, jax.local_devices(backend="cpu")[0])
          else:
            to_wrap = value
          named_array = named_axes.wrap(to_wrap)
          return self._autovisualize_namedarray(
              named_array, path, label, expand_state
          )

        return autovisualize.CustomTreescopeVisualization(
            basic_parts.RenderableAndLineAnnotations(
                renderable=foldable_impl.maybe_defer_rendering(
                    _thunk, _placeholder
                ),
                annotations=common_structures.build_copy_button(path),
            )
        )

      elif isinstance(
          value,
          jax.sharding.PositionalSharding
          | jax.sharding.NamedSharding
          | jax.sharding.Mesh,
      ):
        raw_repr = repr(value)
        repr_oneline = " ".join(line.strip() for line in raw_repr.split("\n"))
        if isinstance(value, jax.sharding.PositionalSharding):
          shardvis = arrayviz.render_sharded_shape(value, value.shape)
        elif isinstance(value, jax.sharding.NamedSharding):
          # Named shardings still act on positional arrays, so show them for
          # the positional shape they require.
          smallest_shape = []
          for part in value.spec:
            if part is None:
              smallest_shape.append(1)
            elif isinstance(part, str):
              smallest_shape.append(value.mesh.shape[part])
            else:
              smallest_shape.append(
                  int(np.prod([value.mesh.shape[a] for a in part]))
              )
          smallest_shape = tuple(smallest_shape)
          shardvis = arrayviz.render_sharded_shape(value, smallest_shape)
        elif isinstance(value, jax.sharding.Mesh):
          shardvis = arrayviz.render_sharded_shape(
              jax.sharding.NamedSharding(
                  value, jax.sharding.PartitionSpec(*value.axis_names)
              ),
              jax.eval_shape(
                  lambda x: named_axes.wrap(x, *value.axis_names),
                  value.device_ids,
              ),
          )
        else:
          assert False  # impossible

        custom_rendering = common_structures.build_custom_foldable_tree_node(
            label=common_styles.AbbreviationColor(
                basic_parts.Text("<" + repr_oneline)
            ),
            contents=basic_parts.siblings(
                basic_parts.FoldCondition(
                    expanded=basic_parts.IndentedChildren.build([shardvis])
                ),
                common_styles.AbbreviationColor(basic_parts.Text(">")),
            ),
            path=path,
            expand_state=part_interface.ExpandState.EXPANDED,
        )
        return autovisualize.CustomTreescopeVisualization(custom_rendering)
      else:
        return None

  @classmethod
  def for_tokenizer(cls, tokenizer: Any):
    """Builds an autovisualizer for a tokenizer.

    This method constructs an ArrayAutovisualizer that annotates integer array
    elements with their token strings. This can then be used to autovisualize
    tokenized arrays.

    Args:
      tokenizer: A tokenizer to use. Either a callable mapping token IDs to
        strings, or a SentencePieceProcessor.

    Returns:
      An ArrayAutovisualizer that annotates integer array elements with their
      token strings.
    """
    if callable(tokenizer):
      return cls(token_lookup_fn=lambda x: repr(tokenizer(x)))
    elif hasattr(tokenizer, "IdToPiece") and hasattr(tokenizer, "GetPieceSize"):

      def lookup(x):
        if x >= 0 and x < tokenizer.GetPieceSize():
          return repr(tokenizer.IdToPiece(x))
        else:
          return f"<out of bounds: {x}>"

      return cls(token_lookup_fn=lookup)
    else:
      raise ValueError(f"Unknown tokenizer type: {tokenizer}")
