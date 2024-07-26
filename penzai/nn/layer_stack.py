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

"""Layer stacks."""

from __future__ import annotations

import collections
import copy
import dataclasses
import enum
from typing import Any, Callable, Hashable

import jax
from penzai.core import named_axes
from penzai.core import selectors
from penzai.core import struct
from penzai.core import tree_util as pz_tree_util
from penzai.core import variables
from penzai.nn import layer as layer_base


class LayerStackVarBehavior(enum.Enum):
  """Behavior of a variable in a layer stack."""

  SHARED = enum.auto()
  PER_LAYER = enum.auto()


@dataclasses.dataclass(frozen=True)
class LayerStackGetAttrKey(jax.tree_util.GetAttrKey):
  """GetAttrKey for LayerStack with extra metadata.

  This allows us to identify whether a given PyTree leaf is contained inside a
  LayerStack, and if so, which axis it is stacked along. This can in turn be
  used to manipulate variables inside a LayerStack in a stack-compatible way.
  """

  name: str
  stack_axis: named_axes.AxisName
  stack_axis_size: int


@struct.pytree_dataclass
class LayerStack(layer_base.Layer):
  """A sequence of layers with identical structure, called under jax.lax.scan.

  This class identifies a sequence of layers that can be efficiently called
  using a `jax.lax.scan` control flow primitive, speeding up compilation times
  under `jax.jit`. Instead of storing separate copies of each layer, this class
  instead stores a single "prototype" layer, whose leaves have an additional
  named axis (the ``stack_axis``) whenever they differ across layers.

  StateVariable instances inside the stacked sublayer are required to have a
  metadata field "layerstack_axes", which maps each axis name to a
  `LayerStackVarBehavior` determining whether it should be shared or split
  across layers. (This is not necessary for Parameters, which are not mutable
  when the layer is called.)

  Attributes:
    stacked_sublayers: A collection of sublayers, each of which have an extra
      named axis.
    stack_axis: The axis name that layer data is stacked along.
    stack_axis_size: The size of the stack axis.
  """

  stacked_sublayers: layer_base.Layer
  stack_axis: named_axes.AxisName = dataclasses.field(
      metadata={"pytree_node": False}
  )
  stack_axis_size: int = dataclasses.field(metadata={"pytree_node": False})

  def __call__(self, argument, /, **side_inputs) -> Any:
    """Calls the stacked sublayers under a `jax.lax.scan`."""

    # Freeze parameters.
    frozen_param_sublayers = variables.freeze_params(self.stacked_sublayers)

    # Extract state variables.
    pure_sublayer, sublayer_vars = variables.unbind_state_vars(
        frozen_param_sublayers
    )
    pure_side_inputs, side_input_vars = variables.unbind_state_vars(side_inputs)

    # Make sure there aren't any other types of variables we don't recognize.
    _, bad_vars = variables.unbind_variables(pure_sublayer, pure_side_inputs)
    if bad_vars:
      raise ValueError(
          "Unsupported variable types found in layer stack:"
          f" {set(type(var) for var in bad_vars)}"
      )

    vars_by_label = {}
    shared_var_states = []
    sliced_var_states = []

    # Side input variables are always shared.
    for var in side_input_vars:
      vars_by_label[var.label] = var
      shared_var_states.append(var.freeze())

    # Attribute variables are either shared or split depending on metadata.
    for var in sublayer_vars:
      if (
          "layerstack_axes" not in var.metadata
          or self.stack_axis not in var.metadata["layerstack_axes"]
      ):
        raise ValueError(
            "All variables in a layer stack must have a `layerstack_axes` "
            "metadata field indicating which axes they are shared over."
        )
      behavior = var.metadata["layerstack_axes"][self.stack_axis]
      if behavior == LayerStackVarBehavior.SHARED:
        bad_leaves = (
            selectors.select(var.value)
            .at_instances_of(named_axes.NamedArrayBase)
            .where(lambda x: self.stack_axis in x.named_shape)
        )
        if bad_leaves.count():
          bad_paths = [
              pz_tree_util.pretty_keystr(path, var.value)
              for path in bad_leaves.selected_by_path.keys()
          ]
          raise ValueError(
              f"Variable {var.label} has shared behavior for layer stack axis"
              f" {self.stack_axis} but the stack axis appears in"
              f" NamedArrays at keypaths: {bad_paths}"
          )

        if var.label in vars_by_label:
          # Shared variables are OK to appear in side inputs (although this is
          # probably rare).
          if vars_by_label[var.label] is not var:
            raise ValueError(
                f"Different variables with same label {var.label} found in side"
                " inputs and sublayers."
            )
        else:
          vars_by_label[var.label] = var
          shared_var_states.append(var.freeze())

      elif behavior == LayerStackVarBehavior.PER_LAYER:
        if var.label in vars_by_label:
          raise ValueError(
              f"Per-layer variable {var.label} cannot also be provided as a"
              " side input."
          )
        var_leaves, var_treedef = jax.tree_util.tree_flatten(
            var.value, is_leaf=named_axes.is_namedarray
        )
        adjusted_leaves = []
        for leaf in var_leaves:
          if (
              not isinstance(leaf, named_axes.NamedArrayBase)
              or self.stack_axis not in leaf.named_shape
          ):
            raise ValueError(
                f"Variable {var.label} has per-layer behavior for layer stack"
                f" axis {self.stack_axis} but the stack axis does not appear in"
                " all leaves of the variable value."
            )
          adjusted_leaves.append(
              leaf.untag_prefix(self.stack_axis).with_positional_prefix()
          )

        vars_by_label[var.label] = var
        sliced_var_states.append(
            variables.StateVariableValue(
                value=var_treedef.unflatten(adjusted_leaves),
                label=var.label,
                metadata=var.metadata,
            )
        )

      else:
        raise ValueError(
            f"Unknown layer stack variable behavior: {behavior} for variable "
            f"with label {var.label}"
        )

    # Separate the array leaves that have the stack axis, and move the stacked
    # axis to be the first positional axis (so that we can scan over it).
    stacked_array_selection = (
        selectors.select(pure_sublayer)
        .at_instances_of(named_axes.NamedArrayBase)
        .where(lambda x: self.stack_axis in x.named_shape)
    )
    named_array_slices = [
        named_array.untag_prefix(self.stack_axis).with_positional_prefix()
        for named_array in stacked_array_selection.get_sequence()
    ]

    def body(carry_data, slice_data):
      cur_arg, shared_var_states = carry_data
      named_array_slices, sliced_var_states = slice_data

      # Insert slices of the named arrays.
      sublayer = stacked_array_selection.set_sequence(named_array_slices)

      # Call with variables and get new variables.
      next_arg, new_var_states = sublayer.stateless_call(
          shared_var_states + sliced_var_states, cur_arg, **pure_side_inputs
      )
      new_var_state_by_label = {var.label: var for var in new_var_states}
      new_shared_var_states = [
          new_var_state_by_label[var.label] for var in shared_var_states
      ]
      new_sliced_var_states = [
          new_var_state_by_label[var.label] for var in sliced_var_states
      ]

      # Ensure results are NamedArray instances (with positional axes at the
      # front) so that we can safely add a new axis.
      new_sliced_var_states = (
          selectors.select(new_sliced_var_states)
          .at_instances_of(named_axes.NamedArrayBase)
          .apply(lambda x: x.with_positional_prefix())
      )

      new_carry = named_axes.order_like(
          (next_arg, new_shared_var_states), carry_data
      )
      return new_carry, new_sliced_var_states

    (final_value, new_shared_var_states), new_sliced_var_states = jax.lax.scan(
        body,
        init=(argument, shared_var_states),
        xs=(named_array_slices, sliced_var_states),
        length=self.stack_axis_size,
        unroll=True,
    )
    for state in new_shared_var_states:
      vars_by_label[state.label].value = state.value
    for state in new_sliced_var_states:
      slice_leaves, treedef = jax.tree_util.tree_flatten(
          state.value, is_leaf=named_axes.is_namedarray
      )
      combined_leaves = []
      for leaf in slice_leaves:
        assert isinstance(leaf, named_axes.NamedArrayBase)
        combined_leaves.append(leaf.tag_prefix(self.stack_axis))
      vars_by_label[state.label].value = treedef.unflatten(combined_leaves)

    return final_value

  def key_for_field(self, field_name: str) -> Hashable:
    """Returns a custom GetAttrKey with layer stack metadata."""
    return LayerStackGetAttrKey(
        name=field_name,
        stack_axis=self.stack_axis,
        stack_axis_size=self.stack_axis_size,
    )

  @classmethod
  def from_sublayer_builder(
      cls,
      builder: Callable[..., layer_base.Layer],
      stack_axis: named_axes.AxisName,
      stack_axis_size: int,
      init_base_rng: jax.Array | None,
      builder_kwargs: dict[str, Any],
  ) -> LayerStack:
    """Builds a layer stack of layers with non-shared parameters.

    This function assumes that all variables returned by this builder are
    defined inside the builder. Returning variables that were already defined
    outside the builder is not supported.

    Args:
      builder: A function that builds a single layer, which must take a keyword
        argument ``init_base_rng``. All variables, as well as all other leaf
        values that depend on this RNG, must be NamedArrays.
      stack_axis: The axis name that layer data is stacked along.
      stack_axis_size: The size of the stack axis.
      init_base_rng: The base RNG for initializing the parameters.
      builder_kwargs: Keyword arguments to pass to the builder.

    Returns:
      A new layer stack. All arrays and variables will be split across the
      stack axis.
    """

    def go(rng):
      sublayer = builder(
          init_base_rng=rng,
          **builder_kwargs,
      )
      unbound_sublayer, var_values = variables.unbind_variables(
          sublayer, freeze=True
      )
      if any(
          not isinstance(leaf, named_axes.NamedArrayBase)
          for leaf in jax.tree_util.tree_leaves(
              var_values, is_leaf=named_axes.is_namedarray
          )
      ):
        raise ValueError(
            "Variables returned by the LayerStack builder must only contain"
            " NamedArrays, not ordinary array data."
        )
      namedarray_selection = selectors.select(
          (unbound_sublayer, var_values)
      ).at_instances_of(named_axes.NamedArrayBase)
      adjusted_namedarrays = collections.OrderedDict({
          k: v.with_positional_prefix()
          for k, v in namedarray_selection.selected_by_path.items()
      })
      return namedarray_selection.remainder, adjusted_namedarrays

    remainder, namedarrays = jax.vmap(
        go, out_axes=(None, 0), axis_size=stack_axis_size
    )(
        None
        if init_base_rng is None
        else jax.random.split(init_base_rng, stack_axis_size)
    )
    stacked_namedarrays = {
        k: v.tag_prefix(stack_axis) for k, v in namedarrays.items()
    }
    stacked_sublayers, stacked_variable_values = selectors.Selection(
        selected_by_path=stacked_namedarrays, remainder=remainder
    ).deselect()

    new_variables = []
    for frozen_var in stacked_variable_values:
      if isinstance(frozen_var, variables.ParameterValue):
        new_variables.append(frozen_var.unfreeze_as_copy())
      elif isinstance(frozen_var, variables.StateVariableValue):
        metadata = copy.deepcopy(frozen_var.metadata)
        if "layerstack_axes" not in metadata:
          metadata["layerstack_axes"] = {}
        if stack_axis in metadata["layerstack_axes"]:
          raise ValueError(
              f"Variable {frozen_var.label} already has layerstack_axes"
              f" metadata for axis {stack_axis}"
          )
        metadata["layerstack_axes"][
            stack_axis
        ] = LayerStackVarBehavior.PER_LAYER
        new_variables.append(
            variables.StateVariable(
                value=frozen_var.value,
                label=frozen_var.label,
                metadata=metadata,
            )
        )
      else:
        raise ValueError(
            f"Unsupported variable type for variable: {frozen_var}"
        )

    return LayerStack(
        stacked_sublayers=variables.bind_variables(
            stacked_sublayers, new_variables
        ),
        stack_axis=stack_axis,
        stack_axis_size=stack_axis_size,
    )


def layerstack_axes_from_keypath(
    keypath: tuple[Any, ...],
) -> dict[named_axes.AxisName, int]:
  """Extracts the stacked axes from a keypath.

  This can be used to initialize new variables for transformations that modify
  layers inside a LayerStack. Generally, if this function returns a non-empty
  dict for a given keypath, then any new variable added should include these
  axis names in its "layerstack_axes" metadata, and (if it is PER_LAYER) should
  include the axis names and sizes in its values.

  Args:
    keypath: A JAX keypath to a subtree of a PyTree.

  Returns:
    A mapping containing the names and sizes of any axes mapped over by
    LayerStack layers.
  """
  result = {}
  for item in keypath:
    if isinstance(item, LayerStackGetAttrKey):
      result[item.stack_axis] = item.stack_axis_size
  return result
