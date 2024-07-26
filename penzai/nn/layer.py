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

"""Penzai's base layer and slot classes and associated utilities.

Layer is the base type for most neural network components in Penzai.
Conceptually, a layer is an object that can be called with

* a single "ordinary input" (either a single array or a pytree), which usually
  comes from the previous layer in the model, and is always passed positionally,
* and a set of named "side inputs", which are passed as keyword arguments,
  represent extra context or state (e.g. attention masks, conditioning
  information, or random number generators), and are generally forwarded between
  layers unchanged.

The purpose of this abstraction is to make it easier to compose logic
of multiple layers together. Combinators such as `Sequential` can be used to
sequence multiple layers, which will feed the output of one layer as the input
to the next, and share the same side inputs across all layers. This simplifies
the "plumbing" of extra information such as attention masks, allowing them to
be used in combination with simple combinators.
"""

from __future__ import annotations

import abc
import typing
from typing import Any, Iterable

from penzai.core import struct
from penzai.core import variables as vars_lib


class Layer(struct.Struct, abc.ABC):
  """Abstract base class for neural network layers and other 1-arg callables."""

  @abc.abstractmethod
  def __call__(self, argument: Any, /, **side_inputs: Any) -> Any:
    """Abstract call method for a layer.

    Layers are model components that take one main input and optional side
    inputs, and produce a single output.
    By convention, almost all model components in a Penzai model are instances
    of Layer, making it possible to easily compose them with other layers and
    wrappers. If a layer needs to take multiple input arrays, its input can be
    a nested data structure.

    Both arguments should be passed positionally by any caller; callers should
    not assume they have particular names, and subclasses of Layer are free to
    rename them.

    Args:
      argument: The primary input to the layer. Usually either an array or a
        nested structure of arrays.
      **side_inputs: Arbitrary side context available to the layer. Each should
        usually be an array, Variable, or structure of arrays or variables.
        Layers *must* accept arbitrary side input keyword arguments and should
        ignore side inputs that they do not use.

    Returns:
      An output value, or possibly a nested structure.
    """
    raise NotImplementedError(
        "__call__ must be overridden for Layer subclasses"
    )

  def bind_variables(
      self,
      variables: Iterable[
          vars_lib.AbstractVariable | vars_lib.AbstractVariableValue
      ],
      allow_unused: bool = False,
  ) -> Layer:
    """Convenience function to bind variables to a layer.

    `layer.bind_variables(variables)` is a simple alias for
    `pz.bind_variables(layer, variables)`.

    Args:
      variables: The collection of variables (or frozen variable values) to
        insert.
      allow_unused: Whether to ignore variables that do not have any matching
        slot (in which case they will not be inserted).

    Returns:
      A copy of this layer with variables (re-)inserted.
    """
    return vars_lib.bind_variables(self, variables, allow_unused)

  @typing.final
  def stateless_call(
      self,
      variable_values: Iterable[vars_lib.AbstractVariableValue],
      argument: Any,
      /,
      **side_inputs: dict[Any, Any],
  ) -> tuple[Any, tuple[vars_lib.AbstractVariableValue, ...]]:
    """Calls a layer with temporary variables, without modifying its state.

    This is a convenience method for:

      * freezing any variables currently inside the model if there are any (so
        that we don't mutate them unexpectedly)
      * creating temporary unfrozen copies of each variable value passed as an
        argument,
      * binding the temporary unfrozen copies to the layer (allowing them to
        be modified while the layer runs),
      * calling the layer,
      * extracting and re-freezing the temporary vars, and
      * returning the result and the updated frozen vars.

    In combination with `variables.unbind_variables`, this can be used to call
    a stateful layer in a functional way.

    Also note that matching variables will *also* be bound inside the argument
    and side inputs. This can simplify the process of calling a model with
    stateful arguments such as random streams.

    Args:
      variable_values: Initial values for each variable that should be mutable
        while the layer runs. These will be substituted for variable slots in
        the layer.
      argument: The argument to pass to the layer. (May also contain variable
        slots that will be bound to variables from ``variable_values``.)
      **side_inputs: Arbitrary side context available to the layer. (May also
        contain variable slots values that will be bound to variables from
        ``variable_values``.)

    Returns:
      A tuple ``(result, updated_vars)`` where ``result`` is the result of
      calling the layer and ``updated_vars`` is a list of updated variable
      values.
    """
    # Freeze any variables remaining in `self` to ensure we don't mutate them.
    # (Usually, this will just freeze the parameters.)
    frozen_self = vars_lib.freeze_variables(self)
    # Substitute in temporary mutable variables based on the input.
    mut_vars = [var.unfreeze_as_copy() for var in variable_values]
    bound_self, bound_arg, bound_sides = vars_lib.bind_variables(
        (frozen_self, argument, side_inputs), mut_vars
    )
    result = bound_self(bound_arg, **bound_sides)
    # Retrieve and freeze the values of the temporary variables.
    return result, tuple(var.freeze() for var in mut_vars)

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    from penzai.nn._treescope_handlers import layer_handler  # pylint: disable=g-import-not-at-top

    return layer_handler.handle_layer(self, path, subtree_renderer)
