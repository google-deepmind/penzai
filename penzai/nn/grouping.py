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

"""Grouping layers, for chaining sequential computations."""

import dataclasses
import typing
from typing import Any, Callable, Sequence

import jax
from penzai.core import selectors
from penzai.core import shapecheck
from penzai.core import struct
from penzai.nn import layer as layer_base


@struct.pytree_dataclass
class Sequential(layer_base.Layer):
  """A group of layers to call sequentially.

  ``Sequential`` is one of the most common layer types to use in a penzai.nn
  model, since many networks can be written as the composition of a number of
  layers. However, you may prefer to use `CheckedSequential` if you can define
  in advance the structure of inputs and outputs your layer will accept.

  A common pattern in penzai is:

  * subclass ``Sequential`` with a different layer name,

  * inherit ``__init__`` and ``__call__`` from ``Sequential``,

  * define a classmethod (often called ``from_config``) that constructs an
    instance of the subclass with its contents.

  This allows the configuration and initialization logic for parts of a network
  (such as a self-attention layer) to be grouped in a single place, without
  affecting the later ability to interactively modify the resulting network.

  Subclasses of ``Sequential`` are NOT allowed to override ``__call__``. If a
  user has a subclass of ``Sequential``, they should be able to assume it just
  calls each child in order. (If you need finer control, consider having a
  ``Sequential`` as a child attribute instead, or just duplicate the relevant
  logic for your own class.)

  Attributes:
    sublayers: A sequence of layers to call in order. These are usually
      pz.nn.Layer instances, but are allowed to be other types of callable
      PyTree as well.
  """

  sublayers: list[layer_base.Layer]

  @typing.final
  def __call__(self, value: Any, **side_inputs) -> Any:
    """Runs each of the sublayers in sequence.

    Args:
      value: The input to the first sublayer.
      **side_inputs: The side inputs for all sublayers.

    Returns:
      The output of the final sublayer.
    """
    for i, layer in enumerate(self.sublayers):
      with jax.named_scope(f"{i}"):
        value = layer(value, **side_inputs)
    return value

  def treescope_color(self) -> str | tuple[str, str]:
    from treescope import formatting_util  # pylint: disable=g-import-not-at-top

    if type(self) is Sequential:  # pylint: disable=unidiomatic-typecheck
      return "#cdcdcd", "color-mix(in oklab, #cdcdcd 25%, white)"
    else:
      type_string = type(self).__module__ + "." + type(self).__qualname__
      accent = formatting_util.color_from_string(type_string)
      return accent, f"color-mix(in oklab, {accent} 25%, white)"


@struct.pytree_dataclass
@typing.final
class NamedGroup(layer_base.Layer):
  """A layer that names an activation or a sequence of layers.

  This layer does not do anything interesting on its own, but exists primarily
  to facilitate manipulation and inspection of a complex network:

  * The name will show up in ``treescope`` when inspecting the network
    interactively, giving context for the wrapped layers.

  * ``NamedGroup`` layers can be selected with ``pz.select`` based on their
    name, using something like ::

      (...).at_instances_of(NamedGroup).where(lambda n: n.name == NAME)

  * When traced in JAX, ``NamedGroup`` layers add their name to the name scope,
    which will be visible in the TensorBoard profiler and in JAXPRs.

  You can also omit the sublayers, in which case this serves as a lightweight
  way to assign a name to an activation (mostly useful in combination with
  `pz.select`).

  Suggestion for when to use ``NamedGroup`` vs subclass `Sequential`: If you
  have a function that builds a particular collection of sub-layers in a
  reusable way, consider subclassing `Sequential` and having that function be a
  constructor classmethod. If you just need to group some sublayers together,
  but want to name them for later reference, just used ``NamedGroup``.

  You shouldn't subclass ``NamedGroup``; either subclass `Sequential` or define
  your own layer.

  Attributes:
    name: The name for the layer.
    sublayers: A sequence of layers to call in order. These are usually
      pz.nn.Layer instances, but are allowed to be other types of callable
      PyTree as well.
  """

  name: str = dataclasses.field(metadata={"pytree_node": False})
  sublayers: Sequence[layer_base.Layer]

  def __call__(self, value: Any, **side_inputs: dict[Any, Any]) -> Any:
    """Runs each of the sublayers in sequence.

    Args:
      value: The input to the first sublayer.
      **side_inputs: The side inputs for all sublayers.

    Returns:
      The output of the final sublayer.
    """
    for i, layer in enumerate(self.sublayers):
      with jax.named_scope(f"{i}"):
        value = layer(value, **side_inputs)
    return value

  def treescope_color(self) -> str | tuple[str, str]:
    from treescope import formatting_util  # pylint: disable=g-import-not-at-top

    accent = formatting_util.color_from_string(self.name)
    return accent, f"color-mix(in oklab, {accent} 25%, white)"


@struct.pytree_dataclass
class CheckedSequential(layer_base.Layer):
  """A group of layers to call sequentially, with known input/output types.

  ``CheckedSequential`` is a "typed" variant of `Sequential`, which is annotated
  with input and output structures. The input and output structures will
  share state variables, which can be used to make assertions about the
  relationship between the shape of the inputs and the shape ouf the outputs.

  Attributes:
    input_like: An input structure, represented as a PyTree of
      `pz.chk.ArraySpec` nodes. This defines the type of input this layer
      expects to receive. Passing anything else will raise an error.
    sublayers: A sequence of layers to call in order. These are usually
      pz.nn.Layer instances, but are allowed to be other types of callable
      PyTree as well.
    output_like: An output structure, represented as a PyTree of
      `pz.chk.ArraySpec` nodes. This defines the type of input this layer will
      produce. Returining anything else will raise an error.
  """

  input_like: shapecheck.StructureAnnotation = dataclasses.field(
      metadata={"pytree_node": False, "treescope_always_collapse": True}
  )
  output_like: shapecheck.StructureAnnotation = dataclasses.field(
      metadata={"pytree_node": False, "treescope_always_collapse": True}
  )
  sublayers: list[layer_base.Layer]

  @typing.final
  def __call__(self, value: Any, **side_inputs: dict[Any, Any]) -> Any:
    """Runs each of the sublayers in sequence.

    Args:
      value: The input to the layer.
      **side_inputs: The side inputs for all sublayers.

    Returns:
      The output of the final sublayer.
    """
    dimvars = shapecheck.check_structure(
        value,
        self.input_like,
        error_prefix="While checking the input of a CheckedSequential:\n",
    )
    for i, layer in enumerate(self.sublayers):
      with jax.named_scope(f"{i}"):
        value = layer(value, **side_inputs)
    shapecheck.check_structure(
        value,
        self.output_like,
        known_vars=dimvars,
        error_prefix="While checking the output of a CheckedSequential:\n",
    )
    return value

  def treescope_color(self) -> str | tuple[str, str]:
    from treescope import formatting_util  # pylint: disable=g-import-not-at-top

    if type(self) is CheckedSequential:  # pylint: disable=unidiomatic-typecheck
      return "#cdcdcd"
    else:
      type_string = type(self).__module__ + "." + type(self).__qualname__
      accent = formatting_util.color_from_string(type_string)
      return accent, "#cdcdcd"


@struct.pytree_dataclass
class Identity(layer_base.Layer):
  """A layer that returns its input unchanged, without any side effects."""

  @typing.final
  def __call__(self, value: Any, **_unused_side_inputs) -> Any:
    """Returns the input unchanged."""
    return value

  def treescope_color(self) -> str | tuple[str, str]:
    return "#cdcdcd", "color-mix(in oklab, #cdcdcd 25%, white)"


@struct.pytree_dataclass
@typing.final
class CheckStructure(layer_base.Layer):
  """A layer that checks the structure of the value passing through it.

  Attributes:
    expected: An expected structure, represented as a PyTree of
      `pz.chk.ArraySpec` nodes. This defines the type of input this layer
      expects to receive. Passing anything else will raise an error.
  """

  expected: Any

  def __call__(self, value: Any, **_unused_side_inputs) -> Any:
    """Checks the structure of the value, then returns it."""
    shapecheck.check_structure(value, self.expected)
    return value

  def treescope_color(self) -> str:
    return "#cdcdcd"


def is_sequential_or_named(tree: Any) -> bool:
  """Checks if a tree is a subclass of `Sequential` or a `NamedGroup`."""
  return isinstance(tree, Sequential | CheckedSequential | NamedGroup)


def is_anonymous_sequential(tree: Any) -> bool:
  """Checks if the type of a node is exactly `Sequential`, not a named subclass."""
  return type(tree) is Sequential  # pylint: disable=unidiomatic-typecheck


def inline_groups(
    tree: Any,
    parent_filter: Callable[[Any], bool],
    child_filter: Callable[[Any], bool],
) -> Any:
  """Inlines sequential nodes into their parents if possible.

  This function finds nodes that match ``child_filter`` within nodes that match
  ``parent_filter``, and splices the sublayers of the child node into the
  parent, removing the child. This can be used to flatten a nested structure of
  `Sequential` or `NamedGroup` objects into a new structer with a smaller depth.

  The logic applies recursively: if a node matches both the parent and the
  child filter, it may be inlined into its parent after its sublayers are
  inlined into it.

  For the common case where you wish to inline "anonymous" groups (instances
  of type `Sequential` but not a more specific subclass of `Sequential`), you
  can use the convenience wrapper `inline_anonymous_sequentials`.

  Args:
    tree: The tree to process.
    parent_filter: A function that returns True on the nodes that we want to
      inline sublayers into.
    child_filter: A function that returns True on the nodes that we want to
      remove and replace with the inlined sequence of its sublayers.

  Returns:
    A copy of ``tree`` that inlines nodes that match ``child_filter`` into their
    parents whenever their parents match ``parent_filter``, as long as they
    are subclasses of `Sequential`, `CheckedSequential`, or `NamedGroup`.
  """

  def _step(subtree):
    # Process children.
    with_processed_children = (
        selectors.select(subtree).at_children().apply(_step)
    )
    # Check for inlining opportunities at this level.
    if is_sequential_or_named(with_processed_children) and parent_filter(
        with_processed_children
    ):
      new_sublayers = []
      for sublayer in with_processed_children.sublayers:
        if is_sequential_or_named(sublayer) and child_filter(sublayer):
          # Inline the child's children into the parent's children.
          new_sublayers.extend(sublayer.sublayers)
        else:
          new_sublayers.append(sublayer)

      # Substitute the new flattened children.
      return dataclasses.replace(
          with_processed_children, sublayers=new_sublayers
      )
    else:
      return with_processed_children

  return _step(tree)


def inline_anonymous_sequentials(tree: Any) -> Any:
  """Inlines instances of `Sequential` (not subclasses) into parent groups."""
  return inline_groups(
      tree,
      parent_filter=is_sequential_or_named,
      child_filter=is_anonymous_sequential,
  )
