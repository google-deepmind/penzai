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

"""Penzai's base layer class and associated utilities.

Layer is the base type for most neural network components in Penzai, but also
used for things that are not necessarily neural networks. The key feature of
a Layer is that it can be called with a single positional argument and produces
a single output. (As such, it must implement ``__call__``.)

The single input is allowed to be a collection or data structure, as is the
output. The purpose of the restriction is to make it easier to compose logic
of multiple layers together. In particular, it's possible to sequentially
run a bunch of layers by passing the output of one layer as the input to the
next. It's also possible to transform layers by nesting their inputs or outputs
inside structures, without worrying about conflicts. For instance, given a
transformation from a layer ``x -> y`` to a layer ``(x, state) -> (y, state)``,
and a transformation from a layer ``x -> y`` to a layer
``(x, random seed) -> y``, you can compose these to transform ``x -> y`` to
``((x, state), random seed) -> (y, state)`` without the two transformations
having to know about each other.

This overall abstraction is inspired by type polymorphism in functional
programming languages, especially as it relates to effect systems.

Layers also make it easier to do shape-checking, by assuming that the input
and output are both single PyTrees.
"""

from __future__ import annotations

import abc
import functools
import typing
from typing import Any, Callable

import jax
from penzai.core import shapecheck
from penzai.core import struct


class MisconfiguredLayerError(Exception):
  """Raised when a layer's attributes are misconfigured.

  Subclasses of Layer can raise ``MisconfiguredLayerError`` in their
  ``input_structure`` or ``output_structure`` methods to indicate that their
  attributes have been set incorrectly and the layer cannot be successfully
  called with any input structure. They can also raise
  ``MisconfiguredLayerError`` in their ``__call__`` method (although this will
  be raised automatically if they use `checked_layer_call`.)
  """


def checked_layer_call(
    func: Callable[[Layer, Any], Any],
) -> Callable[[Layer, Any], Any]:
  """Decorator for ``Layer.__call__`` to add shape-checking and name scopes."""

  @functools.wraps(func)
  def wrapper(self, argument: Any, /) -> Any:
    input_vars = shapecheck.check_structure(
        argument,
        pattern=self.input_structure(),
        error_prefix=(
            "Error while checking the input of a layer of type"
            f" {type(self).__qualname__}:\n"
        ),
    )
    output_structure = self.output_structure()
    with jax.named_scope(type(self).__name__):
      result = func(self, argument)
    shapecheck.check_structure(
        result,
        pattern=output_structure,
        known_vars=input_vars,
        error_prefix=(
            "Error while checking the output of a layer of type"
            f" {type(self).__qualname__}:\n"
        ),
    )
    return result

  wrapper.__penzai_checked_layer_call__ = True
  return wrapper


def unchecked_layer_call(
    func: Callable[[Layer, Any], Any],
) -> Callable[[Layer, Any], Any]:
  """Shape-checking opt-out decorator for ``Layer.__call__``."""
  func.__penzai_checked_layer_call__ = False
  return func


class Layer(struct.Struct, abc.ABC):
  """Abstract base class for neural network layers and other 1-arg callables."""

  @abc.abstractmethod
  def __call__(self, argument: Any, /) -> Any:
    """Abstract call method for a layer.

    Layers are submodels that take a single input and produce a single output.
    By convention, almost all model components in a Penzai model are instances
    of Layer, making it possible to easily compose them with other layers and
    wrappers. If a layer needs to take multiple input arrays, its input can be
    a nested data structure.

    Most subclasses of Layer are encouraged to decorate ``__call__`` with
    `checked_layer_call`, which runs automatic shape checking and adds
    name scopes to aid debugging.

    Args:
      argument: An input value, or possibly a nested structure. Should be passed
        positionally by any caller; caller should not assume this is called
        "argument" exactly. Subclasses of Layer are free to rename this.

    Returns:
      An output value, or possibly a nested structure.
    """
    raise NotImplementedError(
        "__call__ must be overridden for Layer subclasses"
    )

  def input_structure(self) -> shapecheck.StructureAnnotation:
    """Returns the input structure of this layer.

    The input structure of a layer is a PyTree describing the structure the
    layer expects to be called with, using the types from
    `penzai.core.shapecheck`. In particular, it will usually be a PyTree
    with leaves that are either `shapecheck.ArraySpec` nodes or that are
    unchecked `shapecheck.Wildcard` nodes.

    Subclasses of Layer that have complex or configuration-dependent logic in
    ``__call__`` are encouraged to override input_structure. This information
    will be used in two ways:

    * It can give more informative error messages to users when they try to
      call a layer with the wrong input structure.

    * It will be visible in Treescope when the layer is pretty-printed.

    If any ArraySpec contains dimension variables, these dimension
    variables are shared between ``input_structure`` and ``output_structure``.
    This means that the output structure and input structure must have
    consistent sizes if they are annotated with consistent variable names.

    A general suggestion is that each layer should check only the parts
    of its input that it needs to make assumptions about in order to do its
    job well. For instance, low-level operations may want to check the shapes
    of their inputs, but should use general dtypes (like ``jnp.floating``)
    unless they specifically require a specific input dtype. Higher-level
    combinators that contain other layers should only check parts of their input
    that they use (e.g. a layer that unpacks a length-3 tuple should have an
    ``input_structure`` that is a length-3 tuple) but not the parts of their
    input that are passed through to their child layers.

    Note that, if you override this method, you must decorate ``__call__`` with
    `checked_layer_call` to ensure that the input structure is checked (or
    `unchecked_layer_call` to opt out).

    If the attributes of this layer are set incorrectly, you can raise
    `MisconfiguredLayerError` to indicate that the layer cannot b
    successfully called with any input structure.
    """
    return shapecheck.ANY

  def output_structure(self) -> shapecheck.StructureAnnotation:
    """Returns the output structure of this layer.

    The output structure of a layer is a PyTree describing the structure of
    the layer's return value, using the types from
    :mod:`penzai.core.shapecheck`. In particular, it will usually be a PyTree
    with leaves that are either `shapecheck.ArraySpec` nodes or that are
    unchecked `shapecheck.Wildcard` nodes.

    Subclasses of Layer that have complex or configuration-dependent logic in
    ``__call__`` are encouraged to override ``output_structure``. This
    information will be used in two ways:

    * It serves as an assertion that the output matches the expectations of
      the layer, and guards against e.g. accidentally clobbering an axis name.

    * It will be visible in Treescope when the layer is pretty-printed.

    If the attributes of this layer are set incorrectly, you can raise
    `MisconfiguredLayerError` to indicate that calling this layer will
    not succeed at runtime.

    See the documentation for `input_structure` for more details.
    """
    return shapecheck.ANY

  def __init_subclass__(cls, **kwargs):
    """Checks that new subclasses of Layer have wrapped ``__call__`` if needed."""
    super().__init_subclass__(**kwargs)
    if cls.__call__ is not Layer.__call__ and (
        cls.input_structure is not Layer.input_structure
        or cls.output_structure is not Layer.output_structure
    ):
      if not hasattr(cls.__call__, "__penzai_checked_layer_call__"):
        raise TypeError(
            "Subclasses of Layer that override `input_structure` or"
            " `output_structure` should decorate `__call__` with"
            " `checked_layer_call` to ensure that these structures are checked."
            " (If you are checking the inputs and output structures manually,"
            " or want to opt-out for some other reason, you can alternatively"
            " decorate with"
            " `penzai.deprecated.v1.core.layer.unchecked_layer_call`.)"
        )

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    from penzai.deprecated.v1.core._treescope_handlers import layer_handler  # pylint: disable=g-import-not-at-top

    return layer_handler.handle_layers(self, path, subtree_renderer)


# Type alias for an arbitrary callable object with the expected signature.
LayerLike: typing.TypeAlias = Callable[[Any], Any]
