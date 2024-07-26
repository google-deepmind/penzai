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

"""A simple parameter type.

Parameters in Penzai are identified by being instances of `Parameter`. By
default, any array in a model that is not an instance of `Parameter` will be
held constant during optimization.

Additionally, parameters have names, which can be used to identify a single
parameter even after patching or reconfiguring a model. This is useful for
storing parameters in checkpoints, for instance.

By convention, Penzai layers should annotate their parameters as being of type
`ParameterLike`, which is a protocol that defines the interface a parameter
should support. The main implementations of `ParameterLike` are:

* `Parameter`: An ordinary parameter that can be updated during optimization.

* `FrozenParameter`: A parameter that is frozen, which still conforms to the
  `ParameterLike` protocol, but will not be updated by default.

* `UninitializedParameter`: A parameter whose value has not yet been assigned.
  These are usually inserted when a model is being built the first time.

* `ShareableUninitializedParameter`: A temporary marker for an uninitialized
  parameter whose value will be shared by multiple layers. You usually don't
  need to construct or manipulate these yourself; instead, use `mark_shareable`
  and `attach_shared_parameters`.

Additionally, users are free to substitute their own implementations of the
`ParameterLike` protocol to customize how the value of a parameter should be
constructed.

Neural network layers are responsible for assigning unique names to each
parameter they create. The typical way to do this is to use
`add_parameter_prefix` to add prefixes to the parameters inside any of their
inner sublayers. Penzai does not automatically track name scopes, but it will
raise errors if it detects multiple parameters with the same name.
"""

from __future__ import annotations

import abc
import collections
import dataclasses
import hashlib
import typing
from typing import Any, Callable, Generic, Protocol

import jax
import numpy as np
from penzai.core import named_axes
from penzai.core import selectors
from penzai.core import shapecheck
from penzai.core import struct
from penzai.deprecated.v1.core import layer as layer_base
from penzai.deprecated.v1.data_effects import effect_base
from penzai.deprecated.v1.data_effects import side_input

T = typing.TypeVar("T")


################################################################################
# Base parameter protocol
################################################################################


class ParameterLike(Protocol, Generic[T]):
  """Protocol for a parameter-like object.

  ``ParameterLike`` defines the common API for parameters in
  `penzai.deprecated.v1.nn`. Any
  parameterized network layer should annotate its parameters as
  ``ParameterLike``, and should not make assumptions about the exact
  implementation of ``ParameterLike`` used.

  Rules for a ``ParameterLike`` implementation:

  * ``.value_structure`` should always be a concrete structure defining the
    structure of the parameter's value, usually an instance of
    `shapecheck.ArraySpec` (with no dimension variables). Accessing this
    should never raise an exception.

  * ``.value`` should be the concrete value of the parameter if available.
    However, it is allowed for accessing ``.value`` to raise an exception if
    this type is a placeholder for a concrete runtime parameter. In particular,
    this can raise an exception if this parameter is uninitialized or if it is
    a reference to a shared parameter that has not been provided yet.

  ``ParameterLike`` is a protocol, which means implementations of it do not have
  to explicitly subclass it. If you define ``value`` as a dataclass attribute
  rather than a property, you should avoid subclassing ``ParameterLike``
  directly, since Python's ABC system will not see that as an implementation of
  the method.
  """

  @property
  @abc.abstractmethod
  def value(self) -> T:
    """The value of the parameter. May raise an error for some instances."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def value_structure(self) -> shapecheck.StructureAnnotation:
    """The structure of the parameter."""
    raise NotImplementedError


################################################################################
# Parameter renaming traversal
################################################################################


class SupportsParameterRenaming(abc.ABC):
  """Base class that identifies a PyTree node as supporting parameter renaming.

  This class can be used to identify nodes that have parameter names, and which
  should be renamed when `add_parameter_prefix` is called. Subclassing this
  allows new parameter variants to also support renaming.
  """

  @abc.abstractmethod
  def with_renamed_parameters(self, rename_fn: Callable[[str], str]):
    """Returns a copy of this node with parameterss renamed.

    If implemented on a parameter, this should make a copy of the parameter with
    a new name. If implemented on a container of parameters, this should rename
    every parameter, possibly by recursively calling `with_renamed_parameters`
    on its children.

    Args:
      rename_fn: Function mapping old names to new names.

    Returns:
      Copy of ``self`` with renaming applied.
    """
    raise NotImplementedError("with_renamed_parameters not implemented.")


################################################################################
# Common parameter classes
################################################################################


@struct.pytree_dataclass
class Parameter(struct.Struct, Generic[T], SupportsParameterRenaming):
  """A learnable parameter.

  Most initialized layers should use Parameter to store their learnable
  parameters.

  Attributes:
    value: The value of the parameter.
    name: The name of this parameter.
  """

  value: T
  name: str = dataclasses.field(metadata={"pytree_node": False})

  @property
  def value_structure(self) -> shapecheck.StructureAnnotation:
    return shapecheck.as_array_structure(self.value)

  def with_renamed_parameters(
      self, rename_fn: Callable[[str], str]
  ) -> Parameter:
    return dataclasses.replace(self, name=rename_fn(self.name))

  def treescope_color(self) -> str:
    return "#93cce1"


@struct.pytree_dataclass
class FrozenParameter(struct.Struct, Generic[T], SupportsParameterRenaming):
  """A non-learnable parameter.

  This is like `Parameter`, but it is NOT an instance of `Parameter`, so
  ``pz.select(model).at_instances_of(pz.nn.Parameter)`` will not select it.

  Attributes:
    value: The value of the parameter.
    name: The name of this parameter.
  """

  value: T
  name: str = dataclasses.field(metadata={"pytree_node": False})

  @property
  def value_structure(self) -> shapecheck.StructureAnnotation:
    return shapecheck.as_array_structure(self.value)

  def with_renamed_parameters(
      self, rename_fn: Callable[[str], str]
  ) -> FrozenParameter:
    return dataclasses.replace(self, name=rename_fn(self.name))

  def treescope_color(self) -> str:
    return "#a0b0c9"


class UninitializedParameterError(Exception):
  """Raised when an uninitialized parameter is accessed."""


_INFER_VALUE_STRUCTURE = object()


@struct.pytree_dataclass
class UninitializedParameter(
    struct.Struct, Generic[T], SupportsParameterRenaming
):
  """An uninitialized parameter.

  ``UninitializedParameter`` represents a parameter that has not yet been
  initialized, along with an initialization strategy for it. In most cases,
  model-building code should use ``UninitializedParameter`` to construct their
  initial parameters, to make it possible to build a model without initializing
  its parameters immediately.

  Attributes:
    initializer: Callable used to initialize the parameter.
    name: The name to use when checkpointing this parameter, or restoring it
      from a checkpoint.
    value_structure: The structure of the value that will be returned by the
      initializer, but with UninitializedArray in place of any of the actual
      parameters. Usually inferred automatically from ``initializer``.
  """

  initializer: Callable[[jax.Array], T] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  name: str = dataclasses.field(metadata={"pytree_node": False})
  value_structure: shapecheck.StructureAnnotation

  def __init__(
      self,
      initializer: Callable[[jax.Array], T],
      name: str,
      value_structure: Any = _INFER_VALUE_STRUCTURE,
  ):
    """Constructs an uninitialized parameter.

    Args:
      initializer: Initializer to use.
      name: Name to use.
      value_structure: If provided, an explicit value structure that the
        initializer should return. If not provided, will be inferred by using
        `jax.eval_shape`.
    """
    self.initializer = initializer
    self.name = name
    if value_structure is _INFER_VALUE_STRUCTURE:

      def _get_structure(leaf):
        if isinstance(leaf, named_axes.NamedArrayBase):
          return shapecheck.ArraySpec(
              shape=leaf.positional_shape,
              named_shape=leaf.named_shape,
              dtype=leaf.dtype,
          )
        elif isinstance(leaf, jax.ShapeDtypeStruct):
          return shapecheck.ArraySpec(
              shape=leaf.shape,
              dtype=leaf.dtype,
          )
        else:
          raise ValueError(
              "Could not infer parameter shape from initializer! Unrecognized"
              f" leaf: {leaf}"
          )

      value_structure = jax.tree_util.tree_map(
          _get_structure,
          jax.eval_shape(self.initializer, jax.random.PRNGKey(0)),
          is_leaf=named_axes.is_namedarray,
      )
    self.value_structure = value_structure

  def initialize(self, prng_key: jax.Array) -> Parameter[T]:
    """Randomly initializes the parameter.

    Args:
      prng_key: Key to use for initialization.

    Returns:
      A new instance of ``Parameter``.

    Raises:
      ValueError: If the initializer's output does not match the expected
      structure from ``self.value_structure``.
    """
    value = self.initializer(prng_key)
    shapecheck.check_structure(value=value, pattern=self.value_structure)
    return Parameter(self.initializer(prng_key), self.name)

  def as_empty_parameter(self) -> Parameter[T]:
    """Creates a placeholder parameter containing `jax.ShapeDtypeStruct` leaves.

    This can be used to create a model with the right PyTree structure without
    actually initializing the parameters; this is useful for restoring a model
    from a checkpoint, for instance.

    Returns:
      A new instance of Parameter, but with empty `jax.ShapeDtypeStruct` leaves
      instead of initialized arrays.
    """

    placeholder_value = (
        selectors.select(self.value_structure)
        .at_instances_of(shapecheck.ArraySpec)
        .apply(lambda s: s.into_pytree())
    )
    return Parameter(placeholder_value, self.name)

  def initialize_with_value(
      self, value: T, check_structure: bool = True, strict_dtype: bool = True
  ) -> Parameter[T]:
    """Directly initializes the parameter with particular value.

    This can be used to bypass the initializer function and set the value
    directly, which can be useful for loading from checkpoints (for instance).

    Args:
      value: Value to set.
      check_structure: Whether to check that the value matches the expected
        structure.
      strict_dtype: Whether to check that the value matches the expected dtype.
        Ignored unless ``check_structure`` is True.

    Returns:
      An initialized parameter with the given value.
    """
    if check_structure:
      if strict_dtype:
        pattern = self.value_structure
      else:
        pattern = (
            selectors.select(self.value_structure)
            .at_instances_of(shapecheck.ArraySpec)
            .apply(lambda s: dataclasses.replace(s, dtype=np.generic))
        )
      shapecheck.check_structure(value=value, pattern=pattern)
    return Parameter(value, self.name)

  @property
  def value(self):
    r"""Value accessor for compatibility with `ParameterLike`.

    Raises:
      UninitializedParameterError: Since ``UninitializedParameter``\ s are not
        initialized, retrieving their value always raises an error.
    """
    raise UninitializedParameterError(
        "Cannot use the value of an uninitialized parameter! Please initialize"
        " your model's parameters before attempting to use the model."
    )

  def with_renamed_parameters(
      self, rename_fn: Callable[[str], str]
  ) -> UninitializedParameter:
    return dataclasses.replace(self, name=rename_fn(self.name))

  def treescope_color(self) -> str:
    return "#cea03c"


################################################################################
# Parameter sharing
################################################################################


@struct.pytree_dataclass(has_implicitly_inherited_fields=True, init=False)  # pytype: disable=wrong-keyword-args
class ShareableUninitializedParameter(UninitializedParameter):
  """A shareable variant of an uninitialized parameter.

  A ``ShareableUninitializedParameter`` is just like an ordinary
  `UninitializedParameter`, except that they have been tagged as being OK to
  share by name.

  Tagging a parameter as shareable does not actually enable sharing itself,
  because there must only be one copy of each shared parameter in the PyTree to
  ensure that gradients propagate correctly. As such, models or submodels with
  ``ShareableUninitializedParameter`` in them should be transformed using
  `attach_shared_parameters`, which will take ownership of the shareable
  parameters and make them actually shared. If
  ``ShareableUninitializedParameter`` instances with the same name are not bound
  using `attach_shared_parameters`, this will lead to a name conflict during
  initialization.
  """

  @classmethod
  def from_uninitialized(
      cls, uninit: UninitializedParameter
  ) -> ShareableUninitializedParameter:
    """Returns a ``ShareableUninitializedParameter`` equivalent to ``uninit``."""
    return cls(
        initializer=uninit.initializer,
        name=uninit.name,
        value_structure=uninit.value_structure,
    )


@dataclasses.dataclass(frozen=True, order=True)
class SharedParamTag:
  """A tag for a shared parameter.

  Attributes:
    name: The local name for the shared parameter.
  """

  name: str


@struct.pytree_dataclass
class SharedParameterLookup(struct.Struct, Generic[T]):
  """A marker identifying a shared parameter.

  A ``SharedParameterLookup`` acts like a parameter, but does not actually hold
  its value. Instead, it retrieves its parameter using a `SideInputEffect`, and
  expects the value of the parameter to be provided by some external handler,
  as configured by `attach_shared_parameters`.

  Attributes:
    ref: The `SideInputEffect` that provides the value for this parameter.
    value_structure: The structure of the value that will be substituted here.
  """

  ref: side_input.SideInputEffect[ParameterLike[T]]
  value_structure: shapecheck.StructureAnnotation = dataclasses.field(
      metadata={"pytree_node": False}
  )

  @property
  def value(self):
    """Value accessor for compatibility with `ParameterLike`.

    Raises:
      MissingSharedParameterError: Since ``SharedParameterLookup`` does not have
        a value until given one by an `attach_shared_parameters` wrapper,
        retrieving their value always raises an error.
    """
    return self.ref.ask().value

  def treescope_color(self) -> str:
    return effect_base.get_effect_color(side_input.SideInputEffect)


def mark_shareable(submodel: Any):
  """Marks all uninitialized parameters in ``submodel`` as shareable.

  This function is used to annotate the parameters in part of a model as being
  shareable: all `UninitializedParameter` instances will be replaced by
  equivalent `ShareableUninitializedParameter` instances. The submodel can then
  be used multiple times, as long as all uses are wrapped in a single call to
  `attach_shared_parameters`.

  Parameter sharing is opt-in only to avoid accidentally sharing parameters due
  to name collisions. By convention, any caller that calls ``make_shareable`` is
  also responsible for constructing the `attach_shared_parameters` wrapper that
  will actually own the shared parameters.

  Args:
    submodel: Part of a model whose parameters should be shared.

  Returns:
    A copy of ``submodel`` where all `UninitializedParameter` instances have
    been replaced by `ShareableUninitializedParameter` instances.
  """
  return (
      selectors.select(submodel)
      .at_instances_of(UninitializedParameter)
      .apply(ShareableUninitializedParameter.from_uninitialized)
  )


def attach_shared_parameters(
    submodel: layer_base.Layer, *, strict: bool = True
) -> side_input.WithConstantSideInputs:
  """Wraps a submodel in a handler that takes ownership of shareable params.

  Penzai models must have only one copy of each parameter in the model pytree
  to ensure that gradients are computed properly. This function is responsible
  for transforming a model with multiple copies of a shareable parameter into
  a model with a single copy, using the `SideInputEffect` handler.

  Args:
    submodel: Submodel containing `ShareableUninitializedParameter` instances
      that we should take ownership of.
    strict: If True, require that shared parameters have identical initializers
      by Python value. This should always be the case if the same uninitialized
      parameter instance was reused multiple times (e.g. if `mark_shareable` was
      used), and can catch issues if there are accidental name conflicts. If
      False, arbitrarily picks the first initializer seen for each shared
      parameter.

  Returns:
    A new `pz.de.WithConstantSideInputs` handler wraps the original submodel,
      and holds a single `UninitializedParameter` for each uniquely-named
      `ShareableUninitializedParameter` in the original submodel.
  """
  shared_parameters = {}

  def reparent(shareable_uninit: ShareableUninitializedParameter):
    tag = SharedParamTag(shareable_uninit.name)
    if tag in shared_parameters:
      # Make sure the shareable parameters are consistent.
      prev = shared_parameters[tag]
      if prev.value_structure != shareable_uninit.value_structure:
        raise ValueError(
            "Detected incompatible value structures for two shared parameters"
            f" with name {repr(shareable_uninit.name)}!"
        )
      if strict and (prev.initializer != shareable_uninit.initializer):
        raise ValueError(
            "Detected non-identical initializers for two shareable parameters"
            f" with name {repr(shareable_uninit.name)}! This may indicate a"
            " name conflict. If you wish to arbitrarily resolve the"
            " conflict and use the first initializer, pass"
            " `strict=False` to WithSharedParameters.from_shareables_of`"
        )
    else:
      # First time we've seen a shared parameter!
      shared_parameters[tag] = UninitializedParameter(
          initializer=shareable_uninit.initializer,
          name=shareable_uninit.name,
          value_structure=shareable_uninit.value_structure,
      )
    return SharedParameterLookup(
        side_input.SideInputRequest(tag),
        value_structure=shareable_uninit.value_structure,
    )

  reparented_submodel = (
      selectors.select(submodel)
      .at_instances_of(ShareableUninitializedParameter)
      .apply(reparent)
  )
  return side_input.WithConstantSideInputs.handling(
      reparented_submodel,
      shared_parameters,
      handler_id=effect_base.infer_or_check_handler_id(
          "shared_params", reparented_submodel
      ),
  )


################################################################################
# Parameter naming helpers
################################################################################


def _rename_parameters(tree, renamer: Callable[[str], str]):
  """Helper function to rename all `SupportsParameterRenaming` subtrees."""
  return (
      selectors.select(tree)
      .at_instances_of(SupportsParameterRenaming)
      .apply(lambda node: node.with_renamed_parameters(renamer))
  )


def add_parameter_prefix(
    prefix: str,
    tree: Any,
    delimiter: str = ".",
) -> Any:
  """Prepends a prefix to all parameter names inside the tree.

  This function can be used to avoid name conflicts when combining sublayers
  into a larger model layer. For instance, composite layers (subclasses of
  Sequential) can call ``add_parameter_prefix`` from their ``from_config``, to
  ensure that none of their sublayers have conflicting names.

  Args:
    prefix: Prefix to append to all parameters in `tree`.
    tree: Tree containing `Parameter`, `FrozenParameter`, or
      `UninitializedParameter` instances.
    delimiter: Delimiter to add between the prefix and the original name.

  Returns:
    A version of ``tree`` where all `Parameter`, `FrozenParameter`, and
    `UninitializedParameter` instances, and any other subclass of
    `SupportsParameterRenaming`, have their names updated to new names
    with the format ``"{prefix}{delimiter}{original_name}"``.
  """

  def _renamer(name):
    return prefix + delimiter + name

  return _rename_parameters(tree, _renamer)


################################################################################
# Helper functions
################################################################################


def check_no_duplicated_parameters(model: Any) -> None:
  """Checks that there are no duplicated parameters in a model.

  Args:
    model: An arbitrary model.

  Raises:
    ValueError: If two concrete parameters (parameters with a value) have the
    same name.
  """
  parameter_counts = collections.Counter(
      param.name
      for param in (
          selectors.select(model)
          .at_instances_of(Parameter | FrozenParameter | UninitializedParameter)
          .get_sequence()
      )
  )
  duplicates = [key for key, count in parameter_counts.items() if count > 1]
  if duplicates:
    raise ValueError(
        "Found multiple parameters with the same name! Repeated names:"
        f" {duplicates}"
    )


def initialize_parameters(model: Any, prng_key: jax.Array) -> Any:
  """Initializes all parameters in a model.

  Args:
    model: Model to initialize.
    prng_key: Key to use to initialize parameters.

  Returns:
    Copy of ``model``, where each instance of `UninitializedParameter` is
    replaced with a `Parameter` initialized using an independent PRNG key
    derived from ``prng_key``. Keys are derived from the names of the
    parameters, so parameters with the same name will be initialized
    consistently if this method is given the same key.
  """
  check_no_duplicated_parameters(model)

  def name_to_key(name: str):
    hashed_bytes = hashlib.sha1(name.encode("utf-8")).digest()
    offset = int.from_bytes(hashed_bytes[:4], byteorder="big")
    return jax.random.fold_in(prng_key, offset)

  return (
      selectors.select(model)
      .at_instances_of(UninitializedParameter)
      .apply(lambda uninit: uninit.initialize(name_to_key(uninit.name)))
  )
