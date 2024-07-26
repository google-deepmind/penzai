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

"""Utilities for runtime shape-checking.

Penzai is highly dynamic by nature, and is heavily based on having nested
tree structured objects, which is difficult to capture with Python's type
annotations. This module provides a shape-and-structure checking utility that
can be used to check the shapes and dtypes of nested structures while also
simultaneously extracting parts that the program may need to use (e.g. lengths
for axes).
"""

from __future__ import annotations

import collections
from collections.abc import Collection, Mapping
import dataclasses
import itertools
import typing
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import ordered_set
from penzai.core import named_axes
from penzai.core import struct

# Type alias for a structure annotation. Structure annotations can have any
# Python type, but generally have leaves that are either Wildcard or
# ArraySpec.
StructureAnnotation: typing.TypeAlias = Any


@struct.pytree_dataclass
class Wildcard(struct.Struct):
  """Wildcard marker that matches any value or subtree of a PyTree."""

  description: str | None = dataclasses.field(
      default=None, metadata={"pytree_node": False}
  )

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    from penzai.core._treescope_handlers import shapecheck_handlers  # pylint: disable=g-import-not-at-top

    return shapecheck_handlers.handle_arraystructures(
        self, path, subtree_renderer
    )


ANY = Wildcard()


@dataclasses.dataclass(frozen=True)
class RemainingAxisPlaceholder:
  """Sentinel marker for the remainder of axes."""

  pass


@dataclasses.dataclass(frozen=True, order=True)
class DimVar(Mapping):
  """A dimension variable, which will be matched with the shapes in a structure.

  In a slight abuse of syntax, ``DimVar`` can be unpacked using ``*`` or ``**``
  to bind the
  name to a list or dict of variables instead of a single value. In particular,
  if ``foo`` is a ``DimVar``, then ``*foo`` expands to a `MultiDimVar` that will
  capture more than one name, and ``**foo`` expands to a dictionary that maps a
  `MultiDimVar` to the singleton placeholder `RemainingAxisPlaceholder`. It's
  also possible to index a ``DimVar`` with an integer or string key, for
  consistency with the output of `vars_for_axes` and the unpacking constraint
  solver.

  The usual way to create a DimVar is using the `var` or `vars_for_axes`
  functions.

  Attributes:
    name: The unique name for this variable. Variable identity is based on this
      name. May also be a tuple, where the outer name is a name for a collection
      and the inner name is the named shape of a single axis in that collection.
  """

  name: str | tuple[str, str | int]

  def __len__(self) -> int:
    return 1

  def __iter__(self):
    if not isinstance(self.name, str):
      raise TypeError("Cannot iterate over a tuple-named DimVar")
    yield MultiDimVar(self.name)

  def __getitem__(self, key):
    if not isinstance(self.name, str):
      raise TypeError("Cannot index a tuple-named DimVar")
    elif key == MultiDimVar(self.name):
      return RemainingAxisPlaceholder()
    elif isinstance(key, int | str):
      return DimVar((self.name, key))
    else:
      raise KeyError(f"Cannot get key {repr(key)} from a DimVar")


@dataclasses.dataclass(frozen=True, order=True)
class MultiDimVar:
  """A variable standing in for a sequence or dictionary axis names.

  The usual way to create a ``MultiDimVar`` is by unpacking the `var` function,
  e.g. as ``(1, 2, *var("foo"), 5, 6)`` or ``{'a': 1, **var("foo"), 'c': 5}``.
  You can also cast the output of `var` to a tuple or a dict.

  Attributes:
    name: The unique name for this variable. Variable identity is based on this
      name.
  """

  name: str


@dataclasses.dataclass(frozen=True, order=True)
class KnownDim:
  """A dimension with a known value, used to bind a name to a value directly.

  The usual way to create a ``KnownDim`` is by calling `vars_for_axes` with
  a dictionary that maps axis names to known sizes. They are also constructed
  internally by `check_structure` while solving for variables.

  Attributes:
    name: The unique name for a variable bound to this dimension. Variable
      identity is based on this name. Dots in a name have special meaning,
      referring to sub-variables of a larger name.  May also be a tuple, where
      the outer name is a name for a collection and the inner name is the named
      shape of a single axis in that collection.
    size: The size that this variable maps to.
    from_keypath: Optional keypath that indicates where this size was bound.
  """

  name: str | tuple[str, str | int]
  size: int
  from_keypath: str | None = None


def var(name: str) -> DimVar:
  """Creates a variable for an axis shape.

  The resulting variable can be used in place of a single array dimension. It
  can also be unpacked with ``*var(name)`` to replace a sequence of array
  dimensions, or with ``**var(name)`` to replace a subdictionary of named array
  axes.

  Args:
    name: Name for the variable.

  Returns:
    Variable that can be matched with axes. Repeat uses of variables with the
    same name must have the same length. The result can be unpacked with
    ``*var(name)`` to stand in for multiple positional dimensions or with
    ``**var(name)`` to stand in for multiple named axes.
  """
  if not isinstance(name, str):
    raise TypeError(f"Variable names must be strings. Got {repr(name)}")
  return DimVar(name)


def vars_for_axes(
    var_name: str,
    axis_names_or_specs: Collection[str] | Mapping[str, int | None],
) -> dict[str, DimVar | KnownDim]:
  """Creates variables for a known collection of named axes.

  Args:
    var_name: A name for the variable that will store the concrete sizes for all
      of these named axes.
    axis_names_or_specs: Either a collection of axis names, or a mapping from
      axis names to integer axis sizes or to None (for unknown axis sizes).

  Returns:
    A dictionary with the keys from ``axis_names_or_specs``, and values that
    either reflect either the known size from the value of
    ``axis_names_or_specs`` or an unknown dimension. Intended to be passed as
    the ``named_shape`` of an `ArraySpec` or unpacked into a larger dictionary
    of named shapes.
  """
  if not isinstance(axis_names_or_specs, Mapping):
    axis_names_or_specs = {k: None for k in axis_names_or_specs}

  return {
      k: (
          DimVar((var_name, k))
          if size is None
          else KnownDim((var_name, k), size)
      )
      for k, size in axis_names_or_specs.items()
  }


@dataclasses.dataclass(frozen=True)
class _PositionalConstraint:
  """A constraint for matching two positional shapes."""

  value: tuple[int, ...]
  pattern: tuple[int | DimVar | MultiDimVar | KnownDim, ...]

  def extract_vars(self):
    myvars = ordered_set.OrderedSet()
    myvars.update(
        [dim for dim in self.pattern if isinstance(dim, (DimVar, MultiDimVar))]
    )
    return myvars

  def num_multi_dim_vars(self):
    return sum(1 for dim in self.pattern if isinstance(dim, MultiDimVar))


@dataclasses.dataclass(frozen=True)
class _NamedConstraint:
  """A constraint for matching two named shapes."""

  value: dict[named_axes.AxisName, int]
  pattern: dict[
      named_axes.AxisName | MultiDimVar,
      int | DimVar | KnownDim | RemainingAxisPlaceholder,
  ]

  def extract_vars(self):
    myvars = ordered_set.OrderedSet()
    myvars.update(
        [value for value in self.pattern.values() if isinstance(value, DimVar)]
    )
    myvars.update(
        [key for key in self.pattern.keys() if isinstance(key, MultiDimVar)]
    )
    return myvars

  def num_multi_dim_vars(self):
    return sum(1 for key in self.pattern.keys() if isinstance(key, MultiDimVar))


@dataclasses.dataclass(frozen=True)
class _UnsatisfiedConstraint:
  """A constraint that we have already failed to satisfy."""

  msg: str


_MatchConstraint = (
    _PositionalConstraint | _NamedConstraint | _UnsatisfiedConstraint
)


@struct.pytree_dataclass
class ArraySpec(struct.Struct):
  """A non-leaf marker for a (named) array structure.

  This is like a `jax.ShapeDtypeStruct`, but it is an empty PyTree node instead
  of being a leaf, so flattening it produces no children. It also supports
  named axes and dimension variables.

  `ArraySpec` is used for shape checking as well as to annotate the expected
  shape and dtype of uninitialized parameters. It may appear in a model PyTree
  either inside an uninitialized parameter or inside shape-checking layers.

  Note that the ``named_shape`` attribute specifically refers to the named shape
  of a Penzai `NamedArray` or `NamedArrayView`. Some internal JAX transforms
  (e.g. the deprecated ``xmap``) can produce JAX values with their own internal
  ``named_shape`` attribute, but this will not be checked against the
  ``named_shape`` of an ``ArraySpec``.

  Attributes:
    shape: Positional shape of the eventual array that will be inserted here.
      Can include `DimVar` or `MultiDimVar` if it is being used for
      shape-checking.
    dtype: Dtype of the eventual array that will be inserted here. Can be an
      abstract dtype (e.g. `np.floating`, which is actually an abstract base
      class and has type `type`) or a concrete array dtype (e.g.
      ``np.dtype("float32")`` which has type `np.dtype`). Abstract dtypes accept
      any concrete subdtype.
    named_shape: Named shape of the eventual array that will be inserted here.
      Can include a `DimVar` instances as values if it is being used for
      shape-checking. Can also include `MultiDimVar` instances as keys with the
      `RemainingAxisPlaceholder` sentinel as the value, to indicate an arbitrary
      collection of names.
  """

  shape: tuple[int | DimVar | MultiDimVar, ...] = dataclasses.field(
      default=(), metadata={"pytree_node": False}
  )
  dtype: jax.typing.DTypeLike = dataclasses.field(
      default=np.generic, metadata={"pytree_node": False}
  )
  named_shape: Mapping[
      named_axes.AxisName | MultiDimVar,
      int | DimVar | RemainingAxisPlaceholder,
  ] = dataclasses.field(default_factory=dict, metadata={"pytree_node": False})

  @property
  def positional_shape(self) -> tuple[int | DimVar | MultiDimVar, ...]:
    return self.shape

  @classmethod
  def floating_named(
      cls,
      named_shape: Mapping[
          named_axes.AxisName | MultiDimVar,
          int | DimVar | RemainingAxisPlaceholder,
      ],
  ) -> ArraySpec:
    """Returns an `ArraySpec` with this named shape and `np.floating` dtype."""
    return cls(named_shape=named_shape, dtype=np.floating)

  def into_pytree(self) -> jax.ShapeDtypeStruct | named_axes.NamedArray:
    """Converts an ``ArraySpec`` into a (possibly wrapped) PyTree leaf.

    By default, an ``ArraySpec`` has no PyTree children. This method can be
    used to convert it into a PyTree subtree that contains one leaf, a
    `jax.ShapeDtypeStruct`. This can then be used to e.g. restore parameters
    into
    a structure.

    If this structure has a named shape, will return a NamedArray wrapping a
    ``ShapeDtypeStruct``, with the parameter order inferred from the order of
    names in this ``ArraySpec``. Otherwise, will return an ordinary
    ``ShapeDtypeStruct``.

    Returns:
      A PyTree whose structure matches this structure.
    """
    if any(isinstance(dim, DimVar | MultiDimVar) for dim in self.shape) or any(
        isinstance(v, DimVar | RemainingAxisPlaceholder)
        or isinstance(k, MultiDimVar)
        for k, v in self.named_shape.items()
    ):
      raise ValueError(
          "Cannot use `into_pytree` on an ArraySpec with variable dimensions."
      )
    if self.named_shape:
      return named_axes.NamedArray(
          named_axes=collections.OrderedDict(self.named_shape.items()),
          data_array=jax.ShapeDtypeStruct(
              shape=self.shape + tuple(self.named_shape.values()),
              dtype=self.dtype,
          ),
      )
    else:
      return jax.ShapeDtypeStruct(self.shape, self.dtype)

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    from penzai.core._treescope_handlers import shapecheck_handlers  # pylint: disable=g-import-not-at-top

    return shapecheck_handlers.handle_arraystructures(
        self, path, subtree_renderer
    )


def _abstract_leaf(value: Any) -> ArraySpec | Wildcard:
  """Helper function to get an `ArraySpec` view of a leaf."""
  if isinstance(value, named_axes.NamedArrayBase):
    value.check_valid()
    positional_shape = value.positional_shape
    named_shape = value.named_shape
    dtype = value.dtype
  elif isinstance(value, ArraySpec):
    positional_shape = value.shape
    named_shape = value.named_shape
    dtype = value.dtype
  elif hasattr(value, "shape") and hasattr(value, "dtype"):
    positional_shape = value.shape
    named_shape = {}
    dtype = value.dtype
  elif isinstance(value, jax.typing.ArrayLike):
    positional_shape = jnp.shape(value)
    named_shape = {}
    dtype = jnp.result_type(value)
  else:
    return ANY
  return ArraySpec(shape=positional_shape, named_shape=named_shape, dtype=dtype)


def _match_leaf(
    array_structure: ArraySpec,
    value: Any,
) -> list[_MatchConstraint]:
  """Helper function to match a leaf value against an `ArraySpec`.

  Args:
    array_structure: Expected structure to match agains.
    value: Value to match.

  Returns:
    List of constraints that must be satisfied for this value to match the
    structure.
  """
  concrete_structure = _abstract_leaf(value)
  if isinstance(concrete_structure, ArraySpec):
    positional_shape = concrete_structure.shape
    named_shape = concrete_structure.named_shape
    dtype = concrete_structure.dtype
  else:
    return [
        _UnsatisfiedConstraint(
            f"Couldn't match an ArraySpec with a non-arraylike value {value}"
        )
    ]

  constraints = []

  # Check dtype.
  if isinstance(array_structure.dtype, type) and issubclass(
      array_structure.dtype, np.generic
  ):
    if not jnp.issubdtype(dtype, array_structure.dtype):
      constraints.append(
          _UnsatisfiedConstraint(
              "Value has the wrong dtype: expected a sub-dtype of"
              f" {array_structure.dtype} but got dtype {dtype}."
          )
      )
  else:
    expected_dtype = jax.dtypes.canonicalize_dtype(
        array_structure.dtype, allow_extended_dtype=True
    )
    actual_dtype = jax.dtypes.canonicalize_dtype(
        dtype, allow_extended_dtype=True
    )
    if actual_dtype != expected_dtype:
      constraints.append(
          _UnsatisfiedConstraint(
              "Value has the wrong dtype: expected dtype"
              f" {expected_dtype} but got dtype {actual_dtype}."
          )
      )

  constraints.append(
      _PositionalConstraint(
          tuple(positional_shape), tuple(array_structure.shape)
      )
  )
  constraints.append(
      _NamedConstraint(dict(named_shape), dict(array_structure.named_shape))
  )
  return constraints


@dataclasses.dataclass(frozen=True)
class MatchResult(
    Mapping[str, int | tuple[int, ...] | dict[named_axes.AxisName, int]]
):
  """The result of a successful shape check.

  The bound match variables can be extracted in four ways:

  * Use ordinary dict indexing, e.g. ``result["foo"]``

  * Use attribute access, e.g. ``result.foo``

  * Pass multiple keys like a slice to get multiple results, e.g.
    ``result["foo", "bar"]`` is the same as ``(result["foo"], result["bar"])``

  * Convert to a dictionary using e.g. ``dict(result)``

  Attributes:
    _bindings: Resulting bindings from the match.
  """

  _bindings: dict[str, int | tuple[int, ...] | dict[named_axes.AxisName, int]]

  def __len__(self):
    return len(self._bindings)

  def __iter__(self):
    return iter(self._bindings)

  def __getitem__(self, key_or_keys: str | tuple[str, ...]):
    if isinstance(key_or_keys, str):
      return self._bindings[key_or_keys]
    else:
      return tuple(self._bindings[key] for key in key_or_keys)

  def __getattr__(self, key: str):
    if key in self._bindings:
      return self._bindings[key]
    else:
      raise AttributeError(name=key, obj=self)


class StructureMismatchError(Exception):
  """Raised when a `check_structure` call fails."""


@dataclasses.dataclass(frozen=True)
class _Binding:
  """A binding for a variable set by a keypath."""

  source: str
  value: int | tuple[int, ...] | dict[named_axes.AxisName, int]


def _try_match_one(
    keypath,
    pattern: int | DimVar | MultiDimVar | KnownDim,
    value: int | tuple[int, ...] | dict[named_axes.AxisName, int],
    solutions: dict[str | tuple[str, str], _Binding],
) -> str | None:
  """Internal helper to match a pattern with a value.

  Args:
    keypath: Keypath for the current value.
    pattern: Pattern to match.
    value: Value to match.
    solutions: Current solutions for variable bindings. New variable values will
      be added to this.

  Returns:
    None if the match succeeded, otherwise a string describing the failure.
  """
  if isinstance(pattern, DimVar | MultiDimVar):
    if isinstance(pattern, DimVar):
      assert isinstance(value, int)
    else:
      assert isinstance(value, (tuple, dict))
    if pattern.name in solutions:
      if solutions[pattern.name].value != value:
        return (
            f"Size {value} does not match previous size"
            f" {solutions[pattern.name].value} for {_show_dim(pattern)} from"
            f" {solutions[pattern.name].source}"
        )
    else:
      solutions[pattern.name] = _Binding(
          "root" + jax.tree_util.keystr(keypath), value
      )
  elif isinstance(pattern, KnownDim):
    assert isinstance(value, int)
    if pattern.name in solutions:
      if solutions[pattern.name].value != pattern.size:
        return (
            f"Size {pattern.size} does not match previous size"
            f" {solutions[pattern.name].value} for {_show_dim(pattern)} from"
            f" {solutions[pattern.name].source}"
        )
    else:
      solutions[pattern.name] = _Binding(
          "root" + jax.tree_util.keystr(keypath), value
      )
    if value != pattern.size:
      if pattern.from_keypath is not None:
        return (
            f"Size {value} does not match previous"
            f" size {pattern.size} for {_show_dim(pattern)} from"
            f" {pattern.from_keypath}"
        )
      else:
        return (
            f"Actual size {value} does not match expected {_show_dim(pattern)}"
        )
  elif isinstance(pattern, int):
    if value != pattern:
      return f"Actual size {value} does not match expected {_show_dim(pattern)}"
  return None


def _positional_inline_multidimvars(
    constraint: _PositionalConstraint,
    solutions: dict[str | tuple[str, str], _Binding],
) -> tuple[_PositionalConstraint, str]:
  """Simplifies a positional constraint by inlining multivars."""
  new_pattern = []
  inline_summary = []
  for dim in constraint.pattern:
    if isinstance(dim, MultiDimVar) and dim.name in solutions:
      binding = solutions[dim.name]
      assert isinstance(binding.value, tuple)
      for i, val in enumerate(binding.value):
        new_pattern.append(
            KnownDim((dim.name, i), val, from_keypath=binding.source)
        )
      inline_summary.append(
          f"\n  After inlining var({repr(dim.name)}) = {binding.value} from"
          f" {binding.source}"
      )
    else:
      new_pattern.append(dim)
  return (
      _PositionalConstraint(value=constraint.value, pattern=tuple(new_pattern)),
      "".join(inline_summary),
  )


def _named_inline_multidimvars(
    constraint: _NamedConstraint,
    solutions: dict[str | tuple[str, str], _Binding],
) -> tuple[_NamedConstraint | _UnsatisfiedConstraint, str]:
  """Simplifies a named constraint by inlining multivars."""
  new_pattern = {}
  inlinable: list[MultiDimVar] = []
  inline_summary = []
  for key, val in constraint.pattern.items():
    if isinstance(key, MultiDimVar) and key.name in solutions:
      assert val == RemainingAxisPlaceholder()
      inlinable.append(key)
    else:
      new_pattern[key] = val

  for key in inlinable:
    binding = solutions[key.name]
    assert isinstance(binding.value, dict)
    for subkey, subval in binding.value.items():
      assert isinstance(subkey, str)
      if subkey in new_pattern:
        return (
            _UnsatisfiedConstraint(
                "Key conflict while substituting unpacked variable"
                f" {_show_dim(key)} for known named sub-shape"
                f" {binding.value} from"
                f" {binding.source}: name {subkey} is"
                " already present in the shape"
                f" {_summarize_pattern(constraint.pattern)}."
            ),
            "".join(inline_summary),
        )
      new_pattern[subkey] = KnownDim(
          (key.name, subkey), subval, from_keypath=binding.source
      )
    inline_summary.append(
        f"\n  After inlining var({repr(key.name)}) = {binding.value} from"
        f" {binding.source}"
    )
  return (
      _NamedConstraint(value=constraint.value, pattern=new_pattern),
      "".join(inline_summary),
  )


def _show_dim(dim: int | DimVar | MultiDimVar | KnownDim) -> str:
  """Summarizes a dimension variable into a human-readable summary."""
  if isinstance(dim, DimVar | MultiDimVar):
    if isinstance(dim.name, str):
      return f"var({repr(dim.name)})"
    else:
      assert isinstance(dim.name, tuple)
      return f"var({repr(dim.name[0])})[{repr(dim.name[1])}]"
  elif isinstance(dim, KnownDim):
    if isinstance(dim.name, str):
      return f"var({repr(dim.name)}):={dim.size}"
    else:
      assert isinstance(dim.name, tuple)
      return f"var({repr(dim.name[0])})[{repr(dim.name[1])}]:={dim.size}"
  else:
    return repr(dim)


def _summarize_pattern(
    pattern: (
        tuple[int | DimVar | MultiDimVar | KnownDim, ...]
        | dict[
            named_axes.AxisName | MultiDimVar,
            int | DimVar | KnownDim | RemainingAxisPlaceholder,
        ]
    ),
) -> str:
  """Summarizes a pattern into a human-readable summary."""
  if isinstance(pattern, tuple):
    parts = []
    for dim in pattern:
      if isinstance(dim, MultiDimVar):
        parts.append(f"*var({repr(dim.name)})")
      else:
        parts.append(_show_dim(dim))
    if len(parts) == 1:
      return "(" + parts[0] + ",)"
    return "(" + ", ".join(parts) + ")"
  elif isinstance(pattern, dict):
    parts = []
    for key, val in pattern.items():
      if isinstance(key, MultiDimVar) and isinstance(
          val, RemainingAxisPlaceholder
      ):
        parts.append(f"**var({repr(key.name)})")
      else:
        parts.append(f"{repr(key)}: {_show_dim(val)}")
    return "{" + ", ".join(parts) + "}"
  else:
    raise TypeError(f"Bad pattern {pattern}")


def check_structure(
    value: Any,
    pattern: StructureAnnotation,
    known_vars: (
        Mapping[str, int | tuple[int, ...] | dict[named_axes.AxisName, int]]
        | None
    ) = None,
    error_prefix: str = "",
) -> MatchResult:
  """Checks that a structure of values matches a pattern.

  Args:
    value: A PyTree of arrays to check.
    pattern: A PyTree of `ArraySpec` leaves to check against. Each `ArraySpec`
      can include dimension variables (created by ``var(name)``, ``*var(name)``,
      ``**var(name)`` or ``**vars_for_axes(name, spec)``), and any two variables
      with the same name must have the same length. The pattern may also include
      other leaves which will be checked for exact equality.
    known_vars: A previous match result (or other assignment of names to values)
      that the names must be consistent with.
    error_prefix: Optional prefix to prepend to exceptions to indicate the
      context in which shape-checking is being done.

  Returns:
    A match result containing the values for each dimension variable in the
    pattern.

  Raises:
    StructureMismatchError: If the value does not match the pattern.
  """
  all_constraints = []

  def add_constraints(keypath, pattern: Any, value: Any):
    if isinstance(pattern, ArraySpec):
      for constraint in _match_leaf(pattern, value):
        all_constraints.append((keypath, constraint))
    elif isinstance(pattern, Wildcard):
      # Wildcard matches anything.
      pass
    else:
      is_equal = pattern == value
      if isinstance(is_equal, jax.Array | np.ndarray):
        try:
          is_equal = bool(np.all(is_equal))
        except jax.errors.ConcretizationTypeError:
          is_equal = False
      if not is_equal:
        failed = _UnsatisfiedConstraint(
            f"Value {repr(value)} was not equal to the non-ArraySpec"
            f" pattern {repr(pattern)}."
        )
        all_constraints.append((keypath, failed))

  jax.tree_util.tree_map_with_path(
      add_constraints,
      pattern,
      value,
      is_leaf=lambda x: isinstance(x, ArraySpec | Wildcard),
  )

  # Phase 1: Collect and handle constraints that don't require solving for
  # variables.
  failures = []
  variable_constraints = []
  for keypath, constraint in all_constraints:
    if isinstance(constraint, _UnsatisfiedConstraint):
      failures.append((keypath, constraint.msg))
    elif isinstance(constraint, _PositionalConstraint):
      variable_constraints.append((keypath, constraint))
    elif isinstance(constraint, _NamedConstraint):
      variable_constraints.append((keypath, constraint))
    else:
      raise ValueError(
          f"Ungiven constraint type {type(constraint)}: {constraint}"
      )

  # Phase 2: Solve for variable constraints.
  # Solutions maps variable names to _Binding instances.
  if known_vars is None:
    solutions = {}
  else:
    solutions = {
        name: _Binding(
            source=(
                "the known variable assignments (argument `known_vars` to"
                " check_structure)"
            ),
            value=value,
        )
        for name, value in dict(known_vars).items()
    }
  while variable_constraints or failures:
    # Try to solve each constraint in order.
    unsolved_variable_constraints = []
    remaining_variables = set()
    for keypath, constraint in variable_constraints:
      if isinstance(constraint, _PositionalConstraint):
        # Solving a positional constraint.
        # First inline any "unpack" patterns that we already have solved:
        constraint, inline_info = _positional_inline_multidimvars(
            constraint, solutions
        )
        if constraint.num_multi_dim_vars() >= 2:
          # Can't solve with multiple dim vars!
          unsolved_variable_constraints.append((keypath, constraint))
          remaining_variables.update(constraint.extract_vars())
        else:
          # We can solve this positional constraint.
          multi_dim_set = [
              i
              for i, val in enumerate(constraint.pattern)
              if isinstance(val, MultiDimVar)
          ]
          if multi_dim_set:
            # We have a single unpacked variable that should match many dims.
            # We need to match prefixes and suffixes, and assign the rest to
            # the wildcard.
            [prefix_size] = multi_dim_set
            suffix_size = len(constraint.pattern) - prefix_size - 1
            if len(constraint.value) < prefix_size + suffix_size:
              failures.append((
                  keypath,
                  (
                      f"Positional shape {constraint.value} was shorter than"
                      " wildcard pattern"
                      f" {_summarize_pattern(constraint.pattern)}"
                      + inline_info
                  ),
              ))
            else:
              pattern_prefix = constraint.pattern[:prefix_size]
              pattern_suffix = constraint.pattern[prefix_size + 1 :]
              multi_dim = constraint.pattern[prefix_size]
              shape_prefix = constraint.value[:prefix_size]
              shape_suffix = constraint.value[
                  len(constraint.value) - suffix_size :
              ]
              shape_middle = constraint.value[
                  prefix_size : len(constraint.value) - suffix_size
              ]

              subfailures = []
              err = _try_match_one(
                  keypath=keypath,
                  pattern=multi_dim,
                  value=shape_middle,
                  solutions=solutions,
              )
              if err is not None:
                subfailures.append(
                    f"  Unpack pattern *var({multi_dim.name}): " + err
                )
              matched_parts = itertools.chain(
                  zip(range(prefix_size), shape_prefix, pattern_prefix),
                  zip(
                      range(
                          len(constraint.value) - suffix_size,
                          len(constraint.value),
                      ),
                      shape_suffix,
                      pattern_suffix,
                  ),
              )
              for i, sval, spat in matched_parts:
                err = _try_match_one(
                    keypath=keypath,
                    pattern=spat,
                    value=sval,
                    solutions=solutions,
                )
                if err is not None:
                  subfailures.append(f"  Dim {i}: " + err)
              if subfailures:
                failures.append((
                    keypath,
                    (
                        "Positional shape mismatch between value"
                        f" {constraint.value} and pattern"
                        f" {_summarize_pattern(constraint.pattern)}:\n"
                        + "\n".join(subfailures)
                        + inline_info
                    ),
                ))
          else:
            # Matching two sequences without dimension wildcards.
            if len(constraint.value) != len(constraint.pattern):
              failures.append((
                  keypath,
                  (
                      f"Positional shape {constraint.value} had different"
                      " length than pattern"
                      f" {_summarize_pattern(constraint.pattern)}"
                      + inline_info
                  ),
              ))
            else:
              subfailures = []
              for i, (sval, spat) in enumerate(
                  zip(constraint.value, constraint.pattern)
              ):
                err = _try_match_one(
                    keypath=keypath,
                    pattern=spat,
                    value=sval,
                    solutions=solutions,
                )
                if err is not None:
                  subfailures.append(f"  Dim {i}: " + err)
              if subfailures:
                failures.append((
                    keypath,
                    (
                        "Positional shape mismatch between value"
                        f" {constraint.value} and pattern"
                        f" {_summarize_pattern(constraint.pattern)}:\n"
                        + "\n".join(subfailures)
                        + inline_info
                    ),
                ))
      elif isinstance(constraint, _NamedConstraint):
        # Solving a named axis constraint.
        # First inline any "unpack" patterns that we already have solved:
        constraint, inline_info = _named_inline_multidimvars(
            constraint, solutions
        )
        if isinstance(constraint, _UnsatisfiedConstraint):
          failures.append((keypath, constraint.msg + inline_info))
        elif constraint.num_multi_dim_vars() >= 2:
          # Can't solve with multiple dim vars!
          unsolved_variable_constraints.append((keypath, constraint))
          remaining_variables.update(constraint.extract_vars())
        else:
          # Should be able to solve.
          non_dimvar_keys = [
              key
              for key in constraint.pattern.keys()
              if not isinstance(key, MultiDimVar)
          ]
          maybe_dimvar_key = [
              key
              for key in constraint.pattern.keys()
              if isinstance(key, MultiDimVar)
          ]
          remaining_value_keys = ordered_set.OrderedSet(constraint.value.keys())
          subfailures = []
          for key in non_dimvar_keys:
            # Check one key.
            if key in constraint.value:
              err = _try_match_one(
                  keypath=keypath,
                  pattern=constraint.pattern[key],
                  value=constraint.value[key],
                  solutions=solutions,
              )
              if err is not None:
                subfailures.append(f"  Axis {repr(key)}: " + err)
              remaining_value_keys.remove(key)
            else:
              subfailures.append(
                  f"  Axis {repr(key)}: Expected to be present but was missing"
              )
          if maybe_dimvar_key:
            # Match the unpack pattern.
            [dimvar_key] = maybe_dimvar_key
            err = _try_match_one(
                keypath=keypath,
                pattern=dimvar_key,
                value={k: constraint.value[k] for k in remaining_value_keys},
                solutions=solutions,
            )
            if err is not None:
              subfailures.append(
                  f"  Unpack pattern **{repr(dimvar_key)}: " + err
              )
          elif remaining_value_keys:
            # No unpack pattern, but we ran out of keys.
            subfailures.append(
                "  Unexpected names in value's named shape:"
                f" {sorted(remaining_value_keys)}"
            )
          if subfailures:
            failures.append((
                keypath,
                (
                    "Named shape mismatch between value"
                    f" {constraint.value} and pattern"
                    f" {_summarize_pattern(constraint.pattern)}:\n"
                    + "\n".join(subfailures)
                    + inline_info
                ),
            ))
      else:
        raise ValueError(
            f"Unknown constraint type {type(constraint)}: {constraint}"
        )

    # If we encountered an error, or if we were unable to resolve any of the
    # constraints, the match is a failure.
    if failures:
      msgs = [error_prefix + "Mismatch while checking structures:"]
      for keypath, failure in failures:
        msgs.append(f"At root{jax.tree_util.keystr(keypath)}: {failure}")
      if unsolved_variable_constraints:
        msgs.append(
            "There may also be additional mismatches due to remaining"
            " unsolved constraints."
        )
      raise StructureMismatchError("\n".join(msgs))
    elif len(unsolved_variable_constraints) < len(variable_constraints):
      variable_constraints = unsolved_variable_constraints
    else:
      raise StructureMismatchError(
          error_prefix
          + "Could not solve for all variable constraints. This usually means"
          " the variable assignment is ambiguous because multiple unpack"
          " patterns (*var(...) or **var(...)) appeared in the same positional"
          " or named shape.\nUnsolvable variables: "
          + ", ".join(_show_dim(var) for var in sorted(remaining_variables))
          + "\nUnsolvable constraints:"
          + "".join(
              f"\n  {constraint.value} =="
              f" {_summarize_pattern(constraint.pattern)} from"
              f" root{jax.tree_util.keystr(keypath)}"
              for (keypath, constraint) in unsolved_variable_constraints
          )
      )

  # No more constraints, which means this was a successful match. We need to
  # re-associate any variables whose values were tuples.
  final_matches = {}
  match_errors = []
  for name, binding in solutions.items():
    if isinstance(name, tuple):
      if name[0] in solutions:
        # Redundant! We already have this in our solution.
        found = None
        if isinstance(name[1], int):
          if isinstance(solutions[name[0]].value, tuple) and name[1] < len(
              solutions[name[0]].value
          ):
            found = solutions[name[0]].value[name[1]]
        else:
          assert isinstance(name[1], str)
          if isinstance(solutions[name[0]].value, dict):
            found = solutions[name[0]].value.get(name[1])
        if found != binding.value:
          match_errors.append(
              f"  {_show_dim(DimVar(name[0]))}: Solved as"
              f" {solutions[name[0]].value} from"
              f" {solutions[name[0]].source} but"
              f" separately solved {_show_dim(DimVar(name))} as"
              f" {binding.value} from"
              f" {binding.source}"
          )
      else:
        if name[0] not in final_matches:
          final_matches[name[0]] = {}
        # Each match should appear in the solution dict at most once, because
        # if it already exists we just check for it.
        assert name[1] not in final_matches[name[0]]
        final_matches[name[0]][name[1]] = binding.value
    else:
      final_matches[name] = binding.value

  if match_errors:
    raise StructureMismatchError(
        error_prefix
        + "Unexpected conflicts between variable assignments:\n"
        + "\n".join(match_errors)
    )
  return MatchResult(final_matches)


@dataclasses.dataclass
class DimensionVariableSubstitution:
  """A substitution for all of the dimension variables in a structure.

  DimensionVariableSubstitution can be used to inspect and modify the dimension
  variables appearing in an unknown structure. It is intended to be used as
  part of transformations that inspect the input or output structures of layers.

  The substitutions are allowed to contain other dimension variables. For
  instance, you can set `mapping_variables["foo"] = {"bar":1, **var("baz")}` to
  map the original unpacked mapping variable "foo" to the new composite mapping.
  This can be used to e.g. rename variables.

  Attributes:
    size_variables: A map from dimension variable names to their sizes.
    sequence_variables: A map from sequence variable names to their sequence of
      sizes.
    mapping_variables: A map from mapping variable names to their mappings from
      axis name to sizes.
  """

  size_variables: dict[str | tuple[str, str | int], int | DimVar]
  sequence_variables: dict[str, tuple[int | DimVar | MultiDimVar, ...]]
  mapping_variables: dict[
      str, dict[Any | MultiDimVar, int | DimVar | RemainingAxisPlaceholder]
  ]

  def is_empty(self) -> bool:
    """Returns True if there are no variables."""
    return not (
        self.size_variables or self.sequence_variables or self.mapping_variables
    )


def get_dimension_variables(
    structure: StructureAnnotation,
) -> DimensionVariableSubstitution:
  """Returns a collection of all dimension variables in a structure.

  Args:
    structure: The structure to extract variables from. Usually a PyTree with
      `ArraySpec` leaves.

  Returns:
    A `DimensionVariableSubstitution` that contains all dimension variables in
    the structure, mapping them back to themselves, so that calling
    `full_substitute_dimension_variables` would leave the result unchanged
    (except for `KnownDim` instances). Usually, you will want to replace the
    values in the substitution with something else, and keep only the keys.
  """
  result = DimensionVariableSubstitution({}, {}, {})

  def get_leaf(leaf):
    if isinstance(leaf, ArraySpec):
      for dim in leaf.shape:
        if isinstance(dim, DimVar):
          result.size_variables[dim.name] = dim
        elif isinstance(dim, MultiDimVar):
          result.sequence_variables[dim.name] = (dim,)
      for key, val in leaf.named_shape.items():
        if isinstance(val, DimVar):
          result.size_variables[val.name] = val
        elif isinstance(key, MultiDimVar):
          assert isinstance(val, RemainingAxisPlaceholder)
          result.mapping_variables[key.name] = {key: val}

  jax.tree_util.tree_map(
      get_leaf, structure, is_leaf=lambda x: isinstance(x, ArraySpec)
  )
  return result


def full_substitute_dimension_variables(
    structure: StructureAnnotation,
    substitution: DimensionVariableSubstitution,
) -> Any:
  """Substitutes all dimension variables for their known values.

  Args:
    structure: A PyTree structure containing `ArraySpec` leaves.
    substitution: A substitution for the dimension variables. Should have values
      for all variables in the structure, usually with the same keys as returned
      by `get_dimension_variables`.

  Returns:
    A copy of the structure with all dimension variables substituted for their
    known values. Any `KnownDim` instances will also be stripped out and
    replaced
    by their constant sizes, so if the substitution did not contain any new
    dimension variables, the resulting structure will be fully concrete (with
    no variables).
  """

  def subst_leaf(keypath, leaf):
    if isinstance(leaf, ArraySpec):
      new_shape = []
      for dim in leaf.shape:
        if isinstance(dim, DimVar):
          assignment = substitution.size_variables[dim.name]
          new_shape.append(assignment)
        elif isinstance(dim, KnownDim):
          new_shape.append(dim.size)
        elif isinstance(dim, MultiDimVar):
          assignment = substitution.sequence_variables[dim.name]
          new_shape.extend(assignment)
        else:
          new_shape.append(dim)
      new_named_shape = {}
      for key, val in leaf.named_shape.items():
        if isinstance(val, DimVar):
          assignment = substitution.size_variables[val.name]
          new_named_shape[key] = assignment
        elif isinstance(val, KnownDim):
          new_named_shape[key] = val.size
        elif isinstance(key, MultiDimVar):
          assert isinstance(val, RemainingAxisPlaceholder)
          assignment = substitution.mapping_variables[key.name]
          for subkey, subval in assignment.items():
            if subkey in new_named_shape or subkey in leaf.named_shape:
              raise ValueError(
                  "Key conflict while substituting mapping variable"
                  f" {_show_dim(key)} at root{jax.tree_util.keystr(keypath)}:"
                  f" name {subkey} is already present"
              )
            new_named_shape[subkey] = subval
        else:
          new_named_shape[key] = val
      return ArraySpec(
          shape=tuple(new_shape), named_shape=new_named_shape, dtype=leaf.dtype
      )
    else:
      return leaf

  return jax.tree_util.tree_map_with_path(
      subst_leaf, structure, is_leaf=lambda x: isinstance(x, ArraySpec)
  )


def as_array_structure(tree: Any) -> StructureAnnotation:
  """Abstracts a concrete tree of values into a tree of `ArraySpec` nodes.

  Args:
    tree: The tree to get an array structure for.

  Returns:
    A copy of `tree` where each arraylike or `NamedArray` subtree is replaced
    by an `ArraySpec`, and each other leaf is replaced by `ANY`.
  """
  return jax.tree_util.tree_map(
      _abstract_leaf, tree, is_leaf=named_axes.is_namedarray
  )
