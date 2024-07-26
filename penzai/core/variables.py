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

"""Support for named pockets of mutable/shared state within a JAX pytree.

A variable is a mutable Python object that can be stored in a JAX pytree, and
which contains a JAX pytree. It cannot directly be passed through JAX
transformations; instead, it must be extracted before the transformation and
re-inserted afterwards. (Eventually, variables may be extended to natively
support JAX transformations once the necessary JAX APIs are available.)

Variables are used to simplify the handling of parameters and mutable state in
Penzai models, giving an eager interface that follows ordinary Python semantics,
while still allowing mutable state to be safely managed for JAX.

Operations on variables include:

- "Unbinding" them, which extracts each variable and replaces it with it's
  "variable slot" (a placeholder for a variable).
- "Freezing" them, which converts a variable into a frozen variable value, which
  is itself a pytree node.
- "Unfreezing", which converts a frozen variable back into a (new) mutable
  variable.
- "Binding" them, which re-inserts either mutable or frozen variables into the
  pytree in place of their corresponding variable slots.

Every variable must have a unique "slot" value, which uniquely identifies it
within a particular JAX pytree. The same variable Python object
may appear in multiple places in the same pytree, but if there are two different
variable objects with the same slot value, this will cause an error when
variable values are unbound.

Passing Penzai variables through JAX transformations usually involves a
combination of these steps. For instance, to take gradients with respect to
parameters, you can unbind and freeze them, then take gradients w.r.t. those
frozen values, re-binding them inside the function being differentiated. To
"functionalize" a stateful operation, you can bind temporary variables, then
unbind them afterward.

Most Penzai models use two particular types of variable:

* `Parameter`: A model parameter variable, which can be modified using gradient
  descent but isn't modified while the model runs,
* `StateVariable`: A state variable, which can be modified while the model runs
  and can be used to store mutable state.

Parameters and state variables are implemented similarly, but are kept separate
because they may be treated differently by model combinators and user code.
(For instance, when JIT-compiling a particular sublayer, we can often assume
that parameters do not change, even if variables might.)

A Penzai variable is somewhat similar to an NNX Variable (from `flax.nnx`) but
provides a more restricted interface; this allows it to integrate with JAX's
pytrees without tracking a full object graph of Python dependencies.
"""

from __future__ import annotations

import abc
import contextlib
import dataclasses
import functools
import inspect
import typing
from typing import Any, Callable, Generic, Hashable, Iterable, Literal, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from penzai.core import auto_order_types
from penzai.core import selectors
from penzai.core import struct


T = TypeVar("T")

VariableLabel: TypeAlias = Hashable


class VariableConflictError(Exception):
  """Raised when a Variable label is used by multiple Variables."""


class UnboundVariableError(Exception):
  """Raised when attempting to access the value of an unbound variable."""


class AbstractVariable(abc.ABC):
  """Base class for all variables.

  Variables are allowed to be mutable, and should not be registered as PyTree
  nodes.
  """

  @abc.abstractmethod
  def freeze(self) -> AbstractVariableValue:
    """Returns a frozen copy of this variable."""
    raise NotImplementedError("freeze must be overridden by subclasses.")

  @abc.abstractmethod
  def get_slot(self) -> AbstractVariableSlot:
    """Returns the slot that this variable is replaced with when unbound.

    The variable slot will uniquely identify this variable within a pytree.
    """
    raise NotImplementedError("get_slot must be overridden by subclasses.")

  @abc.abstractmethod
  def update(self, new_frozen_value: AbstractVariableValue):
    """Updates the value of this variable to match a frozen variable."""
    raise NotImplementedError("update must be overridden by subclasses.")


class AbstractVariableValue(struct.Struct, abc.ABC):
  """Base class for all frozen variables."""

  @abc.abstractmethod
  def unfreeze_as_copy(self) -> AbstractVariable:
    """Returns a new mutable copy of this variable."""
    raise NotImplementedError(
        "unfreeze_as_copy must be overridden by subclasses."
    )

  @abc.abstractmethod
  def get_slot(self) -> AbstractVariableSlot:
    """Returns the slot that this variable is replaced with when unbound.

    The `get_slot` method of a frozen variable should always return the same
    slot as the `get_slot` method of the variable it was created from (and also
    for the variables created by `unfreeze_as_copy`).
    """
    raise NotImplementedError("get_slot must be overridden by subclasses.")


class AbstractVariableSlot(struct.Struct, abc.ABC):
  """Base class for all variable slots.

  Slots are placeholders for variables that have been unbound from a pytree.
  They must be PyTree nodes, and should also be hashable (i.e. they should not
  contain any array data or mutable types).
  """


#### Core variable manipulation ####


@typing.overload
def unbind_variables(
    tree: Any,
    predicate: Callable[[AbstractVariable], bool] | None = None,
    freeze: Literal[False] = False,
) -> tuple[Any, tuple[AbstractVariable, ...]]:
  ...


@typing.overload
def unbind_variables(
    tree: Any,
    predicate: Callable[[AbstractVariable], bool] | None = None,
    *,
    freeze: Literal[True],
) -> tuple[Any, tuple[AbstractVariableValue, ...]]:
  ...


def unbind_variables(
    tree: Any,
    predicate: Callable[[AbstractVariable], bool] | None = None,
    freeze: bool = False,
) -> tuple[Any, tuple[AbstractVariable | AbstractVariableValue, ...]]:
  """Unbinds variables from a pytree, inserting variable slots in their place.

  This function can be used to extract variables from a pytree before a JAX
  transformation or control-flow primitive. Those vars can either be directly
  updated, or passed through a JAX transformation by calling `freeze` on
  them, calling `unfreeze_as_copy` and then `bind_variables` inside the
  transformation, and then calling `freeze` again before returning the updated
  variable values.

  Args:
    tree: A tree containing variables. Each variable can appear in the tree more
      than once, but if there are two distinct variable objects that have the
      same label (or otherwise would map to the same slot), an error will be
      raised.
    predicate: A function that returns True for variables that should be
      extracted. If None, all variables will be extracted.
    freeze: Whether to return frozen variables instead of mutable variables.

  Returns:
    A tuple ``(tree_with_slots, variables)``, where ``tree_with_slots`` is
    a copy of the original tree with the extracted variables replaced by their
    corresponding slots, and ``variables`` is a collection of
    variables extracted. If `frozen == True`, the returned variables will be
    frozen.

  Raises:
    VariableConflictError: If two variables map to the same slot but are
      different Python objects, or if there is already a conflicting slot in
      the tree.
  """
  if predicate is None:
    predicate = lambda _: True

  leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(
      tree, is_leaf=lambda l: isinstance(l, AbstractVariableSlot)
  )

  slot_labels = set()
  for _, leaf in leaves_with_paths:
    if isinstance(leaf, AbstractVariableSlot):
      slot_labels.add(leaf)

  variable_dict = {}
  variable_keypaths = {}

  new_leaves = []
  for keypath, leaf in leaves_with_paths:
    if isinstance(leaf, AbstractVariable) and predicate(leaf):
      leaf_slot = leaf.get_slot()
      if leaf_slot in slot_labels:
        raise VariableConflictError(
            f"Variable slot {leaf_slot} is already in use in the tree."
        )
      elif leaf_slot in variable_dict:
        if variable_dict[leaf_slot] is leaf:
          new_leaves.append(leaf_slot)
        else:
          raise VariableConflictError(
              f"Found two Variables with the same slot {leaf_slot}! At"
              f" keypaths tree{jax.tree_util.keystr(keypath)},"
              f" tree{jax.tree_util.keystr(variable_keypaths[leaf_slot])}."
          )

      else:
        variable_keypaths[leaf_slot] = keypath
        variable_dict[leaf_slot] = leaf
        new_leaves.append(leaf_slot)
    else:
      new_leaves.append(leaf)

  if freeze:
    extracted = tuple(var.freeze() for var in variable_dict.values())
  else:
    extracted = tuple(variable_dict.values())

  return treedef.unflatten(new_leaves), extracted


def bind_variables(
    tree: Any,
    variables: Iterable[AbstractVariable | AbstractVariableValue],
    allow_unused: bool = False,
    unfreeze_as_copy: bool = False,
) -> Any:
  """Binds variables (mutable or frozen) into the variable slots in a pytree.

  This function re-inserts variable instances into a tree in place of their
  matching variable slots. It is the inverse of `unbind_variables`.

  Args:
    tree: The tree to substitute variables into.
    variables: The collection of variables to insert.
    allow_unused: Whether to ignore variables that do not have any matching slot
      (in which case they will not be inserted).
    unfreeze_as_copy: Whether to unfreeze variable values before inserting them,
      producing a new mutable copy of each input variable. If True, all input
      variables must be instances of `AbstractVariableValue`.

  Returns:
    A copy of ``tree`` with variables re-inserted.
  """
  leaves, treedef = jax.tree_util.tree_flatten(
      tree, is_leaf=lambda l: isinstance(l, AbstractVariableSlot)
  )

  if unfreeze_as_copy:
    orig_variables = variables
    variables = []
    for var in orig_variables:
      if isinstance(var, AbstractVariableValue):
        variables.append(var.unfreeze_as_copy())
      else:
        raise ValueError(
            "unfreeze_as_copy=True is only allowed if all variables are"
            " variable values (e.g. ParameterValue or StateVariableValue)."
        )

  substitution = {}
  for var in variables:
    var_slot = var.get_slot()
    if var_slot in substitution:
      raise ValueError(
          "Cannot substitute multiple variables into the same slot! "
          f"Repeated slot: {var_slot}"
      )
    substitution[var_slot] = var

  seen = set()

  new_leaves = []
  for leaf in leaves:
    if isinstance(leaf, AbstractVariableSlot) and leaf in substitution:
      new_leaves.append(substitution[leaf])
      seen.add(leaf)
    else:
      new_leaves.append(leaf)

  if not allow_unused:
    unseen = set(substitution.keys()) - seen
    if unseen:
      raise ValueError(
          f"Did not find any matching slots for some variables: {unseen}"
      )

  return treedef.unflatten(new_leaves)


def freeze_variables(
    tree: Any,
    predicate: Callable[[AbstractVariable], bool] | None = None,
) -> Any:
  """Replaces each variable in a pytree with a frozen copy.

  The resulting tree will contain frozen variable instances instead of mutable
  variable instances. Frozen variables are themselves pytree nodes, so the
  resulting tree will be safe to pass through JAX transformations if all
  variables are frozen.

  Args:
    tree: A tree containing variables.
    predicate: A function that returns True for variables that should be frozen.
      If None, all variables will be frozen.

  Returns:
    A copy of `tree` but with all variables (or those selected by `predicate`)
    replaced by equivalent frozen instances.
  """
  if predicate is None:
    predicate = lambda _: True
  return (
      selectors.select(tree)
      .at_instances_of(AbstractVariable)
      .where(predicate)
      .apply(lambda var: var.freeze())
  )


def variable_jit(fun, *, donate_variables: bool = False, **jit_kwargs):
  """Variable-aware version of `jax.jit`.

  This function is like `jax.jit`, but adds support for Variables as leaves of
  the input pytree(s).

  Limitations:

  * Closed-over Variables are not supported. All Variables must be passed as
    arguments.
  * Variables always have an unspecified sharding.
  * Variables should not be included in ``static_argnums`` or
    ``static_argnames`` of the jitted function.
  * The keyword argument ``"__penzai_variables"`` is used to track Variables
    and should not be used directly.

  If you run into issues with this wrapper or if you need more control, consider
  using `jax.jit` directly and manually unbinding the Variables before the
  transformation.

  Args:
    fun: The function to be jitted.
    donate_variables: Whether to donate Variables to the jitted function.
    **jit_kwargs: Additional arguments to pass to `jax.jit`. Note: Any donated
      keyword arguments must be configured using ``donate_argnames`` instead of
      ``donate_argnums``.

  Returns:
    A jitted version of ``fun``
  """
  sig = inspect.signature(fun)
  if any(
      param.kind == inspect.Parameter.VAR_KEYWORD
      for param in sig.parameters.values()
  ):
    new_sig = sig
  else:
    new_sig = sig.replace(
        parameters=[
            *sig.parameters.values(),
            inspect.Parameter(
                "__penzai_variables", inspect.Parameter.KEYWORD_ONLY
            ),
        ]
    )

  def inner_fun(*args, **kwargs):
    frozen_variables = kwargs.pop("__penzai_variables")
    mut_vars = [var.unfreeze_as_copy() for var in frozen_variables]
    (rebound_args, rebound_kwargs) = bind_variables((args, kwargs), mut_vars)
    result = fun(*rebound_args, **rebound_kwargs)
    _, bad_vars = unbind_variables(result)
    if bad_vars:
      raise ValueError(
          "Returning a variable from a function transformed by pz.variable_jit"
          " is not allowed. To create new variables under jax.jit, you should"
          " instead return `pz.unbind_variables(..., frozen=True)`, then"
          " rebuild the new variables after with `pz.bind_variables(...,"
          f" unfreeze_as_copy=True)`.\nFound variables: {bad_vars}"
      )
    return result, [var.freeze() for var in mut_vars]

  inner_fun.__signature__ = new_sig

  if donate_variables:
    if "donate_argnames" in jit_kwargs:
      jit_kwargs["donate_argnames"] = (
          *jit_kwargs["donate_argnames"],
          "__penzai_variables",
      )
    else:
      jit_kwargs["donate_argnames"] = ("__penzai_variables",)

  if "out_shardings" in jit_kwargs:
    jit_kwargs["out_shardings"] = (jit_kwargs["out_shardings"], None)

  jitted_inner = jax.jit(inner_fun, **jit_kwargs)

  @functools.wraps(fun)
  def outer_fun(*args, **kwargs):
    (args_slots, kwargs_slots), variables = unbind_variables((args, kwargs))
    frozen_variables = [var.freeze() for var in variables]
    result, new_frozen_variables = jitted_inner(
        *args_slots, __penzai_variables=frozen_variables, **kwargs_slots
    )
    for var, new_frozen in zip(variables, new_frozen_variables):
      var.update(new_frozen)
    return result

  outer_fun.__doc__ = (
      f"Variable-aware JIT-compiled version of {fun.__name__}:\n{fun.__doc__}"
  )
  return outer_fun


#### Basic implementation of a labeled variable ####


class LabeledVariable(AbstractVariable, Generic[T]):
  """Base implementation of a variable with a label, value, and metadata.

  Conceptually, each variable is only valid inside a single JAX "trace",
  corresponding to a JAX transformation or control-flow primitive. Variables
  can be created and modified in a given trace level, and can be read inside
  inner traces (e.g. you can read a variable inside a JAX `cond`), but you
  should generall avoid assigning a value to a variable inside an inner trace,
  because the value may leak.

  This is currently unchecked but may change in the future.

  Attributes:
    label: The unique label for this variable.
    value: The mutable value stored in the variable. Should be a JAX pytree.
    metadata: A dictionary of metadata associated with this variable.
  """

  _raw_value: T
  label: VariableLabel
  metadata: dict[Any, Any]

  def __init__(
      self,
      *,
      label: VariableLabel,
      value: T,
      metadata: dict[Any, Any] | None = None,
  ):
    """Constructs a new variable.

    Args:
      label: The unique label for this variable.
      value: The initial value of the variable.
      metadata: A dictionary of metadata associated with this Variable.
    """
    self.label = label
    self.metadata = metadata or {}
    self.value = value

  @property
  def value(self) -> T:
    return self._raw_value

  @value.setter
  def value(self, new_val: T):
    # Make sure this is a pytree of arrays.
    self._raw_value = jax.tree_util.tree_map(jnp.asarray, new_val)

  def set_value(self, new_val: T):
    """Sets the value of the Variable."""
    self.value = new_val

  def update(self, new_frozen_value: LabeledVariableValue):
    """Updates the value of this variable to match a frozen variable."""
    if self.get_slot() != new_frozen_value.get_slot():
      raise ValueError(
          f"Cannot update variable {self} with incompatible value"
          f"{new_frozen_value}!"
      )
    self.value = new_frozen_value.value

  @abc.abstractmethod
  def treescope_color(self) -> str | tuple[str, str]:
    """Returns a color for this variable in Treescope."""
    raise NotImplementedError(
        "treescope_color must be overridden by subclasses."
    )

  def __repr__(self):
    # Defer to Treescope.
    import treescope  # pylint: disable=g-import-not-at-top

    with treescope.using_expansion_strategy(max_height=1):
      return treescope.render_to_text(self, ignore_exceptions=True)

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    from treescope import repr_lib  # pylint: disable=g-import-not-at-top

    return repr_lib.render_object_constructor(
        type(self),
        {
            "label": self.label,
            "value": self._raw_value,
            "metadata": self.metadata,
        },
        path,
        subtree_renderer,
        roundtrippable=False,
        color=self.treescope_color(),
    )


@struct.pytree_dataclass
class LabeledVariableValue(AbstractVariableValue, Generic[T]):
  """The value of a basic labeled variable, as a frozen JAX pytree.

  Attributes:
    label: The unique label for this variable.
    value: A snapshot of the value stored in the variable. Should be a JAX
      pytree.
    metadata: A dictionary of metadata associated with this variable.
  """

  label: VariableLabel = dataclasses.field(metadata={"pytree_node": False})
  value: T
  metadata: dict[Any, Any] = dataclasses.field(
      default_factory=dict, metadata={"pytree_node": False}
  )


#### Parameters ####


class Parameter(LabeledVariable[T]):
  """A model parameter variable.

  Parameters are variables that are usually updated by optimizers, but not
  updated inside the model itself.

  Attributes:
    label: The unique label for this parameter.
    value: The mutable value stored in the parameter. Should be a JAX pytree.
    metadata: A dictionary of metadata associated with this parameter.
  """

  def freeze(self) -> ParameterValue[T]:
    return ParameterValue(
        value=self.value, label=self.label, metadata=self.metadata
    )

  def get_slot(self) -> ParameterSlot:
    return ParameterSlot(label=self.label)

  def treescope_color(self):
    return "#93cce1"


@struct.pytree_dataclass(has_implicitly_inherited_fields=True)
class ParameterValue(LabeledVariableValue[T]):
  """The value of a Parameter, as a frozen JAX pytree.

  Attributes:
    label: The unique label for this Variable.
    value: A snapshot of the value stored in the Variable. Should be a JAX
      pytree.
    metadata: A dictionary of metadata associated with this Variable.
  """

  def unfreeze_as_copy(self) -> Parameter:
    return Parameter(value=self.value, label=self.label, metadata=self.metadata)

  def get_slot(self) -> ParameterSlot:
    return ParameterSlot(label=self.label)

  def treescope_color(self) -> str:
    return "#a0b0c9"


@struct.pytree_dataclass
class ParameterSlot(AbstractVariableSlot):
  """A slot for a parameter in a model.

  A ParameterSlot identifies a place in a model (or other pytree) where a Penzai
  parameter should be substituted.

  Attributes:
    label: A label that identifies this slot.
  """

  label: VariableLabel = dataclasses.field(metadata={"pytree_node": False})

  @property
  def value(self) -> Any:
    """Raises an error when accessed."""
    raise UnboundVariableError(
        f"Tried to access the .value property of a ParameterSlot: {self}."
    )

  def treescope_color(self):
    return "#93cce1", "#cdcdcd"


#### State variables ####


@dataclasses.dataclass(frozen=True)
class AutoStateVarLabel(auto_order_types.AutoOrderedAcrossTypes):
  """A label for a StateVariable that is unique based on its Python object ID."""

  var_id: int


@dataclasses.dataclass(frozen=True)
class ScopedStateVarLabel(auto_order_types.AutoOrderedAcrossTypes):
  """A label for a StateVariable that is unique within some scope."""

  group: Any
  index: int


_scoped_variable_counter = None
_scoped_variable_group = None


def _next_auto_variable_label(var_id: int) -> VariableLabel:
  """Generates a unique label for a Variable."""
  global _scoped_variable_counter
  if _scoped_variable_counter is None:
    return AutoStateVarLabel(var_id)
  else:
    assert isinstance(_scoped_variable_counter, int)
    ct = _scoped_variable_counter
    _scoped_variable_counter += 1
    return ScopedStateVarLabel(group=_scoped_variable_group, index=ct)


@contextlib.contextmanager
def scoped_auto_state_var_labels(group: Hashable = None):
  # pylint: disable=g-doc-return-or-yield
  """Context manager for using scoped auto-generated StateVariable labels.

  Within this context manager, any StateVariable that does not have an explicit
  label will be assigned a label that is unique within the scope of the context
  manager. This can be used to ensure that the labels are consistent, which can
  avoid recompilation of JAX jitted functions in some cases, since the label is
  considered part of the pytree structure when a Variable is unbound.

  Args:
    group: An optional group tag, which will be part of the automatic labels and
      can be used to disambiguate them.

  Returns:
    A context manager that sets up the scope.
  """
  # pylint: enable=g-doc-return-or-yield
  global _scoped_variable_group
  global _scoped_variable_counter
  orig_group_and_counter = _scoped_variable_group, _scoped_variable_counter
  _scoped_variable_group = group
  _scoped_variable_counter = 0
  try:
    yield
  finally:
    _scoped_variable_group, _scoped_variable_counter = orig_group_and_counter


class StateVariable(LabeledVariable[T]):
  """A mutable state variable.

  StateVariables are variables that can be updated inside the model (and are
  usually not updated by optimizers).

  Attributes:
    label: The unique label for this variable.
    value: The mutable value stored in the variable. Should be a JAX pytree.
    metadata: A dictionary of metadata associated with this parameter.
  """

  def __init__(
      self,
      value: T,
      *,
      label: VariableLabel | None = None,
      metadata: dict[Any, Any] | None = None,
  ):
    """Constructs a new state variable.

    Args:
      value: The initial value of the variable.
      label: The unique label for this variable. If not provided, a unique label
        will be generated automatically.
      metadata: A dictionary of metadata associated with this Variable.
    """
    if label is None:
      label = _next_auto_variable_label(id(self))
    super().__init__(label=label, value=value, metadata=metadata)

  def freeze(self) -> StateVariableValue[T]:
    return StateVariableValue(
        value=self.value, label=self.label, metadata=self.metadata
    )

  def get_slot(self) -> StateVariableSlot:
    return StateVariableSlot(label=self.label)

  def treescope_color(self):
    return "#f57603"


@struct.pytree_dataclass(has_implicitly_inherited_fields=True)
class StateVariableValue(LabeledVariableValue[T]):
  """The value of a StateVariable, as a frozen JAX pytree.

  Attributes:
    value: A snapshot of the value stored in the variable. Should be a JAX
      pytree.
    label: The unique label for this variable.
    metadata: A dictionary of metadata associated with this Variable.
  """

  def unfreeze_as_copy(self) -> StateVariable:
    return StateVariable(
        value=self.value, label=self.label, metadata=self.metadata
    )

  def get_slot(self) -> StateVariableSlot:
    return StateVariableSlot(label=self.label)

  def treescope_color(self):
    return "#d4af27"


@struct.pytree_dataclass
class StateVariableSlot(AbstractVariableSlot):
  """A slot for a parameter in a model.

  A StateVariableSlot identifies a place in a model (or other pytree) where a
  Penzai state variable should be substituted.

  Attributes:
    label: A label that identifies this slot.
  """

  label: VariableLabel = dataclasses.field(metadata={"pytree_node": False})

  @property
  def value(self) -> Any:
    """Raises an error when accessed."""
    raise UnboundVariableError(
        f"Tried to access the .value property of a StateVariableSlot: {self}."
    )

  def treescope_color(self):
    return "#f57603", "#cdcdcd"


#### Convenience functions ####


def _type_filtered_predicate(
    predicate: Callable[[AbstractVariable], bool] | None,
    var_type: type[Any],
) -> Callable[[AbstractVariable], bool]:
  if predicate is None:
    return lambda var: isinstance(var, var_type)
  else:
    return lambda var: isinstance(var, var_type) and predicate(var)


@typing.overload
def unbind_params(
    tree: Any,
    predicate: Callable[[Parameter], bool] | None = None,
    freeze: Literal[False] = False,
) -> tuple[Any, tuple[Parameter, ...]]:
  ...


@typing.overload
def unbind_params(
    tree: Any,
    predicate: Callable[[Parameter], bool] | None = None,
    *,
    freeze: Literal[True],
) -> tuple[Any, tuple[ParameterValue, ...]]:
  ...


def unbind_params(
    tree: Any,
    predicate: Callable[[Parameter], bool] | None = None,
    freeze: bool = False,
) -> tuple[Any, tuple[Parameter | ParameterValue, ...]]:
  r"""Version of `unbind_variables` that only extracts `Parameter`\ s."""
  return unbind_variables(  # type: ignore
      tree,
      predicate=_type_filtered_predicate(predicate, Parameter),
      freeze=freeze,
  )


def freeze_params(
    tree: Any,
    predicate: Callable[[Parameter], bool] | None = None,
) -> Any:
  r"""Version of `freeze_variables` that only freezes `Parameter`\ s."""
  return freeze_variables(  # type: ignore
      tree, predicate=_type_filtered_predicate(predicate, Parameter)
  )


@typing.overload
def unbind_state_vars(
    tree: Any,
    predicate: Callable[[StateVariable], bool] | None = None,
    freeze: Literal[False] = False,
) -> tuple[Any, tuple[StateVariable, ...]]:
  ...


@typing.overload
def unbind_state_vars(
    tree: Any,
    predicate: Callable[[StateVariable], bool] | None = None,
    *,
    freeze: Literal[True],
) -> tuple[Any, tuple[StateVariableValue, ...]]:
  ...


def unbind_state_vars(
    tree: Any,
    predicate: Callable[[StateVariable], bool] | None = None,
    freeze: bool = False,
) -> tuple[Any, tuple[StateVariable | StateVariableValue, ...]]:
  r"""Version of `unbind_variables` that only extracts `StateVariable`\ s."""
  return unbind_variables(  # type: ignore
      tree,
      predicate=_type_filtered_predicate(predicate, StateVariable),
      freeze=freeze,
  )


def freeze_state_vars(
    tree: Any,
    predicate: Callable[[StateVariable], bool] | None = None,
) -> Any:
  r"""Version of `freeze_variables` that only freezes `StateVariable`\ s."""
  return freeze_variables(  # type: ignore
      tree, predicate=_type_filtered_predicate(predicate, StateVariable)
  )
