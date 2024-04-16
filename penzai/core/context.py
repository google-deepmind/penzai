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

"""Helpers for scoped contextual value providers."""
from __future__ import annotations

import contextlib
import dataclasses
import typing
from typing import Any, Generic, TypeVar

T = TypeVar("T")


_INTERACTIVE_CONTEXT_STACK: contextlib.ExitStack | None = None


def enable_interactive_context() -> None:
  """Enables the global interactive context stack.

  By default, ContextualValues can only be changed in a scoped ``with`` block.
  This function turns on interactive mode, enabling those contexts to be set
  persistently in an interactive setting. To turn it back off, see
  `disable_interactive_context`.

  Interactive mode is itself a thin wrapper around a contextlib ``ExitStack``.
  """
  global _INTERACTIVE_CONTEXT_STACK
  if _INTERACTIVE_CONTEXT_STACK is None:
    _INTERACTIVE_CONTEXT_STACK = contextlib.ExitStack()


def disable_interactive_context() -> None:
  """Clears the global interactive context stack and disables interactive mode.

  This function closes all context managers entered in an interactive context,
  restoring all contextual values to their original states.
  """
  global _INTERACTIVE_CONTEXT_STACK
  if _INTERACTIVE_CONTEXT_STACK is not None:
    _INTERACTIVE_CONTEXT_STACK.close()
    _INTERACTIVE_CONTEXT_STACK = None


@dataclasses.dataclass(init=False)
class ContextualValue(Generic[T]):
  """A global value which can be modified in a scoped context.

  Mutable global values can be difficult to reason about, but can also be very
  convenient for reducing boilerplate and focusing on the important parts if
  they are used carefully. Moreover, it's common to only want to change a global
  value within a particular delimited scope, without affecting anything beyond
  that scope.

  This class manages a restricted form of global value access to support this
  use case: read access is allowed anywhere, but updates are only "local", in
  the sense that they apply for the duration of a delimited scope and don't
  affect anything beyond that scope. (This is similar to a Reader monad in
  functional programming languages.)

  If you have many values to set, or you want to set them conditionally,
  consider using a `contextlib.ExitStack`, e.g.

  ::

      with contextlib.ExitStack() as stack:
        stack.enter_context(contextual_one.set_scoped(True))
        if foo:
          stack.enter_context(contextual_two.set_scoped(some_value))
        # logic that uses those values
      # all contexts are exited here

  or just

  ::

      stack = contextlib.ExitStack()
      stack.enter_context(contextual_one.set_scoped(True))
      if foo:
        stack.enter_context(contextual_two.set_scoped(some_value))
      # ... do something ...
      # ... then eventually:
      stack.close()

  If you would like to modify contextual values at the global level (e.g. for
  interactive use), you can call `enable_interactive_context` in this
  module and then use `set_interactive` instead of `set_scoped`;
  this is actually implemented using an internally-managed global ``ExitStack``.

  Attributes:
    __module__: The module where this contextual value is defined.
    __qualname__: The fully-qualified name of the contextual value within its
      module.
    _raw_value: The current value. Should not be used directly; use `get` to get
      it or `set_scoped` to set it within a context.
  """

  __module__: str | None
  __qualname__: str | None
  _raw_value: T

  def __init__(
      self,
      initial_value: T,
      module: str | None = None,
      qualname: str | None = None,
  ):
    """Creates a contextual value.

    Args:
      initial_value: A value to use outside of `set_scoped` contexts.
      module: The module where this contextual value is defined. Usually this
        will be `__name__` (for whichever model it was defined in).
      qualname: The fully-qualified name of the contextual value within its
        module.
    """
    self.__module__ = module
    self.__qualname__ = qualname
    self._raw_value = initial_value

  def get(self) -> T:
    """Retrieves the current value."""
    return self._raw_value

  @contextlib.contextmanager
  def set_scoped(self: ContextualValue[T], new_value: T):
    # pylint: disable=g-doc-return-or-yield
    """Returns a context manager in which the value will be modified.

    This allows scoped modifications to the value, using something like

    ::

      contextual = ContextualValue(10)

      with contextual.set_scoped(3):
        v1 = contextual.get()  # v1 is 3

      v2 = contextual.get()  # v2 is 10

    If you want to conditionally set the value, consider using
    `contextlib.ExitStack`:

    ::

      with contextlib.ExitStack() as exit_stack:
        if condition:
          exit_stack.enter_context(contextual.set_scoped(3))

        # use contextual.get() here

    Args:
      new_value: The new value to use in this context.

    Returns:
      A context manager in which ``new_value`` is set.
    """
    # pylint: enable=g-doc-return-or-yield
    old_value = self._raw_value
    self._raw_value = new_value
    try:
      yield
    finally:
      assert self._raw_value is new_value
      self._raw_value = old_value

  def set_interactive(self: ContextualValue[T], new_value: T) -> None:
    """Sets the value in interactive mode.

    This function should ONLY be called in an interactive setting (e.g. a Colab
    notebook or repl), and must be called after
    `enable_interactive_context` but before
    `disable_interactive_context`.

    `disable_interactive_context` will restore all contexts set using
    `set_interactive` to their original states.

    Args:
      new_value: The new value to use in this context.

    Raises:
      RuntimeError: If interactive contexts have not been enabled.
    """
    if _INTERACTIVE_CONTEXT_STACK is None:
      raise RuntimeError(
          "`set_interactive` should only be used in an interactive setting. To"
          " turn on interactive mode, call"
          " `penzai.enable_interactive_context()`."
      )

    # Pytype gets confused about the types here, so we just tell it not to try.
    new_context = typing.cast(Any, self).set_scoped(new_value)
    _INTERACTIVE_CONTEXT_STACK.enter_context(new_context)
