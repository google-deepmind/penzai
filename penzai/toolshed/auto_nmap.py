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

"""Utility to automatically apply `nmap` to functions in a module.

`pz.nx.nmap` can be used to "lift" any positional JAX function into a named-axis
variant, with simple semantics: operations always vectorize over the named axes,
and the function only sees the positional ones. However, it can be annoying to
wrap each JAX function with `nmap` manually.

This module provides syntactic sugar for automatically wrapping all callables
in a module with `nmap`, by exposing a wrapper object that stands in for the
original module. Accessing an attribute on the wrapper accesses that same
attribute on the underlying module; then, the result is either wrapped in `nmap`
or recursively wrapped using the module wrapper, depending on the type of the
retrieved value.

The result is that, if you define ::

  njax = auto_nmap.wrap_module(jax)
  njnp = auto_nmap.wrap_module(jnp)

then you can write code like ::

  njax.lax.top_k(my_array, k=10)
  njnp.linalg.eigh(my_array)

instead of ::

  pz.nx.wrap(jax.lax.top_k)(my_array, k=10)
  pz.nx.wrap(jnp.linalg.eigh)(my_array)

You can also directly use ordinary array constructors to construct NamedArrays,
e.g. ::

  njax.array([1, 2, 3]).tag("foo")
  njnp.linspace(-0.5, 0.5, 10).tag("bar")
  njax.random.uniform(key, (10, 10)).tag("baz", "qux)

"""

from __future__ import annotations

import dataclasses
import types

from penzai.core import named_axes


@dataclasses.dataclass
class AutoNmapModuleWrapper:
  """Wrapper for a module that automatically applies `nmap` to callables.

  Attributes:
    _module: The module to wrap.
  """

  _module: types.ModuleType

  def __getattr__(self, name: str):
    """Retrieves and wraps an attribute of the underlying module.

    Args:
      name: The attribute name to retrieve.

    Returns:
      Either an AutoNmapModuleWrapper or a `nmap`-ped callable, depending on the
      type of object retrieved from the original module.
    """
    original = getattr(self._module, name)
    if isinstance(original, types.ModuleType):
      return AutoNmapModuleWrapper(original)
    elif callable(original):
      return named_axes.nmap(original)
    else:
      raise AttributeError(
          f"Cannot wrap attribute {name} of {self._module}: it has unrecognized"
          f" type {type(original)} (expected a callable or submodule)"
      )


def wrap_module(module: types.ModuleType) -> AutoNmapModuleWrapper:
  """Wraps a module to automatically apply `named_axes.nmap` to callables.

  The returned object will have similar attributes to `module`, including all
  submodules and all callables. Accessing these on the returned object will
  return an `nmap`-ped version of the original module callable, or a wrapped
  submodule.

  Args:
    module: The module to wrap.

  Returns:
    A wrapper object similar to `module`, such that callable attributes are
    wrapped with `named_axes.nmap`, and submodules are recursively wrapped with
    `wrap_module`.
  """
  return AutoNmapModuleWrapper(module)
