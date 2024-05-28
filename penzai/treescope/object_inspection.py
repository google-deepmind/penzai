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

"""Utilities for inspecting objects."""


from __future__ import annotations
import types
from typing import Any, Callable


def safely_get_real_method(
    obj: Any, method_name: str
) -> Callable[..., Any] | None:
  """Safely retrieves a method directly implemented on a type.

  This function can be used to safely retrieve a "real" method from an object,
  e.g. a method that is actually defined on the object or a superclass. This
  can be used to avoid accidentally trying to call special methods on proxy
  objects or other objects that override `__getattr__`.

  Args:
    obj: The object to retrieve the method from.
    method_name: The name of the method to retrieve.

  Returns:
    The method if it is found, or None if it is not found.
  """
  if not hasattr(type(obj), method_name) or not hasattr(obj, method_name):
    return None

  try:
    retrieved = getattr(obj, method_name)
    if not isinstance(retrieved, types.MethodType):
      return None
    return retrieved
  except Exception:  # pylint: disable=broad-exception-caught
    return None
