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
"""Helper class to define automatic arbitrary orderings across different types.

Dictionaries in JAX need to have sortable keys in order to be manipulated with
`jax.tree_util` or passed through JAX transformations. This makes it difficult
to have a dictionary whose keys are different types. However, such patterns
can be useful for storing multiple types of data in a single dictionary. In
particular, Penzai requires slots and variables to have unique labels, and
frequently uses these labels as keys in dictionaries. These labels can be made
unique by ensuring that different categories of labels are of different types,
but this requires those types to be mutually comparable so that they can pass
through JAX transformations.

This module defines a class `AutoOrderedAcrossTypes` that defines `__lt__`,
`__gt__`, `__le__`, and `__ge__` methods for dataclasses so that:

  * comparisons within a single type are done using the ordinary dataclass rules
    (i.e. ordering like a tuple of their values),
  * comparisons between two different subclasses of `AutoOrderedAcrossTypes` are
    ordered arbitrarily, such that for two types A and B, either all instances
    of A are less than all instances of B or vice versa,
  * any subclass of `AutoOrderedAcrossTypes` is always greater than any string
    or ordinary tuple.

"""
from __future__ import annotations

import dataclasses


class AutoOrderedAcrossTypes:
  """Mixin to define an arbitrary total ordering across dataclass subclasses.

  Subclasses of this class will be comparable with other subclasses of this
  class, using an arbitrary total ordering of their types. Additionally, any
  subclass of this class will be greater than any string or tuple. This makes it
  possible to use subclasses of this class as keys in dictionaries that are
  passed through JAX transformations, potentially in combination with string or
  tuple keys.

  Every subclass of this class *must* be a dataclass with
  ``eq=True, order=False`` (the default), otherwise the provided comparisons
  may not work correctly. (Setting ``eq=False`` may cause inconsistencies
  between equality and ordering comparisons, and setting ``order=True`` will
  override the orderings defined here.)

  An example of how to subclass this class for the common case of making a new
  dictionary key type: ::

    @dataclasses.dataclass(frozen=True)  # WITHOUT order=True
    class MyType(auto_order_types.AutoOrderedAcrossTypes):
      some_field: int
      some_other_field: str

  Here ``MyType`` will have the same ordering behavior as it would have if
  ``order=True`` was set, but it will also be comparable with other subclasses
  of `AutoOrderedAcrossTypes`.
  """

  def _ordering_tuple(self):
    """Builds a tuple that can be used to compare this object to other types."""
    assert dataclasses.is_dataclass(self)
    # Arbitrarily order types by their name, module, and then the id of their
    # type if necessary.
    order_parts = [
        type(self).__qualname__,
        type(self).__module__,
        id(type(self)),
    ]
    # Within a given type, order by the values of the fields, like for an
    # ordinary ordered dataclass.
    for field in dataclasses.fields(self):
      if field.compare:
        order_parts.append(getattr(self, field.name))
    return tuple(order_parts)

  def __lt__(self, other):
    if isinstance(other, str | tuple):
      return False
    elif isinstance(other, AutoOrderedAcrossTypes):
      return self._ordering_tuple() < other._ordering_tuple()
    else:
      return NotImplemented

  def __gt__(self, other):
    if isinstance(other, str | tuple):
      return True
    elif isinstance(other, AutoOrderedAcrossTypes):
      return self._ordering_tuple() > other._ordering_tuple()
    else:
      return NotImplemented

  def __le__(self, other):
    if isinstance(other, str | tuple):
      return False
    elif isinstance(other, AutoOrderedAcrossTypes):
      return self._ordering_tuple() <= other._ordering_tuple()
    else:
      return NotImplemented

  def __ge__(self, other):
    if isinstance(other, str | tuple):
      return True
    elif isinstance(other, AutoOrderedAcrossTypes):
      return self._ordering_tuple() >= other._ordering_tuple()
    else:
      return NotImplemented
