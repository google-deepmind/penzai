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

"""Syntactic sugar utilities for common operations."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class SliceLike:
  """Builds a slice when sliced (e.g. ``pz.slice[1:3] == slice(1, 3, None)``).

  An instance of this class is exposed as `pz.slice`, so that you can easily
  slice named array axes using syntax like ``arr[{"name": pz.slice[1:3]}]``
  instead of ``arr[{"name": slice(1, 3, None)}]``.
  """

  def __getitem__(self, indexer):
    return indexer


slice = SliceLike()  # pylint: disable=redefined-builtin
