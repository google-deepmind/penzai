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

"""Compatibility stub: Stable high-level interface for building object reprs.

New code should instead use `treescope.repr_lib`. This module will be
deprecated and eventually removed.
"""

# pylint: disable=g-importing-member,g-multiple-import,unused-import

from treescope.repr_lib import (
    render_object_constructor,
    render_dictionary_wrapper,
    render_enumlike_item,
)
