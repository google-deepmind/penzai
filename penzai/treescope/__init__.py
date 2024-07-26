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

"""Compatibility stub for treescope.

The treescope pretty-printer has moved into a separate module `treescope`.
This module contains compatibility stubs for the previously-stable parts of the
extension API, which are still available in the `penzai.treescope` namespace.

New code should import from `treescope` directly. This module and its submodules
will be deprecated and eventually removed.
"""

from . import formatting_util
from . import repr_lib
