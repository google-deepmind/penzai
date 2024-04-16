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

"""Treescope handlers for known objects."""

from . import builtin_atom_handler
from . import builtin_structure_handler
from . import canonical_alias_postprocessor
from . import function_reflection_handlers
from . import generic_pytree_handler
from . import generic_repr_handler
from . import hardcoded_structure_handlers
from . import ndarray_handler
from . import penzai
from . import repr_html_postprocessor
from . import shared_value_postprocessor
