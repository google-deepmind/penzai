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

"""Module of aliases for common penzai classes and functions."""

# pylint: disable=g-multiple-import,g-importing-member,unused-import

from penzai.core.auto_order_types import (
    AutoOrderedAcrossTypes,
)
import penzai.core.named_axes as nx
from penzai.core.partitioning import (
    NotInThisPartition,
    combine,
)
from penzai.core.random_stream import (
    RandomStream,
)
from penzai.core.selectors import (
    Selection,
    select,
)
import penzai.core.shapecheck as chk
from penzai.core.struct import (
    Struct,
    StructStaticMetadata,
    PyTreeDataclassSafetyError,
    is_pytree_dataclass_type,
    is_pytree_node_field,
    pytree_dataclass,
)
# pylint: disable=redefined-builtin
from penzai.core.syntactic_sugar import (
    slice,
)
# pylint: enable=redefined-builtin
from penzai.core.tree_util import (
    pretty_keystr,
)
from penzai.core.variables import (
    VariableConflictError,
    UnboundVariableError,
    VariableLabel,
    AbstractVariable,
    AbstractVariableValue,
    AbstractVariableSlot,
    unbind_variables,
    bind_variables,
    freeze_variables,
    variable_jit,
    Parameter,
    ParameterValue,
    ParameterSlot,
    AutoStateVarLabel,
    ScopedStateVarLabel,
    scoped_auto_state_var_labels,
    StateVariable,
    StateVariableValue,
    StateVariableSlot,
    unbind_params,
    freeze_params,
    unbind_state_vars,
    freeze_state_vars,
)
from penzai.treescope._compatibility_setup import (
    show,
    disable_interactive_context,
    enable_interactive_context,
)
from treescope.context import (
    ContextualValue,
)
from treescope.dataclass_util import (
    dataclass_from_attributes,
    init_takes_fields,
)
from treescope.formatting_util import (
    oklch_color,
    color_from_string,
)

from . import nn
from . import ts  # pylint: disable=g-bad-import-order
