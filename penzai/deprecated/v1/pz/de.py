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

"""Common aliases for data effects."""

# pylint: disable=g-multiple-import,g-importing-member,unused-import

from penzai.deprecated.v1.data_effects.effect_base import (
    all_handler_ids,
    EffectHandler,
    EffectRequest,
    EffectRuntimeImpl,
    free_effect_types,
    get_effect_color,
    HandledEffectRef,
    HandlerId,
    infer_or_check_handler_id,
    register_effect_color,
    UnhandledEffectError,
)
from penzai.deprecated.v1.data_effects.local_state import (
    LocalStateEffect,
    InitialLocalStateRequest,
    FrozenLocalStateRequest,
    SharedLocalStateRequest,
    HandledLocalStateRef,
    WithFunctionalLocalState,
    handle_local_states,
    freeze_local_states,
    hoist_shared_state_requests,
    embed_shared_state_requests,
)
from penzai.deprecated.v1.data_effects.random import (
    RandomEffect,
    RandomRequest,
    TaggedRandomRequest,
    HandledRandomRef,
    WithRandomKeyFromArg,
    WithStatefulRandomKey,
    WithFrozenRandomState,
)
from penzai.deprecated.v1.data_effects.side_input import (
    SideInputEffect,
    SideInputRequest,
    HandledSideInputRef,
    WithSideInputsFromInputTuple,
    WithConstantSideInputs,
    HoistedTag,
    hoist_constant_side_inputs,
)
from penzai.deprecated.v1.data_effects.side_output import (
    CollectingSideOutputs,
    HandledSideOutputRef,
    SideOutputEffect,
    SideOutputRequest,
    SideOutputValue,
    TellIntermediate,
)
