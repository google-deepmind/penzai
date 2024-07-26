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

"""Module of aliases for penzai neural networks."""

# pylint: disable=g-multiple-import,g-importing-member,unused-import

from penzai.deprecated.v1.nn.attention import (
    ApplyAttentionMask,
    Attention,
    KVCachingAttention,
)
from penzai.deprecated.v1.nn.basic_ops import (
    CastToDType,
    Elementwise,
    Softmax,
)
from penzai.deprecated.v1.nn.combinators import (
    Residual,
    BranchAndAddTogether,
    BranchAndMultiplyTogether,
)
from penzai.deprecated.v1.nn.dropout import (
    DisabledDropout,
    maybe_dropout,
    StochasticDropout,
)
from penzai.deprecated.v1.nn.embeddings import (
    EmbeddingTable,
    EmbeddingLookup,
    EmbeddingDecode,
    ApplyRoPE,
)
from penzai.deprecated.v1.nn.grouping import (
    CheckedSequential,
    CheckStructure,
    Identity,
    inline_anonymous_sequentials,
    inline_groups,
    is_anonymous_sequential,
    is_sequential_or_named,
    NamedGroup,
    Sequential,
)
from penzai.deprecated.v1.nn.linear_and_affine import (
    AddBias,
    BiasInitializer,
    ConstantRescale,
    NamedEinsum,
    Affine,
    Linear,
    LinearOperatorWeightInitializer,
    LinearInPlace,
    RenameAxes,
    contract,
    variance_scaling_initializer,
    xavier_normal_initializer,
    xavier_uniform_initializer,
    constant_initializer,
    zero_initializer,
)
from penzai.deprecated.v1.nn.parameters import (
    add_parameter_prefix,
    FrozenParameter,
    initialize_parameters,
    SharedParamTag,
    SharedParameterLookup,
    mark_shareable,
    attach_shared_parameters,
    Parameter,
    ParameterLike,
    ShareableUninitializedParameter,
    SupportsParameterRenaming,
    UninitializedParameter,
    UninitializedParameterError,
    check_no_duplicated_parameters,
)
from penzai.deprecated.v1.nn.standardization import (
    LayerNorm,
    Standardize,
    RMSLayerNorm,
    RMSStandardize,
)
