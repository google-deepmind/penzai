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

"""Dataflow combinators for neural networks.

This module defines a number of combinators for expressing dataflow in common
neural networks. These primitives can be composed to construct more complex
behavior.
"""

from __future__ import annotations

from typing import Any

from penzai.core import struct
from penzai.nn import layer as layer_base


@struct.pytree_dataclass
class Residual(layer_base.Layer):
  """A residual block, which runs its sublayers then adds the input.

  ``Residual`` blocks add additional data-flow paths called "skip connections",
  wherein the input to the residual block is saved, and the output of the
  sublayers is treated as a "residual" to add back to the input. When many
  residual blocks are run in order, this produces a "residual stream", with
  each block reading from the stream and then making an additive write to it.

  Residual blocks have non-linear data flow, but in a fairly straightforward
  way. This pattern can be factored out into a block so that it can be expressed
  consistently in more complex models.

  Attributes:
    delta: A block to run and add its output to its input.
  """

  delta: layer_base.Layer

  def __call__(self, value: Any, **side_inputs: dict[Any, Any]) -> Any:
    """Runs each of the sublayers in sequence, then adds back the original input.

    Args:
      value: The input to the block.
      **side_inputs: Side inputs for the block.

    Returns:
      The sum of the input to the residual block and the output of the child.
    """
    delta_value = self.delta(value, **side_inputs)
    return delta_value + value


@struct.pytree_dataclass
class BranchAndAddTogether(layer_base.Layer):
  """A data-flow branch with additive interactions between branches.

  ``BranchAndAddTogether`` can be used to compose multiple operations that all
  produce values of the same shape (or broadcast-compatible shapes). Each
  branch receives the same input, and the outputs of all branches are added
  together.

  Attributes:
    branches: A list of layers that will be run on the same input, and have
      their outputs added together.
  """

  branches: list[layer_base.Layer]

  def __call__(self, arg, **side_inputs):
    if not self.branches:
      raise ValueError('BranchAndAddTogether requires at least one branch.')

    running_product = self.branches[0](arg, **side_inputs)
    for branch in self.branches[1:]:
      running_product += branch(arg, **side_inputs)

    return running_product


@struct.pytree_dataclass
class BranchAndMultiplyTogether(layer_base.Layer):
  """A data-flow branch with multiplicative interactions between branches.

  ``BranchAndMultiplyTogether`` can be used to compose multiple operations that
  all produce values of the same shape (or broadcast-compatible shapes). Each
  branch receives the same input, and the outputs of all branches are multiplied
  together.

  This is a common pattern in "gated" neural networks, where one branch
  computes a gate using a nonlinear activation function, and the other branch
  computes a value that is multiplied by the gate.

  Attributes:
    branches: A list of layers that will be run on the same input, and have
      their outputs multiplied together.
  """

  branches: list[layer_base.Layer]

  def __call__(self, arg, **side_inputs):
    if not self.branches:
      raise ValueError(
          'BranchAndMultiplyTogether requires at least one branch.'
      )

    running_product = self.branches[0](arg, **side_inputs)
    for branch in self.branches[1:]:
      running_product *= branch(arg, **side_inputs)

    return running_product
