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

"""Utilities for gradient checkpointing / rematerialization.

This module provides a wrapper that can be used to rematerialize gradients
through a layer, while correctly accounting for variable states inside the
layer. Rematerialization can be enabled by wrapping a layer in a `Remat` block
using something like::

  (
      pz.select(model)
      .at_instances_of(pz.nn.Attention)  # or another block
      .apply(gradient_checkpointing.Checkpointed)
  )
"""

from __future__ import annotations

from typing import Any, Callable

import jax
from penzai import pz


def _flat_stateless_call(
    body: pz.nn.Layer, state_var_values, argument, side_inputs
):
  return body.stateless_call(state_var_values, argument, **side_inputs)


@pz.pytree_dataclass
class Checkpointed(pz.nn.Layer):
  """Wraps a layer to run with gradient checkpointing.

  The Checkpointed wrapper has the same behavior as the layer it wraps, but
  modifies ``__call__`` so that gradient checkpointing is enabled.

  Since Checkpointed is an ordinary Layer, you can still inspect the contained
  layer and make modifications to it.

  Attributes:
    body: The layer that should run with gradient checkpointing.
    policy: Optional checkpointing policy. (See docs for `jax.checkpoint`
      for details.)
  """

  body: pz.nn.Layer
  policy: Callable[..., bool] | None = None

  def __call__(self, argument: Any, /, **side_inputs) -> Any:
    (unbound_body, unbound_side_inputs), state_vars = pz.unbind_state_vars(
        (pz.freeze_params(self.body), side_inputs)
    )

    checkpointed_call = jax.checkpoint(_flat_stateless_call, policy=self.policy)
    output, new_var_values = checkpointed_call(
        unbound_body,
        tuple(var.freeze() for var in state_vars),
        argument,
        unbound_side_inputs,
    )

    for var, new_value in zip(state_vars, new_var_values):
      var.update(new_value)
    return output
