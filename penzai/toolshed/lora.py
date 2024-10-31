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

"""Low-rank adaptation (LoRA, Hu et al. 2021).

Low-rank adaptation is a parameter-efficient fine-tuning strategy for large
pretrained models. It works by decomposing each original linear operation into
a pretrained full-rank matrix and two low-rank learnable updates.

This implementation is based on the tutorial
:doc:`"LoRA From Scratch" </notebooks/lora_from_scratch>`.
Due to the design conventions of Penzai neural networks,
it is straightforward to substitute LoRA blocks into any model that uses the
`pz.nn.Linear` primitive layer.

See https://arxiv.org/abs/2106.09685 for details on LoRA.
"""

from __future__ import annotations

from typing import Any

import jax
from penzai import pz


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)  # pytype: disable=wrong-keyword-args  # pylint: disable=line-too-long
class LowRankAdapter(pz.nn.Sequential):
  """A LoRA parameter-efficient adaptation block, replacing a Linear layer."""

  @classmethod
  def from_linear(
      cls,
      linear: pz.nn.Linear,
      name: str,
      init_base_rng: jax.Array | None,
      rank: int,
      lowrank_axis: str = "lowrank",
  ) -> LowRankAdapter:
    """Builds a LoRA layer from a `pz.nn.Linear` layer.

    Args:
      linear: The linear layer to adapt.
      name: Name for this layer's parameters. Must be globally unique across all
        LoRA blocks; we recommend using `jax.tree_util.keystr` or
        `pz.pretty_keystr` and setting the name based on the path to the
        original Linear layer being replaced.
      init_base_rng: The base RNG to use for initializing model parameters.
      rank: The rank of the low-rank adapter.
      lowrank_axis: The axis name for low-rank adaptation.

    Returns:
      A LoRA block with uninitialized parameters and the same initial
      behavior as ``linear``.
    """
    return cls([
        pz.nn.BranchAndAddTogether([
            pz.nn.NamedGroup("Pretrained", [linear]),
            pz.nn.NamedGroup(
                "Update",
                [
                    pz.nn.Linear.from_config(
                        name=f"{name}/LoRA_A",
                        init_base_rng=init_base_rng,
                        input_axes=linear.input_axes,
                        output_axes={lowrank_axis: rank},
                        parallel_axes=linear.parallel_axes,
                    ),
                    pz.nn.Linear.from_config(
                        name=f"{name}/LoRA_B",
                        init_base_rng=init_base_rng,
                        input_axes={lowrank_axis: rank},
                        output_axes=linear.output_axes,
                        parallel_axes=linear.parallel_axes,
                        initializer=pz.nn.zero_initializer,
                    ),
                ],
            ),
        ])
    ])


def loraify_linears_in_selection(
    selection: pz.Selection[Any],
    rank: int,
    init_base_rng: jax.Array | None,
) -> Any:
  """Replaces Linear layers inside a selected part of a model with LoRA blocks.

  This function should usually be called after freezing the existing weights
  in the model using something like ::

    pz.nn.at_instances_of(pz.nn.Parameter).apply(
        lambda param: pz.nn.FrozenParameter(param.value, param.name)
    )

  This function returns a copy of the model with new LoRA parameters added, but
  does not modify any existing parameters.

  Args:
    selection: A selection of a model that identifies the parts for which LoRA
      adaptation should be applied. Any `Linear` layers contained within the
      selected part will be replaced.
    rank: The rank of the LoRA blocks to insert.
    init_base_rng: The base RNG to use for initializing the LoRA parameters.

  Returns:
    A copy of the original full model (e.g. of ``selection.deselect()``), but
    where each of the `Linear` layers inside the selected part are replaced with
    new `LowRankAdapter` instances.
  """
  model = selection.deselect()
  return selection.at_instances_of(pz.nn.Linear).apply(
      lambda keypath, lin: LowRankAdapter.from_linear(
          lin,
          name="LoRA:" + pz.pretty_keystr(keypath, model),
          init_base_rng=init_base_rng,
          rank=rank,
      ),
      with_keypath=True,
  )
