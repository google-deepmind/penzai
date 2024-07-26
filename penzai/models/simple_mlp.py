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

"""A simple multi-layer perceptron."""

from __future__ import annotations

from typing import Callable

import jax
from penzai import pz


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class MLP(pz.nn.Sequential):
  """Sequence of Affine layers."""

  @classmethod
  def from_config(
      cls,
      init_base_rng: jax.Array | None,
      feature_sizes: list[int],
      name: str = "mlp",
      activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      feature_axis: str = "features",
  ) -> MLP:
    assert len(feature_sizes) >= 2
    children = []
    for i, (feats_in, feats_out) in enumerate(
        zip(feature_sizes[:-1], feature_sizes[1:])
    ):
      if i:
        children.append(pz.nn.Elementwise(activation_fn))
      children.append(
          pz.nn.Affine.from_config(
              name=f"{name}/Affine_{i}",
              init_base_rng=init_base_rng,
              input_axes={feature_axis: feats_in},
              output_axes={feature_axis: feats_out},
          )
      )
    return cls(sublayers=children)


@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class DropoutMLP(pz.nn.Sequential):
  """Sequence of Affine layers with dropout."""

  @classmethod
  def from_config(
      cls,
      init_base_rng: jax.Array | None,
      feature_sizes: list[int],
      drop_rate: float,
      name: str = "dropout_mlp",
      activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      feature_axis: str = "features",
  ) -> DropoutMLP:
    assert len(feature_sizes) >= 2
    children = []
    for i, (feats_in, feats_out) in enumerate(
        zip(feature_sizes[:-1], feature_sizes[1:])
    ):
      if i:
        children.extend([
            pz.nn.StochasticDropout(drop_rate),
            pz.nn.Elementwise(activation_fn),
        ])
      children.append(
          pz.nn.Affine.from_config(
              name=f"{name}/Affine_{i}",
              init_base_rng=init_base_rng,
              input_axes={feature_axis: feats_in},
              output_axes={feature_axis: feats_out},
          )
      )
    return cls(sublayers=children)
