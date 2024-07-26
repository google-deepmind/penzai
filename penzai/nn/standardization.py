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

"""Layer normalization (stateless) using named axes."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from penzai.core import named_axes
from penzai.core import struct
from penzai.nn import grouping
from penzai.nn import layer as layer_base
from penzai.nn import linear_and_affine

NamedArray = named_axes.NamedArray


@struct.pytree_dataclass
class Standardize(layer_base.Layer):
  """Standardization layer.

  Attributes:
    across: Axis names to standardize across.
    epsilon: Small constant to prevent division by zero.
  """

  across: str | tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  epsilon: float | jax.Array = 1e-6

  def __call__(self, value: NamedArray, **_unused_side_inputs) -> NamedArray:
    """Standardizes a named array across the given axes."""
    # Standardize over all of the (local) positional axes in case `across`
    # has more than one axis. All other axes will be vectorized over
    # automatically.
    across = (self.across,) if isinstance(self.across, str) else self.across
    return named_axes.nmap(jax.nn.standardize)(
        value.untag(*across), axis=None, epsilon=self.epsilon
    ).tag(*across)


@struct.pytree_dataclass
class LayerNorm(grouping.Sequential):
  """Layer normalization layer.

  Layer normalization layers consist of:

  * standardization over a feature axis or axes,

  * a learned parallel rescaling of each feature along those axes,

  * and a learned bias for those axes.

  For flexibility, ``LayerNorm`` is a subclass of `Sequential`.
  """

  sublayers: list[layer_base.Layer]

  @classmethod
  def from_config(
      cls,
      name: str,
      init_base_rng: jax.Array | None,
      across_axes: dict[str, int],
      epsilon: float | jax.Array = 1e-6,
      dtype: jax.typing.DTypeLike = jnp.float32,
  ) -> LayerNorm:
    """Constructs a layer normalization layer.

    Args:
      name: The name of the layer.
      init_base_rng: The base RNG to use for initializing model parameters.
      across_axes: Names and lengths of the axes to normalize over.
      epsilon: Epsilon parameter for the standardization step.
      dtype: Dtype of the scale and shift parameters.

    Returns:
      A newly-constructed ``LayerNorm`` layer.
    """
    return cls([
        Standardize(across=tuple(across_axes.keys()), epsilon=epsilon),
        linear_and_affine.Linear.from_config(
            name=f"{name}/scale",
            init_base_rng=init_base_rng,
            input_axes={},
            output_axes={},
            parallel_axes=across_axes,
            initializer=linear_and_affine.constant_initializer(1.0),
            dtype=dtype,
        ),
        linear_and_affine.AddBias.from_config(
            name=f"{name}/shift",
            init_base_rng=init_base_rng,
            biased_axes=across_axes,
            dtype=dtype,
        ),
    ])


@struct.pytree_dataclass
class RMSStandardize(layer_base.Layer):
  """Root-mean-squared standardization layer.

  As proposed by Zhang & Sennrich (2019): https://arxiv.org/abs/1910.07467.
  This layer does not include the learnable parameter.

  Attributes:
    across: Axis names to standardize across.
    epsilon: Small constant to prevent division by zero.
  """

  across: str | tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  epsilon: float | jax.Array = 1e-6

  def __call__(self, value: NamedArray, **_unused_side_inputs) -> NamedArray:
    """Root-mean-square standardizes a named array across the given axes."""
    across = (self.across,) if isinstance(self.across, str) else self.across

    @named_axes.nmap
    def _rms_standardize(x):
      var = jnp.mean(jnp.square(x))
      return x * jnp.reciprocal(jnp.sqrt(var + self.epsilon))

    return _rms_standardize(value.untag(*across)).tag(*across)


@struct.pytree_dataclass
class RMSLayerNorm(grouping.Sequential):
  """Root-mean-squared layer normalization layer.

  RMS layer normalization layers consist of:

  * root-mean-squared standardization over a feature axis or axes,

  * with a learned parallel rescaling of each feature along those axes.

  As proposed by Zhang & Sennrich (2019): https://arxiv.org/abs/1910.07467.

  For flexibility, ``RMSLayerNorm`` is a subclass of `Sequential`.
  """

  sublayers: list[layer_base.Layer]

  @classmethod
  def from_config(
      cls,
      name: str,
      init_base_rng: jax.Array | None,
      across_axes: dict[str, int],
      epsilon: float | jax.Array = 1e-6,
      dtype: jax.typing.DTypeLike = jnp.float32,
  ) -> RMSLayerNorm:
    """Constructs a RMS layer normalization layer.

    Args:
      name: The name of the layer.
      init_base_rng: The base RNG to use for initializing model parameters.
      across_axes: Names and lengths of the axes to normalize over.
      epsilon: Epsilon parameter for the standardization step.
      dtype: Dtype of the scale and shift parameters.

    Returns:
      A newly-constructed ``RMSLayerNorm`` layer.
    """
    return cls([
        RMSStandardize(
            across=tuple(across_axes.keys()),
            epsilon=jnp.asarray(epsilon, dtype=dtype),
        ),
        linear_and_affine.Linear.from_config(
            name=f"{name}/scale",
            init_base_rng=init_base_rng,
            input_axes={},
            output_axes={},
            parallel_axes=across_axes,
            initializer=linear_and_affine.constant_initializer(1.0),
            dtype=dtype,
        ),
    ])
