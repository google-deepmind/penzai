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
from typing import Any

import jax
import jax.numpy as jnp
from penzai.core import named_axes
from penzai.core import shapecheck
from penzai.core import struct
from penzai.deprecated.v1.core import layer as layer_base
from penzai.deprecated.v1.nn import grouping
from penzai.deprecated.v1.nn import linear_and_affine
from penzai.deprecated.v1.nn import parameters


LayerLike = layer_base.LayerLike
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

  @layer_base.checked_layer_call
  def __call__(self, value: NamedArray) -> NamedArray:
    """Standardizes a named array across the given axes."""
    # Standardize over all of the (local) positional axes in case `across`
    # has more than one axis. All other axes will be vectorized over
    # automatically.
    across = (self.across,) if isinstance(self.across, str) else self.across
    return named_axes.nmap(jax.nn.standardize)(
        value.untag(*across), axis=None, epsilon=self.epsilon
    ).tag(*across)

  def input_structure(self) -> Any:
    if isinstance(self.across, str):
      across = (self.across,)
    else:
      across = self.across
    return shapecheck.ArraySpec.floating_named({
        **shapecheck.var("B"),
        **shapecheck.vars_for_axes("across", across),
    })

  def output_structure(self) -> Any:
    return self.input_structure()


@struct.pytree_dataclass
class LayerNorm(grouping.Sequential):
  """Layer normalization layer.

  Layer normalization layers consist of:

  * standardization over a feature axis or axes,

  * a learned parallel rescaling of each feature along those axes,

  * and a learned bias for those axes.

  For flexibility, ``LayerNorm`` is a subclass of `Sequential`.
  """

  sublayers: list[layer_base.LayerLike]

  @classmethod
  def from_config(
      cls,
      across_axes: dict[str, int],
      epsilon: float | jax.Array = 1e-6,
      dtype: jax.typing.DTypeLike = jnp.float32,
  ) -> LayerNorm:
    """Constructs a layer normalization layer.

    Args:
      across_axes: Names and lengths of the axes to normalize over.
      epsilon: Epsilon parameter for the standardization step.
      dtype: Dtype of the scale and shift parameters.

    Returns:
      A newly-constructed ``LayerNorm`` layer.
    """
    return cls([
        Standardize(across=tuple(across_axes.keys()), epsilon=epsilon),
        parameters.add_parameter_prefix(
            "scale",
            linear_and_affine.Linear.from_config(
                input_axes={},
                output_axes={},
                parallel_axes=across_axes,
                initializer=linear_and_affine.constant_initializer(1.0),
                dtype=dtype,
            ),
        ),
        parameters.add_parameter_prefix(
            "shift",
            linear_and_affine.AddBias.from_config(across_axes, dtype=dtype),
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

  @layer_base.checked_layer_call
  def __call__(self, value: NamedArray) -> NamedArray:
    """Root-mean-square standardizes a named array across the given axes."""
    across = (self.across,) if isinstance(self.across, str) else self.across

    @named_axes.nmap
    def _rms_standardize(x):
      var = jnp.mean(jnp.square(x))
      return x * jnp.reciprocal(jnp.sqrt(var + self.epsilon))

    return _rms_standardize(value.untag(*across)).tag(*across)

  def input_structure(self) -> Any:
    across = (self.across,) if isinstance(self.across, str) else self.across
    return shapecheck.ArraySpec.floating_named({
        **shapecheck.var("B"),
        **shapecheck.vars_for_axes("across", across),
    })

  def output_structure(self) -> Any:
    return self.input_structure()


@struct.pytree_dataclass
class RMSLayerNorm(grouping.Sequential):
  """Root-mean-squared layer normalization layer.

  RMS layer normalization layers consist of:

  * root-mean-squared standardization over a feature axis or axes,

  * with a learned parallel rescaling of each feature along those axes.

  As proposed by Zhang & Sennrich (2019): https://arxiv.org/abs/1910.07467.

  For flexibility, ``RMSLayerNorm`` is a subclass of `Sequential`.
  """

  sublayers: list[layer_base.LayerLike]

  @classmethod
  def from_config(
      cls,
      across_axes: dict[str, int],
      epsilon: float | jax.Array = 1e-6,
      dtype: jax.typing.DTypeLike = jnp.float32,
  ) -> RMSLayerNorm:
    """Constructs a RMS layer normalization layer.

    Args:
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
        parameters.add_parameter_prefix(
            "scale",
            linear_and_affine.Linear.from_config(
                input_axes={},
                output_axes={},
                parallel_axes=across_axes,
                initializer=linear_and_affine.constant_initializer(1.0),
                dtype=dtype,
            ),
        ),
    ])
