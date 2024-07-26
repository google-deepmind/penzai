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

"""Dropout regularization using RandomEffect."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from penzai.core import named_axes
from penzai.core import struct
from penzai.nn import grouping
from penzai.nn import layer


@struct.pytree_dataclass
class StochasticDropout(layer.Layer):
  """Stochastic dropout layer.

  Dropout layers randomly mask out elements with a probability ``drop_rate``,
  and then scale the output up by a factor of ``1 / (1 - drop_rate)``.

  For simplicity, and to avoid having to pass configuration through the model,
  dropout layers are *always* stochastic. To disable dropout, you can remove the
  dropout layers from a model using logic such as ::

    model.select().at_instances_of(StochasticDropout).remove_from_parent()

  or just disable them using ::

    .at_instances_of(StochasticDropout).apply(lambda x: x.disable())

  Note that dropout by default assigns different random dropout masks along
  every axis of the input. If you wish to share masks along different axes and
  thus drop out entire slices at a time, you can add those axis names to
  ``share_across_axes``.

  Attributes:
    drop_rate: Probability of dropping an element.
    share_across_axes: Name or names of axes to share the dropout mask over. A
      single dropout mask will be broadcast across these axes. Any other axes
      will have independently-sampled dropout masks.
    random_stream_input_name: The side input key for the random stream used by
      the model at runtime.
  """

  drop_rate: float
  share_across_axes: tuple[str, ...] = dataclasses.field(
      default=(), metadata={"pytree_node": False}
  )
  random_stream_input_name: str = dataclasses.field(
      default="random_stream", metadata={"pytree_node": False}
  )

  def __call__(
      self, value: named_axes.NamedArray, /, **side_inputs
  ) -> named_axes.NamedArray:
    """Randomly drops out components of the input."""
    rng = side_inputs[self.random_stream_input_name]
    random_mask_axes = {
        name: size
        for name, size in value.named_shape.items()
        if name not in self.share_across_axes
    }
    drop_mask = named_axes.wrap(
        jax.random.bernoulli(
            rng.next_key(),
            p=self.drop_rate,
            shape=tuple(random_mask_axes.values()),
        ),
        *random_mask_axes.keys(),
    )
    return named_axes.nmap(jnp.where)(
        drop_mask, 0, value / (1 - self.drop_rate)
    )

  def disable(self) -> DisabledDropout:
    """Returns a disabled version of this layer."""
    return DisabledDropout(
        drop_rate=self.drop_rate,
        share_across_axes=tuple(self.share_across_axes),
    )


@struct.pytree_dataclass
class DisabledDropout(grouping.Identity):
  """A no-op layer taking the place of a disabled `StochasticDropout` layer.

  This layer can be used to mark a location in a model where dropout could
  be applied, or where dropout was originally applied before disabling it. Its
  attributes are unused, except that they can be used to rebuild a
  `StochasticDropout` layer.

  Attributes:
    drop_rate: Drop rate for the enabled (`StochasticDropout`) version of this
      layer.
    share_across_axes: Shared axis names for the enabled (`StochasticDropout`)
      version of this layer.
    random_stream_input_name: The side input key for the random stream used by
      the model at runtime. (Ignored unless the dropout layer is reenabled.)
  """

  drop_rate: float
  share_across_axes: tuple[str, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  random_stream_input_name: str = dataclasses.field(
      default="random_stream", metadata={"pytree_node": False}
  )

  def enable(self) -> StochasticDropout:
    """Returns a stochastic, enabled version of this layer."""
    return StochasticDropout(
        drop_rate=self.drop_rate,
        share_across_axes=self.share_across_axes,
        random_stream_input_name=self.random_stream_input_name,
    )


def maybe_dropout(
    drop_rate: float | None,
    share_across_axes: tuple[str, ...] = (),
    random_stream_input_name: str = "random_stream",
) -> StochasticDropout | DisabledDropout:
  """Constructs either a stochastic or disabled dropout layer.

  Args:
    drop_rate: Probability of dropping an element. If None, dropout will be
      disabled entirely.
    share_across_axes: Name or names of axes to share the dropout mask over. A
      single dropout mask will be broadcast across these axes.
    random_stream_input_name: Side input key for the random stream.

  Returns:
    A `StochasticDropout` layer if drop_rate is a float, or a `DisabledDropout`
    layer with drop rate 0 if ``drop_rate`` is None.
  """
  if drop_rate is None:
    return DisabledDropout(
        drop_rate=0.0,
        share_across_axes=share_across_axes,
        random_stream_input_name=random_stream_input_name,
    )
  else:
    return StochasticDropout(
        drop_rate=drop_rate,
        share_across_axes=share_across_axes,
        random_stream_input_name=random_stream_input_name,
    )
