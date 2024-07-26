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

"""Tests for linear layers."""

from absl.testing import absltest
from penzai.deprecated.v1 import pz
from penzai.deprecated.v1.toolshed import check_layers_by_tracing


class LinearAndAffineTest(absltest.TestCase):

  def test_rename_axes_one(self):
    layer = pz.nn.RenameAxes(old="foo", new="bar")
    check_layers_by_tracing.check_layer(layer)

  def test_rename_axes_multi(self):
    layer = pz.nn.RenameAxes(old=("foo", "x"), new=("bar", "y"))
    check_layers_by_tracing.check_layer(layer)

  def test_einsum_verbose(self):
    layer = pz.nn.NamedEinsum(
        (
            {"tokens": "t", "heads": "h", "projection": "p"},
            {"kv_tokens": "s", "heads": "h", "projection": "p"},
        ),
        {"heads": "h", "tokens": "t", "kv_tokens": "s"},
    )
    check_layers_by_tracing.check_layer(layer)

  def test_einsum_simple(self):
    layer = pz.nn.NamedEinsum(
        (
            ("tokens", "heads", "projection"),
            ("kv_tokens", "heads", "projection"),
        ),
        ("heads", "tokens", "kv_tokens"),
    )
    check_layers_by_tracing.check_layer(layer)

  def test_einsum_renaming_diagonalizing(self):
    layer = pz.nn.NamedEinsum(
        ({"foo": "i", "bar": "i", "baz": "j"}, {"foo": "j", "qux": "k"}),
        {"x": "i", "y": "k", "z": "j"},
    )
    check_layers_by_tracing.check_layer(layer)

  def test_linear_not_in_place(self):
    layer = pz.nn.Linear.from_config(
        input_axes={"foo": 13},
        output_axes={"bar": 15},
        parallel_axes={"baz": 17},
        parallel_broadcast_axes={"qux": 19},
        rename_outputs_if_necessary=False,
    )
    check_layers_by_tracing.check_layer(layer)

  def test_linear_in_place(self):
    layer = pz.nn.Linear.from_config(
        input_axes={"foo": 13},
        output_axes={"foo": 15},
        parallel_axes={"baz": 17},
        parallel_broadcast_axes={"qux": 19},
        rename_outputs_if_necessary=True,
    )
    check_layers_by_tracing.check_layer(
        layer, pz.nx.zeros({"foo": 13, "baz": 17, "batch": 4})
    )

  def test_add_bias(self):
    layer = pz.nn.AddBias.from_config(
        biased_axes={"foo": 13}, new_output_axes={"bar": 15}
    )
    check_layers_by_tracing.check_layer(layer)

  def test_affine(self):
    layer = pz.nn.Affine.from_config(
        input_axes={"foo": 13},
        output_axes={"foo": 15},
        parallel_axes={"baz": 17},
        parallel_broadcast_axes={"qux": 19},
        rename_outputs_if_necessary=True,
    )
    check_layers_by_tracing.check_layer(
        layer, pz.nx.zeros({"foo": 13, "baz": 17, "batch": 4})
    )

  def test_constant_rescale(self):
    layer = pz.nn.ConstantRescale(3.0)
    check_layers_by_tracing.check_layer(layer)

  def test_residual(self):
    layer = pz.nn.Residual(
        pz.nn.Linear.from_config(
            input_axes={"features": 13},
            output_axes={"features": 13},
            parallel_axes={"heads": 17},
        )
    )
    check_layers_by_tracing.check_layer(
        layer, pz.nx.zeros({"features": 13, "heads": 17, "batch": 4})
    )


if __name__ == "__main__":
  absltest.main()
