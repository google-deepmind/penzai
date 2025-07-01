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
import chex
import jax
from penzai import pz
from penzai.toolshed import jit_wrapper


class LinearAndAffineTest(absltest.TestCase):

  def test_rename_axes_one(self):
    layer = pz.nn.RenameAxes(old="foo", new="bar")
    result = layer(pz.nx.ones({"foo": 2, "baz": 3}))
    pz.chk.check_structure(
        result, pz.chk.ArraySpec(named_shape={"bar": 2, "baz": 3})
    )

  def test_rename_axes_multi(self):
    layer = pz.nn.RenameAxes(old=("foo", "x"), new=("bar", "y"))
    result = layer(pz.nx.ones({"foo": 2, "baz": 3, "x": 4}))
    pz.chk.check_structure(
        result, pz.chk.ArraySpec(named_shape={"bar": 2, "baz": 3, "y": 4})
    )

  def test_einsum_verbose(self):
    layer = pz.nn.NamedEinsum(
        (
            {"tokens": "t", "heads": "h", "projection": "p"},
            {"kv_tokens": "s", "heads": "h", "projection": "p"},
        ),
        {"heads": "h", "tokens": "t", "kv_tokens": "s"},
    )
    result = layer((
        pz.nx.ones({"batch": 1, "tokens": 2, "heads": 3, "projection": 4}),
        pz.nx.ones({"batch": 1, "kv_tokens": 5, "heads": 3, "projection": 4}),
    ))
    pz.chk.check_structure(
        result,
        pz.chk.ArraySpec(
            named_shape={"batch": 1, "heads": 3, "tokens": 2, "kv_tokens": 5}
        ),
    )

  def test_einsum_simple(self):
    layer = pz.nn.NamedEinsum(
        (
            ("tokens", "heads", "projection"),
            ("kv_tokens", "heads", "projection"),
        ),
        ("heads", "tokens", "kv_tokens"),
    )
    result = layer((
        pz.nx.ones({"batch": 1, "tokens": 2, "heads": 3, "projection": 4}),
        pz.nx.ones({"batch": 1, "kv_tokens": 5, "heads": 3, "projection": 4}),
    ))
    pz.chk.check_structure(
        result,
        pz.chk.ArraySpec(
            named_shape={"batch": 1, "heads": 3, "tokens": 2, "kv_tokens": 5}
        ),
    )

  def test_einsum_renaming_diagonalizing(self):
    layer = pz.nn.NamedEinsum(
        ({"foo": "i", "bar": "i", "baz": "j"}, {"foo": "j", "qux": "k"}),
        {"x": "i", "y": "k", "z": "j"},
    )
    result = layer((
        pz.nx.ones({"batch": 1, "foo": 3, "bar": 3, "baz": 4}),
        pz.nx.ones({"batch": 1, "foo": 4, "qux": 5}),
    ))
    pz.chk.check_structure(
        result,
        pz.chk.ArraySpec(named_shape={"batch": 1, "x": 3, "z": 4, "y": 5}),
    )

  def test_linear_not_in_place(self):
    layer = pz.nn.Linear.from_config(
        name="test",
        init_base_rng=jax.random.key(1),
        input_axes={"foo": 3},
        output_axes={"bar": 5},
        parallel_axes={"baz": 7},
        parallel_broadcast_axes={"qux": 11},
        rename_outputs_if_necessary=False,
    )
    result = layer(
        pz.nx.ones({"batch": 1, "foo": 3, "baz": 7}),
    )
    pz.chk.check_structure(
        result,
        pz.chk.ArraySpec(
            named_shape={"batch": 1, "bar": 5, "baz": 7, "qux": 11}
        ),
    )

  def test_linear_in_place(self):
    layer = pz.nn.Linear.from_config(
        name="test",
        init_base_rng=jax.random.key(1),
        input_axes={"foo": 3},
        output_axes={"foo": 5},
        parallel_axes={"baz": 7},
        parallel_broadcast_axes={"qux": 11},
        rename_outputs_if_necessary=True,
    )
    result = layer(
        pz.nx.ones({"batch": 1, "foo": 3, "baz": 7}),
    )
    pz.chk.check_structure(
        result,
        pz.chk.ArraySpec(
            named_shape={"batch": 1, "foo": 5, "baz": 7, "qux": 11}
        ),
    )

  def test_add_bias(self):
    layer = pz.nn.AddBias.from_config(
        name="test",
        init_base_rng=jax.random.key(1),
        biased_axes={"foo": 3},
        new_output_axes={"bar": 5},
    )
    result = layer(
        pz.nx.ones({"batch": 1, "foo": 3}),
    )
    pz.chk.check_structure(
        result,
        pz.chk.ArraySpec(named_shape={"batch": 1, "foo": 3, "bar": 5}),
    )

  def test_affine(self):
    layer = pz.nn.Affine.from_config(
        name="test",
        init_base_rng=jax.random.key(1),
        input_axes={"foo": 3},
        output_axes={"foo": 5},
        parallel_axes={"baz": 7},
        parallel_broadcast_axes={"qux": 11},
        rename_outputs_if_necessary=True,
    )
    result = layer(
        pz.nx.ones({"batch": 1, "foo": 3, "baz": 7}),
    )
    pz.chk.check_structure(
        result,
        pz.chk.ArraySpec(
            named_shape={"batch": 1, "foo": 5, "baz": 7, "qux": 11}
        ),
    )

  def test_conv_shape(self):
    layer = pz.nn.Conv.from_config(
        name="test",
        init_base_rng=jax.random.key(1),
        input_axes={"foo": 3},
        output_axes={"foo": 5},
        convolution_spatial_axes={"height": 3, "width": 3},
        parallel_axes={"baz": 7},
        parallel_broadcast_axes={"qux": 11},
        rename_outputs_if_necessary=True,
    )
    result = layer(
        pz.nx.ones({"batch": 1, "height": 10, "width": 15, "foo": 3, "baz": 7}),
    )
    pz.chk.check_structure(
        result,
        pz.chk.ArraySpec(
            named_shape={
                "batch": 1,
                "height": 10,
                "width": 15,
                "foo": 5,
                "baz": 7,
                "qux": 11,
            }
        ),
    )

  def test_strided_conv_shape(self):
    layer = pz.nn.Conv.from_config(
        name="test",
        init_base_rng=jax.random.key(1),
        input_axes={"foo": 3},
        output_axes={"foo": 5},
        convolution_spatial_axes={"height": 3, "width": 3},
        strides=(2, 2),
        parallel_axes={"baz": 7},
        parallel_broadcast_axes={"qux": 11},
        rename_outputs_if_necessary=True,
    )
    result = layer(
        pz.nx.ones({"batch": 1, "height": 10, "width": 16, "foo": 3, "baz": 7}),
    )
    pz.chk.check_structure(
        result,
        pz.chk.ArraySpec(
            named_shape={
                "batch": 1,
                "height": 5,
                "width": 8,
                "foo": 5,
                "baz": 7,
                "qux": 11,
            }
        ),
    )

  def test_conv_jit_wrapper(self):
    layer = jit_wrapper.Jitted(
        pz.nn.Conv.from_config(
            name="test",
            init_base_rng=jax.random.key(1),
            input_axes={"foo": 3},
            output_axes={"foo": 5},
            convolution_spatial_axes={"height": 3, "width": 3},
            parallel_axes={"baz": 7},
            parallel_broadcast_axes={"qux": 11},
            rename_outputs_if_necessary=True,
        )
    )
    layer(
        pz.nx.ones({"batch": 1, "height": 10, "width": 15, "foo": 3, "baz": 7}),
    )

  def test_conv_value(self):
    inputs = jax.random.normal(
        key=jax.random.PRNGKey(42), shape=(1, 10, 15, 3 * 7)
    )

    pz_inputs = pz.nx.wrap(inputs.reshape(1, 10, 15, 3, 7)).tag(
        "batch", "height", "width", "foo", "baz"
    )

    simple_layer = pz.nn.Conv.from_config(
        name="test",
        init_base_rng=jax.random.key(1),
        input_axes={"foo": 3, "baz": 7},
        output_axes={"foo_out": 5, "baz_out": 11},
        convolution_spatial_axes={"height": 3, "width": 3},
        parallel_axes=None,
        parallel_broadcast_axes=None,
        rename_outputs_if_necessary=True,
    )

    pz_outputs = (
        simple_layer(pz_inputs)
        .untag("batch", "height", "width", "foo_out", "baz_out")
        .reshape((1, 10, 15, 5 * 11))
        .unwrap()
    )

    # build equivalent jax conv
    kernel = (
        simple_layer.kernel.value.untag(
            "height",
            "width",
            "foo",
            "baz",
            "foo_out",
            "baz_out",
        )
        .reshape(3, 3, 3 * 7, 5 * 11)
        .unwrap()
    )
    outputs = jax.lax.conv_general_dilated(
        inputs,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    chex.assert_trees_all_equal(pz_outputs, outputs)

  def test_conv_transpose_shape(self):
    layer = pz.nn.ConvTranspose.from_config(
        name="test",
        init_base_rng=jax.random.key(1),
        input_axes={"foo": 3},
        output_axes={"foo": 5},
        convolution_spatial_axes={"height": 3, "width": 3},
        parallel_axes={"baz": 7},
        parallel_broadcast_axes={"qux": 11},
        rename_outputs_if_necessary=True,
    )
    result = layer(
        pz.nx.ones({"batch": 1, "height": 10, "width": 15, "foo": 3, "baz": 7}),
    )
    pz.chk.check_structure(
        result,
        pz.chk.ArraySpec(
            named_shape={
                "batch": 1,
                "height": 10,
                "width": 15,
                "foo": 5,
                "baz": 7,
                "qux": 11,
            }
        ),
    )

  def test_strided_conv_transpose_shape(self):
    layer = pz.nn.ConvTranspose.from_config(
        name="test",
        init_base_rng=jax.random.key(1),
        input_axes={"foo": 3},
        output_axes={"foo": 5},
        convolution_spatial_axes={"height": 3, "width": 3},
        strides=(2, 2),
        parallel_axes={"baz": 7},
        parallel_broadcast_axes={"qux": 11},
        rename_outputs_if_necessary=True,
    )
    result = layer(
        pz.nx.ones({"batch": 1, "height": 10, "width": 16, "foo": 3, "baz": 7}),
    )
    pz.chk.check_structure(
        result,
        pz.chk.ArraySpec(
            named_shape={
                "batch": 1,
                "height": 20,
                "width": 32,
                "foo": 5,
                "baz": 7,
                "qux": 11,
            }
        ),
    )

  def test_conv_transpose_value(self):
    inputs = jax.random.normal(
        key=jax.random.PRNGKey(42), shape=(1, 10, 15, 3 * 7)
    )

    pz_inputs = pz.nx.wrap(inputs.reshape(1, 10, 15, 3, 7)).tag(
        "batch", "height", "width", "foo", "baz"
    )

    simple_layer = pz.nn.ConvTranspose.from_config(
        name="test",
        init_base_rng=jax.random.key(1),
        input_axes={"foo": 3, "baz": 7},
        output_axes={"foo_out": 5, "baz_out": 11},
        convolution_spatial_axes={"height": 3, "width": 3},
        parallel_axes=None,
        parallel_broadcast_axes=None,
        rename_outputs_if_necessary=True,
    )

    pz_outputs = (
        simple_layer(pz_inputs)
        .untag("batch", "height", "width", "foo_out", "baz_out")
        .reshape((1, 10, 15, 5 * 11))
        .unwrap()
    )

    # build equivalent jax conv transpose
    kernel = (
        simple_layer.kernel.value.untag(
            "height",
            "width",
            "foo",
            "baz",
            "foo_out",
            "baz_out",
        )
        .reshape(3, 3, 3 * 7, 5 * 11)
        .unwrap()
    )
    outputs = jax.lax.conv_transpose(
        inputs,
        kernel,
        padding="SAME",
        strides=(1, 1),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    chex.assert_trees_all_equal(pz_outputs, outputs)

  def test_conv_transposed_jit_wrapper(self):
    layer = jit_wrapper.Jitted(
        pz.nn.ConvTranspose.from_config(
            name="test",
            init_base_rng=jax.random.key(1),
            input_axes={"foo": 3},
            output_axes={"foo": 5},
            convolution_spatial_axes={"height": 3, "width": 3},
            parallel_axes={"baz": 7},
            parallel_broadcast_axes={"qux": 11},
            rename_outputs_if_necessary=True,
        )
    )
    layer(
        pz.nx.ones({"batch": 1, "height": 10, "width": 15, "foo": 3, "baz": 7}),
    )

  def test_constant_rescale(self):
    layer = pz.nn.ConstantRescale(3.0)
    result = layer(pz.nx.ones({"foo": 3}))
    chex.assert_trees_all_equal(result, 3.0 * pz.nx.ones({"foo": 3}))

  def test_residual(self):
    layer = pz.nn.Residual(
        pz.nn.Linear.from_config(
            name="test",
            init_base_rng=jax.random.key(1),
            input_axes={"features": 5},
            output_axes={"features": 5},
            parallel_axes={"heads": 3},
        )
    )
    result = layer(pz.nx.zeros({"features": 5, "heads": 3, "batch": 2}))
    pz.chk.check_structure(
        result,
        pz.chk.ArraySpec(named_shape={"features": 5, "heads": 3, "batch": 2}),
    )


if __name__ == "__main__":
  absltest.main()
