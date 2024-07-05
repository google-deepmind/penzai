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

"""Tests for penzai.named_axes."""

import collections
import re

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from penzai import pz
from penzai.core import named_axes


class NamedAxesTest(parameterized.TestCase):

  def test_wrap(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    named = named_axes.wrap(data)
    chex.assert_trees_all_equal(
        named,
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({}), data_array=data
        ),
    )
    self.assertEqual(named.positional_shape, (2, 3, 5))
    self.assertEqual(named.named_shape, {})
    self.assertEqual(named.dtype, np.int32)

  def test_named_array_tag(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    named = named_axes.wrap(data).tag("foo", "bar", "baz")
    chex.assert_trees_all_equal(
        named,
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({"foo": 2, "bar": 3, "baz": 5}),
            data_array=data,
        ),
    )
    self.assertEqual(named.positional_shape, ())
    self.assertEqual(named.named_shape, {"foo": 2, "bar": 3, "baz": 5})

  def test_named_array_wrap_named(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    named = named_axes.wrap(data, "foo", "bar", "baz")
    chex.assert_trees_all_equal(
        named,
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({"foo": 2, "bar": 3, "baz": 5}),
            data_array=data,
        ),
    )
    self.assertEqual(named.positional_shape, ())
    self.assertEqual(named.named_shape, {"foo": 2, "bar": 3, "baz": 5})

  def test_named_array_untag(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    named = named_axes.wrap(data).tag("foo", "bar", "baz").untag("bar")
    chex.assert_trees_all_equal(
        named,
        named_axes.NamedArrayView(
            data_shape=(2, 3, 5),
            data_axis_for_logical_axis=(1,),
            data_axis_for_name={"foo": 0, "baz": 2},
            data_array=data,
        ),
    )
    self.assertEqual(named.positional_shape, (3,))
    self.assertEqual(named.named_shape, {"foo": 2, "baz": 5})
    self.assertEqual(named.dtype, np.int32)

  def test_named_array_untag_nothing(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    named = named_axes.wrap(data).tag("foo", "bar", "baz").untag()
    self.assertEqual(named.positional_shape, ())
    self.assertEqual(named.named_shape, {"foo": 2, "bar": 3, "baz": 5})

  def test_untag_then_tag(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    named = (
        named_axes.wrap(data)
        .tag("foo", "bar", "baz")
        .untag("bar", "foo")
        .tag("bar2", "foo2")
    )
    chex.assert_trees_all_equal(
        named,
        named_axes.NamedArray(
            named_axes=collections.OrderedDict(
                {"foo2": 2, "bar2": 3, "baz": 5}
            ),
            data_array=data,
        ),
    )
    self.assertEqual(named.positional_shape, ())
    self.assertEqual(
        named.named_shape,
        {"foo2": 2, "bar2": 3, "baz": 5},
    )

    chex.assert_trees_all_equal(named.tag(), named)

  def test_untag_then_tag_prefix(self):
    data = jnp.arange(2 * 3 * 5 * 7).reshape((2, 3, 5, 7))
    arr_c = named_axes.wrap(data).tag("a", "b", "c", "d").untag("c")
    self.assertEqual(arr_c.positional_shape, (5,))
    self.assertEqual(arr_c.named_shape, {"a": 2, "b": 3, "d": 7})
    arr_abc = arr_c.untag_prefix("a", "b")
    self.assertEqual(arr_abc.positional_shape, (2, 3, 5))
    self.assertEqual(arr_abc.named_shape, {"d": 7})
    arr_tagged = arr_abc.tag_prefix("aa", "bb")
    self.assertEqual(arr_tagged.positional_shape, (5,))
    self.assertEqual(arr_tagged.named_shape, {"aa": 2, "bb": 3, "d": 7})

  def test_positional_prefix(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    named = (
        named_axes.wrap(data)
        .tag("foo", "bar", "baz")
        .untag("bar", "foo")
        .with_positional_prefix()
    )
    chex.assert_trees_all_equal(
        named,
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({"baz": 5}),
            data_array=np.array(
                [
                    [[0, 1, 2, 3, 4], [15, 16, 17, 18, 19]],
                    [[5, 6, 7, 8, 9], [20, 21, 22, 23, 24]],
                    [[10, 11, 12, 13, 14], [25, 26, 27, 28, 29]],
                ],
                dtype=np.int32,
            ),
        ),
    )
    self.assertEqual(named.positional_shape, (3, 2))
    self.assertEqual(named.named_shape, {"baz": 5})

    chex.assert_trees_all_equal(named.with_positional_prefix(), named)

  def test_unwrap(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    self.assertIs(named_axes.wrap(data).unwrap(), data)
    self.assertIs(
        named_axes.wrap(data).tag("a", "b", "c").untag("a", "b", "c").unwrap(),
        data,
    )

    chex.assert_trees_all_equal(
        named_axes.wrap(data).tag("a", "b", "c").unwrap("b", "a", "c"),
        jnp.array(
            [
                [[0, 1, 2, 3, 4], [15, 16, 17, 18, 19]],
                [[5, 6, 7, 8, 9], [20, 21, 22, 23, 24]],
                [[10, 11, 12, 13, 14], [25, 26, 27, 28, 29]],
            ],
            dtype=jnp.int32,
        ),
    )

  def test_ordered_as(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    named = named_axes.wrap(data).tag("a", "b", "c").untag("b")
    reordered = named.order_as("a", "c")
    chex.assert_trees_all_equal(
        reordered,
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({"a": 2, "c": 5}),
            data_array=np.array(
                [
                    [[0, 1, 2, 3, 4], [15, 16, 17, 18, 19]],
                    [[5, 6, 7, 8, 9], [20, 21, 22, 23, 24]],
                    [[10, 11, 12, 13, 14], [25, 26, 27, 28, 29]],
                ],
                dtype=np.int32,
            ),
        ),
    )
    self.assertEqual(named.positional_shape, reordered.positional_shape)
    self.assertEqual(named.named_shape, reordered.named_shape)

  def test_invalid_namedarray(self):
    with self.subTest("bad_contents"):
      with self.assertRaises(ValueError):
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({}), data_array=5
        ).check_valid()

    with self.subTest("bad_axes_container"):
      with self.assertRaises(ValueError):
        named_axes.NamedArray(
            named_axes=None, data_array=jnp.arange(5)
        ).check_valid()

    with self.subTest("bad_axes_value"):
      with self.assertRaises(ValueError):
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({"foo": "five"}),
            data_array=jnp.arange(5),
        ).check_valid()

    with self.subTest("bad_axes_2"):
      with self.assertRaises(ValueError):
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({"foo": "five"}),
            data_array=jnp.arange(5),
        ).check_valid()

    with self.subTest("bad_suffix_shape"):
      with self.assertRaises(ValueError):
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({"a": 1, "b": 2}),
            data_array=jnp.zeros([4, 2]),
        ).check_valid()

  def test_invalid_namedarrayview(self):
    with self.subTest("bad_contents"):
      with self.assertRaises(ValueError):
        named_axes.NamedArrayView(
            data_array=5,
            data_shape=(),
            data_axis_for_logical_axis=(),
            data_axis_for_name={},
        ).check_valid()

    with self.subTest("bad_data_shape"):
      with self.assertRaises(ValueError):
        named_axes.NamedArrayView(
            data_array=jnp.zeros([2, 3]),
            data_shape=(3, 4),
            data_axis_for_logical_axis=(1, 0),
            data_axis_for_name={},
        ).check_valid()

    with self.subTest("bad_indices"):
      with self.assertRaises(ValueError):
        named_axes.NamedArrayView(
            data_array=jnp.zeros([2, 3, 4]),
            data_shape=(2, 3, 4),
            data_axis_for_logical_axis=(0, 1, 0),
            data_axis_for_name={"foo": 2, "bar": 1},
        ).check_valid()

  def test_bad_untag(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    with self.subTest("already_untagged"):
      with self.assertRaisesRegex(
          ValueError,
          re.escape(
              "`untag` cannot be used to introduce positional axes for a"
              " NamedArray (or NamedArrayView) that already has positional"
              " axes."
          ),
      ):
        named_axes.wrap(data).tag("a", "b", "c").untag("b").untag("a")

    with self.subTest("repeats"):
      with self.assertRaisesRegex(
          ValueError,
          re.escape("Repeats in `axis_order` are not allowed."),
      ):
        named_axes.wrap(data).tag("a", "b", "c").untag("b", "c", "b")

    with self.subTest("unknown_names"):
      with self.assertRaisesRegex(
          ValueError,
          re.escape(
              "Requested axis names {'bad'} are not present in the array."
          ),
      ):
        named_axes.wrap(data).tag("a", "b", "c").untag("b", "c", "bad")

  def test_bad_tag(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    with self.subTest("wrong_tag_length"):
      with self.assertRaisesRegex(
          ValueError,
          re.escape(
              "There must be exactly as many names given to `tag` as there"
              " are positional axes in the array"
          ),
      ):
        named_axes.wrap(data).tag("a", "b", "c", "d")

    with self.subTest("repeated_tags"):
      with self.assertRaisesRegex(
          ValueError,
          re.escape("Repeated axis names are not allowed"),
      ):
        named_axes.wrap(data).tag("foo", "bar", "foo")

    with self.subTest("tag_conflict"):
      with self.assertRaisesRegex(
          ValueError,
          re.escape("Repeated axis names are not allowed"),
      ):
        named_axes.wrap(data).tag("foo", "bar", "baz").untag("bar").tag("foo")

    with self.subTest("bad_tag"):
      with self.assertRaisesRegex(
          ValueError,
          re.escape("Integers are not allowed as axis names"),
      ):
        named_axes.wrap(data).tag("foo", "bar", 3)

  def test_nmap(self):
    def my_nmappable_function(named_1, a_string, structure, named_2, ordinary):
      self.assertIsInstance(named_1, jax.Array)
      chex.assert_shape(named_1, (5,))

      self.assertEqual(a_string, "a non-arraylike string")

      self.assertIsInstance(structure["a named array"], jax.Array)
      chex.assert_shape(structure["a named array"], ())
      # this number is literally a number, not a tracer, so it could be used
      # as a shape or an axis index.
      self.assertEqual(structure["a number"], 3)
      self.assertIs(structure["an arbitrary object"], jax.nn.relu)

      self.assertIsInstance(named_2, jax.Array)
      chex.assert_shape(named_2, (4, 3))

      self.assertIsInstance(ordinary, jax.Array)
      chex.assert_shape(ordinary, (7,))

      return named_1[:3] + named_2[0, :] + ordinary[:3]

    result = named_axes.nmap(my_nmappable_function)(
        (
            named_axes.wrap(jnp.arange(2 * 3 * 5).reshape((2, 3, 5)))
            .tag("foo", "bar", "baz")
            .untag("baz")
        ),
        "a non-arraylike string",
        structure={
            "a named array": named_axes.wrap(jnp.arange(2)).tag("foo"),
            "a number": 3,
            "an arbitrary object": jax.nn.relu,
        },
        named_2=(
            named_axes.wrap(100 * jnp.ones((3, 2, 3, 4)))
            .tag("bar", "baz", "data1", "data2")
            .untag("data2", "data1")
        ),
        ordinary=jnp.ones((7,)),
    )

    self.assertIsInstance(
        result, named_axes.NamedArray | named_axes.NamedArrayView
    )
    self.assertEqual(result.positional_shape, (3,))
    self.assertEqual(
        result.named_shape,
        {"foo": 2, "bar": 3, "baz": 2},
    )
    chex.assert_trees_all_equal(
        result.with_positional_prefix(),
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({"foo": 2, "bar": 3, "baz": 2}),
            data_array=np.array(
                [
                    [
                        [[101.0, 101.0], [106.0, 106.0], [111.0, 111.0]],
                        [[116.0, 116.0], [121.0, 121.0], [126.0, 126.0]],
                    ],
                    [
                        [[102.0, 102.0], [107.0, 107.0], [112.0, 112.0]],
                        [[117.0, 117.0], [122.0, 122.0], [127.0, 127.0]],
                    ],
                    [
                        [[103.0, 103.0], [108.0, 108.0], [113.0, 113.0]],
                        [[118.0, 118.0], [123.0, 123.0], [128.0, 128.0]],
                    ],
                ],
                dtype=np.dtype("float32"),
            ),
        ),
    )

  def test_indexing_positional(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    named = named_axes.wrap(data).tag("foo", "bar", "baz")

    result = named.untag("baz", "bar")[2, ::-1].with_positional_prefix()
    self.assertEqual(result.named_shape, {"foo": 2})
    self.assertEqual(result.positional_shape, (3,))
    chex.assert_trees_all_equal(
        result,
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({"foo": 2}),
            data_array=np.array(
                [[12, 27], [7, 22], [2, 17]], dtype=np.dtype("int32")
            ),
        ),
    )

  def test_indexing_by_dict(self):
    data = jnp.arange(2 * 3 * 5).reshape((2, 3, 5))
    named = named_axes.wrap(data).tag("foo", "bar", "baz")

    result = named[{"baz": 2, "bar": pz.slice[::-1]}].with_positional_prefix()
    self.assertEqual(result.named_shape, {"foo": 2, "bar": 3})
    self.assertEqual(result.positional_shape, ())
    chex.assert_trees_all_equal(
        result.order_as("foo", "bar"),
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({"foo": 2, "bar": 3}),
            data_array=np.array(
                [[12, 7, 2], [27, 22, 17]], dtype=np.dtype("int32")
            ),
        ),
    )

    result = named.untag("foo")[
        {"baz": 2, "bar": pz.slice[::-1], "qux": None}
    ].tag("foo")
    self.assertEqual(result.named_shape, {"foo": 2, "bar": 3, "qux": 1})
    self.assertEqual(result.positional_shape, ())
    chex.assert_trees_all_equal(
        result.order_as("foo", "bar", "qux"),
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({"foo": 2, "bar": 3, "qux": 1}),
            data_array=np.array(
                [[[12], [7], [2]], [[27], [22], [17]]], dtype=np.dtype("int32")
            ),
        ),
    )

  def test_indexing_batched(self):
    table = named_axes.wrap(jnp.arange(100 * 3).reshape((100, 3))).tag(
        "vocab", "features"
    )
    indices = named_axes.wrap(jnp.array([2, 61, 3, 10, 40])).tag("batch")

    with self.subTest("positional_style"):
      result = table.untag("vocab")[indices]
      self.assertEqual(result.named_shape, {"features": 3, "batch": 5})
      self.assertEqual(result.positional_shape, ())
      chex.assert_trees_all_equal(
          result,
          named_axes.NamedArray(
              named_axes=collections.OrderedDict({"features": 3, "batch": 5}),
              data_array=np.array(
                  [
                      [6, 183, 9, 30, 120],
                      [7, 184, 10, 31, 121],
                      [8, 185, 11, 32, 122],
                  ],
                  dtype=np.dtype("int32"),
              ),
          ),
      )

    with self.subTest("dict_style"):
      result = table[{"vocab": indices}]
      self.assertEqual(result.named_shape, {"features": 3, "batch": 5})
      self.assertEqual(result.positional_shape, ())
      chex.assert_trees_all_equal(
          result,
          named_axes.NamedArray(
              named_axes=collections.OrderedDict({"features": 3, "batch": 5}),
              data_array=np.array(
                  [
                      [6, 183, 9, 30, 120],
                      [7, 184, 10, 31, 121],
                      [8, 185, 11, 32, 122],
                  ],
                  dtype=np.dtype("int32"),
              ),
          ),
      )

  def test_wrapped_instance_methods(self):
    array_a = named_axes.wrap(jnp.arange(2 * 3).reshape((2, 3))).tag(
        "foo", "bar"
    )
    array_b = named_axes.wrap(jnp.arange(5 * 2).reshape((5, 2))).tag(
        "baz", "foo"
    )

    with self.subTest("astype"):
      chex.assert_trees_all_equal(
          array_a.astype(jnp.float32),
          named_axes.NamedArray(
              named_axes=collections.OrderedDict({"foo": 2, "bar": 3}),
              data_array=np.array(
                  [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                  dtype=np.dtype("float32"),
              ),
          ),
      )

    with self.subTest("unop"):
      chex.assert_trees_all_equal(+array_a, array_a)
      chex.assert_trees_all_equal(
          -array_a,
          named_axes.NamedArray(
              named_axes=collections.OrderedDict({"foo": 2, "bar": 3}),
              data_array=np.array(
                  [[0, -1, -2], [-3, -4, -5]], dtype=np.dtype("int32")
              ),
          ),
      )

    with self.subTest("binop_broadcast"):
      chex.assert_trees_all_equal(
          array_a + array_b,
          named_axes.NamedArray(
              named_axes=collections.OrderedDict(
                  {"foo": 2, "bar": 3, "baz": 5}
              ),
              data_array=np.array(
                  [
                      [
                          [0, 2, 4, 6, 8],
                          [1, 3, 5, 7, 9],
                          [2, 4, 6, 8, 10],
                      ],
                      [
                          [4, 6, 8, 10, 12],
                          [5, 7, 9, 11, 13],
                          [6, 8, 10, 12, 14],
                      ],
                  ],
                  dtype=np.dtype("int32"),
              ),
          ),
      )
      chex.assert_trees_all_equal(
          array_a > array_b,
          named_axes.NamedArray(
              named_axes=collections.OrderedDict(
                  {"foo": 2, "bar": 3, "baz": 5}
              ),
              data_array=np.array(
                  [
                      [
                          [False, False, False, False, False],
                          [True, False, False, False, False],
                          [True, False, False, False, False],
                      ],
                      [
                          [True, False, False, False, False],
                          [True, True, False, False, False],
                          [True, True, False, False, False],
                      ],
                  ],
                  dtype=np.dtype("bool"),
              ),
          ),
      )
      chex.assert_trees_all_equal(
          array_a - array_b,
          named_axes.NamedArray(
              named_axes=collections.OrderedDict(
                  {"foo": 2, "bar": 3, "baz": 5}
              ),
              data_array=np.array(
                  [
                      [
                          [0, -2, -4, -6, -8],
                          [1, -1, -3, -5, -7],
                          [2, 0, -2, -4, -6],
                      ],
                      [
                          [2, 0, -2, -4, -6],
                          [3, 1, -1, -3, -5],
                          [4, 2, 0, -2, -4],
                      ],
                  ],
                  dtype=np.dtype("int32"),
              ),
          ),
      )

    with self.subTest("array_method"):
      # Summing over no axes does nothing
      chex.assert_trees_all_equal(array_a.sum(), array_a)
      chex.assert_trees_all_equal(
          array_a.untag("bar").sum(),
          named_axes.NamedArray(
              named_axes=collections.OrderedDict({"foo": 2}),
              data_array=np.array([3, 12], dtype=np.dtype("int32")),
          ),
      )
      chex.assert_trees_all_equal(
          array_a.untag("bar", "foo").sum(axis=1).with_positional_prefix(),
          named_axes.NamedArray(
              named_axes=collections.OrderedDict({}),
              data_array=np.array([3, 5, 7], dtype=np.dtype("int32")),
          ),
      )

    with self.subTest("scalar_conversion_error"):
      with self.assertRaisesRegex(
          ValueError, "Cannot convert a non-scalar NamedArray"
      ):
        _ = int(array_a)

      with self.assertRaisesRegex(
          ValueError, "Cannot convert a non-scalar NamedArray"
      ):
        if array_a:
          pass

    with self.subTest("scalar_conversion"):
      array_scalar = named_axes.wrap(jnp.array(3))
      self.assertEqual(int(array_scalar), 3)
      self.assertTrue(bool(array_scalar))

  @parameterized.parameters("python", "numpy", "jax")
  def test_binop_lifts(self, which):
    named_array = named_axes.wrap(jnp.arange(2 * 3).reshape((2, 3))).tag(
        "foo", "bar"
    )
    if which == "python":
      other = 100
    elif which == "numpy":
      other = np.array(100)
    elif which == "jax":
      other = jnp.array(100)
    else:
      raise ValueError(which)

    with self.subTest("left"):
      chex.assert_trees_all_equal(
          named_array - other,
          named_axes.NamedArray(
              named_axes=collections.OrderedDict({"foo": 2, "bar": 3}),
              data_array=np.array(
                  [[-100, -99, -98], [-97, -96, -95]],
                  dtype=np.dtype("int32"),
              ),
          ),
      )

    with self.subTest("right"):
      chex.assert_trees_all_equal(
          other - named_array,
          named_axes.NamedArray(
              named_axes=collections.OrderedDict({"foo": 2, "bar": 3}),
              data_array=np.array(
                  [[100, 99, 98], [97, 96, 95]], dtype=np.dtype("int32")
              ),
          ),
      )

  def test_named_array_equality(self):
    array_a = named_axes.wrap(jnp.arange(2 * 3).reshape((2, 3))).tag(
        "foo", "bar"
    )
    array_b = named_axes.wrap(jnp.arange(5 * 2).reshape((5, 2))).tag(
        "baz", "foo"
    )

    result = array_a == array_b
    self.assertEqual(result.named_shape, {"foo": 2, "bar": 3, "baz": 5})

  def test_convenience_constructors(self):
    with self.subTest("zeros"):
      chex.assert_trees_all_equal(
          named_axes.zeros({"foo": 2, "bar": 3}, jnp.int32),
          named_axes.wrap(jnp.zeros((2, 3))).tag("foo", "bar"),
      )
    with self.subTest("ones"):
      chex.assert_trees_all_equal(
          named_axes.ones({"foo": 2, "bar": 3}, jnp.int32),
          named_axes.wrap(jnp.ones((2, 3))).tag("foo", "bar"),
      )
    with self.subTest("full"):
      chex.assert_trees_all_equal(
          named_axes.full({"foo": 2, "bar": 3}, 3, jnp.int32),
          named_axes.wrap(jnp.full((2, 3), 3)).tag("foo", "bar"),
      )
    with self.subTest("arange"):
      chex.assert_trees_all_equal(
          named_axes.arange("foo", 5),
          named_axes.wrap(jnp.arange(5)).tag("foo"),
      )
    with self.subTest("random_split"):
      split_key_1 = named_axes.random_split(jax.random.key(42), {"foo": 2})
      split_key_2 = named_axes.random_split(split_key_1, {"bar": 3, "baz": 5})
      values = named_axes.nmap(jax.random.normal)(split_key_2, shape=(7,))
      self.assertEqual(values.named_shape, {"foo": 2, "bar": 3, "baz": 5})
      self.assertEqual(values.positional_shape, (7,))
    with self.subTest("concatenate"):
      chex.assert_trees_all_equal(
          named_axes.concatenate(
              [named_axes.ones({"bar": 3}), named_axes.zeros({"bar": 3})],
              "bar",
          ),
          named_axes.wrap(
              jnp.concatenate([jnp.ones((3,)), jnp.zeros((3,))])
          ).tag("bar"),
      )
    with self.subTest("stack"):
      chex.assert_trees_all_equal(
          named_axes.stack(
              [named_axes.ones({"bar": 3}), named_axes.zeros({"bar": 3})],
              "foo",
          ),
          named_axes.nmap(jnp.stack)(
              [named_axes.ones({"bar": 3}), named_axes.zeros({"bar": 3})]
          ).tag("foo"),
      )
    with self.subTest("unstack"):
      array_a = named_axes.wrap(jnp.arange(2 * 3).reshape((2, 3))).tag(
          "foo", "bar"
      )
      chex.assert_trees_all_equal(
          named_axes.unstack(array_a, "foo"),
          [array_a[{"foo": 0}], array_a[{"foo": 1}]],
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="namedarray_to_namedarray",
          left_structure=named_axes.NamedArray(
              named_axes=collections.OrderedDict([("foo", 3), ("bar", 2)]),
              data_array=jax.ShapeDtypeStruct((5, 6, 3, 2), jnp.int32),
          ),
          right_structure=named_axes.NamedArray(
              named_axes=collections.OrderedDict([("bar", 2), ("foo", 3)]),
              data_array=jax.ShapeDtypeStruct((5, 6, 2, 3), jnp.bool_),
          ),
      ),
      dict(
          testcase_name="view_to_namedarray",
          left_structure=named_axes.NamedArrayView(
              data_shape=(6, 2, 5, 3),
              data_axis_for_logical_axis=(2, 0),
              data_axis_for_name={"foo": 3, "bar": 1},
              data_array=jax.ShapeDtypeStruct((6, 2, 5, 3), jnp.int32),
          ),
          right_structure=named_axes.NamedArray(
              named_axes=collections.OrderedDict([("bar", 2), ("foo", 3)]),
              data_array=jax.ShapeDtypeStruct((5, 6, 2, 3), jnp.bool_),
          ),
      ),
      dict(
          testcase_name="namedarray_to_view",
          left_structure=named_axes.NamedArray(
              named_axes=collections.OrderedDict([("bar", 2), ("foo", 3)]),
              data_array=jax.ShapeDtypeStruct((5, 6, 2, 3), jnp.int32),
          ),
          right_structure=named_axes.NamedArrayView(
              data_shape=(6, 2, 5, 3),
              data_axis_for_logical_axis=(2, 0),
              data_axis_for_name={"foo": 3, "bar": 1},
              data_array=jax.ShapeDtypeStruct((6, 2, 5, 3), jnp.bool_),
          ),
      ),
      dict(
          testcase_name="view_to_view",
          left_structure=named_axes.NamedArrayView(
              data_shape=(3, 2, 5, 6),
              data_axis_for_logical_axis=(2, 3),
              data_axis_for_name={"foo": 0, "bar": 1},
              data_array=jax.ShapeDtypeStruct((3, 2, 5, 6), jnp.int32),
          ),
          right_structure=named_axes.NamedArrayView(
              data_shape=(6, 2, 5, 3),
              data_axis_for_logical_axis=(2, 0),
              data_axis_for_name={"foo": 3, "bar": 1},
              data_array=jax.ShapeDtypeStruct((6, 2, 5, 3), jnp.bool_),
          ),
      ),
  )
  def test_order_like(self, left_structure, right_structure):
    left = jax.tree_util.tree_map(
        lambda leaf: jnp.arange(np.prod(leaf.shape), dtype=leaf.dtype).reshape(
            leaf.shape
        ),
        left_structure,
    )
    right = jax.tree_util.tree_map(jnp.zeros_like, right_structure)
    left_like_right = left.order_like(right)
    # Same content as left.
    chex.assert_trees_all_equal(
        left_like_right.canonicalize(), left.canonicalize()
    )
    # Same structure as right.
    chex.assert_trees_all_equal_structs(left_like_right, right)

  @parameterized.named_parameters(
      dict(
          testcase_name="namedarray_noop",
          value_structure=named_axes.NamedArray(
              named_axes=collections.OrderedDict([("foo", 3), ("bar", 2)]),
              data_array=jax.ShapeDtypeStruct((5, 6, 3, 2), jnp.int32),
          ),
          positional_shape=(5, 6),
          named_shape={"foo": 3, "bar": 2},
      ),
      dict(
          testcase_name="view_noop",
          value_structure=named_axes.NamedArrayView(
              data_shape=(6, 2, 5, 3),
              data_axis_for_logical_axis=(2, 0),
              data_axis_for_name={"foo": 3, "bar": 1},
              data_array=jax.ShapeDtypeStruct((6, 2, 5, 3), jnp.bool_),
          ),
          positional_shape=(5, 6),
          named_shape={"foo": 3, "bar": 2},
      ),
      dict(
          testcase_name="namedarray_broadcasted",
          value_structure=named_axes.NamedArray(
              named_axes=collections.OrderedDict([("foo", 3), ("bar", 2)]),
              data_array=jax.ShapeDtypeStruct((5, 1, 6, 3, 2), jnp.int32),
          ),
          positional_shape=(2, 5, 4, 6),
          named_shape={"bar": 2, "foo": 3, "baz": 7},
      ),
      dict(
          testcase_name="namedarrayview_broadcasted",
          value_structure=named_axes.NamedArrayView(
              data_shape=(5, 1, 6, 3, 2),
              data_axis_for_logical_axis=(0, 1, 2),
              data_axis_for_name={"foo": 3, "bar": 4},
              data_array=jax.ShapeDtypeStruct((5, 1, 6, 3, 2), jnp.int32),
          ),
          positional_shape=(2, 5, 4, 6),
          named_shape={"bar": 2, "foo": 3, "baz": 7},
      ),
  )
  def test_broadcast_to(self, value_structure, positional_shape, named_shape):
    def go(value):
      broadcasted = value.broadcast_to(positional_shape, named_shape)
      broadcasted.check_valid()
      return broadcasted

    broadcasted = jax.eval_shape(go, value_structure)
    self.assertEqual(broadcasted.positional_shape, positional_shape)
    self.assertEqual(dict(broadcasted.named_shape), named_shape)

  def test_tree_order_like(self):
    values = {
        "a": 1.0,
        "b": named_axes.wrap(jnp.arange(6).reshape((2, 3))).tag("foo", "bar"),
        "c": named_axes.wrap(jnp.arange(20).reshape((5, 4))).tag_prefix("baz"),
    }
    target = {
        "a": 2.0,
        "b": named_axes.wrap(jnp.zeros((3, 2))).tag("bar", "foo"),
        "c": (
            named_axes.wrap(jnp.zeros((5, 4)))
            .tag_prefix("baz")
            .with_positional_prefix()
        ),
    }
    values_like_target = named_axes.order_like(values, target)

    chex.assert_trees_all_equal_structs(values_like_target, target)
    chex.assert_trees_all_equal(
        (values["b"].canonicalize(), values["c"].canonicalize()),
        (
            values_like_target["b"].canonicalize(),
            values_like_target["c"].canonicalize(),
        ),
    )

  def test_indexed_update_at(self):
    array = pz.nx.arange("foo", 5) + pz.nx.arange("bar", 4)

    for style in ["positional", "dict"]:
      with self.subTest(f"{style}_style"):

        with self.subTest("set_ordinary"):
          if style == "positional":
            result = array.untag("foo").at[3].set(100).tag("foo")
          else:
            result = array.at[{"foo": 3}].set(100)
          chex.assert_trees_all_equal(
              result.canonicalize(),
              pz.nx.wrap(
                  jnp.array([
                      [0, 1, 2, 3],
                      [1, 2, 3, 4],
                      [2, 3, 4, 5],
                      [100, 100, 100, 100],
                      [4, 5, 6, 7],
                  ])
              )
              .tag("foo", "bar")
              .canonicalize(),
          )

        with self.subTest("add_ordinary"):
          if style == "positional":
            result = array.untag("foo").at[3].add(100).tag("foo")
          else:
            result = array.at[{"foo": 3}].add(100)
          chex.assert_trees_all_equal(
              result.canonicalize(),
              pz.nx.wrap(
                  jnp.array([
                      [0, 1, 2, 3],
                      [1, 2, 3, 4],
                      [2, 3, 4, 5],
                      [103, 104, 105, 106],
                      [4, 5, 6, 7],
                  ])
              )
              .tag("foo", "bar")
              .canonicalize(),
          )

        with self.subTest("add_along_named"):
          if style == "positional":
            result = (
                array.untag("bar")
                .at[2]
                .add(100 * pz.nx.arange("foo", 5))
                .tag("bar")
            )
          else:
            result = array.at[{"bar": 2}].add(100 * pz.nx.arange("foo", 5))
          chex.assert_trees_all_equal(
              result.canonicalize(),
              pz.nx.wrap(
                  jnp.array([
                      [0, 1, 2, 3],
                      [1, 2, 103, 4],
                      [2, 3, 204, 5],
                      [3, 4, 305, 6],
                      [4, 5, 406, 7],
                  ])
              )
              .tag("foo", "bar")
              .canonicalize(),
          )

        with self.subTest("add_new_named"):
          if style == "positional":
            result = (
                array.untag("bar")
                .at[pz.nx.arange("qux", 2) + 1]
                .add(100 + 100 * pz.nx.arange("qux", 2))
                .tag("bar")
            )
          else:
            result = array.at[{"bar": pz.nx.arange("qux", 2) + 1}].add(
                100 + 100 * pz.nx.arange("qux", 2)
            )
          chex.assert_trees_all_equal(
              result.canonicalize(),
              pz.nx.wrap(
                  jnp.array([
                      [
                          [0, 101, 2, 3],
                          [1, 102, 3, 4],
                          [2, 103, 4, 5],
                          [3, 104, 5, 6],
                          [4, 105, 6, 7],
                      ],
                      [
                          [0, 1, 202, 3],
                          [1, 2, 203, 4],
                          [2, 3, 204, 5],
                          [3, 4, 205, 6],
                          [4, 5, 206, 7],
                      ],
                  ])
              )
              .tag("qux", "foo", "bar")
              .canonicalize(),
          )

      with self.subTest("get"):
        result = (
            array.untag("foo")
            .at[jnp.array([1, 1, 2, 100])]
            .get(mode="fill", fill_value=-1)
            .tag("baz")
        )
        chex.assert_trees_all_equal(
            result.canonicalize(),
            pz.nx.wrap(
                jnp.array([
                    [1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [2, 3, 4, 5],
                    [-1, -1, -1, -1],
                ])
            )
            .tag("baz", "bar")
            .canonicalize(),
        )

  def test_complicated_dict_indexed_update(self):
    # Build an array with every type of axis that needs special treatment, and
    # make sure that dict-indexing semantics agrees with
    # nmapped-positional-indexing semantics.
    array_shape = {
        "indexed_scalar": 2,
        "indexed_unnamed": 3,
        "indexed_named": 3,
        "sliced": 3,
        "common": 2,
        "lhs_only": 2,
        "positional_1": 2,
        "positional_2": 2,
    }
    array = (
        pz.nx.wrap(jnp.arange(np.prod(list(array_shape.values()))))
        .reshape((*array_shape.values(),))
        .tag(*array_shape.keys())
        .untag("positional_1", "positional_2")
    )
    index_arr_unnamed = jnp.array([2, 0, 1])
    index_arr_named = pz.nx.wrap(
        jnp.array([
            [0, 0, 1],
            [2, 1, 2],
        ])
    ).tag_prefix("common")

    indexer = {
        "indexed_scalar": 0,
        "sliced": pz.slice[1:],
        "indexed_unnamed": index_arr_unnamed,
        "new": None,
        "indexed_named": index_arr_named,
    }

    with self.subTest("get"):
      result = array[indexer]
      reference = array.untag_prefix(
          "sliced", "indexed_scalar", "indexed_unnamed", "indexed_named"
      )[
          None, pz.slice[1:], 0, index_arr_unnamed, index_arr_named, :, :
      ].tag_prefix(
          "new", "sliced"
      )
      chex.assert_trees_all_equal(
          result.canonicalize(), reference.canonicalize()
      )

    with self.subTest("add_full"):
      update = pz.nx.wrap(
          jnp.arange(2 * 2 * 3 * 2 * 2).reshape((2, 2, 3, 2, 2))
      ).tag_prefix("sliced", "common")[{"new": None}]
      result = array.at[indexer].add(update)

      reference = (
          array.untag_prefix(
              "sliced", "indexed_scalar", "indexed_unnamed", "indexed_named"
          )
          .at[None, pz.slice[1:], 0, index_arr_unnamed, index_arr_named, :, :]
          .add(update.untag_prefix("new", "sliced"))
          .tag_prefix(
              "sliced", "indexed_scalar", "indexed_unnamed", "indexed_named"
          )
      )
      chex.assert_trees_all_equal(
          result.canonicalize(), reference.canonicalize()
      )

    with self.subTest("add_broadcasting_1"):
      update = pz.nx.wrap(jnp.arange(2 * 2).reshape((2, 2))).tag_prefix(
          "common"
      )
      result = array.at[indexer].add(update)

      reference = (
          array.untag_prefix(
              "sliced", "indexed_scalar", "indexed_unnamed", "indexed_named"
          )
          .at[None, pz.slice[1:], 0, index_arr_unnamed, index_arr_named, :, :]
          .add(update)
          .tag_prefix(
              "sliced", "indexed_scalar", "indexed_unnamed", "indexed_named"
          )
      )
      chex.assert_trees_all_equal(
          result.canonicalize(), reference.canonicalize()
      )

    with self.subTest("add_broadcasting_2"):
      update = pz.nx.wrap(jnp.arange(2 * 2 * 2).reshape((2, 2, 2))).tag_prefix(
          "sliced", "common"
      )
      result = array.at[indexer].add(update)

      reference = (
          array.untag_prefix(
              "sliced", "indexed_scalar", "indexed_unnamed", "indexed_named"
          )
          .at[None, pz.slice[1:], 0, index_arr_unnamed, index_arr_named, :, :]
          # The "sliced" axis has to line up with where it would be in the
          # result, and the positional axis has to line up with the *last*
          # positional axis to follow Numpy broadcasting rules.
          .add(update.untag_prefix("sliced")[None, :, None, None, :])
          .tag_prefix(
              "sliced", "indexed_scalar", "indexed_unnamed", "indexed_named"
          )
      )
      chex.assert_trees_all_equal(
          result.canonicalize(), reference.canonicalize()
      )

  def test_scan(self):
    carry = {
        "foo": jnp.array([1, 2, 3]),
        "bar": pz.nx.wrap(jnp.ones([2, 3, 4])).tag_prefix("a", "b"),
    }
    xs = {
        "baz": pz.nx.wrap(jnp.ones([2, 7])).tag("a", "seq"),
        "qux": pz.nx.wrap(jnp.ones([2, 7, 3])).tag_prefix("a", "seq"),
    }

    def f(carry, x):
      self.assertIsInstance(carry["foo"], jax.Array)
      self.assertEqual(carry["foo"].shape, (3,))

      self.assertIsInstance(carry["bar"], pz.nx.NamedArrayBase)
      self.assertEqual(carry["bar"].named_shape, {"a": 2, "b": 3})
      self.assertEqual(carry["bar"].positional_shape, (4,))

      self.assertIsInstance(x["baz"], pz.nx.NamedArrayBase)
      self.assertEqual(x["baz"].named_shape, {"a": 2})
      self.assertEqual(x["baz"].positional_shape, ())

      self.assertIsInstance(x["qux"], pz.nx.NamedArrayBase)
      self.assertEqual(x["qux"].named_shape, {"a": 2})
      self.assertEqual(x["qux"].positional_shape, (3,))

      # Intentionally mess up the axis ordering:
      new_carry = {
          "foo": jnp.array([1, 2, 3]),
          "bar": pz.nx.wrap(jnp.ones([3, 2, 4])).tag_prefix("b", "a"),
      }
      new_output = {
          "out_arr": jnp.zeros((5,)),
          "out_narr": pz.nx.wrap(jnp.ones([4, 5])).tag_prefix("c"),
      }
      return new_carry, new_output

    final_carry, stacked_out = pz.nx.scan(f, "seq", carry, xs)

    self.assertIsInstance(final_carry["foo"], jax.Array)
    self.assertEqual(final_carry["foo"].shape, (3,))

    self.assertIsInstance(final_carry["bar"], pz.nx.NamedArrayBase)
    self.assertEqual(final_carry["bar"].named_shape, {"a": 2, "b": 3})
    self.assertEqual(final_carry["bar"].positional_shape, (4,))

    self.assertIsInstance(stacked_out["out_arr"], pz.nx.NamedArrayBase)
    self.assertEqual(stacked_out["out_arr"].named_shape, {"seq": 7})
    self.assertEqual(stacked_out["out_arr"].positional_shape, (5,))

    self.assertIsInstance(stacked_out["out_narr"], pz.nx.NamedArrayBase)
    self.assertEqual(stacked_out["out_narr"].named_shape, {"seq": 7, "c": 4})
    self.assertEqual(stacked_out["out_narr"].positional_shape, (5,))


if __name__ == "__main__":
  absltest.main()
