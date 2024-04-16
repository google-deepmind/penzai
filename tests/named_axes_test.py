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
        result,
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
        result,
        named_axes.NamedArray(
            named_axes=collections.OrderedDict({"foo": 2, "bar": 3, "qux": 1}),
            data_array=np.array(
                [[[12], [7], [2]], [[27], [22], [17]]], dtype=np.dtype("int32")
            ),
        ),
    )

  def test_indexing_advanced(self):
    table = named_axes.wrap(jnp.arange(100 * 3).reshape((100, 3))).tag(
        "vocab", "features"
    )
    indices = named_axes.wrap(jnp.array([2, 61, 3, 10, 40])).tag("batch")

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

    with self.subTest("binop_scalar"):
      chex.assert_trees_all_equal(
          array_a - 100,
          named_axes.NamedArray(
              named_axes=collections.OrderedDict({"foo": 2, "bar": 3}),
              data_array=np.array(
                  [[-100, -99, -98], [-97, -96, -95]],
                  dtype=np.dtype("int32"),
              ),
          ),
      )
      chex.assert_trees_all_equal(
          100 - array_a,
          named_axes.NamedArray(
              named_axes=collections.OrderedDict({"foo": 2, "bar": 3}),
              data_array=np.array(
                  [[100, 99, 98], [97, 96, 95]], dtype=np.dtype("int32")
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

  def test_named_array_equality(self):
    array_a = named_axes.wrap(jnp.arange(2 * 3).reshape((2, 3))).tag(
        "foo", "bar"
    )
    array_b = named_axes.wrap(jnp.arange(5 * 2).reshape((5, 2))).tag(
        "baz", "foo"
    )

    result = array_a == array_b
    self.assertEqual(result.named_shape, {"foo": 2, "bar": 3, "baz": 5})

  def test_named_array_conversion(self):
    array_a = named_axes.wrap(jnp.arange(6))

    with self.subTest("convert_positional_to_numpy"):
      array_a_as_np = np.array(array_a)
      self.assertEqual(array_a_as_np.shape, (6,))

    with self.subTest("conversion_failure_with_names"):
      with self.assertRaisesRegex(
          ValueError,
          re.escape(
              "Only NamedArray(View)s with no named axes can be converted to"
              " numpy arrays"
          ),
      ):
        _ = np.array(array_a.tag("foo"))

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


if __name__ == "__main__":
  absltest.main()
