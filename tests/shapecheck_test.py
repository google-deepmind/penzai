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

"""Tests for shapecheck."""

import textwrap

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from penzai import pz


class ShapecheckTest(absltest.TestCase):

  def test_same_empty(self):
    match = pz.chk.check_structure(value=[], pattern=[])
    self.assertEqual(dict(match), {})

  def test_same_objects(self):
    match = pz.chk.check_structure(
        value=["a", (1, 2, 3), np.array([3, 5]), ("foo", "bar")],
        pattern=["a", (1, 2, 3), np.array([3, 5]), pz.chk.ANY],
    )
    self.assertEqual(dict(match), {})

  def test_object_mismatches(self):
    err = textwrap.dedent("""\
    Mismatch while checking structures:
    At root[0]: Value 'a' was not equal to the non-ArraySpec pattern 'b'.
    At root[1][0]: Value 1 was not equal to the non-ArraySpec pattern 3.
    At root[1][2]: Value 3 was not equal to the non-ArraySpec pattern 1.
    At root[2]: Value array([3, 5]) was not equal to the non-ArraySpec pattern array([3, 6]).
    At root[3]: Value ('foo', 'bar') was not equal to the non-ArraySpec pattern 100.
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value=["a", (1, 2, 3), np.array([3, 5]), ("foo", "bar")],
          pattern=["b", (3, 2, 1), np.array([3, 6]), 100],
      )

  def test_simple_array(self):
    match = pz.chk.check_structure(
        value={"a": jax.ShapeDtypeStruct(shape=(1, 2, 3), dtype=jnp.float32)},
        pattern={"a": pz.chk.ArraySpec(shape=(1, 2, 3))},
    )
    self.assertEqual(dict(match), {})

  def test_array_structure_wrong(self):
    err = textwrap.dedent("""\
    Mismatch while checking structures:
    At root: Couldn't match an ArraySpec with a non-arraylike value {'a': ShapeDtypeStruct(shape=(1, 2, 2), dtype=float32)}
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value={"a": jax.ShapeDtypeStruct(shape=(1, 2, 2), dtype=jnp.float32)},
          pattern=pz.chk.ArraySpec(shape=(1, 2, 3)),
      )

  def test_bad_shapes_dtypes(self):
    err = textwrap.dedent("""\
    Mismatch while checking structures:
    At root['b']: Value has the wrong dtype: expected a sub-dtype of <class 'numpy.floating'> but got dtype int32.
    At root['a']: Positional shape mismatch between value (1, 2, 2) and pattern (1, 2, 3):
      Dim 2: Actual size 2 does not match expected 3
    At root['c']: Positional shape (1, 2, 3) had different length than pattern (1, 2, 3, 4)
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value={
              "a": jax.ShapeDtypeStruct(shape=(1, 2, 2), dtype=jnp.float32),
              "b": jax.ShapeDtypeStruct(shape=(6, 3), dtype=jnp.int32),
              "c": jax.ShapeDtypeStruct(shape=(1, 2, 3), dtype=jnp.float32),
          },
          pattern={
              "a": pz.chk.ArraySpec(shape=(1, 2, 3)),
              "b": pz.chk.ArraySpec(shape=(6, 3), dtype=np.floating),
              "c": pz.chk.ArraySpec(shape=(1, 2, 3, 4), dtype=jnp.float32),
          },
      )

  def test_simple_named(self):
    match = pz.chk.check_structure(
        value={
            "a": pz.nx.zeros(
                {"foo": 3, "bar": 4, "baz": 5}, dtype=jnp.float32
            ).untag("baz"),
        },
        pattern={
            "a": pz.chk.ArraySpec(shape=(5,), named_shape={"foo": 3, "bar": 4})
        },
    )
    self.assertEqual(dict(match), {})

  def test_bad_named_shapes(self):
    err = textwrap.dedent("""\
    Mismatch while checking structures:
    At root['b']: Value has the wrong dtype: expected a sub-dtype of <class 'numpy.floating'> but got dtype int32.
    At root['a']: Positional shape (5,) had different length than pattern ()
    At root['c']: Named shape mismatch between value {'foo': 3, 'bar': 4, 'baz': 5} and pattern {'foo': 3, 'bar': 4, 'qux': 6}:
      Axis 'qux': Expected to be present but was missing
      Unexpected names in value's named shape: ['baz']
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value={
              "a": pz.nx.zeros(
                  {"foo": 3, "bar": 4, "baz": 5}, dtype=jnp.float32
              ).untag("baz"),
              "b": pz.nx.zeros({"foo": 3, "bar": 4, "baz": 5}, dtype=jnp.int32),
              "c": pz.nx.zeros(
                  {"foo": 3, "bar": 4, "baz": 5}, dtype=jnp.float32
              ),
          },
          pattern={
              "a": pz.chk.ArraySpec(named_shape={"foo": 3, "bar": 4}),
              "b": pz.chk.ArraySpec(
                  named_shape={"foo": 3, "bar": 4, "baz": 5}, dtype=np.floating
              ),
              "c": pz.chk.ArraySpec(
                  named_shape={"foo": 3, "bar": 4, "qux": 6}, dtype=jnp.float32
              ),
          },
      )

  def test_solve_dimension_vars(self):
    match = pz.chk.check_structure(
        value={
            "a": jax.ShapeDtypeStruct(shape=(1, 2, 3), dtype=jnp.float32),
            "b": jax.ShapeDtypeStruct(shape=(4, 3), dtype=jnp.int32),
            "c": jax.ShapeDtypeStruct(
                shape=(6, 7, 8, 9, 10), dtype=jnp.float32
            ),
            "d": jax.ShapeDtypeStruct(shape=(11, 12, 13), dtype=jnp.float32),
            "e": jax.ShapeDtypeStruct(shape=(11, 12, 13), dtype=jnp.float32),
            "f": jax.ShapeDtypeStruct(shape=(5, 5, 5), dtype=jnp.float32),
            "g": jax.ShapeDtypeStruct(shape=(6, 7), dtype=jnp.float32),
        },
        pattern={
            "a": pz.chk.ArraySpec(shape=(pz.chk.var("a"), 2, pz.chk.var("c"))),
            "b": pz.chk.ArraySpec(shape=(4, pz.chk.var("b"))),
            "c": pz.chk.ArraySpec(
                shape=(6, *pz.chk.var("c1"), 9, pz.chk.var("c2")),
                dtype=jnp.float32,
            ),
            "d": pz.chk.ArraySpec(
                shape=(*pz.chk.var("d1"), 11, 12, 13), dtype=jnp.float32
            ),
            "e": pz.chk.ArraySpec(
                shape=(11, 12, 13, *pz.chk.var("d2")), dtype=jnp.float32
            ),
            "f": pz.chk.ArraySpec(
                shape=(pz.chk.var("f"), pz.chk.var("f"), pz.chk.var("f")),
                dtype=jnp.float32,
            ),
            "g": pz.chk.ArraySpec(shape=pz.chk.var("g"), dtype=jnp.float32),
        },
    )
    self.assertEqual(
        dict(match),
        {
            "a": 1,
            "c": 3,
            "b": 3,
            "c1": (7, 8),
            "c2": 10,
            "d1": (),
            "d2": (),
            "f": 5,
            "g": (6, 7),
        },
    )

  def test_mismached_positional_dimension_vars(self):
    err = textwrap.dedent("""\
    Mismatch while checking structures:
    At root['a']: Positional shape mismatch between value (3, 4, 5) and pattern (var('x'), 4, var('x')):
      Dim 2: Size 5 does not match previous size 3 for var('x') from root['a']
    At root['b']: Positional shape mismatch between value (5, 6) and pattern (var('x'), 6):
      Dim 0: Size 5 does not match previous size 3 for var('x') from root['a']
    At root['d']: Positional shape mismatch between value (5, 6, 7) and pattern (var('y')[0]:=2, var('y')[1]:=3, 7):
      Dim 0: Size 5 does not match previous size 2 for var('y')[0]:=2 from root['c']
      Dim 1: Size 6 does not match previous size 3 for var('y')[1]:=3 from root['c']
      After inlining var('y') = (2, 3) from root['c']
    At root['e']: Positional shape (5, 6, 7) had different length than pattern (var('y')[0]:=2, var('y')[1]:=3)
      After inlining var('y') = (2, 3) from root['c']
    At root['f']: Positional shape (1, 2, 3, 4) was shorter than wildcard pattern (1, 2, 3, *var('z'), 3, 4)
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value={
              "a": jax.ShapeDtypeStruct(shape=(3, 4, 5), dtype=jnp.float32),
              "b": jax.ShapeDtypeStruct(shape=(5, 6), dtype=jnp.float32),
              "c": jax.ShapeDtypeStruct(shape=(1, 2, 3, 4), dtype=jnp.float32),
              "d": jax.ShapeDtypeStruct(shape=(5, 6, 7), dtype=jnp.float32),
              "e": jax.ShapeDtypeStruct(shape=(5, 6, 7), dtype=jnp.float32),
              "f": jax.ShapeDtypeStruct(shape=(1, 2, 3, 4), dtype=jnp.float32),
          },
          pattern={
              "a": pz.chk.ArraySpec(
                  shape=(pz.chk.var("x"), 4, pz.chk.var("x"))
              ),
              "b": pz.chk.ArraySpec(shape=(pz.chk.var("x"), 6)),
              "c": pz.chk.ArraySpec(shape=(1, *pz.chk.var("y"), 4)),
              "d": pz.chk.ArraySpec(shape=(*pz.chk.var("y"), 7)),
              "e": pz.chk.ArraySpec(shape=pz.chk.var("y")),
              "f": pz.chk.ArraySpec(shape=(1, 2, 3, *pz.chk.var("z"), 3, 4)),
          },
      )

  def test_multiple_unpack_iterative_solve(self):
    match = pz.chk.check_structure(
        value={
            "a": jax.ShapeDtypeStruct(shape=(1, 2, 3, 4), dtype=jnp.float32),
            "b": jax.ShapeDtypeStruct(
                shape=(5, 6, 7, 3, 4, 5, 6, 7), dtype=jnp.float32
            ),
            "c": jax.ShapeDtypeStruct(
                shape=(1, 5, 6, 7, 10), dtype=jnp.float32
            ),
        },
        pattern={
            "a": pz.chk.ArraySpec(shape=(*pz.chk.var("x"), *pz.chk.var("y"))),
            "b": pz.chk.ArraySpec(
                shape=(*pz.chk.var("z"), *pz.chk.var("y"), *pz.chk.var("z"))
            ),
            "c": pz.chk.ArraySpec(shape=(1, *pz.chk.var("z"), 10)),
        },
    )
    self.assertEqual(dict(match), {"z": (5, 6, 7), "y": (3, 4), "x": (1, 2)})

  def test_positional_unpack_stuck(self):
    err = textwrap.dedent("""\
    Could not solve for all variable constraints. This usually means the variable assignment is ambiguous because multiple unpack patterns (*var(...) or **var(...)) appeared in the same positional or named shape.
    Unsolvable variables: var('x'), var('y'), var('z')
    Unsolvable constraints:
      (1, 2, 3, 4) == (*var('x'), *var('y')) from root['a']
      (5, 6, 7, 3, 4, 5, 6, 7) == (*var('z'), *var('y'), *var('z')) from root['b']
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value={
              "a": jax.ShapeDtypeStruct(shape=(1, 2, 3, 4), dtype=jnp.float32),
              "b": jax.ShapeDtypeStruct(
                  shape=(5, 6, 7, 3, 4, 5, 6, 7), dtype=jnp.float32
              ),
          },
          pattern={
              "a": pz.chk.ArraySpec(shape=(*pz.chk.var("x"), *pz.chk.var("y"))),
              "b": pz.chk.ArraySpec(
                  shape=(*pz.chk.var("z"), *pz.chk.var("y"), *pz.chk.var("z"))
              ),
          },
      )

  def test_early_error_remaining_unsolved(self):
    err = textwrap.dedent("""\
    Mismatch while checking structures:
    At root['d']: Positional shape (1, 2, 3) had different length than pattern (4,)
    There may also be additional mismatches due to remaining unsolved constraints.
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value={
              "a": jax.ShapeDtypeStruct(shape=(1, 2, 3, 4), dtype=jnp.float32),
              "b": jax.ShapeDtypeStruct(
                  shape=(5, 6, 7, 3, 4, 5, 6, 7), dtype=jnp.float32
              ),
              "c": jax.ShapeDtypeStruct(
                  shape=(1, 5, 6, 7, 10), dtype=jnp.float32
              ),
              "d": jax.ShapeDtypeStruct(shape=(1, 2, 3), dtype=jnp.float32),
          },
          pattern={
              "a": pz.chk.ArraySpec(shape=(*pz.chk.var("x"), *pz.chk.var("y"))),
              "b": pz.chk.ArraySpec(
                  shape=(*pz.chk.var("z"), *pz.chk.var("y"), *pz.chk.var("z"))
              ),
              "c": pz.chk.ArraySpec(shape=(1, *pz.chk.var("z"), 10)),
              "d": pz.chk.ArraySpec(shape=(4,)),
          },
      )

  def test_named_axis_dimension_variables(self):
    match = pz.chk.check_structure(
        value={
            "a": pz.chk.ArraySpec(
                named_shape={"a": 1, "b": 2, "c": 3}, dtype=jnp.float32
            ),
            "b": pz.chk.ArraySpec(
                named_shape={"a": 4, "b": 3}, dtype=jnp.int32
            ),
            "c": pz.chk.ArraySpec(
                named_shape={"a": 6, "b": 7, "c": 8, "d": 9, "e": 10},
                dtype=jnp.float32,
            ),
            "d": pz.chk.ArraySpec(
                named_shape={"a": 11, "b": 12, "c": 13}, dtype=jnp.float32
            ),
            "e": pz.chk.ArraySpec(
                named_shape={"a": 11, "b": 12, "c": 13}, dtype=jnp.float32
            ),
            "f": pz.chk.ArraySpec(
                named_shape={"a": 5, "b": 5, "c": 5}, dtype=jnp.float32
            ),
        },
        pattern={
            "a": pz.chk.ArraySpec(
                named_shape={"a": pz.chk.var("a"), "b": 2, "c": pz.chk.var("c")}
            ),
            "b": pz.chk.ArraySpec(named_shape={"a": 4, "b": pz.chk.var("b")}),
            "c": pz.chk.ArraySpec(
                named_shape={
                    "a": 6,
                    "d": 9,
                    "e": pz.chk.var("c2"),
                    **pz.chk.var("c1"),
                },
                dtype=jnp.float32,
            ),
            "d": pz.chk.ArraySpec(
                named_shape={**pz.chk.var("d"), "a": 11, "b": 12, "c": 13},
                dtype=jnp.float32,
            ),
            "e": pz.chk.ArraySpec(
                named_shape={**pz.chk.var("e")}, dtype=jnp.float32
            ),
            "f": pz.chk.ArraySpec(
                named_shape={
                    "a": pz.chk.var("f"),
                    "b": pz.chk.var("f"),
                    "c": pz.chk.var("f"),
                },
                dtype=jnp.float32,
            ),
        },
    )
    self.assertEqual(
        dict(match),
        {
            "c": 3,
            "a": 1,
            "b": 3,
            "c2": 10,
            "c1": {"c": 8, "b": 7},
            "d": {},
            "e": {"c": 13, "b": 12, "a": 11},
            "f": 5,
        },
    )

  def test_named_unpack_substitution_conflict(self):
    err = textwrap.dedent("""\
    Mismatch while checking structures:
    At root['c']: Key conflict while substituting unpacked variable var('x') for known named sub-shape {'c': 3, 'd': 4} from root['b']: name c is already present in the shape {'a': 1, 'c': 3, **var('x')}.
    At root['d']: Key conflict while substituting unpacked variable var('x') for known named sub-shape {'c': 3, 'd': 4} from root['b']: name c is already present in the shape {**var('u'), **var('x')}.
      After inlining var('u') = {'a': 1, 'c': 3} from root['a']
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value={
              "a": pz.chk.ArraySpec(
                  named_shape={"a": 1, "b": 2, "c": 3, "d": 4}
              ),
              "b": pz.chk.ArraySpec(
                  named_shape={"a": 1, "b": 2, "c": 3, "d": 4}
              ),
              "c": pz.chk.ArraySpec(
                  named_shape={"a": 1, "b": 2, "c": 3, "d": 4}
              ),
              "d": pz.chk.ArraySpec(
                  named_shape={"a": 1, "b": 2, "c": 3, "d": 4}
              ),
          },
          pattern={
              "a": pz.chk.ArraySpec(
                  named_shape={**pz.chk.var("u"), "b": 2, "d": 4}
              ),
              "b": pz.chk.ArraySpec(
                  named_shape={"a": 1, "b": 2, **pz.chk.var("x")}
              ),
              "c": pz.chk.ArraySpec(
                  named_shape={"a": 1, "c": 3, **pz.chk.var("x")}
              ),
              "d": pz.chk.ArraySpec(
                  named_shape={**pz.chk.var("u"), **pz.chk.var("x")}
              ),
          },
      )

  def test_named_axis_inconsistent_shapes(self):
    err = textwrap.dedent("""\
    Mismatch while checking structures:
    At root['a']: Named shape mismatch between value {'a': 3, 'b': 4, 'c': 7} and pattern {'a': var('x'), 'b': 4, 'c': var('x')}:
      Axis 'c': Size 7 does not match previous size 3 for var('x') from root['a']
    At root['b']: Named shape mismatch between value {'a': 5, 'b': 6} and pattern {'a': var('x'), 'b': 6}:
      Axis 'a': Size 5 does not match previous size 3 for var('x') from root['a']
    At root['d']: Named shape mismatch between value {'b': 6, 'c': 7, 'd': 4} and pattern {'d': 4, 'b': var('y')['b']:=2, 'c': var('y')['c']:=3}:
      Axis 'b': Size 6 does not match previous size 2 for var('y')['b']:=2 from root['c']
      Axis 'c': Size 7 does not match previous size 3 for var('y')['c']:=3 from root['c']
      After inlining var('y') = {'b': 2, 'c': 3} from root['c']
    At root['e']: Named shape mismatch between value {'b': 6, 'c': 7, 'd': 4} and pattern {'b': var('y')['b']:=2, 'c': var('y')['c']:=3}:
      Axis 'b': Size 2 does not match previous size 6 for var('y')['b']:=2 from root['d']
      Axis 'c': Size 3 does not match previous size 7 for var('y')['c']:=3 from root['d']
      Unexpected names in value's named shape: ['d']
      After inlining var('y') = {'b': 2, 'c': 3} from root['c']
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value={
              "a": pz.chk.ArraySpec(named_shape={"a": 3, "b": 4, "c": 7}),
              "b": pz.chk.ArraySpec(named_shape={"a": 5, "b": 6}),
              "c": pz.chk.ArraySpec(
                  named_shape={"a": 1, "b": 2, "c": 3, "d": 4}
              ),
              "d": pz.chk.ArraySpec(named_shape={"b": 6, "c": 7, "d": 4}),
              "e": pz.chk.ArraySpec(named_shape={"b": 6, "c": 7, "d": 4}),
          },
          pattern={
              "a": pz.chk.ArraySpec(
                  named_shape={
                      "a": pz.chk.var("x"),
                      "b": 4,
                      "c": pz.chk.var("x"),
                  }
              ),
              "b": pz.chk.ArraySpec(named_shape={"a": pz.chk.var("x"), "b": 6}),
              "c": pz.chk.ArraySpec(
                  named_shape={"a": 1, **pz.chk.var("y"), "d": 4}
              ),
              "d": pz.chk.ArraySpec(named_shape={**pz.chk.var("y"), "d": 4}),
              "e": pz.chk.ArraySpec(named_shape={**pz.chk.var("y")}),
          },
      )

  def test_named_multiple_unpack_iterative_solve(self):
    match = pz.chk.check_structure(
        value={
            "a": pz.chk.ArraySpec(named_shape={"a": 1, "b": 2, "c": 3, "d": 4}),
            "b": pz.chk.ArraySpec(named_shape={"a": 1, "b": 2, "e": 5, "f": 6}),
            "c": pz.chk.ArraySpec(named_shape={"e": 5, "f": 6}),
        },
        pattern={
            "a": pz.chk.ArraySpec(
                named_shape={**pz.chk.var("x"), **pz.chk.var("y")}
            ),
            "b": pz.chk.ArraySpec(
                named_shape={**pz.chk.var("z"), **pz.chk.var("y")}
            ),
            "c": pz.chk.ArraySpec(named_shape=pz.chk.var("z")),
        },
    )
    self.assertEqual(
        dict(match),
        {
            "z": {"f": 6, "e": 5},
            "y": {"b": 2, "a": 1},
            "x": {"d": 4, "c": 3},
        },
    )

  def test_named_unpack_stuck(self):
    err = textwrap.dedent("""\
    Could not solve for all variable constraints. This usually means the variable assignment is ambiguous because multiple unpack patterns (*var(...) or **var(...)) appeared in the same positional or named shape.
    Unsolvable variables: var('x'), var('y'), var('z')
    Unsolvable constraints:
      {'a': 1, 'b': 2, 'c': 3, 'd': 4} == {**var('x'), **var('y')} from root['a']
      {'a': 1, 'b': 2, 'e': 5, 'f': 6} == {**var('z'), **var('y')} from root['b']
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value={
              "a": pz.chk.ArraySpec(
                  named_shape={"a": 1, "b": 2, "c": 3, "d": 4}
              ),
              "b": pz.chk.ArraySpec(
                  named_shape={"a": 1, "b": 2, "e": 5, "f": 6}
              ),
          },
          pattern={
              "a": pz.chk.ArraySpec(
                  named_shape={**pz.chk.var("x"), **pz.chk.var("y")}
              ),
              "b": pz.chk.ArraySpec(
                  named_shape={**pz.chk.var("z"), **pz.chk.var("y")}
              ),
          },
      )

  def test_vars_for_axes(self):
    match = pz.chk.check_structure(
        value={
            "a": pz.chk.ArraySpec(named_shape={"a": 1, "b": 2, "c": 3, "d": 4}),
            "b": pz.chk.ArraySpec(named_shape={"a": 1, "b": 2, "c": 3, "d": 4}),
        },
        pattern={
            "a": pz.chk.ArraySpec(
                named_shape={
                    "a": 1,
                    **pz.chk.vars_for_axes("x", {"b": 2, "c": None, "d": None}),
                }
            ),
            "b": pz.chk.ArraySpec(
                named_shape={
                    "a": 1,
                    **pz.chk.vars_for_axes("y", ("b", "c", "d")),
                }
            ),
        },
    )
    self.assertEqual(
        dict(match),
        {
            "x": {"d": 4, "c": 3, "b": 2},
            "y": {"d": 4, "c": 3, "b": 2},
        },
    )

  def test_mismatched_vars_for_axes(self):
    err = textwrap.dedent("""\
    Mismatch while checking structures:
    At root['a']: Named shape mismatch between value {'a': 1, 'b': 2, 'c': 3, 'd': 4} and pattern {'a': 1, 'b': var('x')['b']:=4, 'c': var('x')['c'], 'd': var('x')['d']}:
      Axis 'b': Actual size 2 does not match expected var('x')['b']:=4
    At root['c']: Named shape mismatch between value {'a': 1, 'b': 2, 'c': 3, 'd': 5} and pattern {'a': 1, 'b': var('y')['b'], 'c': var('y')['c'], 'd': var('y')['d']}:
      Axis 'd': Size 5 does not match previous size 4 for var('y')['d'] from root['b']
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value={
              "a": pz.chk.ArraySpec(
                  named_shape={"a": 1, "b": 2, "c": 3, "d": 4}
              ),
              "b": pz.chk.ArraySpec(
                  named_shape={"a": 1, "b": 2, "c": 3, "d": 4}
              ),
              "c": pz.chk.ArraySpec(
                  named_shape={"a": 1, "b": 2, "c": 3, "d": 5}
              ),
          },
          pattern={
              "a": pz.chk.ArraySpec(
                  named_shape={
                      "a": 1,
                      **pz.chk.vars_for_axes(
                          "x", {"b": 4, "c": None, "d": None}
                      ),
                  }
              ),
              "b": pz.chk.ArraySpec(
                  named_shape={
                      "a": 1,
                      **pz.chk.vars_for_axes("y", ("b", "c", "d")),
                  }
              ),
              "c": pz.chk.ArraySpec(
                  named_shape={
                      "a": 1,
                      **pz.chk.vars_for_axes("y", ("b", "c", "d")),
                  }
              ),
          },
      )

  def test_consistent_overlap_vars_for_axes_and_unpack(self):
    match = pz.chk.check_structure(
        value={
            "a": pz.chk.ArraySpec(named_shape={"a": 1, "b": 2, "c": 3, "d": 4}),
            "b": pz.chk.ArraySpec(named_shape={"a": 1, "b": 2, "c": 3, "d": 4}),
        },
        pattern={
            "a": pz.chk.ArraySpec(
                named_shape={
                    "a": 1,
                    **pz.chk.vars_for_axes("x", {"b": 2, "c": None, "d": None}),
                }
            ),
            "b": pz.chk.ArraySpec(named_shape={"a": 1, **pz.chk.var("x")}),
        },
    )
    self.assertEqual(dict(match), {"x": {"d": 4, "c": 3, "b": 2}})

  def test_inconsistent_overlap_vars_for_axes_and_unpack(self):
    err = textwrap.dedent("""\
    Unexpected conflicts between variable assignments:
      var('x'): Solved as {'e': 2, 'f': 3, 'g': 4} from root['b'] but separately solved var('x')['b'] as 2 from root['a']
      var('x'): Solved as {'e': 2, 'f': 3, 'g': 4} from root['b'] but separately solved var('x')['c'] as 3 from root['a']
      var('x'): Solved as {'e': 2, 'f': 3, 'g': 4} from root['b'] but separately solved var('x')['d'] as 4 from root['a']
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value={
              "a": pz.chk.ArraySpec(
                  named_shape={"a": 1, "b": 2, "c": 3, "d": 4}
              ),
              "b": pz.chk.ArraySpec(
                  named_shape={"a": 1, "e": 2, "f": 3, "g": 4}
              ),
          },
          pattern={
              "a": pz.chk.ArraySpec(
                  named_shape={
                      "a": 1,
                      **pz.chk.vars_for_axes(
                          "x", {"b": 2, "c": None, "d": None}
                      ),
                  }
              ),
              "b": pz.chk.ArraySpec(named_shape={"a": 1, **pz.chk.var("x")}),
          },
      )

  def test_any_structure(self):
    match = pz.chk.check_structure(
        value={
            "a": pz.chk.ArraySpec(named_shape={"a": 1, "b": 2, "c": 3, "d": 4}),
            "b": [
                pz.chk.ArraySpec(named_shape={"a": 1, "b": 2, "c": 3, "d": 4}),
                3,
            ],
        },
        pattern={
            "a": pz.chk.ArraySpec(
                named_shape={**pz.chk.var("u"), "b": 2, "d": 4}
            ),
            "b": pz.chk.ANY,
        },
    )
    self.assertEqual(dict(match), {"u": {"c": 3, "a": 1}})

  def test_multi_match_inconsistent(self):
    match_1 = pz.chk.check_structure(
        value={
            "a": jax.ShapeDtypeStruct(shape=(3, 4, 5), dtype=jnp.float32),
        },
        pattern={
            "a": pz.chk.ArraySpec(shape=(pz.chk.var("x"), 4, 5)),
        },
    )
    err = textwrap.dedent("""\
    Mismatch while checking structures:
    At root['b']: Positional shape mismatch between value (5, 6) and pattern (var('x'), 6):
      Dim 0: Size 5 does not match previous size 3 for var('x') from the known variable assignments (argument `known_vars` to check_structure)
    """).rstrip()
    with self.assertRaisesWithLiteralMatch(pz.chk.StructureMismatchError, err):
      pz.chk.check_structure(
          value={
              "b": jax.ShapeDtypeStruct(shape=(5, 6), dtype=jnp.float32),
          },
          pattern={
              "b": pz.chk.ArraySpec(shape=(pz.chk.var("x"), 6)),
          },
          known_vars=match_1,
      )

  def test_multi_match_consistent(self):
    match_1 = pz.chk.check_structure(
        value={
            "a": jax.ShapeDtypeStruct(shape=(3, 4, 5), dtype=jnp.float32),
        },
        pattern={
            "a": pz.chk.ArraySpec(shape=(pz.chk.var("x"), pz.chk.var("y"), 5)),
        },
    )
    match_2 = pz.chk.check_structure(
        value={
            "b": jax.ShapeDtypeStruct(shape=(3, 6), dtype=jnp.float32),
        },
        pattern={
            "b": pz.chk.ArraySpec(shape=(pz.chk.var("x"), pz.chk.var("z"))),
        },
        known_vars=match_1,
    )
    self.assertEqual(dict(match_2), {"x": 3, "y": 4, "z": 6})

  def test_structure_into_pytree(self):
    structures = [
        pz.chk.ArraySpec(shape=(1, 2, 3), dtype=jnp.float32),
        pz.chk.ArraySpec(named_shape={"foo": 4, "bar": 5}, dtype=jnp.int64),
        pz.chk.ArraySpec(
            shape=(1, 2, 3), named_shape={"foo": 4, "bar": 5}, dtype=jnp.int32
        ),
    ]
    pytrees = [x.into_pytree() for x in structures]
    pz.chk.check_structure(value=pytrees, pattern=structures)

  def test_get_and_substitute_dimension_variables(self):
    struct = {
        "a": pz.chk.ArraySpec(shape=(1, 2, pz.chk.var("a"))),
        "b": pz.chk.ArraySpec(shape=(3, *pz.chk.var("b"))),
        "c": pz.chk.ArraySpec(named_shape={"foo": 4, "bar": pz.chk.var("c")}),
        "d": pz.chk.ArraySpec(named_shape={"foo": 4, **pz.chk.var("d")}),
        "e": pz.chk.ArraySpec(
            named_shape={
                "w": 1,
                **pz.chk.vars_for_axes("e", {"x": 2, "y": None, "z": None}),
            }
        ),
        "b2": pz.chk.ArraySpec(shape=(3, *pz.chk.var("b"))),
    }

    with self.subTest("get"):
      retrieved = pz.chk.get_dimension_variables(struct)
      self.assertEqual(
          retrieved,
          pz.chk.DimensionVariableSubstitution(
              size_variables={
                  "a": pz.chk.var("a"),
                  "c": pz.chk.var("c"),
                  ("e", "y"): pz.chk.var("e")["y"],
                  ("e", "z"): pz.chk.var("e")["z"],
              },
              sequence_variables={"b": (*pz.chk.var("b"),)},
              mapping_variables={"d": {**pz.chk.var("d")}},
          ),
      )

    with self.subTest("substitute"):
      substitutions = pz.chk.DimensionVariableSubstitution(
          size_variables={
              "a": 10,
              "c": 13,
              ("e", "y"): 16,
              ("e", "z"): 17,
          },
          sequence_variables={"b": (11, 12)},
          mapping_variables={"d": {"bar": 14, "baz": 15}},
      )
      res = pz.chk.full_substitute_dimension_variables(struct, substitutions)
      expected = {
          "a": pz.chk.ArraySpec(shape=(1, 2, 10)),
          "b": pz.chk.ArraySpec(shape=(3, 11, 12)),
          "c": pz.chk.ArraySpec(named_shape={"foo": 4, "bar": 13}),
          "d": pz.chk.ArraySpec(named_shape={"foo": 4, "bar": 14, "baz": 15}),
          "e": pz.chk.ArraySpec(named_shape={"w": 1, "x": 2, "y": 16, "z": 17}),
          "b2": pz.chk.ArraySpec(shape=(3, 11, 12)),
      }
      self.assertEqual(res, expected)


if __name__ == "__main__":
  absltest.main()
