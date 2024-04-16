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

"""Tests for the treescope renderer and handlers."""

import ast
import collections
import functools
import textwrap
import types
from typing import Any, Callable
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from penzai import pz
import penzai.core.selectors
import penzai.core.struct
from tests.fixtures import treescope_examples_fixture as fixture_lib
from penzai.treescope import autovisualize
from penzai.treescope import default_renderer
from penzai.treescope import selection_rendering
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import layout_algorithms
from penzai.treescope.foldable_representation import part_interface
from penzai.treescope.handlers import function_reflection_handlers


class TreescopeRendererTest(parameterized.TestCase):

  def test_renderer_interface(self):
    renderer = default_renderer.active_renderer.get()

    rendering = renderer.to_text({"key": "value"})
    self.assertEqual(rendering, "{'key': 'value'}")

    rendering = renderer.to_html({"key": "value"})
    self.assertIsInstance(rendering, str)

    rendering = renderer.to_foldable_representation({"key": "value"})
    self.assertIsInstance(
        rendering, part_interface.RenderableAndLineAnnotations
    )

  def test_high_level_interface(self):
    rendering = default_renderer.render_to_text({"key": "value"})
    self.assertEqual(rendering, "{'key': 'value'}")

    rendering = default_renderer.render_to_html({"key": "value"})
    self.assertIsInstance(rendering, str)

  def test_error_recovery(self):
    def handler_that_crashes(node, path, subtree_renderer):
      del path, subtree_renderer
      if node == "trigger handler error":
        raise RuntimeError("handler error!")
      return NotImplemented

    def hook_that_crashes(node, path, node_renderer):
      del path, node_renderer
      if node == "trigger hook error":
        raise RuntimeError("hook error!")
      return NotImplemented

    renderer = default_renderer.active_renderer.get().extended_with(
        handlers=[handler_that_crashes], wrapper_hooks=[hook_that_crashes]
    )

    rendering = renderer.to_foldable_representation([1, 2, 3, "foo", 4])
    layout_algorithms.expand_to_depth(rendering.renderable, 1)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(
            basic_parts.build_full_line_with_annotations(rendering)
        ),
        "[\n  1,\n  2,\n  3,\n  'foo',\n  4,\n]",
    )

    with self.assertRaisesWithLiteralMatch(RuntimeError, "handler error!"):
      _ = renderer.to_foldable_representation(
          [1, 2, 3, "trigger handler error", 4]
      )

    with self.assertRaisesWithLiteralMatch(RuntimeError, "hook error!"):
      _ = renderer.to_foldable_representation(
          [1, 2, 3, "trigger hook error", 4]
      )

    with warnings.catch_warnings(record=True) as recorded:
      rendering = renderer.to_foldable_representation(
          [1, 2, 3, "trigger handler error", "trigger hook error", 4],
          ignore_exceptions=True,
      )
    layout_algorithms.expand_to_depth(rendering.renderable, 1)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(
            basic_parts.build_full_line_with_annotations(rendering)
        ),
        "[\n  1,\n  2,\n  3,\n  'trigger handler error',\n  'trigger hook"
        " error',\n  4,\n]",
    )
    self.assertLen(recorded, 2)
    self.assertEqual(type(recorded[0]), warnings.WarningMessage)
    self.assertContainsInOrder(
        [
            "Ignoring error while formatting value",
            "with",
            "<locals>.handler_that_crashes",
        ],
        str(recorded[0].message),
    )
    self.assertEqual(type(recorded[1]), warnings.WarningMessage)
    self.assertContainsInOrder(
        [
            "Ignoring error inside wrapper hook",
            "<locals>.hook_that_crashes",
        ],
        str(recorded[1].message),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="literals",
          target=[False, True, None, Ellipsis, NotImplemented],
          expected_collapsed="[False, True, None, Ellipsis, NotImplemented]",
      ),
      dict(
          testcase_name="numbers",
          target=[1234, 1234.0],
          expected_collapsed="[1234, 1234.0]",
      ),
      dict(
          testcase_name="strings",
          target=["some string", b"some bytes"],
          expected_collapsed="['some string', b'some bytes']",
      ),
      dict(
          testcase_name="multiline_string",
          target="some string\n    with \n newlines in it",
          expected_collapsed="'some string\\n    with \\n newlines in it'",
          expected_expanded=(
              "  'some string\\n'\n  '    with \\n'\n  ' newlines in it'\n"
          ),
      ),
      dict(
          testcase_name="enums",
          target=[
              fixture_lib.MyTestEnum.FOO,
              fixture_lib.MyTestEnum.BAR,
          ],
          expected_collapsed="[MyTestEnum.FOO, MyTestEnum.BAR]",
          expected_expanded=textwrap.dedent("""\
              [
                MyTestEnum.FOO,  # value: 1
                MyTestEnum.BAR,  # value: 2
              ]"""),
          expected_roundtrip=textwrap.dedent("""\
              [
                tests.fixtures.treescope_examples_fixture.MyTestEnum.FOO,  # value: 1
                tests.fixtures.treescope_examples_fixture.MyTestEnum.BAR,  # value: 2
              ]"""),
      ),
      dict(
          testcase_name="empty_dict",
          target={},
          expected_collapsed="{}",
      ),
      dict(
          testcase_name="dict_with_contents",
          target={"a": "b", (1, 2, 3, 4): (5, 6, 7, 8), (): (9, 10)},
          expected_collapsed=(
              "{'a': 'b', (1, 2, 3, 4): (5, 6, 7, 8), (): (9, 10)}"
          ),
          expected_expanded=textwrap.dedent("""\
              {
                'a': 'b',
                (1, 2, 3, 4):
                  (5, 6, 7, 8),
                (): (9, 10),
              }"""),
      ),
      dict(
          testcase_name="dict_subclass",
          target=collections.OrderedDict({"a": "b", (): (9, 10)}),
          expected_collapsed="OrderedDict({'a': 'b', (): (9, 10)})",
          expected_expanded=textwrap.dedent("""\
              OrderedDict({
                'a': 'b',
                (): (9, 10),
              })"""),
          expected_roundtrip=textwrap.dedent("""\
              collections.OrderedDict({
                'a': 'b',
                (): (9, 10),
              })"""),
      ),
      dict(
          testcase_name="tuple_empty",
          target=(),
          expected_collapsed="()",
      ),
      dict(
          testcase_name="tuple_singleton",
          target=(1,),
          expected_collapsed="(1,)",
          expected_expanded=textwrap.dedent("""\
              (
                1,
              )"""),
      ),
      dict(
          testcase_name="tuple_multiple",
          target=(1, 2, 3),
          expected_collapsed="(1, 2, 3)",
          expected_expanded=textwrap.dedent("""\
              (
                1,
                2,
                3,
              )"""),
      ),
      dict(
          testcase_name="list_empty",
          target=[],
          expected_collapsed="[]",
      ),
      dict(
          testcase_name="list_singleton",
          target=[1],
          expected_collapsed="[1]",
          expected_expanded=textwrap.dedent("""\
              [
                1,
              ]"""),
      ),
      dict(
          testcase_name="list_multiple",
          target=[1, 2, 3],
          expected_collapsed="[1, 2, 3]",
          expected_expanded=textwrap.dedent("""\
              [
                1,
                2,
                3,
              ]"""),
      ),
      dict(
          testcase_name="sets_empty",
          target=[set(), frozenset()],
          expected_collapsed="[set(), frozenset({})]",
          expected_expanded=textwrap.dedent("""\
              [
                set(),
                frozenset({}),
              ]"""),
      ),
      dict(
          testcase_name="set_with_items",
          target={1, 2, 3},
          expected_collapsed="{1, 2, 3}",
          expected_expanded=textwrap.dedent("""\
              {
                1,
                2,
                3,
              }"""),
      ),
      dict(
          testcase_name="frozenset_with_items",
          target=frozenset({1, 2, 3}),
          expected_collapsed="frozenset({1, 2, 3})",
          expected_expanded=textwrap.dedent("""\
              frozenset({
                1,
                2,
                3,
              })"""),
      ),
      dict(
          testcase_name="simplenamespace",
          target=types.SimpleNamespace(foo="bar", baz="qux"),
          expected_collapsed="SimpleNamespace(foo='bar', baz='qux')",
          expected_expanded=textwrap.dedent("""\
              SimpleNamespace(
                foo='bar',
                baz='qux',
              )"""),
      ),
      dict(
          testcase_name="nametuple",
          target=fixture_lib.SomeNamedtupleClass(foo="baz", bar="qux"),
          expected_collapsed="SomeNamedtupleClass(foo='baz', bar='qux')",
          expected_expanded=textwrap.dedent("""\
              SomeNamedtupleClass(
                foo='baz',
                bar='qux',
              )"""),
          expected_roundtrip=textwrap.dedent("""\
              tests.fixtures.treescope_examples_fixture.SomeNamedtupleClass(
                foo='baz',
                bar='qux',
              )"""),
      ),
      dict(
          testcase_name="dataclass",
          target=fixture_lib.DataclassWithTwoChildren(foo="baz", bar="qux"),
          expected_collapsed="DataclassWithTwoChildren(foo='baz', bar='qux')",
          expected_expanded=textwrap.dedent("""\
              DataclassWithTwoChildren(
                foo='baz',
                bar='qux',
              )"""),
          expected_roundtrip=textwrap.dedent("""\
              tests.fixtures.treescope_examples_fixture.DataclassWithTwoChildren(
                foo='baz',
                bar='qux',
              )"""),
      ),
      dict(
          testcase_name="dataclass_empty",
          target=fixture_lib.EmptyDataclass(),
          expected_collapsed="EmptyDataclass()",
          expected_expanded="EmptyDataclass()",
          expected_roundtrip=(
              "tests.fixtures.treescope_examples_fixture.EmptyDataclass()"
          ),
      ),
      dict(
          testcase_name="penzai_layer",
          target=fixture_lib.ExampleLayer(100),
          expected_collapsed="ExampleLayer(foo=100)",
          expected_expanded=textwrap.dedent("""\
              ExampleLayer(
                foo=100,
              )"""),
          expected_roundtrip=textwrap.dedent("""\
              tests.fixtures.treescope_examples_fixture.ExampleLayer(
                foo=100,
              )"""),
      ),
      dict(
          testcase_name="ndarray_small",
          target=np.array([1, 2, 4, 8, 16]),
          expected_collapsed="<numpy.array([ 1,  2,  4,  8, 16])>",
          expected_expanded="<numpy.array([ 1,  2,  4,  8, 16])>",
      ),
      dict(
          testcase_name="ndarray_large",
          target=np.arange(3 * 7).reshape((3, 7)),
          expected_collapsed=(
              "<numpy.ndarray int64(3, 7) [≥0, ≤20] zero:1 nonzero:20>"
          ),
          expected_expanded=textwrap.dedent("""\
              # numpy.ndarray int64(3, 7) [≥0, ≤20] zero:1 nonzero:20
                array([[ 0,  1,  2,  3,  4,  5,  6],
                       [ 7,  8,  9, 10, 11, 12, 13],
                       [14, 15, 16, 17, 18, 19, 20]])
              """),
      ),
      dict(
          testcase_name="jax_array_large",
          target_builder=lambda: jnp.arange(3 * 7).reshape((3, 7)),
          expected_collapsed=(
              "<jax.Array int32(3, 7) [≥0, ≤20] zero:1 nonzero:20>"
          ),
          expected_expanded=textwrap.dedent("""\
              # jax.Array int32(3, 7) [≥0, ≤20] zero:1 nonzero:20
                Array([[ 0,  1,  2,  3,  4,  5,  6],
                       [ 7,  8,  9, 10, 11, 12, 13],
                       [14, 15, 16, 17, 18, 19, 20]], dtype=int32)
              """),
      ),
      dict(
          testcase_name="named_array_jax",
          target_builder=lambda: penzai.core.named_axes.NamedArray(
              named_axes=collections.OrderedDict({"bar": 5, "baz": 7}),
              data_array=jnp.arange(3 * 5 * 7).reshape((3, 5, 7)),
          ),
          expected_collapsed=(
              "<NamedArray int32(3 | bar:5, baz:7) [≥0, ≤104] zero:1"
              " nonzero:104 (wrapping jax.Array)>"
          ),
          expected_expanded=textwrap.dedent("""\
              NamedArray(  # int32(3 | bar:5, baz:7) [≥0, ≤104] zero:1 nonzero:104
                named_axes=OrderedDict({'bar': 5, 'baz': 7}),
                data_array=<jax.Array int32(3, 5, 7) [≥0, ≤104] zero:1 nonzero:104>,
              )"""),
      ),
      dict(
          testcase_name="named_array_np",
          target_builder=lambda: penzai.core.named_axes.NamedArray(
              named_axes=collections.OrderedDict({"bar": 5, "baz": 7}),
              data_array=jnp.arange(3 * 5 * 7).reshape((3, 5, 7)),
          ),
          expected_collapsed=(
              "<NamedArray int32(3 | bar:5, baz:7) [≥0, ≤104] zero:1"
              " nonzero:104 (wrapping jax.Array)>"
          ),
          expected_expanded=textwrap.dedent("""\
              NamedArray(  # int32(3 | bar:5, baz:7) [≥0, ≤104] zero:1 nonzero:104
                named_axes=OrderedDict({'bar': 5, 'baz': 7}),
                data_array=<jax.Array int32(3, 5, 7) [≥0, ≤104] zero:1 nonzero:104>,
              )"""),
      ),
      dict(
          testcase_name="named_array_view_jax",
          target_builder=lambda: penzai.core.named_axes.NamedArrayView(
              data_shape=(3, 5, 7),
              data_axis_for_logical_axis=(1,),
              data_axis_for_name={"foo": 0, "baz": 2},
              data_array=jnp.arange(3 * 5 * 7).reshape((3, 5, 7)),
          ),
          expected_collapsed=(
              "<NamedArrayView int32(5 | foo:3, baz:7) [≥0, ≤104] zero:1"
              " nonzero:104 (wrapping jax.Array)>"
          ),
          expected_expanded=textwrap.dedent("""\
              NamedArrayView(  # int32(5 | foo:3, baz:7) [≥0, ≤104] zero:1 nonzero:104
                data_shape=(3, 5, 7),
                data_axis_for_logical_axis=(1,),
                data_axis_for_name={'foo': 0, 'baz': 2},
                data_array=<jax.Array int32(3, 5, 7) [≥0, ≤104] zero:1 nonzero:104>,
              )"""),
      ),
      dict(
          testcase_name="well_known_function",
          target=default_renderer.render_to_text,
          expected_collapsed="render_to_text",
          expected_roundtrip_collapsed=(
              "penzai.treescope.default_renderer.render_to_text"
          ),
      ),
      dict(
          testcase_name="well_known_type",
          target=penzai.core.struct.Struct,
          expected_collapsed="Struct",
          expected_roundtrip_collapsed="penzai.core.struct.Struct",
      ),
      dict(
          testcase_name="ast_nodes",
          target=ast.parse("print(1, 2)").body[0],
          expand_depth=3,
          expected_expanded=textwrap.dedent("""\
              Expr(
                value=Call(
                  func=Name(
                    id='print',
                    ctx=Load(),
                  ),
                  args=[
                    Constant(value=1, kind=None),
                    Constant(value=2, kind=None),
                  ],
                  keywords=[],
                ),
              )"""),
      ),
      dict(
          testcase_name="dtype_standard",
          target=np.dtype(np.float32),
          expected_collapsed="dtype('float32')",
          expected_roundtrip_collapsed="numpy.dtype('float32')",
      ),
      dict(
          testcase_name="dtype_extended",
          target=np.dtype(jnp.bfloat16),
          expected_collapsed="dtype('bfloat16')",
          expected_roundtrip_collapsed="numpy.dtype('bfloat16')",
      ),
      dict(
          testcase_name="jax_precision",
          target=[jax.lax.Precision.HIGHEST],
          expected_collapsed="[Precision.HIGHEST]",
          expected_expanded=textwrap.dedent("""\
              [
                Precision.HIGHEST,  # value: 2
              ]"""),
          expected_roundtrip_collapsed="[jax.lax.Precision.HIGHEST]",
      ),
      dict(
          testcase_name="shapecheck_ArraySpec",
          target=pz.chk.ArraySpec(
              shape=(1, pz.chk.var("one"), 2, *pz.chk.var("many")),
              named_shape={
                  "foo": pz.chk.var("foo"),
                  **pz.chk.var("bar"),
                  **pz.chk.vars_for_axes("baz", {"x": 2, "y": None}),
              },
              dtype=np.floating,
          ),
          expected_collapsed=(
              "<ArraySpec floating(1, {one}, 2, *{many…} | foo:{foo},"
              " **{bar:…}, x:2, y:{baz/y})>"
          ),
          expand_depth=2,
          expected_expanded=textwrap.dedent("""\
              ArraySpec(
                shape=(
                  1,
                  DimVar(name='one'),
                  2,
                  MultiDimVar(name='many'),
                ),
                dtype=# numpy.floating
                  <class 'numpy.floating'>
                ,
                named_shape={
                  'foo': DimVar(name='foo'),
                  MultiDimVar(name='bar'):
                    RemainingAxisPlaceholder(),
                  'x': KnownDim(name=('baz', 'x'), size=2, from_keypath=None),
                  'y': DimVar(name=('baz', 'y')),
                },
              )"""),
          expected_roundtrip_collapsed=(
              "penzai.core.shapecheck.ArraySpec(shape=(1,"
              " penzai.core.shapecheck.DimVar(name='one'), 2,"
              " penzai.core.shapecheck.MultiDimVar(name='many')),"
              " dtype=numpy.floating, named_shape={'foo':"
              " penzai.core.shapecheck.DimVar(name='foo'),"
              " penzai.core.shapecheck.MultiDimVar(name='bar'):"
              " penzai.core.shapecheck.RemainingAxisPlaceholder(), 'x':"
              " penzai.core.shapecheck.KnownDim(name=('baz', 'x'), size=2,"
              " from_keypath=None), 'y':"
              " penzai.core.shapecheck.DimVar(name=('baz', 'y'))})"
          ),
      ),
      dict(
          testcase_name="layer_annotations",
          target=pz.nn.Sequential([
              pz.nn.Identity(),
              pz.de.WithRandomKeyFromArg.handling(
                  fixture_lib.LayerThatHoldsStuff({
                      "a": pz.de.RandomRequest(),
                      "b": pz.de.SideInputRequest(tag="side"),
                  }),
                  handler_id="foo",
              ),
              pz.nn.Identity(),
          ]),
          expand_depth=5,
          expected_collapsed=(
              "Sequential(sublayers=[Identity(),"
              " WithRandomKeyFromArg(handler_id='foo',"
              " body=LayerThatHoldsStuff(stuff={'a':"
              " HandledRandomRef(handler_id='foo'), 'b':"
              " SideInputRequest(tag='side')})),"
              " Identity()])"
          ),
          expected_expanded=textwrap.dedent("""\
              Sequential(
                #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                # Unhandled effects: SideInputEffect
                #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                sublayers=[
                  Identity(),
                  WithRandomKeyFromArg(
                    #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                    # Input: (
                      <input to body>,
                      ArraySpec(
                        shape=(),
                        dtype=jax.dtypes.prng_key,
                        named_shape={},
                      ),
                    )
                    # Output: <output from body>
                    #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                    handler_id='foo',
                    body=LayerThatHoldsStuff(
                      #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                      # Input: {
                        'input': <ArraySpec any(1, 2, 3)>,
                      }
                      # Output: {
                        'output': <ArraySpec any(| foo:5)>,
                      }
                      #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                      stuff={
                        'a': HandledRandomRef(handler_id='foo'), # Handled by WithRandomKeyFromArg
                        'b': SideInputRequest(tag='side'),
                      },
                    ),
                  ),
                  Identity(),
                ],
              )"""),
          expected_roundtrip=textwrap.dedent("""\
              penzai.nn.grouping.Sequential(
                # #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                # # Unhandled effects: penzai.data_effects.side_input.SideInputEffect
                # #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                sublayers=[
                  penzai.nn.grouping.Identity(),
                  penzai.data_effects.random.WithRandomKeyFromArg(
                    # #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                    # # Input: (
                    #   penzai.core.shapecheck.Wildcard('input to body'),
                    #   penzai.core.shapecheck.ArraySpec(
                    #     shape=(),
                    #     dtype=jax.dtypes.prng_key,
                    #     named_shape={},
                    #   ),
                    # )
                    # # Output: penzai.core.shapecheck.Wildcard('output from body')
                    # #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                    handler_id='foo',
                    body=tests.fixtures.treescope_examples_fixture.LayerThatHoldsStuff(
                      # #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                      # # Input: {
                      #   'input': penzai.core.shapecheck.ArraySpec(shape=(1, 2, 3), dtype=numpy.generic, named_shape={}),
                      # }
                      # # Output: {
                      #   'output': penzai.core.shapecheck.ArraySpec(shape=(), dtype=numpy.generic, named_shape={'foo': 5}),
                      # }
                      # #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                      stuff={
                        'a': penzai.data_effects.random.HandledRandomRef(handler_id='foo'), # Handled by penzai.data_effects.random.WithRandomKeyFromArg
                        'b': penzai.data_effects.side_input.SideInputRequest(tag='side'),
                      },
                    ),
                  ),
                  penzai.nn.grouping.Identity(),
                ],
              )"""),
      ),
  )
  def test_object_rendering(
      self,
      *,
      target: Any = None,
      target_builder: Callable[[], Any] | None = None,
      expected_collapsed: str | None = None,
      expected_expanded: str | None = None,
      expected_roundtrip: str | None = None,
      expected_roundtrip_collapsed: str | None = None,
      expand_depth: int = 1,
  ):
    if target_builder is not None:
      assert target is None
      target = target_builder()

    renderer = default_renderer.active_renderer.get()
    # Render it to IR.
    rendering = basic_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )

    # Collapse all foldables.
    layout_algorithms.expand_to_depth(rendering, 0)

    if expected_collapsed is not None:
      with self.subTest("collapsed"):
        self.assertEqual(
            foldable_impl.render_to_text_as_root(rendering),
            expected_collapsed,
        )

    if expected_roundtrip_collapsed is not None:
      with self.subTest("roundtrip_collapsed"):
        self.assertEqual(
            foldable_impl.render_to_text_as_root(rendering, roundtrip=True),
            expected_roundtrip_collapsed,
        )

    layout_algorithms.expand_to_depth(rendering, expand_depth)

    if expected_expanded is not None:
      with self.subTest("expanded"):
        self.assertEqual(
            foldable_impl.render_to_text_as_root(rendering),
            expected_expanded,
        )

    if expected_roundtrip is not None:
      with self.subTest("roundtrip"):
        self.assertEqual(
            foldable_impl.render_to_text_as_root(rendering, roundtrip=True),
            expected_roundtrip,
        )

    # Render to HTML; make sure it doesn't raise any errors.
    with self.subTest("html_no_errors"):
      _ = foldable_impl.render_to_html_as_root(rendering)

  def test_closure_rendering(self):
    def outer_fn(x):
      def inner_fn(y):
        return x + y

      return inner_fn

    closure = outer_fn(100)

    renderer = default_renderer.active_renderer.get()
    # Enable closure rendering (currently disabled by default)
    renderer = renderer.extended_with(
        handlers=[
            functools.partial(
                function_reflection_handlers.handle_code_objects_with_reflection,
                show_closure_vars=True,
            )
        ]
    )
    # Render it to IR.
    rendering = basic_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(closure)
    )

    layout_algorithms.expand_to_depth(rendering, 1)

    self.assertContainsInOrder(
        [
            "<function",
            "test_closure_rendering.<locals>.outer_fn.<locals>.inner_fn at 0x",
            ">",
            "# Closure variables:",
            "{'x': 100}",
            "# Defined at line ",
            " of ",
            "tests/treescope_renderer_test.py",
        ],
        foldable_impl.render_to_text_as_root(rendering),
    )

  def test_fallback_repr_pytree_node(self):
    target = [fixture_lib.UnknownPytreeNode(1234, 5678)]
    renderer = default_renderer.active_renderer.get()
    rendering = basic_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        "[<custom repr for UnknownPytreeNode: x=1234, y=5678>]",
    )

    layout_algorithms.expand_to_depth(rendering, 2)
    rendered_text = foldable_impl.render_to_text_as_root(rendering)
    self.assertEqual(
        "\n".join(
            line.rstrip() for line in rendered_text.splitlines(keepends=True)
        ),
        textwrap.dedent(f"""\
            [
              <custom repr for UnknownPytreeNode: x=1234, y=5678>
                #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                # PyTree children:
                  GetAttrKey(name='x'): 1234,
                  'custom_key': 5678,
                #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
              ,  # {object.__repr__(target[0])}
            ]"""),
    )

  def test_fallback_repr_one_line(self):
    target = [fixture_lib.UnknownObjectWithOneLineRepr()]
    renderer = default_renderer.active_renderer.get()
    rendering = basic_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        "[<custom repr for UnknownObjectWithOneLineRepr>]",
    )
    layout_algorithms.expand_to_depth(rendering, 2)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        textwrap.dedent(f"""\
            [
              <custom repr for UnknownObjectWithOneLineRepr>,  # {object.__repr__(target[0])}
            ]"""),
    )

  def test_fallback_repr_multiline_idiomatic(self):
    target = [fixture_lib.UnknownObjectWithMultiLineRepr()]
    renderer = default_renderer.active_renderer.get()
    rendering = basic_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        "[<custom repr↩  for↩  UnknownObjectWithMultiLineRepr↩>]",
    )
    layout_algorithms.expand_to_depth(rendering, 2)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        textwrap.dedent(f"""\
            [
              <custom repr
                for
                UnknownObjectWithMultiLineRepr
              >,  # {object.__repr__(target[0])}
            ]"""),
    )

  def test_fallback_repr_multiline_unidiomatic(self):
    target = [fixture_lib.UnknownObjectWithBadMultiLineRepr()]
    renderer = default_renderer.active_renderer.get()
    rendering = basic_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        f"[{object.__repr__(target[0])}]",
    )
    layout_algorithms.expand_to_depth(rendering, 2)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        textwrap.dedent(f"""\
            [
              # {object.__repr__(target[0])}
                Non-idiomatic
                multiline
                object
              ,
            ]"""),
    )

  def test_fallback_repr_basic(self):
    target = [fixture_lib.UnknownObjectWithBuiltinRepr()]
    renderer = default_renderer.active_renderer.get()
    rendering = basic_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        f"[{repr(target[0])}]",
    )
    self.assertContainsInOrder(
        [
            (
                "penzai.treescope.copypaste_fallback.NotRoundtrippable(original_repr='<tests.fixtures.treescope_examples_fixture.UnknownObjectWithBuiltinRepr"
                " object at 0x"
            ),
            ">', original_id=",
            (
                ", original_type=tests.fixtures.treescope_examples_fixture"
                ".UnknownObjectWithBuiltinRepr)"
            ),
        ],
        foldable_impl.render_to_text_as_root(rendering, roundtrip=True),
    )
    layout_algorithms.expand_to_depth(rendering, 2)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        textwrap.dedent(f"""\
            [
              {repr(target[0])},
            ]"""),
    )

  def test_ndarray_roundtrip_fallback(self):
    target = np.array([1, 2, 3])
    renderer = default_renderer.active_renderer.get()
    rendering = basic_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertContainsInOrder(
        [
            (
                "penzai.treescope.copypaste_fallback"
                ".NotRoundtrippable(original_repr='array([1, 2, 3])', "
                "original_id="
            ),
            ", original_type=numpy.ndarray)",
        ],
        foldable_impl.render_to_text_as_root(rendering, roundtrip=True),
    )

  def test_shared_values(self):
    shared = ["bar"]
    target = [shared, shared, {"foo": shared}]
    renderer = default_renderer.active_renderer.get()
    rendering = basic_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 3)
    rendered_text = foldable_impl.render_to_text_as_root(rendering)
    # Rendering may contain trailing whitespace; remove it before checking the
    # value, since it's not important.
    self.assertEqual(
        "\n".join(
            line.rstrip() for line in rendered_text.splitlines(keepends=True)
        ),
        textwrap.dedent(f"""\
            [
              [
                'bar',
              ], # Repeated python obj at 0x{id(shared):x}
              [
                'bar',
              ], # Repeated python obj at 0x{id(shared):x}
              {{
                'foo': [
                  'bar',
                ], # Repeated python obj at 0x{id(shared):x}
              }},
            ]"""),
    )

  def test_autovisualizer(self):
    target = [1, 2, "foo", 3, 4, 5, [6, 7]]

    def autovisualizer_for_test(node, path):
      if isinstance(node, str):
        return autovisualize.CustomTreescopeVisualization(
            part_interface.RenderableAndLineAnnotations(
                basic_parts.Text("(visualiation for foo goes here)"),
                basic_parts.Text(" # annotation for vis for foo"),
            ),
        )
      elif path == (jax.tree_util.SequenceKey(4),):
        return autovisualize.IPythonVisualization(
            types.SimpleNamespace(_repr_html_=lambda: "(html rendering)"),
            replace=True,
        )
      elif path == (jax.tree_util.SequenceKey(5),):
        return autovisualize.IPythonVisualization(
            types.SimpleNamespace(_repr_html_=lambda: "(html rendering)"),
            replace=False,
        )
      elif path == (jax.tree_util.SequenceKey(6),):
        return autovisualize.ChildAutovisualizer(inner_autovisualizer)

    def inner_autovisualizer(node, path):
      del path
      if node == 6:
        return autovisualize.CustomTreescopeVisualization(
            part_interface.RenderableAndLineAnnotations(
                basic_parts.Text("(child visualiation of 6 goes here)"),
                basic_parts.Text(" # annotation for vis for 6"),
            ),
        )

    with autovisualize.active_autovisualizer.set_scoped(
        autovisualizer_for_test
    ):
      renderer = default_renderer.active_renderer.get()
      rendering = basic_parts.build_full_line_with_annotations(
          renderer.to_foldable_representation(target)
      )
      layout_algorithms.expand_to_depth(rendering, 3)
      rendered_text = foldable_impl.render_to_text_as_root(rendering)
      rendered_text_as_roundtrip = foldable_impl.render_to_text_as_root(
          rendering, roundtrip=True
      )

    self.assertEqual(
        rendered_text,
        textwrap.dedent("""\
            [
              1,
              2,
              (visualiation for foo goes here), # annotation for vis for foo
              3,
              <Visualization of int:
                <rich HTML visualization>
              >,
              5
              #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
              <rich HTML visualization>
              #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
              ,
              [
                (child visualiation of 6 goes here), # annotation for vis for 6
                7,
              ],
            ]"""),
    )

    self.assertEqual(
        rendered_text_as_roundtrip,
        textwrap.dedent("""\
            [
              1,
              2,
              'foo',  # Visualization hidden in roundtrip mode
              3,
              4,  # Visualization hidden in roundtrip mode
              5
              # #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
              # <rich HTML visualization>
              # #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
              ,
              [
                6,  # Visualization hidden in roundtrip mode
                7,
              ],
            ]"""),
    )

  def test_selection_rendering(self):
    # Renders a selection. Note that the public interface to selection rendering
    # (using Selection.show_selection()) renders to HTML only, but it's easier
    # to test the text version.
    selection = penzai.core.selectors.select({
        "a": 1,
        "b": fixture_lib.StructWithOneChild(10),
        "c": [
            {"value": fixture_lib.StructWithOneChild(12)},
            {"value": fixture_lib.StructWithOneChild(13)},
            {"value": 3},
        ],
    }).at(lambda root: (root["b"], root["c"][1]["value"].foo))

    with self.subTest("visible_selection"):
      rendered_ir = (
          selection_rendering.render_selection_to_foldable_representation(
              selection
          )
      )
      rendered_text = foldable_impl.render_to_text_as_root(rendered_ir)
      self.assertEqual(
          "\n".join(
              line.rstrip() for line in rendered_text.splitlines(keepends=True)
          ),
          textwrap.dedent("""\
              pz.select(
                {
                  'a': 1,
                  'b':
                  #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                  # Selected:
                  StructWithOneChild(foo=10)
                  #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                  ,
                  'c': [
                    {'value': StructWithOneChild(foo=12)},
                    {
                      'value': StructWithOneChild(
                        foo=
                        #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                        # Selected:
                        13
                        #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                        ,
                      ),
                    },
                    {'value': 3},
                  ],
                }
              ).at_keypaths(<2 subtrees, highlighted above>)"""),
      )

      # Expanding the hidden keypaths:
      for foldable in rendered_ir.foldables_in_this_part():
        foldable.set_expand_state(part_interface.ExpandState.EXPANDED)
      rendered_text = foldable_impl.render_to_text_as_root(rendered_ir)
      self.assertEqual(
          "\n".join(
              line.rstrip() for line in rendered_text.splitlines(keepends=True)
          ),
          textwrap.dedent("""\
              pz.select(
                {
                  'a': 1,
                  'b':
                  #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                  # Selected:
                  StructWithOneChild(foo=10)
                  #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                  ,
                  'c': [
                    {'value': StructWithOneChild(foo=12)},
                    {
                      'value': StructWithOneChild(
                        foo=
                        #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                        # Selected:
                        13
                        #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                        ,
                      ),
                    },
                    {'value': 3},
                  ],
                }
              ).at_keypaths(# 2 subtrees, highlighted above
                (
                  (DictKey(key='b'),),
                  (
                    DictKey(key='c'),
                    SequenceKey(idx=1),
                    DictKey(key='value'),
                    GetAttrKey(name='foo'),
                  ),
                )
              )"""),
      )

    with self.subTest("selection_expansion"):
      rendered_ir = (
          selection_rendering.render_selection_to_foldable_representation(
              selection, visible_selection=False
          )
      )
      self.assertEqual(
          foldable_impl.render_to_text_as_root(rendered_ir),
          textwrap.dedent("""\
              {
                'a': 1,
                'b': StructWithOneChild(foo=10),
                'c': [
                  {'value': StructWithOneChild(foo=12)},
                  {
                    'value': StructWithOneChild(
                      foo=13,
                    ),
                  },
                  {'value': 3},
                ],
              }"""),
      )

  def test_balanced_layout(self):
    renderer = default_renderer.active_renderer.get()
    some_nested_object = fixture_lib.DataclassWithOneChild([
        ["foo"] * 4,
        ["12345678901234567890"] * 5,
        {"a": 1, "b": 2, "c": [{"bar": "baz"} for _ in range(5)]},
        [list(range(10)) for _ in range(6)],
    ])

    def render_and_expand(**kwargs):
      rendering = renderer.to_foldable_representation(
          some_nested_object
      ).renderable
      layout_algorithms.expand_for_balanced_layout(rendering, **kwargs)
      return foldable_impl.render_to_text_as_root(rendering)

    with self.subTest("no_max_height"):
      self.assertEqual(
          render_and_expand(max_height=None, target_width=60),
          textwrap.dedent("""\
              DataclassWithOneChild(
                foo=[
                  ['foo', 'foo', 'foo', 'foo'],
                  [
                    '12345678901234567890',
                    '12345678901234567890',
                    '12345678901234567890',
                    '12345678901234567890',
                    '12345678901234567890',
                  ],
                  {
                    'a': 1,
                    'b': 2,
                    'c': [
                      {'bar': 'baz'},
                      {'bar': 'baz'},
                      {'bar': 'baz'},
                      {'bar': 'baz'},
                      {'bar': 'baz'},
                    ],
                  },
                  [
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  ],
                ],
              )"""),
      )

    with self.subTest("medium_max_height"):
      self.assertEqual(
          render_and_expand(max_height=20, target_width=60),
          textwrap.dedent("""\
              DataclassWithOneChild(
                foo=[
                  ['foo', 'foo', 'foo', 'foo'],
                  [
                    '12345678901234567890',
                    '12345678901234567890',
                    '12345678901234567890',
                    '12345678901234567890',
                    '12345678901234567890',
                  ],
                  {
                    'a': 1,
                    'b': 2,
                    'c': [{'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}],
                  },
                  [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                ],
              )"""),
      )

    with self.subTest("small_max_height"):
      self.assertEqual(
          render_and_expand(max_height=10, target_width=60),
          textwrap.dedent("""\
              DataclassWithOneChild(
                foo=[
                  ['foo', 'foo', 'foo', 'foo'],
                  ['12345678901234567890', '12345678901234567890', '12345678901234567890', '12345678901234567890', '12345678901234567890'],
                  {'a': 1, 'b': 2, 'c': [{'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}]},
                  [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                ],
              )"""),
      )

    with self.subTest("long_target_width"):
      self.assertEqual(
          render_and_expand(max_height=None, target_width=150),
          textwrap.dedent("""\
              DataclassWithOneChild(
                foo=[
                  ['foo', 'foo', 'foo', 'foo'],
                  ['12345678901234567890', '12345678901234567890', '12345678901234567890', '12345678901234567890', '12345678901234567890'],
                  {'a': 1, 'b': 2, 'c': [{'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}]},
                  [
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  ],
                ],
              )"""),
      )

  def test_balanced_layout_after_manual_expansion(self):
    renderer = default_renderer.active_renderer.get()
    some_nested_object = [
        fixture_lib.DataclassWithOneChild(
            [["foo"] * 4, (["baz"] * 5, ["qux"] * 5)]
        )
    ]

    rendering = renderer.to_foldable_representation(
        some_nested_object
    ).renderable
    layout_algorithms.expand_for_balanced_layout(
        rendering,
        max_height=3,
        target_width=40,
        recursive_expand_height_for_collapsed_nodes=10,
    )

    # Initially collapsed due to height constraint.
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        textwrap.dedent("""\
        [
          DataclassWithOneChild(foo=[['foo', 'foo', 'foo', 'foo'], (['baz', 'baz', 'baz', 'baz', 'baz'], ['qux', 'qux', 'qux', 'qux', 'qux'])]),
        ]"""),
    )

    # But expands multiple levels once we expand the collapsed node manually.
    # (In the browser, this would be done by clicking the expand marker.)
    target_foldable = (
        rendering.foldables_in_this_part()[0]
        .as_expanded_part()
        .foldables_in_this_part()[0]
    )
    self.assertEqual(
        target_foldable.get_expand_state(),
        part_interface.ExpandState.COLLAPSED,
    )
    target_foldable.set_expand_state(part_interface.ExpandState.EXPANDED)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        textwrap.dedent("""\
            [
              DataclassWithOneChild(
                foo=[
                  ['foo', 'foo', 'foo', 'foo'],
                  (
                    ['baz', 'baz', 'baz', 'baz', 'baz'],
                    ['qux', 'qux', 'qux', 'qux', 'qux'],
                  ),
                ],
              ),
            ]"""),
    )

  def test_balanced_layout_relaxes_height_constraint_once(self):
    renderer = default_renderer.active_renderer.get()
    some_nested_object = [
        fixture_lib.DataclassWithOneChild(
            [fixture_lib.DataclassWithOneChild(["abcdefghik"] * 20)]
        )
    ]

    # With a relaxed only-child expansion constraint (the default), we still
    # expand the large list because it's the only nontrivial object.
    rendering = renderer.to_foldable_representation(
        some_nested_object
    ).renderable
    layout_algorithms.expand_for_balanced_layout(
        rendering, max_height=10, target_width=40
    )
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        textwrap.dedent("""\
            [
              DataclassWithOneChild(
                foo=[
                  DataclassWithOneChild(
                    foo=[
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                    ],
                  ),
                ],
              ),
            ]"""),
    )

    # Without a relaxed only-child expansion constraint, it stays collapsed.
    rendering = renderer.to_foldable_representation(
        some_nested_object
    ).renderable
    layout_algorithms.expand_for_balanced_layout(
        rendering,
        max_height=10,
        target_width=40,
        relax_height_constraint_for_only_child=False,
    )
    self.assertEqual(
        foldable_impl.render_to_text_as_root(rendering),
        textwrap.dedent("""\
            [
              DataclassWithOneChild(
                foo=[
                  DataclassWithOneChild(
                    foo=['abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik'],
                  ),
                ],
              ),
            ]"""),
    )


if __name__ == "__main__":
  absltest.main()
