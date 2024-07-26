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

"""Tests for the Penzai's treescope handlers."""

import collections
import dataclasses
import textwrap
from typing import Any, Callable

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from penzai import pz
from penzai.core._treescope_handlers import selection_rendering
import penzai.core.selectors
import penzai.core.struct
from penzai.deprecated.v1 import pz as pz_v1
from tests.treescope.fixtures import treescope_examples_fixture as fixture_lib
import treescope
from treescope import layout_algorithms
from treescope import lowering
from treescope import rendering_parts


@dataclasses.dataclass
class CustomReprHTMLObject:
  repr_html: str

  def _repr_html_(self):
    return self.repr_html


class TreescopeRendererTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="deprecated_v1_penzai_layer",
          target=fixture_lib.ExampleLayer(100),
          expected_collapsed="ExampleLayer(foo=100)",
          expected_expanded=textwrap.dedent("""\
              ExampleLayer(
                foo=100,
              )"""),
          expected_roundtrip=textwrap.dedent("""\
              tests.treescope.fixtures.treescope_examples_fixture.ExampleLayer(
                foo=100,
              )"""),
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
          testcase_name="deprecated_v1_layer_annotations",
          target=pz_v1.nn.Sequential([
              pz_v1.nn.Identity(),
              pz_v1.de.WithRandomKeyFromArg.handling(
                  fixture_lib.LayerThatHoldsStuff({
                      "a": pz_v1.de.RandomRequest(),
                      "b": pz_v1.de.SideInputRequest(tag="side"),
                  }),
                  handler_id="foo",
              ),
              pz_v1.nn.Identity(),
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
              penzai.deprecated.v1.nn.grouping.Sequential(
                # #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                # # Unhandled effects: penzai.deprecated.v1.data_effects.side_input.SideInputEffect
                # #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                sublayers=[
                  penzai.deprecated.v1.nn.grouping.Identity(),
                  penzai.deprecated.v1.data_effects.random.WithRandomKeyFromArg(
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
                    body=tests.treescope.fixtures.treescope_examples_fixture.LayerThatHoldsStuff(
                      # #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                      # # Input: {
                      #   'input': penzai.core.shapecheck.ArraySpec(shape=(1, 2, 3), dtype=numpy.generic, named_shape={}),
                      # }
                      # # Output: {
                      #   'output': penzai.core.shapecheck.ArraySpec(shape=(), dtype=numpy.generic, named_shape={'foo': 5}),
                      # }
                      # #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
                      stuff={
                        'a': penzai.deprecated.v1.data_effects.random.HandledRandomRef(handler_id='foo'), # Handled by penzai.deprecated.v1.data_effects.random.WithRandomKeyFromArg
                        'b': penzai.deprecated.v1.data_effects.side_input.SideInputRequest(tag='side'),
                      },
                    ),
                  ),
                  penzai.deprecated.v1.nn.grouping.Identity(),
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

    renderer = treescope.active_renderer.get()
    # Render it to IR.
    rendering = rendering_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )

    # Collapse all foldables.
    layout_algorithms.expand_to_depth(rendering, 0)

    if expected_collapsed is not None:
      with self.subTest("collapsed"):
        self.assertEqual(
            lowering.render_to_text_as_root(rendering),
            expected_collapsed,
        )

    if expected_roundtrip_collapsed is not None:
      with self.subTest("roundtrip_collapsed"):
        self.assertEqual(
            lowering.render_to_text_as_root(rendering, roundtrip=True),
            expected_roundtrip_collapsed,
        )

    layout_algorithms.expand_to_depth(rendering, expand_depth)

    if expected_expanded is not None:
      with self.subTest("expanded"):
        self.assertEqual(
            lowering.render_to_text_as_root(rendering),
            expected_expanded,
        )

    if expected_roundtrip is not None:
      with self.subTest("roundtrip"):
        self.assertEqual(
            lowering.render_to_text_as_root(rendering, roundtrip=True),
            expected_roundtrip,
        )

    # Render to HTML; make sure it doesn't raise any errors.
    with self.subTest("html_no_errors"):
      _ = lowering.render_to_html_as_root(rendering)

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
    }).at(lambda root: (root["b"], root["c"][1]["value"].foo), multiple=True)

    with self.subTest("visible_selection"):
      rendered_ir = selection.__treescope_root_repr__()
      rendered_text = lowering.render_to_text_as_root(rendered_ir)
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
        foldable.set_expand_state(rendering_parts.ExpandState.EXPANDED)
      rendered_text = lowering.render_to_text_as_root(rendered_ir)
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
          lowering.render_to_text_as_root(rendered_ir),
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


if __name__ == "__main__":
  absltest.main()
