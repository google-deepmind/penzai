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

"""Tests for treescope's foldable representation."""

import dataclasses
import io
from typing import Any, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from penzai.treescope import canonical_aliases
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import part_interface

CSSStyleRule = part_interface.CSSStyleRule


@dataclasses.dataclass(frozen=True)
class MockFoldableTreeNode(part_interface.FoldableTreeNode):
  tag: str

  def get_expand_state(self):
    raise NotImplementedError()

  def set_expand_state(self, value):
    pass

  def as_expanded_part(self):
    raise NotImplementedError()

  def html_setup_parts(self, *args, **kwargs):
    raise NotImplementedError()

  def render_to_html(self, *args, **kwargs):
    raise NotImplementedError()

  def render_to_text(self, *args, **kwargs):
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class MockRenderableTreePart(part_interface.RenderableTreePart):
  tag: StopAsyncIteration

  def _compute_collapsed_width(self) -> int:
    return 100

  def _compute_newlines_in_expanded_parent(self) -> int:
    return 10

  def _compute_tags_in_this_part(self) -> frozenset[Any]:
    return frozenset([self.tag])

  def foldables_in_this_part(self):
    return (MockFoldableTreeNode(self.tag),)

  def html_setup_parts(self, context):
    return {part_interface.CSSStyleRule(f'[mock rule {self.tag}]')}

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    b = 'y' if at_beginning_of_line else 'n'
    stream.write(f'[mock {self.tag}, b:{b}]')

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    e = 'y' if expanded_parent else 'n'
    r = 'y' if roundtrip_mode else 'n'
    stream.write(f'[mock {self.tag}, e:{e} r:{r}]')


mock_context = part_interface.HtmlContextForSetup(
    collapsed_selector='[mock collapsed_selector]',
    roundtrip_selector='[mock roundtrip_selector]',
    hyperlink_hover_selector='[mock hyperlink_hover_selector]',
    hyperlink_clicked_selector='[mock hyperlink_clicked_selector]',
    hyperlink_clicked_tick_selector='[mock hyperlink_clicked_tick_selector]',
    hyperlink_target_selector='[mock hyperlink_target_selector]',
)


class TaggedGroupForTest(basic_parts.BaseTaggedGroup):
  """Test subclass of BaseTaggedGroup."""

  def _tags(self) -> frozenset[Any]:
    return frozenset({'test_tag'})


class SpanGroupForTest_OneRule(basic_parts.BaseSpanGroup):  # pylint: disable=invalid-name
  """Test subclass of BaseSpanGroup."""

  def _span_css_class(self) -> str:
    return 'test_span_css_class'

  def _span_css_rule(self, context):
    return CSSStyleRule('test_span_css_rule')


class SpanGroupForTest_TwoRules(basic_parts.BaseSpanGroup):  # pylint: disable=invalid-name
  """Test subclass of BaseSpanGroup."""

  def _span_css_class(self) -> str:
    return 'test_span_css_class'

  def _span_css_rule(self, context):
    return {
        CSSStyleRule('test_span_css_rule'),
        CSSStyleRule('test_span_css_rule_2'),
    }


class BoxForTest(basic_parts.BaseBoxWithOutline):
  """Test subclass of BaseBoxWithOutline."""

  def _box_css_class(self) -> str:
    return 'test_box_css_class'

  def _box_css_rule(self, context):
    return CSSStyleRule('test_span_css_rule')


class RepresentationPartsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_part',
          part=basic_parts.EmptyPart(),
          expected_width=0,
          expected_newlines=0,
          expected_tags=frozenset({}),
          expected_foldables=(),
          expected_setup_parts=set(),
          expected_text_collapsed='',
          expected_text_expanded='',
          expected_text_roundtrip_collapsed='',
          expected_text_roundtrip_expanded='',
          expected_html='',
          expected_html_at_begining='',
      ),
      dict(
          testcase_name='single_line_text',
          part=basic_parts.Text('some text with characters < > &'),
          expected_width=31,
          expected_newlines=0,
          expected_tags=frozenset({}),
          expected_foldables=(),
          expected_setup_parts=set(),
          expected_text_collapsed='some text with characters < > &',
          expected_text_expanded='some text with characters < > &',
          expected_text_roundtrip_collapsed='some text with characters < > &',
          expected_text_roundtrip_expanded='some text with characters < > &',
          expected_html='some text with characters &lt; &gt; &amp;',
          expected_html_at_begining='some text with characters &lt; &gt; &amp;',
      ),
      dict(
          testcase_name='siblings',
          part=basic_parts.Siblings([
              MockRenderableTreePart('A'),
              MockRenderableTreePart('B'),
              MockRenderableTreePart('C'),
          ]),
          expected_width=300,
          expected_newlines=30,
          expected_tags=frozenset({'C', 'B', 'A'}),
          expected_foldables=[
              MockFoldableTreeNode(tag='A'),
              MockFoldableTreeNode(tag='B'),
              MockFoldableTreeNode(tag='C'),
          ],
          expected_setup_parts={
              CSSStyleRule(rule='[mock rule A]'),
              CSSStyleRule(rule='[mock rule B]'),
              CSSStyleRule(rule='[mock rule C]'),
          },
          expected_text_collapsed=(
              '[mock A, e:n r:n][mock B, e:n r:n][mock C, e:n r:n]'
          ),
          expected_text_expanded=(
              '[mock A, e:y r:n][mock B, e:y r:n][mock C, e:y r:n]'
          ),
          expected_text_roundtrip_collapsed=(
              '[mock A, e:n r:y][mock B, e:n r:y][mock C, e:n r:y]'
          ),
          expected_text_roundtrip_expanded=(
              '[mock A, e:y r:y][mock B, e:y r:y][mock C, e:y r:y]'
          ),
          expected_html='[mock A, b:n][mock B, b:n][mock C, b:n]',
          expected_html_at_begining='[mock A, b:y][mock B, b:n][mock C, b:n]',
      ),
      dict(
          testcase_name='tagged_group',
          part=TaggedGroupForTest(MockRenderableTreePart('A')),
          expected_width=100,
          expected_newlines=10,
          expected_tags=frozenset({'test_tag', 'A'}),
          expected_foldables=(MockFoldableTreeNode(tag='A'),),
          expected_setup_parts={CSSStyleRule(rule='[mock rule A]')},
          expected_text_collapsed='[mock A, e:n r:n]',
          expected_text_expanded='[mock A, e:y r:n]',
          expected_text_roundtrip_collapsed='[mock A, e:n r:y]',
          expected_text_roundtrip_expanded='[mock A, e:y r:y]',
          expected_html='[mock A, b:n]',
          expected_html_at_begining='[mock A, b:y]',
      ),
      dict(
          testcase_name='span_group_1',
          part=SpanGroupForTest_OneRule(MockRenderableTreePart('A')),
          expected_width=100,
          expected_newlines=10,
          expected_tags=frozenset({'A'}),
          expected_foldables=(MockFoldableTreeNode(tag='A'),),
          expected_setup_parts={
              CSSStyleRule(rule='test_span_css_rule'),
              CSSStyleRule(rule='[mock rule A]'),
          },
          expected_text_collapsed='[mock A, e:n r:n]',
          expected_text_expanded='[mock A, e:y r:n]',
          expected_text_roundtrip_collapsed='[mock A, e:n r:y]',
          expected_text_roundtrip_expanded='[mock A, e:y r:y]',
          expected_html=(
              '<span class="test_span_css_class">[mock A, b:n]</span>'
          ),
          expected_html_at_begining=(
              '<span class="test_span_css_class">[mock A, b:y]</span>'
          ),
      ),
      dict(
          testcase_name='span_group_2',
          part=SpanGroupForTest_TwoRules(MockRenderableTreePart('A')),
          expected_width=100,
          expected_newlines=10,
          expected_tags=frozenset({'A'}),
          expected_foldables=(MockFoldableTreeNode(tag='A'),),
          expected_setup_parts={
              CSSStyleRule(rule='test_span_css_rule'),
              CSSStyleRule(rule='test_span_css_rule_2'),
              CSSStyleRule(rule='[mock rule A]'),
          },
          expected_text_collapsed='[mock A, e:n r:n]',
          expected_text_expanded='[mock A, e:y r:n]',
          expected_text_roundtrip_collapsed='[mock A, e:n r:y]',
          expected_text_roundtrip_expanded='[mock A, e:y r:y]',
          expected_html=(
              '<span class="test_span_css_class">[mock A, b:n]</span>'
          ),
          expected_html_at_begining=(
              '<span class="test_span_css_class">[mock A, b:y]</span>'
          ),
      ),
      dict(
          testcase_name='fold_condition_collapsed',
          part=basic_parts.FoldCondition(
              collapsed=MockRenderableTreePart(tag='A')
          ),
          expected_width=100,
          expected_newlines=0,
          expected_tags=frozenset({'A'}),
          expected_foldables=(),
          expected_setup_parts={CSSStyleRule(rule='[mock rule A]'), ...},
          expected_text_collapsed='[mock A, e:n r:n]',
          expected_text_expanded='',
          expected_text_roundtrip_collapsed='[mock A, e:n r:y]',
          expected_text_roundtrip_expanded='',
          expected_html='<span class="when_collapsed">[mock A, b:n]</span>',
          expected_html_at_begining=(
              '<span class="when_collapsed">[mock A, b:y]</span>'
          ),
      ),
      dict(
          testcase_name='fold_condition_expanded',
          part=basic_parts.FoldCondition(
              expanded=MockRenderableTreePart(tag='B')
          ),
          expected_width=0,
          expected_newlines=10,
          expected_tags=frozenset({'B'}),
          expected_foldables=(MockFoldableTreeNode(tag='B'),),
          expected_setup_parts={CSSStyleRule(rule='[mock rule B]'), ...},
          expected_text_collapsed='',
          expected_text_expanded='[mock B, e:y r:n]',
          expected_text_roundtrip_collapsed='',
          expected_text_roundtrip_expanded='[mock B, e:y r:y]',
          expected_html='<span class="when_expanded">[mock B, b:n]</span>',
          expected_html_at_begining=(
              '<span class="when_expanded">[mock B, b:y]</span>'
          ),
      ),
      dict(
          testcase_name='fold_condition_both',
          part=basic_parts.FoldCondition(
              expanded=MockRenderableTreePart(tag='A'),
              collapsed=MockRenderableTreePart(tag='B'),
          ),
          expected_width=100,
          expected_newlines=10,
          expected_tags=frozenset({'B', 'A'}),
          expected_foldables=(MockFoldableTreeNode(tag='A'),),
          expected_setup_parts={
              CSSStyleRule(rule='[mock rule A]'),
              CSSStyleRule(rule='[mock rule B]'),
              ...,
          },
          expected_text_collapsed='[mock B, e:n r:n]',
          expected_text_expanded='[mock A, e:y r:n]',
          expected_text_roundtrip_collapsed='[mock B, e:n r:y]',
          expected_text_roundtrip_expanded='[mock A, e:y r:y]',
          expected_html=(
              '<span class="when_collapsed">[mock B, b:n]</span><span'
              ' class="when_expanded">[mock A, b:n]</span>'
          ),
          expected_html_at_begining=(
              '<span class="when_collapsed">[mock B, b:y]</span><span'
              ' class="when_expanded">[mock A, b:y]</span>'
          ),
      ),
      dict(
          testcase_name='roundtrip_condition_roundtrip',
          part=basic_parts.RoundtripCondition(
              roundtrip=MockRenderableTreePart(tag='A')
          ),
          expected_width=0,
          expected_newlines=0,
          expected_tags=frozenset({'A'}),
          expected_foldables=(),
          expected_setup_parts={CSSStyleRule(rule='[mock rule A]'), ...},
          expected_text_collapsed='',
          expected_text_expanded='',
          expected_text_roundtrip_collapsed='[mock A, e:n r:y]',
          expected_text_roundtrip_expanded='[mock A, e:y r:y]',
          expected_html='<span class="when_roundtrip">[mock A, b:n]</span>',
          expected_html_at_begining=(
              '<span class="when_roundtrip">[mock A, b:y]</span>'
          ),
      ),
      dict(
          testcase_name='roundtrip_condition_not_roundtrip',
          part=basic_parts.RoundtripCondition(
              not_roundtrip=MockRenderableTreePart(tag='B')
          ),
          expected_width=100,
          expected_newlines=10,
          expected_tags=frozenset({'B'}),
          expected_foldables=(MockFoldableTreeNode(tag='B'),),
          expected_setup_parts={CSSStyleRule(rule='[mock rule B]'), ...},
          expected_text_collapsed='[mock B, e:n r:n]',
          expected_text_expanded='[mock B, e:y r:n]',
          expected_text_roundtrip_collapsed='',
          expected_text_roundtrip_expanded='',
          expected_html='<span class="when_not_roundtrip">[mock B, b:n]</span>',
          expected_html_at_begining=(
              '<span class="when_not_roundtrip">[mock B, b:y]</span>'
          ),
      ),
      dict(
          testcase_name='roundtrip_condition_both',
          part=basic_parts.RoundtripCondition(
              roundtrip=MockRenderableTreePart(tag='A'),
              not_roundtrip=MockRenderableTreePart(tag='B'),
          ),
          expected_width=100,
          expected_newlines=10,
          expected_tags=frozenset({'B', 'A'}),
          expected_foldables=(MockFoldableTreeNode(tag='B'),),
          expected_setup_parts={
              CSSStyleRule(rule='[mock rule A]'),
              CSSStyleRule(rule='[mock rule B]'),
              ...,
          },
          expected_text_collapsed='[mock B, e:n r:n]',
          expected_text_expanded='[mock B, e:y r:n]',
          expected_text_roundtrip_collapsed='[mock A, e:n r:y]',
          expected_text_roundtrip_expanded='[mock A, e:y r:y]',
          expected_html=(
              '<span class="when_roundtrip">[mock A, b:n]</span><span'
              ' class="when_not_roundtrip">[mock B, b:n]</span>'
          ),
          expected_html_at_begining=(
              '<span class="when_roundtrip">[mock A, b:y]</span><span'
              ' class="when_not_roundtrip">[mock B, b:y]</span>'
          ),
      ),
      dict(
          testcase_name='summarizable_condition_summary',
          part=basic_parts.SummarizableCondition(
              summary=MockRenderableTreePart(tag='A')
          ),
          expected_width=100,
          expected_newlines=0,
          expected_tags=frozenset({'A'}),
          expected_foldables=(),
          expected_setup_parts={CSSStyleRule(rule='[mock rule A]'), ...},
          expected_text_collapsed='[mock A, e:n r:n]',
          expected_text_expanded='',
          expected_text_roundtrip_collapsed='',
          expected_text_roundtrip_expanded='',
          expected_html=(
              '<span class="when_collapsed_and_not_roundtrip">[mock A,'
              ' b:n]</span>'
          ),
          expected_html_at_begining=(
              '<span class="when_collapsed_and_not_roundtrip">[mock A,'
              ' b:y]</span>'
          ),
      ),
      dict(
          testcase_name='summarizable_condition_detail',
          part=basic_parts.SummarizableCondition(
              detail=MockRenderableTreePart(tag='B')
          ),
          expected_width=0,
          expected_newlines=10,
          expected_tags=frozenset({'B'}),
          expected_foldables=(MockFoldableTreeNode(tag='B'),),
          expected_setup_parts={CSSStyleRule(rule='[mock rule B]'), ...},
          expected_text_collapsed='',
          expected_text_expanded='[mock B, e:y r:n]',
          expected_text_roundtrip_collapsed='[mock B, e:n r:y]',
          expected_text_roundtrip_expanded='[mock B, e:y r:y]',
          expected_html=(
              '<span class="when_expanded_or_roundtrip">[mock B, b:n]</span>'
          ),
          expected_html_at_begining=(
              '<span class="when_expanded_or_roundtrip">[mock B, b:y]</span>'
          ),
      ),
      dict(
          testcase_name='summarizable_condition_both',
          part=basic_parts.SummarizableCondition(
              summary=MockRenderableTreePart(tag='A'),
              detail=MockRenderableTreePart(tag='B'),
          ),
          expected_width=100,
          expected_newlines=10,
          expected_tags=frozenset({'B', 'A'}),
          expected_foldables=(MockFoldableTreeNode(tag='B'),),
          expected_setup_parts={
              CSSStyleRule(rule='[mock rule A]'),
              CSSStyleRule(rule='[mock rule B]'),
              ...,
          },
          expected_text_collapsed='[mock A, e:n r:n]',
          expected_text_expanded='[mock B, e:y r:n]',
          expected_text_roundtrip_collapsed='[mock B, e:n r:y]',
          expected_text_roundtrip_expanded='[mock B, e:y r:y]',
          expected_html=(
              '<span class="when_collapsed_and_not_roundtrip">[mock A,'
              ' b:n]</span><span class="when_expanded_or_roundtrip">[mock B,'
              ' b:n]</span>'
          ),
          expected_html_at_begining=(
              '<span class="when_collapsed_and_not_roundtrip">[mock A,'
              ' b:y]</span><span class="when_expanded_or_roundtrip">[mock B,'
              ' b:y]</span>'
          ),
      ),
      dict(
          testcase_name='on_separate_lines',
          part=basic_parts.OnSeparateLines([
              MockRenderableTreePart('A'),
              MockRenderableTreePart('B'),
              MockRenderableTreePart('C'),
          ]),
          indent=4,
          expected_width=300,
          expected_newlines=34,
          expected_tags=frozenset({'C', 'B', 'A'}),
          expected_foldables=[
              MockFoldableTreeNode(tag='A'),
              MockFoldableTreeNode(tag='B'),
              MockFoldableTreeNode(tag='C'),
          ],
          expected_setup_parts={
              CSSStyleRule(rule='[mock rule A]'),
              CSSStyleRule(rule='[mock rule B]'),
              CSSStyleRule(rule='[mock rule C]'),
              ...,
          },
          expected_text_collapsed=(
              '[mock A, e:n r:n][mock B, e:n r:n][mock C, e:n r:n]'
          ),
          expected_text_expanded=(
              '\n    [mock A, e:y r:n]'
              '\n    [mock B, e:y r:n]'
              '\n    [mock C, e:y r:n]'
              '\n    '
          ),
          expected_text_roundtrip_collapsed=(
              '[mock A, e:n r:y][mock B, e:n r:y][mock C, e:n r:y]'
          ),
          expected_text_roundtrip_expanded=(
              '\n    [mock A, e:y r:y]'
              '\n    [mock B, e:y r:y]'
              '\n    [mock C, e:y r:y]'
              '\n    '
          ),
          expected_html=(
              '<span class="separate_lines_children"><span'
              ' class="separate_lines_child">[mock A, b:y]</span><span'
              ' class="separate_lines_child">[mock B, b:y]</span><span'
              ' class="separate_lines_child">[mock C, b:y]</span></span>'
          ),
          expected_html_at_begining=(
              '<span class="separate_lines_children"><span'
              ' class="separate_lines_child">[mock A, b:y]</span><span'
              ' class="separate_lines_child">[mock B, b:y]</span><span'
              ' class="separate_lines_child">[mock C, b:y]</span></span>'
          ),
      ),
      dict(
          testcase_name='indented_children',
          part=basic_parts.IndentedChildren([
              MockRenderableTreePart('A'),
              MockRenderableTreePart('B'),
              MockRenderableTreePart('C'),
          ]),
          indent=4,
          expected_width=300,
          expected_newlines=34,
          expected_tags=frozenset({'C', 'B', 'A'}),
          expected_foldables=[
              MockFoldableTreeNode(tag='A'),
              MockFoldableTreeNode(tag='B'),
              MockFoldableTreeNode(tag='C'),
          ],
          expected_setup_parts={
              CSSStyleRule(rule='[mock rule A]'),
              CSSStyleRule(rule='[mock rule B]'),
              CSSStyleRule(rule='[mock rule C]'),
              ...,
          },
          expected_text_collapsed=(
              '[mock A, e:n r:n][mock B, e:n r:n][mock C, e:n r:n]'
          ),
          expected_text_expanded=(
              '\n      [mock A, e:y r:n]'
              '\n      [mock B, e:y r:n]'
              '\n      [mock C, e:y r:n]'
              '\n    '
          ),
          expected_text_roundtrip_collapsed=(
              '[mock A, e:n r:y][mock B, e:n r:y][mock C, e:n r:y]'
          ),
          expected_text_roundtrip_expanded=(
              '\n      [mock A, e:y r:y]'
              '\n      [mock B, e:y r:y]'
              '\n      [mock C, e:y r:y]'
              '\n    '
          ),
          expected_html=(
              '<span class="indented_children"><span'
              ' class="indented_child">[mock A, b:y]</span><span'
              ' class="indented_child">[mock B, b:y]</span><span'
              ' class="indented_child">[mock C, b:y]</span></span>'
          ),
          expected_html_at_begining=(
              '<span class="indented_children"><span'
              ' class="indented_child">[mock A, b:y]</span><span'
              ' class="indented_child">[mock B, b:y]</span><span'
              ' class="indented_child">[mock C, b:y]</span></span>'
          ),
      ),
      dict(
          testcase_name='box_with_outline',
          part=BoxForTest(MockRenderableTreePart('A')),
          indent=4,
          expected_width=100,
          expected_newlines=12,
          expected_tags=frozenset({'A'}),
          expected_foldables=(MockFoldableTreeNode(tag='A'),),
          expected_setup_parts={
              CSSStyleRule(rule='[mock rule A]'),
              ...,
          },
          expected_text_collapsed='[mock A, e:n r:n]',
          expected_text_expanded=(
              '\n    #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮'
              '\n'
              '\n    [mock A, e:y r:n]'
              '\n    #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯'
              '\n    '
          ),
          expected_text_roundtrip_collapsed='[mock A, e:n r:y]',
          expected_text_roundtrip_expanded=(
              '\n    #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮'
              '\n'
              '\n    [mock A, e:y r:y]'
              '\n    #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯'
              '\n    '
          ),
          expected_html=(
              '<span class="outerbox_for_outline"><span class="box_with_outline'
              ' test_box_css_class">[mock A, b:y]</span></span>'
          ),
          expected_html_at_begining=(
              '<span class="outerbox_for_outline"><span class="box_with_outline'
              ' test_box_css_class">[mock A, b:y]</span></span>'
          ),
      ),
      dict(
          testcase_name='with_hover_tooltip',
          part=basic_parts.WithHoverTooltip(
              MockRenderableTreePart('A'), tooltip='test_tooltip'
          ),
          indent=4,
          expected_width=100,
          expected_newlines=10,
          expected_tags=frozenset({'A'}),
          expected_foldables=(MockFoldableTreeNode(tag='A'),),
          expected_setup_parts={
              CSSStyleRule(rule='[mock rule A]'),
              ...,
          },
          expected_text_collapsed='[mock A, e:n r:n]',
          expected_text_expanded='[mock A, e:y r:n]',
          expected_text_roundtrip_collapsed='[mock A, e:n r:y]',
          expected_text_roundtrip_expanded='[mock A, e:y r:y]',
          expected_html=(
              '<span class="has_hover_tooltip"'
              ' data-tooltip="test_tooltip">[mock A, b:n]</span>'
          ),
          expected_html_at_begining=(
              '<span class="has_hover_tooltip"'
              ' data-tooltip="test_tooltip">[mock A, b:y]</span>'
          ),
      ),
      dict(
          testcase_name='scoped_unselectable',
          part=basic_parts.ScopedSelectableAnnotation(
              MockRenderableTreePart('A')
          ),
          indent=4,
          expected_width=100,
          expected_newlines=10,
          expected_tags=frozenset({'A'}),
          expected_foldables=(MockFoldableTreeNode(tag='A'),),
          expected_setup_parts={
              CSSStyleRule(rule='[mock rule A]'),
              ...,
          },
          expected_text_collapsed='[mock A, e:n r:n]',
          expected_text_expanded='[mock A, e:y r:n]',
          expected_text_roundtrip_collapsed='',
          expected_text_roundtrip_expanded='\n    # [mock A, e:y r:y]\n    ',
          expected_html=(
              '<span tabindex="-1" class="scoped_unselectable">[mock A,'
              ' b:n]</span>'
          ),
          expected_html_at_begining=(
              '<span tabindex="-1" class="scoped_unselectable">[mock A,'
              ' b:y]</span>'
          ),
      ),
  )
  def test_parts(
      self,
      *,
      part: basic_parts.RenderableTreePart,
      expected_width: int | type(NotImplemented),
      expected_newlines: int,
      expected_tags: set[Any],
      expected_foldables: Sequence[basic_parts.FoldableTreeNode],
      expected_setup_parts: set[Any],
      expected_text_collapsed: str,
      expected_text_expanded: str,
      expected_text_roundtrip_collapsed: str,
      expected_text_roundtrip_expanded: str,
      expected_html: str,
      expected_html_at_begining: str,
      indent: int = 0,
  ):
    with self.subTest('collapsed_width'):
      if expected_width is not NotImplemented:
        self.assertEqual(part.collapsed_width, expected_width)

    with self.subTest('newlines_in_expanded_parent'):
      self.assertEqual(part.newlines_in_expanded_parent, expected_newlines)

    with self.subTest('tags_in_this_part'):
      self.assertEqual(part.tags_in_this_part, expected_tags)

    with self.subTest('foldables_in_this_part'):
      self.assertEqual(part.foldables_in_this_part(), expected_foldables)

    with self.subTest('setup_parts'):
      if isinstance(expected_setup_parts, set) and ... in expected_setup_parts:
        self.assertContainsSubset(
            expected_setup_parts - {...}, part.html_setup_parts(mock_context)
        )
      else:
        self.assertEqual(
            expected_setup_parts, part.html_setup_parts(mock_context)
        )

    # Check rendering in text mode.
    with self.subTest('text_collapsed'):
      stream = io.StringIO()
      part.render_to_text(
          stream,
          expanded_parent=False,
          indent=indent,
          roundtrip_mode=False,
          render_context={},
      )
      self.assertEqual(stream.getvalue(), expected_text_collapsed)

    with self.subTest('text_expanded'):
      stream = io.StringIO()
      part.render_to_text(
          stream,
          expanded_parent=True,
          indent=indent,
          roundtrip_mode=False,
          render_context={},
      )
      self.assertEqual(stream.getvalue(), expected_text_expanded)

    with self.subTest('text_roundtrip_collapsed'):
      stream = io.StringIO()
      part.render_to_text(
          stream,
          expanded_parent=False,
          indent=indent,
          roundtrip_mode=True,
          render_context={},
      )
      self.assertEqual(stream.getvalue(), expected_text_roundtrip_collapsed)

    with self.subTest('text_roundtrip_expanded'):
      stream = io.StringIO()
      part.render_to_text(
          stream,
          expanded_parent=True,
          indent=indent,
          roundtrip_mode=True,
          render_context={},
      )
      self.assertEqual(stream.getvalue(), expected_text_roundtrip_expanded)

    with self.subTest('html'):
      stream = io.StringIO()
      part.render_to_html(stream, at_beginning_of_line=False, render_context={})
      self.assertEqual(stream.getvalue(), expected_html)

    with self.subTest('html_at_beginning'):
      stream = io.StringIO()
      part.render_to_html(stream, at_beginning_of_line=True, render_context={})
      self.assertEqual(stream.getvalue(), expected_html_at_begining)

  def test_multiline_text_part(self):
    part = basic_parts.Text('some text\nsecond line\nthird line')
    self.assertEqual(part.newlines_in_expanded_parent, 2)

    stream = io.StringIO()
    part.render_to_text(
        stream,
        expanded_parent=True,
        indent=4,
        roundtrip_mode=False,
        render_context={},
    )
    self.assertEqual(
        stream.getvalue(),
        'some text\n    second line\n    third line',
    )

  def test_build_siblings(self):
    part = basic_parts.siblings(
        'foo',
        MockRenderableTreePart(tag='A'),
        'bar',
        basic_parts.Siblings(
            [MockRenderableTreePart('B'), MockRenderableTreePart('C')],
        ),
        basic_parts.EmptyPart(),
        'baz',
    )
    self.assertEqual(
        part,
        basic_parts.Siblings(
            children=(
                basic_parts.Text(text='foo'),
                MockRenderableTreePart(tag='A'),
                basic_parts.Text(text='bar'),
                MockRenderableTreePart(tag='B'),
                MockRenderableTreePart(tag='C'),
                basic_parts.Text(text='baz'),
            )
        ),
    )

  def test_build_siblings_with_annotations(self):
    part = basic_parts.siblings_with_annotations(
        'foo',
        basic_parts.RenderableAndLineAnnotations(
            MockRenderableTreePart(tag='A'),
            MockRenderableTreePart(tag='annotation1'),
        ),
        'bar',
        basic_parts.Siblings(
            [MockRenderableTreePart('B'), MockRenderableTreePart('C')],
        ),
        basic_parts.RenderableAndLineAnnotations(
            MockRenderableTreePart(tag='D'),
            MockRenderableTreePart(tag='annotation2'),
        ),
        basic_parts.EmptyPart(),
        'baz',
        extra_annotations=[
            MockRenderableTreePart(tag='annotation3'),
        ],
    )
    self.assertEqual(
        part,
        basic_parts.RenderableAndLineAnnotations(
            renderable=basic_parts.Siblings(
                children=(
                    basic_parts.Text(text='foo'),
                    MockRenderableTreePart(tag='A'),
                    basic_parts.Text(text='bar'),
                    MockRenderableTreePart(tag='B'),
                    MockRenderableTreePart(tag='C'),
                    MockRenderableTreePart(tag='D'),
                    basic_parts.Text(text='baz'),
                )
            ),
            annotations=basic_parts.Siblings(
                children=(
                    MockRenderableTreePart(tag='annotation1'),
                    MockRenderableTreePart(tag='annotation2'),
                    MockRenderableTreePart(tag='annotation3'),
                )
            ),
        ),
    )

  def test_build_full_line_with_annotations(self):
    part = basic_parts.build_full_line_with_annotations(
        'foo',
        basic_parts.RenderableAndLineAnnotations(
            MockRenderableTreePart(tag='A'),
            MockRenderableTreePart(tag='annotation1'),
        ),
        'bar',
        basic_parts.RenderableAndLineAnnotations(
            MockRenderableTreePart(tag='B'),
            MockRenderableTreePart(tag='annotation2'),
        ),
    )
    self.assertEqual(
        part,
        basic_parts.Siblings(
            children=(
                basic_parts.Text(text='foo'),
                MockRenderableTreePart(tag='A'),
                basic_parts.Text(text='bar'),
                MockRenderableTreePart(tag='B'),
                basic_parts.FoldCondition(
                    collapsed=basic_parts.EmptyPart(),
                    expanded=basic_parts.Siblings(
                        children=(
                            MockRenderableTreePart(tag='annotation1'),
                            MockRenderableTreePart(tag='annotation2'),
                        )
                    ),
                ),
            )
        ),
    )

  def test_build_children(self):
    children = [
        'foo',
        basic_parts.RenderableAndLineAnnotations(
            MockRenderableTreePart(tag='A'),
            MockRenderableTreePart(tag='annotation2'),
        ),
        'bar',
        MockRenderableTreePart(tag='B'),
    ]
    with self.subTest('OnSeparateLines'):
      self.assertEqual(
          basic_parts.OnSeparateLines.build(children),
          basic_parts.OnSeparateLines(
              children=[
                  basic_parts.Siblings(
                      children=(basic_parts.Text(text='foo'),)
                  ),
                  basic_parts.Siblings(
                      children=(
                          MockRenderableTreePart(tag='A'),
                          basic_parts.FoldCondition(
                              collapsed=basic_parts.EmptyPart(),
                              expanded=basic_parts.Siblings(
                                  children=(
                                      MockRenderableTreePart(tag='annotation2'),
                                  )
                              ),
                          ),
                      )
                  ),
                  basic_parts.Siblings(
                      children=(basic_parts.Text(text='bar'),)
                  ),
                  basic_parts.Siblings(
                      children=(MockRenderableTreePart(tag='B'),)
                  ),
              ]
          ),
      )
    with self.subTest('IndentedChildren'):
      self.assertEqual(
          basic_parts.IndentedChildren.build(children),
          basic_parts.IndentedChildren(
              children=[
                  basic_parts.Siblings(
                      children=(basic_parts.Text(text='foo'),)
                  ),
                  basic_parts.Siblings(
                      children=(
                          MockRenderableTreePart(tag='A'),
                          basic_parts.FoldCondition(
                              collapsed=basic_parts.EmptyPart(),
                              expanded=basic_parts.Siblings(
                                  children=(
                                      MockRenderableTreePart(tag='annotation2'),
                                  )
                              ),
                          ),
                      )
                  ),
                  basic_parts.Siblings(
                      children=(basic_parts.Text(text='bar'),)
                  ),
                  basic_parts.Siblings(
                      children=(MockRenderableTreePart(tag='B'),)
                  ),
              ]
          ),
      )
    with self.subTest('IndentedChildren_comma_separated'):
      self.assertEqual(
          basic_parts.IndentedChildren.build(children, comma_separated=True),
          basic_parts.IndentedChildren(
              children=[
                  basic_parts.Siblings(
                      children=(
                          basic_parts.Text(text='foo'),
                          basic_parts.Text(text=','),
                          basic_parts.FoldCondition(
                              collapsed=basic_parts.Text(text=' '),
                              expanded=basic_parts.EmptyPart(),
                          ),
                      )
                  ),
                  basic_parts.Siblings(
                      children=(
                          MockRenderableTreePart(tag='A'),
                          basic_parts.Text(text=','),
                          basic_parts.FoldCondition(
                              collapsed=basic_parts.Text(text=' '),
                              expanded=basic_parts.EmptyPart(),
                          ),
                          basic_parts.FoldCondition(
                              collapsed=basic_parts.EmptyPart(),
                              expanded=basic_parts.Siblings(
                                  children=(
                                      MockRenderableTreePart(tag='annotation2'),
                                  )
                              ),
                          ),
                      )
                  ),
                  basic_parts.Siblings(
                      children=(
                          basic_parts.Text(text='bar'),
                          basic_parts.Text(text=','),
                          basic_parts.FoldCondition(
                              collapsed=basic_parts.Text(text=' '),
                              expanded=basic_parts.EmptyPart(),
                          ),
                      )
                  ),
                  basic_parts.Siblings(
                      children=(
                          MockRenderableTreePart(tag='B'),
                          basic_parts.FoldCondition(
                              collapsed=basic_parts.EmptyPart(),
                              expanded=basic_parts.Text(text=','),
                          ),
                      )
                  ),
              ]
          ),
      )

  def test_render_as_root(self):
    part = basic_parts.siblings(
        basic_parts.OnSeparateLines.build(
            [MockRenderableTreePart('A'), '', '', MockRenderableTreePart('B')]
        )
    )
    with self.subTest('text_root_default'):
      self.assertEqual(
          foldable_impl.render_to_text_as_root(part),
          '[mock A, e:y r:n]\n[mock B, e:y r:n]\n',
      )
    with self.subTest('text_root_no_strip'):
      self.assertEqual(
          foldable_impl.render_to_text_as_root(
              part, strip_whitespace_lines=False
          ),
          '\n[mock A, e:y r:n]\n\n\n[mock B, e:y r:n]\n',
      )
    with self.subTest('text_root_roundtrip'):
      self.assertEqual(
          foldable_impl.render_to_text_as_root(part, roundtrip=True),
          '[mock A, e:y r:y]\n[mock B, e:y r:y]\n',
      )
    with self.subTest('html_root'):
      # We don't test the full HTML source since it includes implementation
      # details; just test that it executes correctly and does contain the
      # rendered objects somewhere.
      html_output = foldable_impl.render_to_html_as_root(part)
      self.assertContainsInOrder(
          ['[mock A, b:y]', '[mock B, b:y]'], html_output
      )

  def test_build_qualified_type_name(self):
    canonical_aliases.update_lazy_aliases()
    part = common_structures.maybe_qualified_type_name(jax.ShapeDtypeStruct)
    self.assertEqual(
        foldable_impl.render_to_text_as_root(part),
        'ShapeDtypeStruct',
    )
    self.assertEqual(
        foldable_impl.render_to_text_as_root(part, roundtrip=True),
        'jax.ShapeDtypeStruct',
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='build_one_line_tree_node_1',
          part=common_structures.build_one_line_tree_node(
              line='Some contents',
          ),
          expected_collapsed='Some contents',
          expected_expanded='Some contents',
      ),
      dict(
          testcase_name='build_one_line_tree_node_2',
          part=common_structures.build_one_line_tree_node(
              line=basic_parts.RenderableAndLineAnnotations(
                  MockRenderableTreePart('A'),
                  annotations=basic_parts.Text('# line comment'),
              ),
              path=(),
          ),
          # No foldable, so nothing to collapse; child is still "expanded".
          expected_collapsed='[mock A, e:y r:n]',
          expected_expanded='[mock A, e:y r:n]',
          expected_annotations='# line comment',
      ),
      dict(
          testcase_name='build_foldable_tree_node_from_children_1',
          part=common_structures.build_foldable_tree_node_from_children(
              prefix='Wrapper(',
              children=[
                  'foo',
                  MockRenderableTreePart('A'),
              ],
              suffix=')',
              comma_separated=False,
          ),
          expected_collapsed='Wrapper(foo[mock A, e:n r:n])',
          expected_expanded='Wrapper(\n  foo\n  [mock A, e:y r:n]\n)',
      ),
      dict(
          testcase_name='build_foldable_tree_node_from_children_2',
          part=common_structures.build_foldable_tree_node_from_children(
              prefix='Wrapper(',
              children=[
                  'foo',
                  MockRenderableTreePart('A'),
              ],
              suffix=')',
              path=(),
              comma_separated=True,
          ),
          expected_collapsed='Wrapper(foo, [mock A, e:n r:n])',
          expected_expanded='Wrapper(\n  foo,\n  [mock A, e:y r:n],\n)',
      ),
      dict(
          testcase_name='build_foldable_tree_node_from_children_3',
          part=common_structures.build_foldable_tree_node_from_children(
              prefix='Wrapper(',
              children=[
                  'foo',
                  basic_parts.RenderableAndLineAnnotations(
                      MockRenderableTreePart('A'),
                      annotations=basic_parts.Text('# line comment'),
                  ),
              ],
              suffix=')',
              path=(),
              comma_separated=True,
              force_trailing_comma=True,
          ),
          expected_collapsed='Wrapper(foo, [mock A, e:n r:n],)',
          expected_expanded=(
              'Wrapper(\n  foo,\n  [mock A, e:y r:n],# line comment\n)'
          ),
      ),
      dict(
          testcase_name='build_custom_foldable_tree_node',
          part=common_structures.build_custom_foldable_tree_node(
              label=basic_parts.Text('foo'),
              contents=basic_parts.siblings(
                  'bar',
                  basic_parts.OnSeparateLines.build([
                      'foo',
                      basic_parts.RenderableAndLineAnnotations(
                          MockRenderableTreePart('A'),
                          annotations=basic_parts.Text('# line comment'),
                      ),
                  ]),
              ),
              path=(),
          ),
          expected_collapsed='foobarfoo[mock A, e:n r:n]',
          expected_expanded='foobar\nfoo\n[mock A, e:y r:n]# line comment\n',
      ),
  )
  def test_render_common_structure(
      self,
      part: basic_parts.RenderableAndLineAnnotations,
      *,
      expected_collapsed: str,
      expected_expanded: str,
      expected_annotations: str = '',
  ):
    with self.subTest('annotations'):
      self.assertEqual(
          foldable_impl.render_to_text_as_root(part.annotations),
          expected_annotations,
      )

    for foldable in part.renderable.foldables_in_this_part():
      foldable.set_expand_state(basic_parts.ExpandState.COLLAPSED)

    with self.subTest('text_collapsed'):
      self.assertEqual(
          foldable_impl.render_to_text_as_root(part.renderable),
          expected_collapsed,
      )

    for foldable in part.renderable.foldables_in_this_part():
      foldable.set_expand_state(basic_parts.ExpandState.EXPANDED)

    with self.subTest('text_expanded'):
      self.assertEqual(
          foldable_impl.render_to_text_as_root(part.renderable),
          expected_expanded,
      )


if __name__ == '__main__':
  absltest.main()
