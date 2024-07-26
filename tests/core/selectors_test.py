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

"""Tests for selectors.

See also `partitioning_test` for tests of `Selection.partition`, and
`treescope_selection_rendering_test` for rendering of selections.
"""

import collections
import dataclasses
from typing import Any

from absl.testing import absltest
import jax
from penzai import pz
import penzai.core.selectors


@dataclasses.dataclass
class CustomLeaf:
  """A leaf type for use in tests."""

  tag: int


def make_example_object():
  return {
      'a': 1,
      'b': CustomLeaf(10),
      'c': [
          {'value': CustomLeaf(12)},
          {'value': CustomLeaf(13)},
          {'value': 3},
      ],
  }


@dataclasses.dataclass
class SELECTED_PART:  # pylint: disable=invalid-name
  """(Visually-distinctive) marker class for the selected subtree in tests."""

  value: Any


class SelectorsTest(absltest.TestCase):

  def test_select(self):
    self.assertEqual(
        pz.select(make_example_object()),
        pz.Selection(
            selected_by_path=collections.OrderedDict({
                (): {
                    'a': 1,
                    'b': CustomLeaf(tag=10),
                    'c': [
                        {'value': CustomLeaf(tag=12)},
                        {'value': CustomLeaf(tag=13)},
                        {'value': 3},
                    ],
                }
            }),
            remainder=penzai.core.selectors.SelectionHole(path=()),
        ),
    )

  def test_select_deselect(self):
    self.assertEqual(
        pz.select(make_example_object()).at_instances_of(int).deselect(),
        make_example_object(),
    )

  def test_select_at_accessor__example_1(self):
    selection = pz.select(make_example_object()).at(lambda root: root['c'])
    self.assertEqual(
        selection,
        pz.Selection(
            selected_by_path=collections.OrderedDict({
                (jax.tree_util.DictKey(key='c'),): [
                    {'value': CustomLeaf(tag=12)},
                    {'value': CustomLeaf(tag=13)},
                    {'value': 3},
                ]
            }),
            remainder={
                'a': 1,
                'b': CustomLeaf(tag=10),
                'c': penzai.core.selectors.SelectionHole(
                    path=(jax.tree_util.DictKey(key='c'),)
                ),
            },
        ),
    )
    self.assertEqual(
        selection.apply(SELECTED_PART),
        {
            'a': 1,
            'b': CustomLeaf(tag=10),
            'c': SELECTED_PART(
                value=[
                    {'value': CustomLeaf(tag=12)},
                    {'value': CustomLeaf(tag=13)},
                    {'value': 3},
                ]
            ),
        },
    )

  def test_select_at_accessor__example_2(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at(lambda root: root['c'][1]['value'])
            .apply(SELECTED_PART)
        ),
        {
            'a': 1,
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': CustomLeaf(tag=12)},
                {'value': SELECTED_PART(value=CustomLeaf(tag=13))},
                {'value': 3},
            ],
        },
    )

  def test_select_at_accessor_multiple_and_chaining(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at(lambda root: (root['c'][0], root['c'][1]), multiple=True)
            .at(lambda subtree: subtree['value'])
            .apply(SELECTED_PART)
        ),
        {
            'a': 1,
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': SELECTED_PART(value=CustomLeaf(tag=12))},
                {'value': SELECTED_PART(value=CustomLeaf(tag=13))},
                {'value': 3},
            ],
        },
    )

  def test_select_at_accessor_deprecated_auto_multiple(self):
    with self.assertWarns(UserWarning) as cm:
      self.assertEqual(
          (
              pz.select(make_example_object())
              .at(lambda root: (root['c'][0], root['c'][1]))
              .apply(SELECTED_PART)
          ),
          {
              'a': 1,
              'b': CustomLeaf(tag=10),
              'c': [
                  SELECTED_PART({'value': CustomLeaf(tag=12)}),
                  SELECTED_PART({'value': CustomLeaf(tag=13)}),
                  {'value': 3},
              ],
          },
      )
    self.assertIn(
        'Returning a collection of nodes from the accessor function for'
        ' `Selection.at` without passing `multiple=True` is deprecated.',
        str(cm.warning),
    )

  def test_select_at_accessor_fails_not_multiple(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        lambda exc: (
            'accessor_fn returned a value that was not found in the tree'
            in str(exc)
        ),
    ):
      _ = (
          pz.select(make_example_object())
          .at(lambda root: (root['c'][0], root['c'][1]), multiple=False)
          .apply(SELECTED_PART)
      )

  def test_select_at_accessor_can_select_singletons(self):
    # The singletons None and () have the same ID even when flattened and
    # unflattened by jax.tree_util, so they need special handling to be able
    # to select.
    thing_with_singletons = [None, None, None, None, (), (), (), ()]
    self.assertEqual(
        (
            pz.select(thing_with_singletons)
            .at(lambda r: (r[1], r[2], r[5], r[7]), multiple=True)
            .apply(SELECTED_PART)
        ),
        [
            None,
            SELECTED_PART(None),
            SELECTED_PART(None),
            None,
            (),
            SELECTED_PART(()),
            (),
            SELECTED_PART(()),
        ],
    )

  def test_select_at_instances_of__example_1(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(int)
            .apply(SELECTED_PART)
        ),
        {
            'a': SELECTED_PART(value=1),
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': CustomLeaf(tag=12)},
                {'value': CustomLeaf(tag=13)},
                {'value': SELECTED_PART(value=3)},
            ],
        },
    )

  def test_select_at_instances_of__example_2(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(list)
            .apply(SELECTED_PART)
        ),
        {
            'a': 1,
            'b': CustomLeaf(tag=10),
            'c': SELECTED_PART(
                value=[
                    {'value': CustomLeaf(tag=12)},
                    {'value': CustomLeaf(tag=13)},
                    {'value': 3},
                ]
            ),
        },
    )

  def test_select_at_instances_of__example_3(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(dict)
            .apply(SELECTED_PART)
        ),
        SELECTED_PART(
            value={
                'a': 1,
                'b': CustomLeaf(tag=10),
                'c': [
                    {'value': CustomLeaf(tag=12)},
                    {'value': CustomLeaf(tag=13)},
                    {'value': 3},
                ],
            }
        ),
    )

  def test_select_at_subtrees_where(self):
    predicate = lambda node: isinstance(node, CustomLeaf) and node.tag <= 10
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_subtrees_where(predicate)
            .apply(SELECTED_PART)
        ),
        {
            'a': 1,
            'b': SELECTED_PART(value=CustomLeaf(tag=10)),
            'c': [
                {'value': CustomLeaf(tag=12)},
                {'value': CustomLeaf(tag=13)},
                {'value': 3},
            ],
        },
    )

  def test_select_at_subtrees_where_with_keypath(self):
    def predicate(path, node):
      return isinstance(node, CustomLeaf) and len(path) == 3

    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_subtrees_where(predicate, with_keypath=True)
            .apply(SELECTED_PART)
        ),
        {
            'a': 1,
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': SELECTED_PART(value=CustomLeaf(tag=12))},
                {'value': SELECTED_PART(value=CustomLeaf(tag=13))},
                {'value': 3},
            ],
        },
    )

  def test_select_at_subtrees_where_with_absolute_keypath(self):
    seen_keypaths = []

    def predicate(path, node):
      if isinstance(node, CustomLeaf):
        seen_keypaths.append(path)
        return True
      else:
        return False

    self.assertEqual(
        (
            pz.select(make_example_object())
            .at(lambda root: root['c'])
            .at_subtrees_where(
                predicate, with_keypath=True, absolute_keypath=False
            )
            .apply(SELECTED_PART)
        ),
        {
            'a': 1,
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': SELECTED_PART(value=CustomLeaf(tag=12))},
                {'value': SELECTED_PART(value=CustomLeaf(tag=13))},
                {'value': 3},
            ],
        },
    )
    self.assertEqual(
        seen_keypaths,
        [
            (
                jax.tree_util.SequenceKey(idx=0),
                jax.tree_util.DictKey(key='value'),
            ),
            (
                jax.tree_util.SequenceKey(idx=1),
                jax.tree_util.DictKey(key='value'),
            ),
        ],
    )

    seen_keypaths = []
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at(lambda root: root['c'])
            .at_subtrees_where(
                predicate, with_keypath=True, absolute_keypath=True
            )
            .apply(SELECTED_PART)
        ),
        {
            'a': 1,
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': SELECTED_PART(value=CustomLeaf(tag=12))},
                {'value': SELECTED_PART(value=CustomLeaf(tag=13))},
                {'value': 3},
            ],
        },
    )
    self.assertEqual(
        seen_keypaths,
        [
            (
                jax.tree_util.DictKey(key='c'),
                jax.tree_util.SequenceKey(idx=0),
                jax.tree_util.DictKey(key='value'),
            ),
            (
                jax.tree_util.DictKey(key='c'),
                jax.tree_util.SequenceKey(idx=1),
                jax.tree_util.DictKey(key='value'),
            ),
        ],
    )

  def test_select_at_subtrees_where_innermost(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_subtrees_where(
                lambda node: isinstance(node, dict), innermost=True
            )
            .apply(SELECTED_PART)
        ),
        {
            'a': 1,
            'b': CustomLeaf(tag=10),
            'c': [
                SELECTED_PART(value={'value': CustomLeaf(tag=12)}),
                SELECTED_PART(value={'value': CustomLeaf(tag=13)}),
                SELECTED_PART(value={'value': 3}),
            ],
        },
    )

  def test_select_at_equal_to__example_1(self):
    self.assertEqual(
        (pz.select(make_example_object()).at_equal_to(1).apply(SELECTED_PART)),
        {
            'a': SELECTED_PART(value=1),
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': CustomLeaf(tag=12)},
                {'value': CustomLeaf(tag=13)},
                {'value': 3},
            ],
        },
    )

  def test_select_at_equal_to__example_2(self):
    self.assertEqual(
        (
            pz.select({'foo': 'foo', 'bar': [1, 2, 3]})
            .at_equal_to([1, 2, 3])
            .apply(SELECTED_PART)
        ),
        {'bar': SELECTED_PART(value=[1, 2, 3]), 'foo': 'foo'},
    )

  def test_select_at_keypaths(self):
    keypaths = [
        (
            jax.tree_util.DictKey(key='c'),
            jax.tree_util.SequenceKey(idx=2),
            jax.tree_util.DictKey(key='value'),
        ),
        (jax.tree_util.DictKey(key='c'), jax.tree_util.SequenceKey(idx=0)),
    ]
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_keypaths(keypaths)
            .apply(SELECTED_PART)
        ),
        {
            'a': 1,
            'b': CustomLeaf(tag=10),
            'c': [
                SELECTED_PART(value={'value': CustomLeaf(tag=12)}),
                {'value': CustomLeaf(tag=13)},
                {'value': SELECTED_PART(value=3)},
            ],
        },
    )

  def test_select_at_children(self):
    self.assertEqual(
        (pz.select(make_example_object()).at_children().apply(SELECTED_PART)),
        {
            'a': SELECTED_PART(value=1),
            'b': SELECTED_PART(value=CustomLeaf(tag=10)),
            'c': SELECTED_PART(
                value=[
                    {'value': CustomLeaf(tag=12)},
                    {'value': CustomLeaf(tag=13)},
                    {'value': 3},
                ]
            ),
        },
    )

  def test_select_at_pytree_leaves__example_1(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_pytree_leaves()
            .apply(SELECTED_PART)
        ),
        {
            'a': SELECTED_PART(value=1),
            'b': SELECTED_PART(value=CustomLeaf(tag=10)),
            'c': [
                {'value': SELECTED_PART(value=CustomLeaf(tag=12))},
                {'value': SELECTED_PART(value=CustomLeaf(tag=13))},
                {'value': SELECTED_PART(value=3)},
            ],
        },
    )

  def test_select_at_pytree_leaves__example_2(self):
    self.assertEqual(
        (
            pz.select([1, 2, (), 3, None, 4, {'a': 5, 'b': {}}])
            .at_pytree_leaves()
            .apply(SELECTED_PART)
        ),
        [
            SELECTED_PART(value=1),
            SELECTED_PART(value=2),
            (),
            SELECTED_PART(value=3),
            None,
            SELECTED_PART(value=4),
            {'a': SELECTED_PART(value=5), 'b': {}},
        ],
    )

  def test_select_at_childless(self):
    self.assertEqual(
        (
            pz.select([1, 2, (), 3, None, 4, {'a': 5, 'b': {}}])
            .at_childless()
            .apply(SELECTED_PART)
        ),
        [
            SELECTED_PART(value=1),
            SELECTED_PART(value=2),
            SELECTED_PART(value=()),
            SELECTED_PART(value=3),
            SELECTED_PART(value=None),
            SELECTED_PART(value=4),
            {'a': SELECTED_PART(value=5), 'b': SELECTED_PART(value={})},
        ],
    )

  def test_select_where(self):
    self.assertEqual(
        (
            pz.select(list(range(10)))
            .at_instances_of(int)
            .where(lambda x: x % 2 == 0)
            .apply(SELECTED_PART)
        ),
        [
            SELECTED_PART(value=0),
            1,
            SELECTED_PART(value=2),
            3,
            SELECTED_PART(value=4),
            5,
            SELECTED_PART(value=6),
            7,
            SELECTED_PART(value=8),
            9,
        ],
    )

  def test_pick_nth_selected(self):
    self.assertEqual(
        (
            pz.select(list(range(10)))
            .at_instances_of(int)
            .where(lambda x: x % 2 == 0)
            .pick_nth_selected(2)
            .apply(SELECTED_PART)
        ),
        [0, 1, 2, 3, SELECTED_PART(value=4), 5, 6, 7, 8, 9],
    )

  def test_invert__example_1(self):
    predicate = lambda node: isinstance(node, CustomLeaf) and node.tag <= 10
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_subtrees_where(predicate)
            .invert()
            .apply(SELECTED_PART)
        ),
        {
            'a': SELECTED_PART(value=1),
            'b': CustomLeaf(tag=10),
            'c': SELECTED_PART(
                value=[
                    {'value': CustomLeaf(tag=12)},
                    {'value': CustomLeaf(tag=13)},
                    {'value': 3},
                ]
            ),
        },
    )

  def test_invert__example_2(self):
    self.assertEqual(
        (
            pz.select([1, 'a', 2, 'b', 3, 'c', 4, 'd'])
            .at_instances_of(int)
            .pick_nth_selected(2)
            .invert()
            .apply(SELECTED_PART)
        ),
        [
            SELECTED_PART(value=1),
            SELECTED_PART(value='a'),
            SELECTED_PART(value=2),
            SELECTED_PART(value='b'),
            3,
            SELECTED_PART(value='c'),
            SELECTED_PART(value=4),
            SELECTED_PART(value='d'),
        ],
    )

  def test_refine(self):
    # Select field "a" of Foo and field "b" of Bar

    @pz.pytree_dataclass
    class Foo(pz.Struct):
      a: Any
      b: Any

    @pz.pytree_dataclass
    class Bar(pz.Struct):
      a: Any
      b: Any

    def refine_fn(value):
      if isinstance(value, Foo):
        return value.select().at(lambda x: x.a)
      else:
        return value.select().at(lambda x: x.b)

    foos_and_bars = [
        Foo(1, 2),
        Bar(3, 4),
        Foo(5, 6),
        Bar(7, 8),
        None,
    ]

    self.assertEqual(
        (
            pz.select(foos_and_bars)
            .at_instances_of((Foo, Bar))
            .refine(refine_fn)
            .apply(SELECTED_PART)
        ),
        [
            Foo(a=SELECTED_PART(value=1), b=2),
            Bar(a=3, b=SELECTED_PART(value=4)),
            Foo(a=SELECTED_PART(value=5), b=6),
            Bar(a=7, b=SELECTED_PART(value=8)),
            None,
        ],
    )

  def test_selection_size_summaries(self):
    selection_1 = pz.select([1, 2]).at_instances_of(int)
    self.assertEqual(selection_1.count(), 2)
    self.assertLen(selection_1, 2)
    self.assertFalse(selection_1.is_empty())
    self.assertTrue(bool(selection_1))

    selection_2 = pz.select([1, 2]).at_instances_of(str)
    self.assertEqual(selection_2.count(), 0)
    self.assertLen(selection_2, 0)  # pylint: disable=g-generic-assert
    self.assertTrue(selection_2.is_empty())
    self.assertFalse(bool(selection_2))

  def test_selection_get(self):
    self.assertEqual(
        pz.select(make_example_object()).at(lambda root: root['c'][0]).get(),
        {'value': CustomLeaf(tag=12)},
    )

  def test_selection_get_sequence(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(CustomLeaf)
            .get_sequence()
        ),
        (CustomLeaf(tag=10), CustomLeaf(tag=12), CustomLeaf(tag=13)),
    )

  def test_selection_get_by_path(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(CustomLeaf)
            .get_by_path()
        ),
        collections.OrderedDict({
            (jax.tree_util.DictKey(key='b'),): CustomLeaf(tag=10),
            (
                jax.tree_util.DictKey(key='c'),
                jax.tree_util.SequenceKey(idx=0),
                jax.tree_util.DictKey(key='value'),
            ): CustomLeaf(tag=12),
            (
                jax.tree_util.DictKey(key='c'),
                jax.tree_util.SequenceKey(idx=1),
                jax.tree_util.DictKey(key='value'),
            ): CustomLeaf(tag=13),
        }),
    )

  def test_selection_set(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(CustomLeaf)
            .set('REPLACED')
        ),
        {
            'a': 1,
            'b': 'REPLACED',
            'c': [
                {'value': 'REPLACED'},
                {'value': 'REPLACED'},
                {'value': 3},
            ],
        },
    )

  def test_selection_set_sequence(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(CustomLeaf)
            .set_sequence(['REPLACED_1', 'REPLACED_2', 'REPLACED_3'])
        ),
        {
            'a': 1,
            'b': 'REPLACED_1',
            'c': [
                {'value': 'REPLACED_2'},
                {'value': 'REPLACED_3'},
                {'value': 3},
            ],
        },
    )

  def test_selection_set_by_path_dict(self):
    selection = pz.select(make_example_object()).at_instances_of(CustomLeaf)
    new_values = {
        k: f'REPLACED: tag={v.tag}' for k, v in selection.get_by_path().items()
    }
    self.assertEqual(
        selection.set_by_path(new_values),
        {
            'a': 1,
            'b': 'REPLACED: tag=10',
            'c': [
                {'value': 'REPLACED: tag=12'},
                {'value': 'REPLACED: tag=13'},
                {'value': 3},
            ],
        },
    )

  def test_selection_set_by_path_fn(self):
    selection = pz.select(make_example_object()).at_instances_of(CustomLeaf)
    replacer = lambda path: f'REPLACED: {jax.tree_util.keystr(path)}'
    self.assertEqual(
        selection.set_by_path(replacer),
        {
            'a': 1,
            'b': "REPLACED: ['b']",
            'c': [
                {'value': "REPLACED: ['c'][0]['value']"},
                {'value': "REPLACED: ['c'][1]['value']"},
                {'value': 3},
            ],
        },
    )

  def test_select_and_set_by_path(self):
    selection = pz.select(make_example_object()).at_instances_of(CustomLeaf)
    new_values = {
        k: f'REPLACED: tag={v.tag}' for k, v in selection.get_by_path().items()
    }
    self.assertEqual(
        pz.select(make_example_object()).select_and_set_by_path(new_values),
        {
            'a': 1,
            'b': 'REPLACED: tag=10',
            'c': [
                {'value': 'REPLACED: tag=12'},
                {'value': 'REPLACED: tag=13'},
                {'value': 3},
            ],
        },
    )

  def test_selection_apply(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(int)
            .apply(lambda x: x + 1000)
        ),
        {
            'a': 1001,
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': CustomLeaf(tag=12)},
                {'value': CustomLeaf(tag=13)},
                {'value': 1003},
            ],
        },
    )

  def test_selection_apply_keep_selected(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(int)
            .apply(lambda x: x + 1000, keep_selected=True)
            .apply(SELECTED_PART)
        ),
        {
            'a': SELECTED_PART(value=1001),
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': CustomLeaf(tag=12)},
                {'value': CustomLeaf(tag=13)},
                {'value': SELECTED_PART(value=1003)},
            ],
        },
    )

  def test_selection_apply_with_keypath(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(int)
            .apply(
                lambda path, value: (  # pylint: disable=g-long-lambda
                    f'path={jax.tree_util.keystr(path)}, value={value}'
                ),
                with_keypath=True,
            )
        ),
        {
            'a': "path=['a'], value=1",
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': CustomLeaf(tag=12)},
                {'value': CustomLeaf(tag=13)},
                {'value': "path=['c'][2]['value'], value=3"},
            ],
        },
    )

  def test_selection_apply_with_keypath_keep_selected(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(int)
            .apply(
                lambda path, value: (  # pylint: disable=g-long-lambda
                    f'path={jax.tree_util.keystr(path)}, value={value}'
                ),
                with_keypath=True,
                keep_selected=True,
            )
            .apply(SELECTED_PART)
        ),
        {
            'a': SELECTED_PART(value="path=['a'], value=1"),
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': CustomLeaf(tag=12)},
                {'value': CustomLeaf(tag=13)},
                {
                    'value': SELECTED_PART(
                        value="path=['c'][2]['value'], value=3"
                    )
                },
            ],
        },
    )

  def test_selection_apply_with_selected_index(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(int)
            .apply_with_selected_index(
                lambda index, value: f'index={index}, value={value}'
            )
        ),
        {
            'a': 'index=0, value=1',
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': CustomLeaf(tag=12)},
                {'value': CustomLeaf(tag=13)},
                {'value': 'index=1, value=3'},
            ],
        },
    )

  def test_selection_apply_with_selected_index_keep_selected(self):
    self.assertEqual(
        (
            pz.select(make_example_object())
            .at_instances_of(int)
            .apply_with_selected_index(
                lambda index, value: f'index={index}, value={value}',
                keep_selected=True,
            )
            .apply(SELECTED_PART)
        ),
        {
            'a': SELECTED_PART(value='index=0, value=1'),
            'b': CustomLeaf(tag=10),
            'c': [
                {'value': CustomLeaf(tag=12)},
                {'value': CustomLeaf(tag=13)},
                {'value': SELECTED_PART(value='index=1, value=3')},
            ],
        },
    )

  def test_selection_insert_before(self):
    self.assertEqual(
        (
            pz.select({'a': list(range(10)), 'b': list(range(10, 20))})
            .at_instances_of(int)
            .where(lambda x: x % 4 == 0)
            .insert_before('before a multiple of 4')
        ),
        {
            'a': [
                'before a multiple of 4',
                0,
                1,
                2,
                3,
                'before a multiple of 4',
                4,
                5,
                6,
                7,
                'before a multiple of 4',
                8,
                9,
            ],
            'b': [
                10,
                11,
                'before a multiple of 4',
                12,
                13,
                14,
                15,
                'before a multiple of 4',
                16,
                17,
                18,
                19,
            ],
        },
    )

  def test_selection_insert_after(self):
    self.assertEqual(
        (
            pz.select({'a': list(range(10)), 'b': list(range(10, 20))})
            .at_instances_of(int)
            .where(lambda x: x % 4 == 0)
            .insert_after('after a multiple of 4')
        ),
        {
            'a': [
                0,
                'after a multiple of 4',
                1,
                2,
                3,
                4,
                'after a multiple of 4',
                5,
                6,
                7,
                8,
                'after a multiple of 4',
                9,
            ],
            'b': [
                10,
                11,
                12,
                'after a multiple of 4',
                13,
                14,
                15,
                16,
                'after a multiple of 4',
                17,
                18,
                19,
            ],
        },
    )

  def test_selection_remove(self):
    self.assertEqual(
        (
            pz.select({'a': list(range(10)), 'b': list(range(10, 20))})
            .at_instances_of(int)
            .where(lambda x: x % 4 == 0)
            .remove_from_parent()
        ),
        {'a': [1, 2, 3, 5, 6, 7, 9], 'b': [10, 11, 13, 14, 15, 17, 18, 19]},
    )

  def test_selection_apply_and_inline(self):
    self.assertEqual(
        (
            pz.select({'a': list(range(10)), 'b': list(range(10, 20))})
            .at_instances_of(int)
            .where(lambda x: x % 4 == 0)
            .apply_and_inline(
                lambda x: ['before', f'the value was {x}', 'after']
            )
        ),
        {
            'a': [
                'before',
                'the value was 0',
                'after',
                1,
                2,
                3,
                'before',
                'the value was 4',
                'after',
                5,
                6,
                7,
                'before',
                'the value was 8',
                'after',
                9,
            ],
            'b': [
                10,
                11,
                'before',
                'the value was 12',
                'after',
                13,
                14,
                15,
                'before',
                'the value was 16',
                'after',
                17,
                18,
                19,
            ],
        },
    )

  def test_select_selected_selection(self):
    # This is probably never useful, but it works.
    meta_meta_selection = pz.select(
        pz.select(
            pz.select(make_example_object()).at_instances_of(CustomLeaf)
        ).at(lambda seln: seln.remainder['c'])
    ).at_instances_of(int)

    self.assertEqual(
        meta_meta_selection.deselect().deselect().deselect(),
        make_example_object(),
    )

    self.assertEqual(
        meta_meta_selection.remainder.remainder.remainder,
        {
            'a': penzai.core.selectors.SelectionHole(
                path=(
                    jax.tree_util.GetAttrKey('remainder'),
                    jax.tree_util.GetAttrKey('remainder'),
                    jax.tree_util.DictKey(key='a'),
                )
            ),
            'b': penzai.core.selectors.SelectionQuote(
                quoted=penzai.core.selectors.SelectionQuote(
                    quoted=penzai.core.selectors.SelectionHole(
                        path=(jax.tree_util.DictKey(key='b'),)
                    )
                )
            ),
            'c': penzai.core.selectors.SelectionQuote(
                quoted=penzai.core.selectors.SelectionHole(
                    path=(
                        jax.tree_util.GetAttrKey('remainder'),
                        jax.tree_util.DictKey(key='c'),
                    )
                )
            ),
        },
    )

    # This part only works if the "selected level" wrappers are pytrees. Note
    # that directly modifying the contents of selections by selecting them
    # with other selections may not always work in general.

    @pz.pytree_dataclass
    class SELECTED_LEVEL_1(pz.Struct):  # pylint: disable=invalid-name
      value: Any

    @pz.pytree_dataclass
    class SELECTED_LEVEL_2(pz.Struct):  # pylint: disable=invalid-name
      value: Any

    @pz.pytree_dataclass
    class SELECTED_LEVEL_3(pz.Struct):  # pylint: disable=invalid-name
      value: Any

    self.assertEqual(
        (
            meta_meta_selection.apply(SELECTED_LEVEL_1)
            .apply(SELECTED_LEVEL_2)
            .apply(SELECTED_LEVEL_3)
        ),
        {
            'a': SELECTED_LEVEL_1(value=1),
            'b': SELECTED_LEVEL_3(value=CustomLeaf(tag=10)),
            'c': SELECTED_LEVEL_2(
                value=[
                    {'value': SELECTED_LEVEL_3(value=CustomLeaf(tag=12))},
                    {'value': SELECTED_LEVEL_3(value=CustomLeaf(tag=13))},
                    {'value': SELECTED_LEVEL_1(value=3)},
                ]
            ),
        },
    )


if __name__ == '__main__':
  absltest.main()
