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

"""Tags and utility functions for working with partitioned models.

Partitioning models allows different sets of leaves to be manipulated
independently by JAX transformations, optimizers, and other logic. The
partitioning logic in Penzai is inspired by the similar system in `Equinox`_,
but:

- All partitions are created using selectors, in particular by
  calling `pz.core.selectors.Selection.partition`, making it possible to
  partition based on many use-case-specific criteria.

- Nodes that are removed by the partition use the sentinel `NotInThisPartition`
  as a tag for parts that have been removed. This makes it obvious which parts
  of a partition have been removed. (By default, Equinox uses ``None`` for this,
  but ``None`` may also have other meanings in a PyTree structure.)

- Partitions should be combined with the `combine` function in this module
  rather than Equinox, so that `NotInThisPartition` nodes are identified
  correctly.

.. _Equinox: https://docs.kidger.site/equinox/api/manipulation/
"""

import functools
from typing import Any

import jax
from penzai.core import struct
from penzai.core import tree_util


@struct.pytree_dataclass
class NotInThisPartition(struct.Struct):
  """Sentinel object that identifies subtrees removed by `partition`.

  This object appears in place of subtrees that were partitioned into a separate
  partition by the `partition` method of a selector. To restore the original
  tree, call `combine` with all of the separate partitions.
  """

  def treescope_color(self) -> str:
    return "orange"


def combine(*partitions: Any) -> Any:
  """Combines leaves from multiple partitions.

  This function can be used to reverse the results of calling `partition` on
  a selector. It walks two or more PyTrees, and combines their leaves by
  detecting and replacing instances of `NotInThisPartition`. The partitions can
  have `NotInThisPartition` replacing any subtree, not just leaves.

  A common use case for this function is to recombine parts of an input model
  after splitting them to handle them differently in JAX transformations. See
  the documentation for `penzai.core.selectors.Selection.partition` for more
  details.

  This function is inspired by Equinox's `eqx.combine`, but uses
  `NotInThisPartition` as the sentinel instead of None, and supports
  `NotInThisPartition` at arbitrary subtree locations instead of only having
  None at leaves. (Partitioning is also somewhat less important in Penzai than
  in Equinox because all PyTree leaves are arraylike by convention; partitioning
  is only necessary when different parts of the tree need special treatment.)

  Args:
    *partitions: Partitions to combine. All partitions should have the same
      PyTree structure except that, for each PyTree leaf, exactly one of the
      input partitions actually has that leaf, and every other partition has
      `NotInThisPartition` sentinels in place of that leaf or one of its
      ancestors.

  Returns:
    Combined version of all partitions, which takes the concrete value from
    each partition instead of the `NotInThisPartition` sentinel.
  """
  if not partitions:
    # NotInThisPartition() is the identity for `combine` and thus a reasonable
    # return value for combining the empty set.
    return NotInThisPartition()

  def combine_pairwise_subtrees(keypath, tree_a, tree_b):
    if isinstance(tree_a, NotInThisPartition):
      return tree_b
    elif isinstance(tree_b, NotInThisPartition):
      return tree_a
    else:
      flattened_a = tree_util.tree_flatten_exactly_one_level(tree_a)
      flattened_b = tree_util.tree_flatten_exactly_one_level(tree_b)
      if flattened_a is not None and flattened_b is not None:
        keys_and_subtrees_a, treedef_a = flattened_a
        keys_and_subtrees_b, treedef_b = flattened_b
        if treedef_a == treedef_b:
          combined_children = []
          for (key_a, subtree_a), (key_b, subtree_b) in zip(
              keys_and_subtrees_a, keys_and_subtrees_b
          ):
            assert key_a == key_b
            combined_children.append(
                combine_pairwise_subtrees(
                    keypath + (key_a,), subtree_a, subtree_b
                )
            )
          return treedef_a.unflatten(combined_children)

      # If we got here, tree structures aren't compatible.
      raise ValueError(
          f"Expected the nodes at path {jax.tree_util.keystr(keypath)} to have"
          " compatible structures, or for one of them to be"
          f" NotInThisPartition. However, got {tree_a} and {tree_b}, which are"
          " incompatible."
      )

  return functools.reduce(
      functools.partial(combine_pairwise_subtrees, ()), partitions
  )
