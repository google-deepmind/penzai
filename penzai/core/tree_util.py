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

"""Additional tree functionality related to `jax.tree_util`."""

from __future__ import annotations

from typing import Any, Optional

import jax

PyTreeDef = jax.tree_util.PyTreeDef


def tree_flatten_exactly_one_level(
    tree: Any,
) -> Optional[tuple[list[tuple[Any, Any]], PyTreeDef]]:
  """Flattens a PyTree exactly one level, or returns None if it's not a PyTree.

  Args:
    tree: Tree to flatten.

  Returns:
    If ``tree`` has any children, returns a tuple ``(children, treedef)`` where
    children is a list of ``(key, child)`` pairs. Otherwise, returns ``None``.
  """
  paths_and_subtrees, treedef = jax.tree_util.tree_flatten_with_path(
      tree, is_leaf=lambda subtree: subtree is not tree
  )
  leaf_treedef = jax.tree_util.tree_structure(1)
  if treedef == leaf_treedef:
    return None

  keys_and_subtrees = [
      (key, subtree) for ((key,), subtree) in paths_and_subtrees
  ]
  return keys_and_subtrees, treedef


def pretty_keystr(keypath: tuple[Any, ...], tree: Any) -> str:
  """Constructs a pretty name from a keypath and an object.

  This can be used to construct human-readable names for locations inside a
  PyTree.

  Args:
    keypath: Keypath to process.
    tree: Tree that this keypath indexes into.

  Returns:
    A human-readable string like
    "``Foo.sublayers[0]/Bar.body/Baz.param/Parameter.value"``
    instead of ``".sublayers[0].body.param.value"``
  """
  parts = []
  for key in keypath:
    if isinstance(
        key, jax.tree_util.GetAttrKey | jax.tree_util.FlattenedIndexKey
    ):
      parts.extend(("/", type(tree).__name__))
    split = tree_flatten_exactly_one_level(tree)
    assert split is not None
    tree = dict(split[0])[key]
    parts.append(str(key))
  if parts and parts[0] == "/":
    parts = parts[1:]
  return "".join(parts)
