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

"""A toolkit for pytree manipulation."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import typing
from typing import Any, Callable, Collection, Generic, Iterable, Literal, Mapping, Sequence
import warnings

import jax
import numpy as np
from penzai.core import partitioning
from penzai.core import struct
from penzai.core import tree_util as penzai_tree_util


KeyPath = tuple[Any, ...]


SelectedSubtree = typing.TypeVar("SelectedSubtree")
OtherSubtree = typing.TypeVar("OtherSubtree")
T = typing.TypeVar("T")


@struct.pytree_dataclass
class SelectionHole(struct.Struct):
  """A hole in a structure, taking the place of a selected subtree.

  When building a selection, the nodes that are selected are moved out of the
  original tree for easier processing. They are replaced by a ``SelectionHole``,
  which points to the node that was here originally.

  A ``SelectionHole`` is a PyTree with no children. This ensures that the
  selected elements are actually "removed" from the tree from JAX's point of
  view.

  Users should not need to create a ``SelectionHole`` directly, and should
  instead use the `select(...)` function and other selector traversals. However,
  you may see a ``SelectionHole`` if inspecting the contents of a `Selection`
  object.

  Note that the `Selection` machinery assumes that the selected PyTree nodes do
  not require their children to be a specific type, so that we can insert
  ``SelectionHole`` in arbitrary places in the tree.
  If a node makes strong assumptions about the types of its children, it may
  not be possible to select those children, since rebuilding that node with
  a ``SelectionHole`` may fail.

  See
  https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization
  for more information on how to implement your PyTrees to avoid this problem.

  Attributes:
    path: Keypath to this hole. Used to index back into the selected components.
  """

  path: KeyPath = dataclasses.field(metadata={"pytree_node": False})


@struct.pytree_dataclass
class SelectionQuote(struct.Struct):
  """Marks a particular subtree as relating to an inner selection.

  ``SelectionQuote`` is primarily used to handle the situation where a
  `Selection` contains another `Selection`. In this situation, the inner
  `Selection` holes must be kept distinct from the outer `Selection` holes,
  so that only the outer `Selection`'s holes are modified by operations on the
  selection. When this occurs, the inner `Selection`'s `SelectionHole` instances
  are wrapped in a ``SelectionQuote``, so that they aren't processed until the
  outer `Selection` is resolved.

  Note that quoting is only applied to the `remainder` tree, since this is the
  tree where we expect to find a `SelectionHole`. If there are `SelectionHole`
  instances inside the selected subtrees themselves, these are not quoted, since
  we never look at those subtrees when rebuilding the tree from a selection.

  One situation where selections-of-selections may appear is when using
  `treescope` to visualize a `Selection`. To
  support even higher levels of nesting, or trees where the user has inserted
  their own `SelectionHole` or ``SelectionQuote`` for some reason, we also
  quote ``SelectionQuote``; it's not clear that there are many uses for this
  but `penzai` supports it regardless.

  Users should not need to create a ``SelectionQuote`` directly, and should
  instead use the `select(...)` function and other selector traversals. However,
  you may see ``SelectionQuote`` if inspecting the contents of a `Selection`
  object.

  .. note:: Aside for programming languages geeks:
    ``SelectionQuote`` can be seen as a Peano-arithmetic "successor" function
    for De Bruijn levels, a particular convention for encoding variable binding
    in the lambda calculus, with `SelectionHole` representing zero. (De Bruijn
    levels are the conceptual opposite of De Bruijn indices, which use lower
    indices for the innermost variables instead of for the outermost variables.)

    See https://randall-holmes.github.io/Watson/babydocs/node26.html for a bit
    of related discussion.

    This is related to the interpretation of `Selection` as
    a lens, for which the `remainder` tree conceptually represents the
    partially-applied setter function.
  """

  quoted: SelectionHole | SelectionQuote = dataclasses.field(
      metadata={"pytree_node": False}
  )


@dataclasses.dataclass(frozen=True)
class _InProgressSelectionBoundary:
  """Helper object for building a selection.

  Not intended to be exposed to user code! If you're seeing this and you don't
  expect to, please file an issue.

  This class is only used temporarily, while building a selection. It denotes
  the boundary between the selected part and the unselected part, allowing
  us to pull out the selected nodes using an ordinary ``tree_map`` call. All
  public selection-creation functions assume without checking that their
  input does NOT already contain any instances of this class; if they do,
  the resulting selection may be incorrect.

  Attributes:
    selected: The subtree we are selecting.
  """

  selected: Any


@dataclasses.dataclass(frozen=False)
class _LeafWrapper:
  """Helper object for tagging a leaf of a PyTree during ``.at(...)``.

  Attributes:
    wrapped_leaf: The leaf being wrapped.
  """

  wrapped_leaf: Any


def _is_hole_or_quote(subtree: Any) -> bool:
  """Checks if we need special handling for this subtree."""
  return isinstance(subtree, (SelectionHole, SelectionQuote))


@struct.pytree_dataclass
class Selection(Generic[SelectedSubtree], struct.Struct):
  """A selected subset of nodes within a larger PyTree.

  Penzai selectors (such as ``.at(...)``) return ``Selection`` objects, which
  indicate a specific subset of nodes within a larger PyTree, allowing those
  nodes to be pulled out and modified in a functional way.

  Selected nodes are required to be non-overlapping: no selected node can be
  the ancestor of any other selected node in the same selection.

  For convenience, a ``Selection`` is also a PyTree, and its leaves are the same
  as the leaves in the original PyTree, but they are likely to be in a different
  order.

  .. note:: Aside for functional programming geeks:
    A ``Selection`` is conceptually related to an "optic", specifically a
    "lens". If you're familiar with optics, you can
    think of a ``Selection`` as a partially-applied lens: it allows either
    retrieving the selected values, or setting the selected values in the
    structure. (If you're not familiar with optics, you can ignore this.)

  Attributes:
    selected_by_path: A mapping whose values are the selected parts from the
      original structure, and whose keys are the keypaths for those parts (as
      registered with JAX's PyTree registry). This is an `OrderedDict` to
      prevent JAX from trying to sort the keys, which may be arbitrarily
      hashable objects without an ordering.
    remainder: The rest of the structure. The locations where the selected
      components were are marked with `SelectionHole` nodes. If the remainder
      also includes a ``Selection`` itself, the remainder may also include
      `SelectionQuote` nodes.
  """

  selected_by_path: collections.OrderedDict[KeyPath, SelectedSubtree]
  remainder: Any

  def count(self) -> int:
    """Returns the number of elements in the selection."""
    return len(self.selected_by_path)

  def __len__(self) -> int:
    """Returns the number of elements in the selection."""
    return len(self.selected_by_path)

  def is_empty(self) -> bool:
    """Returns True if the selection is empty."""
    return not self.selected_by_path

  def deselect(self) -> Any:
    """Rebuilds the tree, forgetting which nodes were selected.

    Returns:
      A copy of `remainder` with the holes filled by the values in
      `selected_by_path`. If called on an ordinary selection, this rebuilds
      the original tree.
    """

    def rebuild(subtree) -> Any:
      """Processes nodes in the remainder to rebuild the tree."""
      if isinstance(subtree, SelectionHole):
        # Pull out the value for each hole.
        return self.selected_by_path[subtree.path]
      elif isinstance(subtree, SelectionQuote):
        # Unquote one level of quoting.
        return subtree.quoted
      else:
        # An ordinary PyTree leaf in the remainder; leave it as is.
        return subtree

    with _wrap_selection_errors(self):
      return jax.tree_util.tree_map(
          rebuild, self.remainder, is_leaf=_is_hole_or_quote
      )

  def get(self) -> SelectedSubtree:
    """Returns the result of a singleton selection.

    Returns:
      The selected subtree from this selection.

    Raises:
      ValueError: If this selection does not have exactly one selected subtree.
    """
    if len(self.selected_by_path) != 1:
      raise ValueError(
          "Selection.get() can only be called on selections with one selected"
          f" element, but there were {len(self.selected_by_path)}. Consider"
          " using .selected_by_path instead."
      )

    (value,) = self.selected_by_path.values()
    return value

  def get_keypaths(self) -> tuple[KeyPath, ...]:
    """Returns the collection of selected key paths."""
    return tuple(self.selected_by_path.keys())

  # pytype: disable=invalid-annotation

  @typing.overload
  def apply(
      self,
      fn: Callable[[SelectedSubtree], Any],
      *,
      keep_selected: Literal[False] = False,
      with_keypath: Literal[False] = False,
  ) -> Any:
    ...

  @typing.overload
  def apply(
      self,
      fn: Callable[[SelectedSubtree], OtherSubtree],
      *,
      keep_selected: Literal[True],
      with_keypath: Literal[False] = False,
  ) -> Selection[OtherSubtree]:
    ...

  @typing.overload
  def apply(
      self,
      fn: Callable[[KeyPath, SelectedSubtree], Any],
      *,
      with_keypath: Literal[True],
      keep_selected: Literal[False] = False,
  ) -> Any:
    ...

  @typing.overload
  def apply(
      self,
      fn: Callable[[KeyPath, SelectedSubtree], OtherSubtree],
      *,
      with_keypath: Literal[True],
      keep_selected: Literal[True],
  ) -> Selection[OtherSubtree]:
    ...

  # pytype: enable=invalid-annotation

  def apply(
      self,
      fn: Callable[..., Any],
      *,
      with_keypath: bool = False,
      keep_selected: bool = False,
  ) -> Any:
    """Replaces each selected node with the result of applying this function.

    Args:
      fn: Function to apply to each selected node. This function should take a
        PyTree (if with_keypath=False) or a KeyPath and a PyTree (if
        with_keypath=True) and return a replacement PyTree.
      with_keypath: Whether to pass the keypath as the first argument to the
        callable.
      keep_selected: Whether to keep the nodes selected. If True, returns the
        modified selection; if False, rebuilds the tree after replacing.

    Returns:
      Either a modified `Selection` (if keep_selected=True) or a rebuilt version
      of the original tree, each with the replacements applied.
    """
    if with_keypath:
      new_values = collections.OrderedDict(
          [(k, fn(k, v)) for k, v in self.selected_by_path.items()]
      )
    else:
      new_values = collections.OrderedDict(
          [(k, fn(v)) for k, v in self.selected_by_path.items()]
      )
    new_selection = Selection(
        selected_by_path=new_values, remainder=self.remainder
    )
    if keep_selected:
      return new_selection
    else:
      return new_selection.deselect()

  def set(self, replacement: Any) -> Any:
    """Replaces the selected subtree(s) with a fixed replacement.

    Args:
      replacement: The pytree to replace with.

    Returns:
      A modified version of the original tree, with this replacement in place
      of any selected subtrees.
    """
    return self.apply(lambda _: replacement)

  def get_by_path(self) -> collections.OrderedDict[KeyPath, SelectedSubtree]:
    """Retrieves the selected subtree(s) based on their path(s).

    Returns:
      A dictionary of selected nodes, indexed by their path
    """
    return self.selected_by_path

  def set_by_path(
      self, replacer: Mapping[KeyPath, Any] | Callable[[KeyPath], Any]
  ) -> Any:
    """Replaces the selected subtree(s) based on their path(s).

    If you need both the value and the key, see
    ``.apply(fn, with_keypath=True)``.

    Args:
      replacer: A mapping from key paths to replacements, or a function that
        builds such a mapping. Passing ``self.selection_by_path`` will return
        the original tree unchanged.

    Returns:
      A modified version of the original tree, with replacements taken from the
      replacer.
    """
    if callable(replacer):
      replacer = {k: replacer(k) for k in self.selected_by_path.keys()}
    return self.apply(lambda k, v: replacer[k], with_keypath=True)

  def select_and_set_by_path(
      self, replacements_by_path: dict[KeyPath, Any]
  ) -> Any:
    """Selects subtrees and replaces them based on relative keypaths.

    Convenience method that combines `at_keypaths` and `set_by_path`.

    Args:
      replacements_by_path: A mapping from key paths to replacements. Key paths
        are relative to the current selected nodes.

    Returns:
      A modified version of the original tree, with replacements taken from the
      replacer.
    """

    def go(node):
      return (
          select(node)
          .at_keypaths(replacements_by_path.keys())
          .set_by_path(replacements_by_path)
      )

    return self.apply(go)

  def get_sequence(self) -> tuple[SelectedSubtree, ...]:
    """Gets the selected subtree(s) in order.

    Convenience wrapper for ``.selected_by_path.values()``.

    Returns:
      A tuple containing the selected subtrees.
    """
    return tuple(self.selected_by_path.values())

  def set_sequence(self, replacements: Iterable[Any]) -> Any:
    """Replaces the selected subtree(s) in order.

    Args:
      replacements: An iterable of PyTrees to insert at the selected locations,
        in order.

    Returns:
      A modified version of the original tree, with replacements taken from the
      iterable.
    """
    replacer = {}
    for keypath, replacement in zip(self.selected_by_path.keys(), replacements):
      replacer[keypath] = replacement
    return self.set_by_path(replacer)

  def flatten_selected_selections(
      self: "Selection[Selection[SelectedSubtree]]",
  ) -> "Selection[SelectedSubtree]":
    """Flattens a selection whose selected values are all selections.

    This function takes a selection for which all of the selected values are
    already selections, and merges them into a single selection that selects
    all of the values from each individual selection.

    You can use this to build more complex selections by chaining your own
    logic. For instance, if you have written a function ``f`` that selects part
    of a tree, you can run

    ::

      selection.apply(f, keep_selected=True).flatten_selected_selections()

    to "broadcast" that logic across all of the already-selected subtrees in
    the original selection.

    See also `refine`, which allows you to express similar transformations more
    directly.

    Returns:
      A flattened selection object.
    """
    # Strategy: Replace the values of each inner selection by a boundary,
    # deselect everything, then re-select at the boundary.

    def process_subselection(subselection: Selection) -> Any:
      if not isinstance(subselection, Selection):
        raise ValueError(
            "flatten_selected_selections can only be called on Selections for"
            " which all values in `.selected_by_path` are also Selections. Got"
            f" {subselection}"
        )
      return subselection.apply(_InProgressSelectionBoundary)

    with _wrap_selection_errors(self):
      return _build_selection_from_boundary(self.apply(process_subselection))

  def refine(
      self, selector_fn: "Callable[[Any], Selection[OtherSubtree]]"
  ) -> "Selection[OtherSubtree]":
    """Refines a selection by selecting within each selected subtree.

    Although selectors can already be refined by making additional calls,
    chained calls generally treat all selected subtrees the same way. In
    contrast, this method allows each selected node to be processed
    independently. Additionally, similar to `apply`, the additional logic is
    free to modify the subtree as it goes.

    Args:
      selector_fn: A function that takes a selected subtree from this selection
        and returns a new selection object, usually a selection of some nodes in
        the input subtree.

    Returns:
      A new selection that selects every node selected by ``selector_fn``, but
      in the context of the original tree rather than the individual selected
      subtrees.
    """
    return self.apply(
        selector_fn, keep_selected=True
    ).flatten_selected_selections()

  def at(
      self,
      accessor_fn: Callable[[SelectedSubtree], Any | Collection[Any]],
      multiple: bool | None = None,
  ) -> "Selection":
    """Selects a specific child of each selected node.

    ``Selection.at(...)`` allows you to modify a tree with an almost-imperative
    style while maintaining a functional interface, similar to the
    ``Array.at[...]`` syntax for ordinary NDArrays. It takes a callable that
    picks out a subtree of the tree, and returns a new selection that selects
    the part that was picked out.

    For instance, if you have an object

    ::
      obj = Foo(bar=[1, 2, {"baz": 5}])

    you could select the 5 using

    ::

      pz.select(obj).at(lambda x: x.bar[2]["baz"])

    Args:
      accessor_fn: A function which takes each element of the current selection
        and returns a single node within that selection (if ``multiple`` is
        False) or a collection of nodes (if ``multiple`` is True). This function
        must be structural; it must depend only on the PyTree structure of its
        input and not on the actual values or Python IDs of the leaves. It will
        be called with a copy of the object where every PyTree leaf and every
        empty PyTree node (e.g. an empty tuple or the None singleton) are
        wrapped with an internal wrapper object.
      multiple: Whether `accessor_fn` returns a collection of nodes to select,
        rather than a single node. If `None`, first tries to find it as a single
        node, and if that fails, tries to find it as a collection of nodes but
        emits a warning.

    Returns:
      A modified selection that selects the specific child of each node in the
      original selection (or the set of nodes if ``multiple`` was True).
    """

    def _is_leaf_or_childless(node):
      result = penzai_tree_util.tree_flatten_exactly_one_level(node)
      if result is None:
        return True
      else:
        children, _ = result
        return not children

    def _unwrap(l: _LeafWrapper):
      assert isinstance(l, _LeafWrapper)
      return l.wrapped_leaf

    # This logic is based on equinox.tree_at, but kept separate to avoid
    # depending on equinox.
    def _process_one(node, multiple=multiple):
      # Make a pytree copy of the node, and wrap the leaves so that they are
      # unique objects.
      uniquified_copy = jax.tree_util.tree_map(
          _LeafWrapper, node, is_leaf=_is_leaf_or_childless
      )
      # Run the accessor.
      needle_or_needles = accessor_fn(uniquified_copy)
      if multiple:
        needles = needle_or_needles
      else:
        needles = (needle_or_needles,)
      needle_ids = set(id(x) for x in needles)
      # Find the needle by Python ID; each needle should only appear once.
      leaves_or_needles, treedef = jax.tree_util.tree_flatten(
          uniquified_copy, is_leaf=lambda t: id(t) in needle_ids
      )
      new_leaves = []
      found_ids = set()
      for leaf_or_needle in leaves_or_needles:
        if id(leaf_or_needle) in needle_ids:
          if id(leaf_or_needle) in found_ids:
            raise ValueError(
                "accessor_fn returned a value that appeared twice in the input"
                " tree! This should not happen; Penzai should have ensured each"
                " value appears only once. Please file an issue! Value was:"
                f" {leaf_or_needle}"
            )
          found_ids.add(id(leaf_or_needle))
          new_leaves.append(
              _InProgressSelectionBoundary(
                  jax.tree_util.tree_map(_unwrap, leaf_or_needle)
              )
          )
        else:
          new_leaves.append(_unwrap(leaf_or_needle))
      missing_needles = [
          needle for needle in needles if id(needle) not in found_ids
      ]
      if missing_needles:
        if multiple is None and isinstance(needle_or_needles, Collection):
          try:
            result = _process_one(node, multiple=True)
          except ValueError:
            pass
          else:
            warnings.warn(
                "Returning a collection of nodes from the accessor function for"
                " `Selection.at` without passing `multiple=True` is deprecated."
                " If you intended to select multiple nodes, please pass"
                " `multiple=True` to `Selection.at`."
            )
            return result

        if multiple:
          raise ValueError(
              "accessor_fn returned a value that was not found in the tree! The"
              " return value must be a collection of PyTree nodes or leaves "
              f" from the provided argument PyTree. Missing: {missing_needles}"
          )
        else:
          assert len(missing_needles) == 1
          raise ValueError(
              "accessor_fn returned a value that was not found in the tree! The"
              " return value must be a single PyTree node or leaf from the"
              f" provided argument PyTree. Missing: {missing_needles[0]}"
          )
      return treedef.unflatten(new_leaves)

    with_boundary = self.apply(_process_one)
    with _wrap_selection_errors(self):
      return _build_selection_from_boundary(with_boundary)

  def at_pytree_leaves(self) -> "Selection":
    """Selects all PyTree leaves of each selected subtree.

    This selects all of the leaves of the PyTree according to `jax.tree_util`,
    giving the most-specific selection expressible with a `Selection` object.
    (Note that, if any objects in the tree are not registered as JAX PyTree
    nodes, they will be selected in their entirety even if they contain children
    when printed out by treescope.)

    Returns:
      A new selection that selects every leaf of each selected subtree.
    """
    add_boundary = functools.partial(
        jax.tree_util.tree_map, _InProgressSelectionBoundary
    )
    return _build_selection_from_boundary(self.apply(add_boundary))

  def at_children(self) -> "Selection":
    """Selects all direct children of each selected subtree.

    This can be used to implement recursive tree traversals in a generic way,
    using something like::

      def traverse(subtree):
        # ... process the subtree before recursive call ...
        subtree = select(subtree).at_children().apply(traverse)
        # ... process the subtree after the recursive call ...
        return subtree

      new_value = traverse(value)

    Returns:
      A new selection that selects every direct child of a selected subtree.
      If any leaves were previously selected, those leaves will no longer be
      selected (since they have no children).
    """

    def process(subtree):
      maybe_children = penzai_tree_util.tree_flatten_exactly_one_level(subtree)
      if maybe_children:
        keyed_children, treedef = maybe_children
        return treedef.unflatten(
            [_InProgressSelectionBoundary(child) for _, child in keyed_children]
        )
      else:
        return subtree

    with _wrap_selection_errors(self):
      return _build_selection_from_boundary(self.apply(process))

  @typing.overload
  def where(
      self: "Selection[SelectedSubtree]",
      filter_fn: Callable[[SelectedSubtree], bool],
      *,
      with_keypath: Literal[False] = False,
  ) -> "Selection[SelectedSubtree]":
    ...

  @typing.overload
  def where(
      self: "Selection[SelectedSubtree]",
      filter_fn: Callable[[KeyPath, SelectedSubtree], bool],
      *,
      with_keypath: Literal[True],
  ) -> "Selection[SelectedSubtree]":
    ...

  def where(
      self: "Selection[SelectedSubtree]",
      filter_fn: Callable[..., bool],
      *,
      with_keypath: bool = False,
  ) -> "Selection[SelectedSubtree]":
    """Filters to only a subset of selected nodes based on a condition.

    Args:
      filter_fn: Function to determine whether to keep a node in the selection.
        This function should take a PyTree (if ``with_keypath=False``) or a
        KeyPath and a PyTree (if ``with_keypath=True``).
      with_keypath: Whether to pass the keypath as the first argument to the
        callable.

    Returns:
      A new selection that includes only the selected parts where ``filter_fn``
      evaluates to true.
    """
    with _wrap_selection_errors(self):
      keep = _InProgressSelectionBoundary
      if with_keypath:
        new_with_boundary = self.apply(
            lambda k, v: keep(v) if filter_fn(k, v) else v, with_keypath=True
        )
      else:
        new_with_boundary = self.apply(lambda v: keep(v) if filter_fn(v) else v)

      return _build_selection_from_boundary(new_with_boundary)

  @typing.overload
  def at_subtrees_where(
      self,
      filter_fn: Callable[[SelectedSubtree], bool],
      *,
      with_keypath: Literal[False] = False,
      absolute_keypath: bool = False,
      innermost: bool = False,
  ) -> "Selection":
    ...

  @typing.overload
  def at_subtrees_where(
      self,
      filter_fn: Callable[[KeyPath, SelectedSubtree], bool],
      *,
      with_keypath: Literal[True],
      absolute_keypath: bool = False,
      innermost: bool = False,
  ) -> "Selection":
    ...

  def at_subtrees_where(
      self,
      filter_fn: Callable[..., bool],
      *,
      with_keypath: bool = False,
      absolute_keypath: bool = False,
      innermost: bool = False,
  ) -> "Selection":
    """Selects subtrees of selected nodes where a function evaluates to True.

    Note that a selection cannot contain a node that is the descendant of
    another selected node. If ``innermost=False``, we return the outermost
    node, whereas if ``innermost=True`` we return the innermost.

    If you want to apply a modification to *all* matches of a function, even
    if they are nested, you can use a pattern like ::

      selection = select(value)
      while not selection.empty():
        selection = selection.at_subtrees_where(foo).apply(
            bar, keep_selected=True)
      new_value = selection.deselect()

    More complex modifications can also be made using a manual traversal, e.g.::

      def traverse(subtree):
        # ... process the subtree before recursive call ...
        subtree = select(subtree).at_children().apply(traverse)
        # ... process the subtree after the recursive call ...
        return subtree

      new_value = traverse(value)

    Args:
      filter_fn: A function determining which subtrees to select. Should be
        deterministic, and may be called more than once. This function should
        take a PyTree (if ``with_keypath=False``) or a KeyPath and a PyTree (if
        ``with_keypath=True``).
      with_keypath: Whether to pass a keypath as the first argument to the
        callable.
      absolute_keypath: Whether to pass the keypath relative to the root of the
        original tree (if True) or the keypath relative to the currently
        selected node (if False). Ignored if ``with_keypath`` is False.
      innermost: Whether to select the innermost subtree(s) for which the filter
        function is true, instead of the first subtrees encountered.

    Returns:
      A new selection that selects the desired subtrees.
    """

    def safe_filter_fn(*args):
      result = filter_fn(*args)
      if not isinstance(result, bool):
        raise TypeError(
            "Filter function for at_subtrees_where must return a bool, not"
            f" {type(result)}"
        )
      return result

    with _wrap_selection_errors(self):
      if with_keypath or innermost:
        if with_keypath:
          wrapped_filter_fn = safe_filter_fn
        if not with_keypath:
          wrapped_filter_fn = lambda _, s: safe_filter_fn(s)

        def process_subtree(keypath, leaf_or_subtree) -> tuple[bool, Any]:
          """Recursively walks subtrees one level at a time.

          Args:
            keypath: Keypath to this subtree
            leaf_or_subtree: The subtree (or a leaf)

          Returns:
            (found_any, processed_pytree)
          """
          if not innermost:
            found_here = wrapped_filter_fn(keypath, leaf_or_subtree)
            if found_here:
              # Select it.
              return True, _InProgressSelectionBoundary(leaf_or_subtree)

          maybe_children = penzai_tree_util.tree_flatten_exactly_one_level(
              leaf_or_subtree
          )
          if maybe_children:
            # Recursive step.
            keyed_children, treedef = maybe_children
            new_children = []
            any_descendant_selected = False
            for key, child in keyed_children:
              found_in_child, new_child = process_subtree(
                  keypath + (key,), child
              )
              new_children.append(new_child)
              any_descendant_selected = (
                  any_descendant_selected or found_in_child
              )

            if not any_descendant_selected and wrapped_filter_fn(
                keypath, leaf_or_subtree
            ):
              # No descendant was selected, so select this tree.
              return True, _InProgressSelectionBoundary(leaf_or_subtree)
            elif any_descendant_selected:
              # A descendant was selected, use the descendant.
              return True, treedef.unflatten(new_children)
            else:
              # Didn't find anything.
              return False, leaf_or_subtree
          else:
            # This is a leaf.
            return False, leaf_or_subtree

        def process_selected(keypath, selected_subtree):
          """Processes one of the selected subtrees."""
          if absolute_keypath:
            # Start with this selection's keypath
            _, result = process_subtree(keypath, selected_subtree)
            return result
          else:
            # Start with the empty keypath (relative to this selection)
            _, result = process_subtree((), selected_subtree)
            return result

        return _build_selection_from_boundary(
            self.apply(process_selected, with_keypath=True)
        )

      else:

        def process_leaf_or_filtered(leaf_or_subtree):
          """Processes leaves or values where filter_fn is True."""
          if safe_filter_fn(leaf_or_subtree):
            return _InProgressSelectionBoundary(leaf_or_subtree)
          else:
            return leaf_or_subtree

        def process_selected(selected_subtree):
          """Processes one of the selected subtrees."""
          return jax.tree_util.tree_map(
              process_leaf_or_filtered, selected_subtree, is_leaf=safe_filter_fn
          )

        return _build_selection_from_boundary(self.apply(process_selected))

  def at_instances_of(
      self,
      cls: type[OtherSubtree] | tuple[type[OtherSubtree], ...],
      innermost: bool = False,
  ) -> "Selection[OtherSubtree]":
    """Selects subtrees that are an instance of the given type.

    Convenience wrapper for::

      .at_subtrees_where(lambda subtree: isinstance(subtree, cls))

    Args:
      cls: The class (or tuple of classes) to retrieve instances of.
      innermost: Whether to return the innermost instances of the class (instead
        of the outermost).

    Returns:
      A refined selection that selects instances of this class within the
      original selection. If instances of this class are nested, only selects
      the outermost (if ``innermost=False``) or the innermost (if
      ``innermost=True``), but never both.
    """
    return self.at_subtrees_where(
        lambda subtree: isinstance(subtree, cls), innermost=innermost
    )

  def at_equal_to(self, template: OtherSubtree) -> Selection[OtherSubtree]:  # pytype: disable=invalid-annotation
    """Selects subtrees that are equal to a particular object.

    Mostly a convenience wrapper for ::

      .at_subtrees_where(lambda subtree: template == subtree)

    but also skips `jax.Array`, `np.ndarray`, and
    `penzai.core.named_axes.NamedArray`, since they override ``==`` to return
    arrays.

    Args:
      template: The object to select occurrences of.

    Returns:
      A refined selection that selects instances of this class that compare
      equal to this object (with other on the left).
    """
    # Lazy import to avoid circular dependencies
    import penzai.core.named_axes  # pylint: disable=g-import-not-at-top

    bypass_equal_types = (
        jax.Array,
        np.ndarray,
        penzai.core.named_axes.NamedArrayBase,
    )
    if isinstance(template, bypass_equal_types):
      raise ValueError(
          "Cannot use at_equal_to to check for equality of an array, since"
          " arrays override the == operator. Consider using `at_subtrees_where`"
          " instead."
      )

    def _check_equal(subtree):
      if isinstance(subtree, bypass_equal_types):
        return False
      else:
        return bool(subtree == template)

    return self.at_subtrees_where(_check_equal)

  def partition(self, at_leaves: bool = False) -> tuple[Any, Any]:
    """Partitions the tree into ``(selected_tree, remainder_tree)`` parts.

    This function can be used to separate out the selected components of a tree
    into their own separate tree, so that JAX functions and other JAX libraries
    can process them like ordinary PyTrees. It splits its input into two
    disjoint trees ``(selected_tree, remainder_tree)``, where ``selected_tree``
    only contains the leaves that were selected, and `remainder_tree` only
    contains the remainder. The parts that were removed are identified using a
    sentinel `pz.NotInThisPartition` object, which has no PyTree children.

    The main use case for ``partition`` is to identify subsets of models that
    should be treated in different ways by JAX API functions. For instance, if
    you want to take a gradient with respect to a specific subset of parameters,
    you can select those parameters, call ``partition`` to separate them from
    the rest, then call `jax.grad` and use `argnums` to identify the partition
    of interest. Similarly, if you want to donate only a subset of the state to
    `jax.jit`, you can partition it and then use JAX's ``donate_argnums``
    argument to `jax.jit` to identify the parts you want to donate. Inside the
    function, you can then use `pz.combine` to rebuild the original tree.

    It is possible to repeatedly call ``partition`` to split a tree into
    more than two parts. In particular, you can select the ``remainder_tree``,
    target some additional nodes, and call ``.partition()`` again, repeating
    this process as needed. All of the partitioned trees can then be re-combined
    using a single call to `pz.combine`.

    Note that `NotInThisPartition` is a PyTree node with no children, which
    means that partitioned trees are safe to pass through JAX transformations,
    and the set of leaves in the two partitioned trees together are the same as
    the set of leaves in the original selected tree.

    This function is inspired by Equinox's `equinox.partition`, but is designed
    to work with Penzai's selector system. Unlike `equinox.partition`, missing
    nodes are identified with the `pz.NotInThisPartition` sentinel, and can
    replace arbitrary PyTree subtrees instead of just leaves. (Partitioning is
    also somewhat less important in Penzai than in Equinox because all PyTree
    leaves are arraylike by convention; partitioning is only necessary when
    different parts of the tree need special treatment e.g. for ``argnums`` or
    ``donate_argnums`` parameters.)

    Args:
      at_leaves: Whether to do the partitioning at the leaf level, so that the
        returned trees have exactly the same structure. (Note that `pz.combine`
        is OK with entire subtrees missing, so this is not necessary, but can
        make the partitions easier to manipulate manually if desired.) If False,
        the entire selected subtrees will be replaced by `NotInThisPartition` in
        the remainder tree.

    Returns:
      A tuple ``(selected_tree, remainder_tree)``, where both trees have the
      same structure (if ``at_leaves=True``) or the same prefix (if
      ``at_leaves=False``) except that `NotInThisPartition` is used to replace
      parts that are in the other partition.
    """
    if at_leaves:
      selected_tree = (
          self.invert()
          .at_pytree_leaves()
          .set(partitioning.NotInThisPartition())
      )
      remainder_tree = self.at_pytree_leaves().set(
          partitioning.NotInThisPartition()
      )
    else:
      selected_tree = self.invert().set(partitioning.NotInThisPartition())
      remainder_tree = self.set(partitioning.NotInThisPartition())

    return selected_tree, remainder_tree

  def at_keypaths(self, keypaths: Collection[KeyPath]) -> "Selection":
    """Selects nodes by their keypaths (relative to the current selection).

    Args:
      keypaths: A collection of keypaths.

    Returns:
      A new selection where any node whose keypath is in the given selection
      is selected. Note that if any path in ``keypaths`` is a prefix of another,
      only the shorter prefix will be used, since selected nodes cannot be
      nested.
    """
    return self.at_subtrees_where(
        lambda keypath, _: keypath in keypaths,
        with_keypath=True,
        absolute_keypath=False,
    )

  def invert(self) -> "Selection":
    """Inverts a selection, selecting subtrees with no selected children.

    ``selection.invert()`` selects the largest set of subtrees such that those
    subtrees do NOT contain any selected children in the original selection.
    In other words, it selects the common ancestors of all unselected nodes,
    without selecting any selected nodes.

    Returns:
      An inverted selection.
    """
    # Strategy: Select any node whose keypath is NOT a prefix of
    # any of the original selection's keypaths. But we have to be careful to
    # not include the children of the original selection by accident (since
    # their paths aren't prefixes of the original selection anymore).
    original_keypath_set = set(self.selected_by_path.keys())

    all_prefixes = set()
    for keypath in self.selected_by_path.keys():
      for i in range(len(keypath) + 1):
        all_prefixes.add(keypath[:i])

    def equal_or_not_a_prefix(keypath, _):
      return keypath in original_keypath_set or keypath not in all_prefixes

    return (
        select(self.deselect())
        .at_subtrees_where(
            equal_or_not_a_prefix,
            with_keypath=True,
        )
        .where(
            lambda keypath, _: keypath not in all_prefixes,
            with_keypath=True,
        )
    )

  def at_childless(self) -> "Selection":
    """Selects all PyTree nodes with no children, including PyTree leaves.

    This is different than `at_pytree_leaves` in that it additionally selects
    pytree nodes that are childless, e.g. empty lists, ``None``, and structures
    without any PyTree children. Those nodes are not considered leaves according
    to JAX, but it may still be useful to select them, e.g. for visualization
    purposes.

    Returns:
      A new selection that selects every childless node of each selected
      subtree.
    """
    return self.at_subtrees_where(lambda x: select(x).at_children().is_empty())

  @typing.overload
  def apply_and_inline(
      self,
      fn: Callable[[SelectedSubtree], Iterable[Any]],
      *,
      with_keypath: Literal[False] = False,
  ) -> Any:
    ...

  @typing.overload
  def apply_and_inline(
      self,
      fn: Callable[[KeyPath, SelectedSubtree], Iterable[Any]],
      *,
      with_keypath: Literal[True],
  ) -> Any:
    ...

  def apply_and_inline(
      self,
      fn: Callable[..., Iterable[Any]],
      *,
      with_keypath: bool = False,
  ) -> Any:
    """Replaces selected list/tuple items with a sequence of new items.

    This function should only be called when the selected elements are all
    children of lists or tuples. It removes the original selected element from
    the sequence, then inserts the results of calling ``fn`` at that location.
    This is similar to the "flatmap" operation in some other languages.

    Args:
      fn: Function to apply to each selected node. This function should take a
        PyTree (if ``with_keypath=False``) or a KeyPath and a PyTree (if
        ``with_keypath=True``). It should return a replacement iterable of
        PyTrees, each of which will be inserted into the parent container in
        place of the selected node.
      with_keypath: Whether to pass the keypath as the first argument to the
        callable.

    Returns:
      A replacement of the original tree with the selected elements replaced
      with the outputs of ``fn``.

    Raises:
      ValueError: If any selected node is not the child of a list or tuple.
    """
    error_extra_if_callable = (
        "\n\nIf you are trying to replace a layer (or other callable) with a"
        " sequence of callables, consider"
        " replacing it with a Sequential layer instead, using something like"
        " `.apply(lambda x: Sequential(...))` instead of"
        " `.apply_and_inline(lambda x: ...)`."
    )

    def process_subtree(subtree, keypath, parent):
      if isinstance(subtree, SelectionHole):
        # Holes inside of lists or tuples will be processed by their parent.
        # If we get here, that's an error.
        if callable(subtree):
          # Give some more help if the subtree is callable, since that might
          # mean they are trying to insert logic before/after this node.
          error_extra = error_extra_if_callable
        else:
          error_extra = ""
        if parent is None:
          raise ValueError(
              "apply_and_inline requires all selected nodes to be children of"
              " lists or tuples, but the root node was selected."
              + error_extra
          )
        else:
          raise ValueError(
              "apply_and_inline requires all selected nodes to be children of"
              " lists or tuples, but the node at keypath"
              f" {jax.tree_util.keystr(keypath)} was contained in a node of"
              f" type {type(parent).__name__}"
              + error_extra
          )
      elif isinstance(subtree, SelectionQuote):
        # We are resolving the selection, so unquote here.
        return subtree.quoted
      elif isinstance(subtree, (list, tuple)):
        # Process and inline any selected children.
        new_children = []
        for i, child in enumerate(subtree):
          child_keypath = keypath + (jax.tree_util.SequenceKey(i),)
          if isinstance(child, SelectionHole):
            child_value = self.selected_by_path[child.path]
            if with_keypath:
              replacements = fn(child_keypath, child_value)
            else:
              replacements = fn(child_value)
            new_children.extend(replacements)
          else:
            # Not a selected node; process recursively.
            new_children.append(
                process_subtree(child, keypath=child_keypath, parent=subtree)
            )
        return type(subtree)(new_children)
      else:
        # Recursively process children.
        maybe_children = penzai_tree_util.tree_flatten_exactly_one_level(
            subtree
        )
        if maybe_children is None:
          return subtree
        else:
          children_by_key, treedef = maybe_children
          return treedef.unflatten([
              process_subtree(child, keypath + (key,), subtree)
              for key, child in children_by_key
          ])

    return process_subtree(self.remainder, (), None)

  def remove_from_parent(self) -> Any:
    """Removes selected nodes from their parent sequence (a list or tuple).

    This is a convenience wrapper for ``.apply_and_inline(lambda x: ())``.

    Returns:
      A copy of the original tree, but with all selected nodes removed from
      their parents.

    Raises:
      ValueError: If any selected nodes are not children of a list or tuple.
    """
    return self.apply_and_inline(lambda x: ())

  def insert_after(self, value: Any, and_select: bool = False) -> Any:
    """Inserts copies of ``value`` after each selected node.

    Args:
      value: Value to insert after each selected node.
      and_select: If True, selects the newly-inserted value for additional
        modification.

    Returns:
      If ``and_select`` is False, a copy of the original tree, but with
      ``value`` inserted after each selected node. If ``and_select`` is True, a
      new selection selecting the inserted nodes.

    Raises:
      ValueError: If any selected nodes are not children of a list or tuple.
    """
    if and_select:
      return _build_selection_from_boundary(
          self.apply_and_inline(
              lambda x: (x, _InProgressSelectionBoundary(value))
          )
      )
    return self.apply_and_inline(lambda x: (x, value))

  def insert_before(self, value: Any, and_select: bool = False) -> Any:
    """Inserts copies of ``value`` before each selected node.

    Args:
      value: Value to insert before each selected node.
      and_select: If True, selects the newly-inserted value for additional
        modification.

    Returns:
      If ``and_select`` is False, a copy of the original tree, but with
      ``value`` inserted before each selected node. If ``and_select`` is True, a
      new selection selecting the inserted nodes.

    Raises:
      ValueError: If any selected nodes are not children of a list or tuple.
    """
    if and_select:
      return _build_selection_from_boundary(
          self.apply_and_inline(
              lambda x: (_InProgressSelectionBoundary(value), x)
          )
      )
    return self.apply_and_inline(lambda x: (value, x))

  def show_selection(self):
    """Renders the selection in IPython.

    This method is intended to visualize the selection object itself, and
    renders boxes around the selected nodes.

    This method should only be used when IPython is available.
    """
    # Import selection_rendering here to avoid a circular import.
    from penzai.core._treescope_handlers import selection_rendering  # pylint: disable=g-import-not-at-top

    selection_rendering.display_selection_streaming(
        self, visible_selection=True
    )

  def show_value(self):
    """Renders the original tree in IPython, expanding up to the selected nodes.

    This method is intended to visualize a value but emphasizing the selected
    parts, where the selection is used to determine what to focus on by default
    but isn't actually an object we care about.

    This method should only be used when IPython is available.
    """
    # Import selection_rendering here to avoid a circular import.
    from penzai.core._treescope_handlers import selection_rendering  # pylint: disable=g-import-not-at-top

    selection_rendering.display_selection_streaming(
        self, visible_selection=False
    )

  def assert_count_is(self, count: int) -> Selection:
    """Checks that an expected number of nodes is selected.

    Args:
      count: The expected number of nodes.

    Returns:
      The original selection unchanged.

    Raises:
      AssertionError: If the selection does not have this many nodes.
    """
    assert self.count() == count
    return self

  def pick_nth_selected(self, n: int | Sequence[int]) -> Selection:
    """Filters a selection to only the nth selected node.

    ``n`` is taken relative to the linearized sequence of currently selected
    nodes, in the sense that ::

      my_selection.get_sequence()[n] == my_selection.pick_nth(n).get()

    Args:
      n: The index of the selected nodes that should remain selected. If this is
        a sequence, all nodes in the sequence will be selected.

    Returns:
      A new selection that includes only the ``n``th node from the original
      selection.  For instance, if the original selection has 5 nodes selected,
      setting ``n = 1`` would produce a new selection with only the node at
      index 1 selected.
    """
    if isinstance(n, int):
      indices = (n,)
    else:
      indices = n

    with _wrap_selection_errors(self):
      keep = _InProgressSelectionBoundary
      new_selected_by_path = collections.OrderedDict({
          key: keep(value) if i in indices else value
          for i, (key, value) in enumerate(self.selected_by_path.items())
      })
      return _build_selection_from_boundary(
          Selection(
              selected_by_path=new_selected_by_path, remainder=self.remainder
          ).deselect()
      )

  def apply_with_selected_index(
      self,
      fn: Callable[[int, SelectedSubtree], Any],
      keep_selected: bool = False,
  ) -> Any | Selection:
    """Applies a function, passing both the selected nodes and their indices.

    Indices are taken relative to the linearized sequence of currently selected
    nodes. In other words, if there are five nodes selected, then ``fn`` will
    be called with the numbers 0 through 4 inclusive, regardless of the specific
    keypaths to the nodes.

    Args:
      fn: Function to call. This function will be passed both the index of the
        selected node and the value, and should return something to replace the
        value with.
      keep_selected: Whether to keep the node selected after the transformation.

    Returns:
      The tree or modified selection after inserting the results of ``fn``,
      depending on ``keep_selected``.
    """
    new_selected_by_path = collections.OrderedDict({
        key: fn(i, value)
        for i, (key, value) in enumerate(self.selected_by_path.items())
    })
    new_selection = Selection(
        selected_by_path=new_selected_by_path, remainder=self.remainder
    )
    if keep_selected:
      return new_selection
    else:
      return new_selection.deselect()

  def __treescope_root_repr__(self):
    """Renders this selection as the root object in a treescope rendering."""
    from penzai.core._treescope_handlers import selection_rendering  # pylint: disable=g-import-not-at-top

    return selection_rendering.render_selection_to_foldable_representation(
        self, visible_selection=True, ignore_exceptions=True
    )


@contextlib.contextmanager
def _wrap_selection_errors(selection: "Selection"):
  """Context manager to add informative details to selection errors."""
  try:
    yield
  except Exception as exc:
    new_message = (
        "An error occurred while building a Selection. This can happen when"
        " PyTree nodes make assumptions about the types of their children,"
        " since Selections replace subtrees with sentinel elements. It can also"
        " happen if the value being selected isn't a valid JAX PyTree (in which"
        " case `jax.tree_utils.tree_flatten` will also fail)."
    )

    # Check for known failure cases
    any_partial = False

    def _check(subtree) -> bool:
      nonlocal any_partial
      if isinstance(subtree, jax.tree_util.Partial):
        any_partial = True  # pylint: disable=unused-variable
        return True
      return False

    jax.tree_util.tree_map(_check, selection.deselect(), is_leaf=_check)

    if any_partial:
      new_message += (
          "\n\nIn this case, the error may have been caused by a"
          " jax.tree_util.Partial instance, which requires its PyTree children"
          " to be a list and a dictionary. This means the direct children of a"
          " Partial cannot be selected. If you want to avoid this edge case,"
          " consider replacing your jax.tree_utils.Partial instances with"
          " penzai.experimental.safe_partial.Partial instances, using something"
          " like"
          " `select(your_tree).at_instances_of(jax.tree_util.Partial)"
          ".apply(penzai.experimental.safe_partial.Partial.from_jax)`"
      )

    raise ValueError(new_message) from exc


def _build_selection_from_boundary(tree_with_boundary: Any) -> Selection:
  """Internal function to build a selection from a boundary.

  This function, like the `_InProgressSelectionBoundary` class, is an
  implementation detail and should not be relied upon by user code.

  Args:
    tree_with_boundary: A PyTree, with `_InProgressSelectionBoundary` nodes
      wrapping the subtrees that we wish to select. An unchecked requirement is
      that no `_InProgressSelectionBoundary` node should contain another
      `_InProgressSelectionBoundary` node; if this happens the boundary nodes
      will leak back into the returned selection.

  Returns:
    A selection, such that each occurrence of `_InProgressSelectionBoundary` is
    transformed into a selected subtree.
  """
  selected_by_path = collections.OrderedDict()

  def process_selected(keypath, leaf):
    """Process PyTree leaves or subtrees where _is_hole_or_quote is True."""
    if isinstance(leaf, _InProgressSelectionBoundary):
      # Select this object and stop processing the tree.
      selected_by_path[keypath] = leaf.selected
      return SelectionHole(keypath)
    elif _is_hole_or_quote(leaf):
      # This is already a hole or a quote, but we shouldn't treat it like a
      # hole or quote for the purposes of this selection. Escape them by adding
      # a level of quoting.
      return SelectionQuote(leaf)
    else:
      # This is just an ordinary leaf; keep it in the remainder.
      return leaf

  remainder = jax.tree_util.tree_map_with_path(
      process_selected, tree_with_boundary, is_leaf=_is_hole_or_quote
  )
  return Selection(selected_by_path=selected_by_path, remainder=remainder)


## Top-level API


def select(tree: Any) -> Selection:
  """Wraps a tree in a singleton selection for processing.

  Args:
    tree: A tree to select.

  Returns:
    A selection of the entire tree. Useful because it allows chaining additional
    selection methods defined on Selection.
  """
  return _build_selection_from_boundary(_InProgressSelectionBoundary(tree))
