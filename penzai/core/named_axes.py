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

"""A lightweight and minimal implementation of named axes.

As argued by "Tensors Considered Harmful", relying on axis indices for complex
tensor operations can be brittle and difficult to read. This has led to a
number of proposals for indexing axes by name instead of by position. However,
due to the large API surface for NDArray manipulation, building a fully-featured
named axis implementation requires making named-axis versions of many individual
operations.

This module provides a lightweight implementation of named axes using a
"locally positional" style. The key idea is to reuse positional-axis operations
in their original form, but use *local* bindings of positional axes to the
underlying named axes; positional axes then become aliases for particular
named axes within a local context.

To see how this works, suppose we want to implement dot-product attention
between queries and keys. We'd start with two named arrays::

  queries = pz.nx.wrap(...).tag("batch", "query_pos", "heads", "embed")
  keys = pz.nx.wrap(...).tag("batch", "key_pos", "heads", "embed")

We then contract them against each other, discarding the "embed" dimension
and broadcasting jointly over the others (e.g. "bqhf, bkhf -> bqkh" in einsum
notation). In our "locally positional" style, we could write this as::

  dot_prods = nmap(jnp.dot)(queries.untag("embed"), keys.untag("embed"))

Here `jnp.dot` is called with two one-axis views of ``queries`` and ``keys``,
respectively. (More specifically, it is called with `jax.vmap` tracers that
have a single logical axis and three implicitly-broadcasted axes.) We could just
as easily use our own function::

  def my_dot(a, b):
    print("a:", a)
    print("b:", b)
    print("a.shape:", a.shape)
    print("b.shape:", b.shape)
    return jnp.dot(a, b)

  dot_prods = nmap(my_dot)(queries.untag("embed"), keys.untag("embed"))

We can similarly apply ``softmax`` over one of the axes::

  attn_weights = nmap(jax.nn.softmax)(
      dot_prods.untag("key_pos")).tag("key_pos")

In this case, we need to "tag" the positional axis produced by softmax with a
name, and we choose to give it the same name as the original axis.

One advantage of the locally-positional style is that it does not require
wrapping/modifying any of the functions in the numpy/JAX API to take axis names;
instead, the primitives are written in terms of ordinary positional-axis logic.
This means that the full API surface for named axes can be very small. It
also means that it's easy to "drop down" into positional-axis code and do more
complex modifications (e.g. slicing, updating) without losing the readability
or flexibility of named-axis code.

The locally-positional style is fairly similar to the notation used in
the paper `"Named Tensor Notation"`_ (Chiang, Rush, and Barak, 2022), in which
ordinary mathematical notation is extended with subscripts to identify which
axis or axes they should operate over. In both cases, any names that do NOT
appear as part of the operation are implicitly vectorized over. The primary
difference is that named axes are specified (by ``untag``) separately for each
argument instead of being necessarily shared; this simplifies operations that
act over different names for each argument or that produce new axis names as
outputs.

.. _"Named Tensor Notation": https://arxiv.org/abs/2102.13196

For more information, see the named axis tutorial in ``penzai/notebooks``.
"""

from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import operator
import typing
from typing import Any, Callable, Hashable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import ordered_set
from penzai.core import struct


# Axis names are almost always strings, but can be arbitrary hashable objects
# (except integers, which aren't allowed to avoid confusion with positional
# axes.)
AxisName: typing.TypeAlias = Hashable


class TmpPosAxisMarker:
  """A marker object used to temporarily assign names to positional axes.

  Every ``TmpPosAxisMarker`` is unique, comparing equal only to itself, so it is
  always safe to bind a ``TmpPosAxisMarker`` to a positional axis without
  worrying about axis name conflicts.
  """


def nmap(fun: Callable[..., Any]) -> Callable[..., Any]:
  """Automatically vectorizes ``fun`` over named axes of NamedArray inputs.

  ``nmap`` is a "named-axis vectorizing map". It wraps an ordinary
  positional-axis-based function so that it accepts NamedArrays as input and
  produces NamedArrays as output, and vectorizes over all of the named axes,
  calling the original function with positionally-indexed slices corresponding
  to each argument's `positional_shape`.

  Unlike `jax.vmap`, the axes to vectorize over are inferred
  automatically from the named axes in the NamedArray / NamedArrayView, rather
  than being specified as part of the mapping transformation. Specifically, each
  axis name that appears in any of the arguments is vectorized over jointly
  across all arguments that include that axis name, and is then included as an
  axis name in the output. To make an axis visible to ``fun``, you can call
  `untag` on the argument and pass the axis name(s) of interest; ``fun`` will
  then see those axes as positional axes instead of mapping over them.

  `untag` and ``nmap`` are together the primary ways to apply individual
  operations to axes of a NamedArray. `tag` can then be used on the result to
  re-bind names to positional axes.

  Within ``fun``, any mapped-over axes will be accessible using standard JAX
  collective operations like ``psum``, although doing this is usually
  unnecessary.

  Args:
    fun: Function to vectorize by name. This can take arbitrary arguments (even
      non-JAX-arraylike arguments or "static" axis sizes), but must produce a
      PyTree of JAX ArrayLike outputs.

  Returns:
    An automatically-vectorized version of ``fun``, which can optionally be
    called with NamedArrays (or NamedArrayViews) instead of ordinary arrays, and
    which will always return NamedArrays (or NamedArrayViews) for each of its
    output leaves. Any argument (or PyTree leaf of an argument) that is a
    NamedArray(View) will have its named axes vectorized over; ``fun`` will then
    be called with batch tracers corresponding to slices of the input array that
    are shaped like ``named_array_arg.positional_shape``. Every axis name that
    appeared in any input will also appear in every output.
  """
  if hasattr(fun, "__name__"):
    fun_name = fun.__name__
  else:
    fun_name = repr(fun)
  if hasattr(fun, "__doc__"):
    fun_doc = fun.__doc__
  else:
    fun_doc = None
  return _nmap_with_doc(fun, fun_name, fun_doc)


def _nmap_with_doc(
    fun: Callable[..., Any], fun_name: str, fun_doc: str | None = None
) -> Callable[..., Any]:
  """Builds a nmap-wrapped function with a docstring."""

  @functools.wraps(fun)
  def wrapped_fun(*args, **kwargs):
    arg_leaves_and_paths, arg_treedef = jax.tree_util.tree_flatten_with_path(
        (args, kwargs),
        is_leaf=lambda node: isinstance(node, NamedArray | NamedArrayView),
    )
    arg_leaves = [leaf for _, leaf in arg_leaves_and_paths]
    # Extract any argument leaves that were NamedArrays or NamedArrayViews. The
    # rest of the arguments will just be closed over, so they don't have to be
    # arraylike. We also check named shapes here.
    # To simplify the implementation, we ensure that all arguments are views
    # rather than positional-prefix NamedArrays; this can always be done without
    # any device computations.
    named_array_arg_leaves = []
    known_sizes = {}
    bad_names = []
    for leaf in arg_leaves:
      if isinstance(leaf, NamedArray | NamedArrayView):
        for name, size in leaf.named_shape.items():
          if name in known_sizes:
            if known_sizes[name] != size and name not in bad_names:
              bad_names.append(name)
          else:
            known_sizes[name] = size
        named_array_arg_leaves.append(leaf.as_namedarrayview())

    if bad_names:
      msg = [
          f"Inconsistent named axes in a call to nmap({fun}) for axes"
          f" {bad_names}:"
      ]
      for keypath, leaf in arg_leaves_and_paths:
        if isinstance(leaf, NamedArray | NamedArrayView):
          assert keypath
          if keypath[0] == jax.tree_util.SequenceKey(0):
            prefix = f"args{jax.tree_util.keystr(keypath[1:])}"
          elif keypath[0] == jax.tree_util.SequenceKey(1):
            prefix = f"kwargs{jax.tree_util.keystr(keypath[1:])}"
          else:
            # Should never happen!
            prefix = f"tree{jax.tree_util.keystr(keypath)}"
          msg.append(f"  {prefix}.named_shape == {leaf.named_shape}")
      raise ValueError("\n".join(msg))

    # Prepare a version of the function that accepts tracers for each of those
    # extracted arguments, and rebuilds the full arguments to `fun`.
    def flat_array_fun(batch_tracers):
      # Replace each NamedArray with its batch tracer.
      batch_tracers_stack = batch_tracers[::-1]
      new_arg_leaves = []
      for leaf in arg_leaves:
        if isinstance(leaf, NamedArray | NamedArrayView):
          # Substitute the tracer.
          new_arg_leaves.append(batch_tracers_stack.pop())
        else:
          new_arg_leaves.append(leaf)
      # Call the function.
      args, kwargs = jax.tree_util.tree_unflatten(arg_treedef, new_arg_leaves)
      return fun(*args, **kwargs)

    # Collect the axis names. This determines what we will be vectorizing over.
    # We use an ordered set to guarantee a deterministic ordering based on the
    # arguments, since sets have nondeterministic iteration order in Python.
    all_names = ordered_set.OrderedSet()
    for named_arg in named_array_arg_leaves:
      all_names.update(named_arg.data_axis_for_name.keys())

    all_names = list(all_names)

    # Recursively vectorize over each axis.
    def recursive_vectorize_step(current_views, remaining_names):
      if not remaining_names:
        # All names have been processed, so none of the args should have names.
        # Unwrap them and call the function.
        return flat_array_fun([view.unwrap() for view in current_views])

      # Otherwise, we still have names to vectorize over. Pop one name off the
      # stack and vectorize over it as needed.
      vmap_name = remaining_names[0]
      reduced_views = []
      vmap_axes = []
      for view in current_views:
        if vmap_name in view.data_axis_for_name:
          vmap_axis = view.data_axis_for_name[vmap_name]
          vmap_axes.append(vmap_axis)

          # pylint: disable=cell-var-from-loop
          def _shift_axis(other_axis):
            assert other_axis != vmap_axis
            if other_axis < vmap_axis:
              return other_axis
            else:
              return other_axis - 1

          # pylint: enable=cell-var-from-loop

          # We are temporarily constructing an "invalid" view here because
          # data_array will still have the extra axis. But after running `vmap`,
          # it will be valid again.
          reduced_views.append(
              NamedArrayView(
                  data_array=view.data_array,
                  data_axis_for_name={
                      name: _shift_axis(data_axis)
                      for name, data_axis in view.data_axis_for_name.items()
                      if name != vmap_name
                  },
                  data_axis_for_logical_axis=tuple(
                      _shift_axis(data_axis)
                      for data_axis in view.data_axis_for_logical_axis
                  ),
                  data_shape=(
                      view.data_shape[:vmap_axis]
                      + view.data_shape[vmap_axis + 1 :]
                  ),
              )
          )
        else:
          # This argument doesn't have this axis, so don't map over anything.
          vmap_axes.append(None)
          reduced_views.append(view)

      return jax.vmap(
          functools.partial(
              recursive_vectorize_step,
              remaining_names=remaining_names[1:],
          ),
          in_axes=(vmap_axes,),
          out_axes=0,
          axis_name=vmap_name,
      )(reduced_views)

    # Run the function.
    result_data = recursive_vectorize_step(named_array_arg_leaves, all_names)

    # Wrap all leaves in NamedArray or NamedArrayView, assigning the names from
    # `all_names` to their mapped-over axes. The mapped-over named axes always
    # end up at the front, followed by positional axes, so if there are any
    # positional axes we need to return a NamedArrayView.
    def handle_result(leaf):
      leaf = jnp.array(leaf)
      if leaf.ndim == len(all_names):
        return NamedArray(
            data_array=leaf,
            named_axes=collections.OrderedDict(zip(all_names, leaf.shape)),
        )
      else:
        assert leaf.ndim > len(all_names)
        return NamedArrayView(
            data_array=leaf,
            data_shape=leaf.shape,
            data_axis_for_name={name: i for i, name in enumerate(all_names)},
            data_axis_for_logical_axis=tuple(range(len(all_names), leaf.ndim)),
        )

    return jax.tree_util.tree_map(handle_result, result_data)

  docstr = (
      f"Name-vectorized version of `{fun_name}`. Takes similar arguments as"
      f" `{fun_name}` but accepts and returns NamedArrays (or NamedArrayViews)"
      " in place of regular arrays."
  )
  if fun_doc:
    docstr += f"\n\nOriginal documentation:\n\n{fun_doc}"
  wrapped_fun.__doc__ = docstr
  return wrapped_fun


def _swapped_binop(binop):
  """Swaps the order of operations for a binary operation."""

  def swapped(x, y):
    return binop(y, x)

  return swapped


def _wrap_scalar_conversion(scalar_conversion):
  """Wraps a scalar conversion operator on a named array."""

  def wrapped_scalar_conversion(self: NamedArrayBase):
    if self.named_shape or self.positional_shape:
      raise ValueError(
          "Cannot convert a non-scalar NamedArray or NamedArrayView with"
          f" {scalar_conversion}"
      )
    return scalar_conversion(self.unwrap())

  return wrapped_scalar_conversion


def _wrap_array_method(name):
  """Wraps an array method on a named array."""

  def func(array, *args, **kwargs):
    return getattr(array, name)(*args, **kwargs)

  array_method = getattr(jax.Array, name)
  wrapped_func = nmap(func)
  functools.update_wrapper(
      wrapped_func,
      array_method,
      assigned=("__name__", "__qualname__", "__annotations__"),
      updated=(),
  )
  wrapped_func.__module__ = __name__
  wrapped_func.__doc__ = (
      "Name-vectorized version of array method"
      f" `{name} <numpy.ndarray.{name}>`. Takes similar arguments as"
      f" `{name} <numpy.ndarray.{name}>` but accepts and returns NamedArrays"
      " (or NamedArrayViews) in place of regular arrays."
  )
  return wrapped_func


@struct.pytree_dataclass
class _StaticThunk(struct.Struct):
  value: Any = dataclasses.field(metadata={"pytree_node": False})

  def unwrap(self):
    return self.value


@struct.pytree_dataclass
class _DynamicThunk(struct.Struct):
  value: Any

  def unwrap(self):
    return self.value


@struct.pytree_dataclass
class _SliceThunk(struct.Struct):
  start: Any = dataclasses.field(metadata={"pytree_node": False})
  stop: Any = dataclasses.field(metadata={"pytree_node": False})
  step: Any = dataclasses.field(metadata={"pytree_node": False})

  def unwrap(self):
    return slice(self.start, self.stop, self.step)


@functools.partial(
    jax.jit,
    static_argnames=[
        "indices_are_sorted",
        "unique_indices",
        "mode",
        "fill_value",
    ],
)
@nmap
def _jitted_nmapped_getitem(
    array: jax.Array,
    index_thunks: tuple[_StaticThunk | _DynamicThunk | _SliceThunk, ...],
    *,
    indices_are_sorted=False,
    unique_indices=False,
    mode=None,
    fill_value=None,
):
  """JIT-compiled helper for getitem."""
  indexer = tuple(thunk.unwrap() for thunk in index_thunks)
  return array.at[indexer].get(
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      mode=mode,
      fill_value=fill_value,
  )


@functools.partial(
    jax.jit,
    static_argnames=["method", "indices_are_sorted", "unique_indices", "mode"],
)
@nmap
def _jitted_nmapped_update(
    array: jax.Array,
    index_thunks: tuple[_StaticThunk | _DynamicThunk | _SliceThunk, ...],
    values: jax.Array,
    method: str,
    *,
    indices_are_sorted=False,
    unique_indices=False,
    mode=None,
):
  """JIT-compiled helper for in-place updates."""
  indexer = tuple(thunk.unwrap() for thunk in index_thunks)
  return getattr(array.at[indexer], method)(
      values,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      mode=mode,
  )


@dataclasses.dataclass
class _IndexUpdateHelper:
  """Helper property for index update functionality.

  Lifts the ``jax.Array.at[...]`` syntax to also work for Penzai NamedArrays.
  """

  array: NamedArrayBase

  def __getitem__(self, index) -> _IndexUpdateRef:
    return _IndexUpdateRef(self.array, index)


@dataclasses.dataclass
class _IndexUpdateRef:
  """Helper object to index or update at an (advanced) index.

  Attributes:
    array: The array to index or update.
    indexer: The index to use.
  """

  array: NamedArrayBase
  indexer: Any

  def _partition_dict_index(self):
    """Helper to partition the indices of a dict-style index."""
    indexer = self.indexer
    assert isinstance(indexer, dict)
    # Our strategy will be to convert the indexed axis names into
    # positional axes, with
    # - named axes that are being introduced coming first,
    # - named axes that are being sliced (and thus preserved) coming next,
    # - named axes that are being removed or indexed using advanced indexing
    #   coming after,
    # - and finally, the positional axes of the original array.
    # We can then use ordinary positional indexing logic to retrieve the
    # result.
    sliced_axes = []
    introduced_axes = []
    indexed_axes = []

    for name, index in self.indexer.items():
      if index is None:
        # New axis.
        introduced_axes.append(name)
      elif isinstance(index, slice):
        # Sliced axis.
        sliced_axes.append(name)
      elif isinstance(index, int) or (
          isinstance(index, jax.Array | np.ndarray | NamedArrayBase)
          and jnp.issubdtype(index.dtype, np.integer)
      ):
        # Indexed axis.
        indexed_axes.append(name)
      else:
        raise TypeError(
            "Unsupported index for a named axis using dict-style index:"
            " expected a slice, an integer, an integer array, or None, but"
            f" got {index}"
        )

    input_prefix_order = (*sliced_axes, *indexed_axes)
    slice_order = (*introduced_axes, *sliced_axes, *indexed_axes)
    output_prefix_order = (*introduced_axes, *sliced_axes)

    return input_prefix_order, slice_order, output_prefix_order

  def get(self, **kwargs) -> NamedArrayBase:
    """Get values from the array."""
    if isinstance(self.indexer, dict):
      # Dict indexing => desugar it to positional indexing over the requested
      # names.
      input_prefix_order, slice_order, output_prefix_order = (
          self._partition_dict_index()
      )
      return (
          self.array.untag_prefix(*input_prefix_order)
          .at[tuple(self.indexer[name] for name in slice_order)]
          .get(**kwargs)
          .tag_prefix(*output_prefix_order)
      )

    else:
      # Normal indexing => map it over any named axes.
      # We always jit-compile `getitem`, because eagerly `vmap`-ing (and, by
      # extension, `nmap`-ing) a gather operation can lead to spurious
      # transposes that waste device memory when indexing into a large array.
      # We have to do a bit of trickery to deal with slices and non-jittable
      # array data.
      indexer = self.indexer
      if not isinstance(indexer, tuple):
        indexer = (indexer,)
      index_thunks = []
      for c in indexer:
        if isinstance(c, jax.Array | np.ndarray | NamedArrayBase | int):
          index_thunks.append(_DynamicThunk(c))
        elif isinstance(c, slice):
          index_thunks.append(_SliceThunk(c.start, c.stop, c.step))
        else:
          index_thunks.append(_StaticThunk(c))

      return _jitted_nmapped_getitem(self.array, tuple(index_thunks), **kwargs)

  def _nmap_update_op(self, method: str, value, kwargs) -> NamedArrayBase:
    """Updates values in the array."""
    if isinstance(self.indexer, dict):
      # Dict indexing => desugar it to positional indexing over the requested
      # names.
      input_prefix_order, slice_order, output_prefix_order = (
          self._partition_dict_index()
      )

      # Make sure the provided value has the necessary axes, by broadcasting it
      # to the positional shape the result would have, and adding new named
      # axes that would be introduced into the result. But keep the axes of
      # length 1.
      if not isinstance(value, NamedArrayBase):
        value = wrap(value)

      result_structure = jax.eval_shape(self.get)
      assert isinstance(result_structure, NamedArrayBase)

      value_shape = value.positional_shape
      result_shape = result_structure.positional_shape
      num_new_positional_axes = len(result_shape) - len(value_shape)
      if num_new_positional_axes < 0 or not all(
          vd == 1 or vd == rd
          for vd, rd in zip(
              value_shape, result_shape[len(result_shape) - len(value_shape) :]
          )
      ):
        raise ValueError(
            "Cannot provide updates with positional shape"
            f" {value_shape} for an index whose result shape is"
            f" {result_shape}! Update shape must be a"
            " suffix of the result shape (or broadcastable to it)."
        )
      if num_new_positional_axes:
        value = value[(None,) * num_new_positional_axes + (...,)]

      new_names = {
          name: None
          for name in output_prefix_order
          if name not in value.named_shape
      }
      if new_names:
        value = value[new_names]

      # pylint: disable=protected-access
      return (
          self.array.untag_prefix(*input_prefix_order)
          .at[tuple(self.indexer[name] for name in slice_order)]
          ._nmap_update_op(
              method, value.untag_prefix(*output_prefix_order), kwargs
          )
          .tag_prefix(*input_prefix_order)
      )
      # pylint: enable=protected-access

    else:
      # Normal indexing => map it over any named axes.
      # We always jit-compile advanced updates, because eagerly `vmap`-ing (and,
      # by extension, `nmap`-ing) a gather operation can lead to spurious
      # transposes that waste device memory when indexing into a large array.
      # We have to do a bit of trickery to deal with slices and non-jittable
      # array data.
      indexer = self.indexer
      if not isinstance(indexer, tuple):
        indexer = (indexer,)
      index_thunks = []
      for c in indexer:
        if isinstance(c, jax.Array | np.ndarray | NamedArrayBase | int):
          index_thunks.append(_DynamicThunk(c))
        elif isinstance(c, slice):
          index_thunks.append(_SliceThunk(c.start, c.stop, c.step))
        else:
          index_thunks.append(_StaticThunk(c))

      return _jitted_nmapped_update(
          self.array, tuple(index_thunks), value, method, **kwargs
      )

  def set(self, values, /, **kwargs):
    return self._nmap_update_op("set", values, kwargs)

  def apply(self, values, /, **kwargs):
    return self._nmap_update_op("apply", values, kwargs)

  def add(self, values, /, **kwargs):
    return self._nmap_update_op("add", values, kwargs)

  def multiply(self, values, /, **kwargs):
    return self._nmap_update_op("multiply", values, kwargs)

  def mul(self, values, /, **kwargs):
    return self._nmap_update_op("mul", values, kwargs)

  def divide(self, values, /, **kwargs):
    return self._nmap_update_op("divide", values, kwargs)

  def div(self, values, /, **kwargs):
    return self._nmap_update_op("div", values, kwargs)

  def power(self, values, /, **kwargs):
    return self._nmap_update_op("power", values, kwargs)

  def min(self, values, /, **kwargs):
    return self._nmap_update_op("min", values, kwargs)  # pylint: disable=protected-access

  def max(self, values, /, **kwargs):
    return self._nmap_update_op("max", values, kwargs)  # pylint: disable=protected-access


class NamedArrayBase(abc.ABC):
  """Base class for named arrays and their transposed views."""

  # Abstract methods.

  @property
  @abc.abstractmethod
  def dtype(self) -> np.dtype:
    """The dtype of the wrapped array."""

  @abc.abstractmethod
  def check_valid(self) -> None:
    """Checks that the names in the array are correct."""

  @property
  @abc.abstractmethod
  def named_shape(self) -> Mapping[AxisName, int]:
    """A mapping of axis names to their sizes."""

  @property
  @abc.abstractmethod
  def positional_shape(self) -> tuple[int, ...]:
    """A tuple of axis sizes for any anonymous axes."""

  @abc.abstractmethod
  def unwrap(self, *names: AxisName) -> jax.Array:
    """Unwraps this array, possibly mapping axis names to positional axes.

    Unwrap can be called either on arrays with no named axes, or arrays with
    no positional axes (in which case ``names`` should be a permutation of its
    axis names).

    Args:
      *names: Sequence of axis names to map to positional axes, if this array
        has named axes. Shortand for ``untag(*names).unwrap()``.

    Returns:
      An equivalent ordinary positional array.

    Raises:
      ValueError: If the array has a mixture of positional and named axes, or if
        the names do not match the named axes.
    """

  @abc.abstractmethod
  def with_positional_prefix(self) -> NamedArray:
    """Ensures a view is a `NamedArray` by moving positional axes.

    The resulting `NamedArray` has the same named and positional shapes as
    this object, but the data array may be transposed so that all the positional
    axes are in the front. This makes it possible to manipulate those prefix
    axes safely using `jax.tree_util` or scan/map over them using JAX
    control flow primitives.

    Returns:
      An equivalent `NamedArray` for this view, or the original `NamedArray`
      if it already was one.
    """

  @abc.abstractmethod
  def as_namedarrayview(self) -> NamedArrayView:
    """Converts into a `NamedArrayView`, keeping positional axes.

    This function is usually not necessary for ordinary named-array
    manipulation, since `NamedArray` and `NamedArrayView` define the same
    methods. However, it can be useful for simplifying library code that wishes
    to access the fields of `NamedArrayView` directly, or handle arbitrary
    named array objects without handling each case separately.

    Converting a `NamedArray` to a `NamedArrayView` never involves any
    device computations. (The reverse is not true).

    Returns:
      An equivalent `NamedArrayView` for this array if it isn't one already.
    """

  @abc.abstractmethod
  def untag(self, *axis_order: AxisName) -> NamedArray | NamedArrayView:
    """Produces a positional view of the requested axis names.

    `untag` can only be called on a `NamedArray` or `NamedArrayView` that
    does not have any positional axes. It produces a new `NamedArrayView` where
    the axes with the requested names (the arguments to this function) are now
    treated as positional in the given order.

    If you want to use `untag` on an array that already has positional axes,
    you can use `untag_prefix` instead.

    Args:
      *axis_order: Axis names to make positional, in the order they should
        appear in the positional view.

    Raises:
      ValueError: If this array already has positional axes, or if the provided
        axis ordering is not valid.

    Returns:
      A view with the given axes treated as positional for the purposes of
      later calls to `apply`, `nmap`, or `with_positional_prefix`. If passed
      an empty axis order, returns an ordinary NamedArray with no positional
      axes.
    """

  @abc.abstractmethod
  def tag(self, *names) -> NamedArray:
    """Attaches names to the positional axes of an array or view.

    Args:
      *names: Axis names to assign to each positional axis in the array or view.
        Must be the same length as `positional_shape`; if you only want to tag a
        subset of axes, use `tag_prefix` instead.

    Raises:
      ValueError: If the names are invalid, or if they aren't the same length
        as `positional_shape`.

    Returns:
      A NamedArray with the given names assigned to the positional axes, and no
      remaining positional axes.
    """

  # Inherited methods that can already be implemented in terms of the above.

  def untag_prefix(self, *axis_order: AxisName) -> NamedArray | NamedArrayView:
    """Adds the requested axes to the front of the array's positional axes.

    This is a version of `untag` that can be called on NamedArrays or
    NamedArrayViews that already have positional axes.

    Args:
      *axis_order: Axis names to make positional, in the order they should
        appear in the positional view.

    Returns:
      A view with the given axes treated as positional, followed by the existing
      positional axes.
    """
    # We implement `untag_prefix` using `untag` with temporary axis
    # identifiers.
    tmp_axis_ids = [TmpPosAxisMarker() for _ in self.positional_shape]
    return self.tag(*tmp_axis_ids).untag(*axis_order, *tmp_axis_ids)

  def tag_prefix(self, *axis_order: AxisName) -> NamedArray | NamedArrayView:
    """Attaches names to the first positional axes in an array or view.

    This is a version of `tag` that allows you to name only a subset of the
    array's positional axes.

    Args:
      *axis_order: Axis names to make positional, in the order they should
        appear in the positional view.

    Returns:
      A NamedArray or view with the given names assigned to the first positional
      axes, and whose positional shape includes only the suffix of axes that
      have not been given names.
    """
    # We implement `tag_prefix` using `tag` with temporary axis
    # identifiers.
    tmp_axis_ids = [
        TmpPosAxisMarker()
        for _ in range(len(self.positional_shape) - len(axis_order))
    ]
    return self.tag(*axis_order, *tmp_axis_ids).untag(*tmp_axis_ids)

  def order_as(self, *axis_order: AxisName) -> NamedArray:
    """Ensures that the named axes are stored in this order, keeping them named.

    This function can be used if it is important for the axis names to appear in
    a consistent order, e.g. to ensure that two `NamedArray` instances have
    exactly the same PyTree structure.

    If you want a canonical ordering for a named array that doesn't involve
    knowing all the axis names in advance, you could do something like
    ``array.order_as(*sorted(array.named_shape.keys()))``.

    See also `order_like`.

    Args:
      *axis_order: Axis names in the order they should appear in the data array.
        Must be a permutation of all of the axis names in this array.

    Returns:
      Equivalent `NamedArray` whose data array contains the positional axes
      followed by the named axes in the given order.
    """
    # Create temporary "names" for the positional axes by creating new objects,
    # which are hashable but only compare equal by ID and are thus
    # guaranteed unique.
    tmp_names = [TmpPosAxisMarker() for _ in self.positional_shape]

    data_array = self.tag(*tmp_names).untag(*tmp_names, *axis_order).unwrap()
    return (
        NamedArray.wrap(data_array)
        .tag(*tmp_names, *axis_order)
        .untag(*tmp_names)
        .with_positional_prefix()
    )

  def order_like(
      self, other: NamedArray | NamedArrayView
  ) -> NamedArray | NamedArrayView:
    """Ensures that this array's PyTree structure matches another array's.

    This can be used to ensure that one named array has the same PyTree
    structure as another, so that the two can be jointly processed by
    non-namedarray-aware tree functions (e.g. `jax.tree_util` functions,
    `jax.lax.cond`, `jax.jvp`, etc).

    To ensure compatibility of entire PyTrees, you can use something like::

      jax.tree_util.tree_map(
          lambda a, b: a.order_like(b), tree1, tree2,
          is_leaf=pz.nx.is_namedarray,
      )

    Args:
      other: Another named array or named array view. Must have the same set of
        named axes as ``self``. If ``other`` is a `NamedArrayView`, ``other``
        must also have the same number of positional axes.

    Returns:
      A new `NamedArray` or `NamedArrayView` that has the content of ``self``
      but is possibly transposed to have the axes appear in the same order as
      ``other`` in the data array. If the arrays have the same named and
      positional shapes, the result will have the same PyTree structure as
      ``other``.
    """
    self.check_valid()
    other.check_valid()
    if isinstance(other, NamedArray):
      return self.order_as(*other.named_shape.keys())
    elif isinstance(other, NamedArrayView):
      if len(self.positional_shape) != len(other.positional_shape):
        raise ValueError(
            "Calling `order_like` with a NamedArrayView requires the two"
            " arrays to have the same number of positional axes, but got"
            f" positional shapes {self.positional_shape=},"
            f" {other.positional_shape=}"
        )
      if set(self.named_shape.keys()) != set(other.named_shape.keys()):
        raise ValueError(
            "Calling `order_like` with a NamedArrayView requires the two"
            " arrays to have the axis names, but got"
            f" named shapes {self.named_shape=}, {other.named_shape=}"
        )

      self_view = self.as_namedarrayview()
      new_to_old_data_axis = {}
      for old_data_axis, new_data_axis in zip(
          self_view.data_axis_for_logical_axis, other.data_axis_for_logical_axis
      ):
        new_to_old_data_axis[new_data_axis] = old_data_axis
      for name, new_data_axis in other.data_axis_for_name.items():
        new_to_old_data_axis[new_data_axis] = self_view.data_axis_for_name[name]
      new_data_array = jnp.transpose(
          self_view.data_array,
          [new_to_old_data_axis[i] for i in range(self_view.data_array.ndim)],
      )
      return NamedArrayView(
          data_shape=new_data_array.shape,
          data_axis_for_logical_axis=other.data_axis_for_logical_axis,
          data_axis_for_name=other.data_axis_for_name,
          data_array=new_data_array,
      )
    else:
      raise TypeError(
          "`order_like` requires a NamedArray or NamedArrayView, but got"
          f" {type(other).__name__}"
      )

  def broadcast_to(
      self,
      positional_shape: Sequence[int] = (),
      named_shape: Mapping[AxisName, int] | None = None,
  ) -> NamedArrayBase:
    """Broadcasts a named array to a possibly-larger shape.

    Args:
      positional_shape: Desired positional shape for the array. Will be
        broadcast using numpy broadcasting rules.
      named_shape: Desired named shape for the array. Will be broadcast using
        `nmap`-style vectorizing rules (e.g. new named axes will be introduced
        if missing, but length-1 axes will not be broadcast).

    Returns:
      A named array that has the given positional and named shapes. Note that
      if this array has axis names that are not in ``named_shape``, these will
      be preserved in the answer as well.
    """
    if named_shape is None:
      named_shape = {}
    named_shape = dict(named_shape)
    if (
        self.positional_shape == tuple(positional_shape)
        and dict(self.named_shape) == named_shape
    ):
      return self

    # Trick: create a size-zero array with the right shape so that we can
    # broadcast using nmap's vectorization rules.
    prototype_data = jnp.zeros(
        tuple(named_shape.values()) + tuple(positional_shape) + (0,)
    )
    assert prototype_data.size == 0
    prototype = NamedArray.wrap(prototype_data).tag_prefix(*named_shape.keys())
    return nmap(lambda a, b: jnp.broadcast_to(a, b.shape[:-1]))(self, prototype)

  def broadcast_like(
      self, other: NamedArrayBase | jax.typing.ArrayLike
  ) -> NamedArrayBase:
    """Broadcasts a named array to be compatible with another.

    Args:
      other: Another named array.

    Returns:
      A named array that has the same positional and named shapes as ``other``
      (although it may also include extra named axes).
    """
    if isinstance(other, NamedArrayBase):
      return self.broadcast_to(other.positional_shape, other.named_shape)
    else:
      shape = jnp.shape(other)
      return self.broadcast_to(shape, {})

  def canonicalize(self) -> NamedArray:
    """Ensures that the named axes are stored in a canonical order.

    Returns:
      Equivalent `NamedArray` whose data array contains the positional axes
      followed by the named axes in sorted order.
    """
    return self.order_as(*sorted(self.named_shape.keys()))

  # Indexing.
  def __getitem__(self, indexer) -> NamedArray | NamedArrayView:
    """Retrieves slices from an indexer.

    `NamedArray` and `NamedArrayView` can be indexed in two different ways,
    depending on whether the axes you wish to index are positional or named.

    To index positional axes, you can use ordinary Numpy-style indexing.
    Indexing operations will be automatically vectorized over all of the named
    axes. For instance, an embedding lookup could look something like::

      embedding_table.untag("vocab")[token_ids]

    which first untags the "vocab" named axis as positional, then indexes into
    that axis using another array (which can be a `NamedArray` or an ordinary
    array).

    The result of positional indexing follows the combination of `nmap` and
    Numpy indexing semantics: positional axis ordering is determined by Numpy
    basic/advanced indexing rules, and any named axes in the input or in the
    slices will be jointly vectorized over.

    To index named axes, you can use a dictionary mapping axis names to indices
    or slices. For instance, you can use ::

      my_array[{"position": 1, "feature": pz.slice[2:5]}]

    Here ``pz.slice[2:5]`` is syntactic sugar for ``slice(2, 5, None)``.

    The semantics of dict-style indexing are based on Numpy indexing rules,
    except that they apply to the named axes instead of positional axes. In
    general, dict-style indexing will behave like positional indexing, where
    the requested axes are mapped to positional axes, then indexed, then mapped
    back to named axes where applicable. For instance, ::

      # Slice "foo" in place, and index into "bar" and "baz".
      named_array[{"foo": pz.slice[2:5], "bar": 1, "baz": indexer_array}]

    will behave like ::

      # Slice the first axis, and index into the next two. Then restore the
      # axis name "foo".
      named_array.untag_prefix("foo", "bar", "baz")[2:5, 1, indexer_array, ...]
        .tag_prefix("foo")

    Specifically:

    * Axis names that map to an integer will be indexed into, and those names
      will not appear in the output.
    * Axis names that map to a slice object (like ``pz.slice[2:5]``) will be
      sliced into, and preserved in the output with a smaller size.
    * Axis names that map to None (or `np.newaxis`) must not appear in the
      input array. These axis names will be introduced and will have size 1.
    * Axis names that map to a Numpy or JAX array with non-empty (positional)
      shape and integer dtype will follow Numpy advanced indexing rules. All
      such arrays will be broadcast together and iterated over as one, and
      interpreted as a sequence of indices into each named axis. The result will
      have new positional axes at the front (matching the shapes of the advanced
      index arrays), followed by the existing positional axes of the input (if
      any).
    * Axis names that map to a Penzai named array will be vectorized over using
      `nmap` rules: those names will be vectorized over jointly with the array
      if they are present, and will be introduced into the result if they are
      not present.

    The resulting array's positional shape will first include any new axes
    introduced by advanced indexing, followed by the existing positional axes
    of the input array (if any). The array's named shape will be the input
    array's named shape, minus any axes that were indexed into (without using
    a slice), plus any names that were introduced using `None`/`np.newaxis`,
    plus the union of all names used in named array indices that aren't already
    present.

    Note that Numpy-style advanced indexing can be difficult to reason about
    due to the axis ordering of positional axes. We recommend indexing using
    either integers or NamedArrays with an empty positional shape, which will
    always introduce named axes instead of positional ones.

    Args:
      indexer: Either a normal Numpy-style indexer into the positional axes, or
        a mapping from a subset of axes names to the indices or slice it should
        return.

    Returns:
      A slice of the array.
    """
    self.check_valid()
    return self.at[indexer].get()

  # In-place mutation with at-set syntax.
  @property
  def at(self) -> _IndexUpdateHelper:
    """Helper property for index update functionality.

    Lifts the ``jax.Array.at[...]`` syntax to also work for Penzai NamedArrays.

    Similar to direct indexing, `NamedArray.at[...]` can be indexed in two
    different ways, depending on whether the axes you wish to index are
    positional or named.

    When indexing positional axes, this operation follows the combination of
    `nmap` and ``jax.Array.at[...]`` semantics. In particular, ::

      named_array.at[index].set(value)

    is equivalent to ::

      nmap(lambda arr, i, v: arr.at[i].set(v))(named_array, index, value)

    The resulting array will have the same positional shape as the input array,
    and will have a named shape that is the union of the named shape of the
    input array and the names used in the indexer.

    The semantics of dict-style indexing are similar, except that they apply to
    the named axes instead of the positional ones. In general, dict-style
    indexing will behave like positional indexing, where the requested axes
    are mapped to positional axes, then indexed, then mapped back to named axes
    where applicable. In other words, ::

      # Update part of "foo" in place, and index into "bar" and "baz".
      named_array.at[{
          "foo": pz.slice[2:5], "bar": 1, "baz": indexer_array
      }].set(value)

    will behave like ::

      result_structure = jax.eval_shape(lambda: named_array[{
          "foo": pz.slice[2:5], "bar": 1, "baz": indexer_array
      }])

      # Update positionally, but expect the name "foo" in `value`, and make
      # sure it maps correctly to "foo" in the input.
      named_array.untag_prefix("foo", "bar", "baz")
        .at[2:5, 1, indexer_array, ...]
        .set(value.broadcast_like(result_structure).untag_prefix("foo"))
        .tag_prefix("foo", "bar", "baz")

    Specifically:

    * Axis names that are sliced by ``index_dict`` (e.g. mapped to a slice
      object) can appear in the ``value``. These will be used to update the
      corresponding slices of the array.
    * Axis names that do not appear in ``index_dict`` will be broadcast against
      the corresponding axis names in ``value`` if they exist, following the
      semantics of `nmap`.
    * The positional axes of ``value`` will be broadcast against the result of
      slicing the input array. This means that the suffix axes of ``value``
      will correspond to positional axes of the input array, and the prefix
      axes will correspond to new axes introduced by advanced Numpy indexing.
      Often, the input ``named_array`` will have no positional axes, in which
      case the positional axes of ``value`` will be broadcast against the
      positional axes of the indexer arrays.

    Note that, in order to update multiple positions along the same axis, you
    will need to use Numpy-style advanced indexing, by indexing with an array
    with a *positional* axis. For instance, to update indices 2 and 4 along
    axis "foo", you can do ::

      # Option 1: dict-style
      named_array.at[{ "foo": jnp.array([2, 4]) }].set( jnp.array([100, 101]) )

      # Option 2: positional-style
      named_array.untag("foo").at[jnp.array([2, 4])]
        .set(jnp.array([100, 101])).tag("foo")

    The following, will instead create a new axis "bar" with one update in each
    dimension: ::

      # Option 1: dict-style
      named_array.at[{
          "foo": pz.nx.wrap(jnp.array([2, 4])).tag("bar")
      }].set(
          pz.nx.wrap(jnp.array([100, 101])).tag("bar")
      )

      # Option 2: positional-style
      named_array.untag("foo").at[
          pz.nx.wrap(jnp.array([2, 4])).tag("bar")
      ].set(
          pz.nx.wrap(jnp.array([100, 101])).tag("bar")
      ).tag("foo")

    The reason for this difference is that the named axis "bar" is *vectorized*
    over, and the behavior needs to be consistent regardless of whether "bar"
    appears in ``named_array`` or not.
    """
    self.check_valid()
    return _IndexUpdateHelper(self)

  # Iteration. Note that we *must* implement this to avoid Python simply trying
  # to run __getitem__ until it raises IndexError, because we won't raise
  # IndexError (since JAX clips array indices).
  def __iter__(self):
    if not self.positional_shape:
      raise ValueError("Cannot iterate over an array with no positional axes!")
    for i in range(self.positional_shape[0]):
      yield self[i]

  # Rendering
  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    """Treescope handler for named arrays."""
    from penzai.core._treescope_handlers import named_axes_handlers  # pylint: disable=g-import-not-at-top

    return named_axes_handlers.handle_named_arrays(self, path, subtree_renderer)

  def __treescope_ndarray_adapter__(self):
    """Treescope handler for named arrays."""
    from penzai.core._treescope_handlers import named_axes_handlers  # pylint: disable=g-import-not-at-top

    return named_axes_handlers.NamedArrayAdapter()

  # Convenience wrappers: Elementwise infix operators.
  __lt__ = _nmap_with_doc(operator.lt, "jax.Array.__lt__")
  __le__ = _nmap_with_doc(operator.le, "jax.Array.__le__")
  __eq__ = _nmap_with_doc(operator.eq, "jax.Array.__eq__")
  __ne__ = _nmap_with_doc(operator.ne, "jax.Array.__ne__")
  __ge__ = _nmap_with_doc(operator.ge, "jax.Array.__ge__")
  __gt__ = _nmap_with_doc(operator.gt, "jax.Array.__gt__")

  __add__ = _nmap_with_doc(operator.add, "jax.Array.__add__")
  __sub__ = _nmap_with_doc(operator.sub, "jax.Array.__sub__")
  __mul__ = _nmap_with_doc(operator.mul, "jax.Array.__mul__")
  __truediv__ = _nmap_with_doc(operator.truediv, "jax.Array.__truediv__")
  __floordiv__ = _nmap_with_doc(operator.floordiv, "jax.Array.__floordiv__")
  __mod__ = _nmap_with_doc(operator.mod, "jax.Array.__mod__")
  __divmod__ = _nmap_with_doc(divmod, "jax.Array.__divmod__")
  __pow__ = _nmap_with_doc(operator.pow, "jax.Array.__pow__")
  __lshift__ = _nmap_with_doc(operator.lshift, "jax.Array.__lshift__")
  __rshift__ = _nmap_with_doc(operator.rshift, "jax.Array.__rshift__")
  __and__ = _nmap_with_doc(operator.and_, "jax.Array.__and__")
  __or__ = _nmap_with_doc(operator.or_, "jax.Array.__or__")
  __xor__ = _nmap_with_doc(operator.xor, "jax.Array.__xor__")

  __radd__ = _nmap_with_doc(_swapped_binop(operator.add), "jax.Array.__radd__")
  __rsub__ = _nmap_with_doc(_swapped_binop(operator.sub), "jax.Array.__rsub__")
  __rmul__ = _nmap_with_doc(_swapped_binop(operator.mul), "jax.Array.__rmul__")
  __rtruediv__ = _nmap_with_doc(
      _swapped_binop(operator.truediv), "jax.Array.__rtruediv__"
  )
  __rfloordiv__ = _nmap_with_doc(
      _swapped_binop(operator.floordiv), "jax.Array.__rfloordiv__"
  )
  __rmod__ = _nmap_with_doc(_swapped_binop(operator.mod), "jax.Array.__rmod__")
  __rdivmod__ = _nmap_with_doc(_swapped_binop(divmod), "jax.Array.__rdivmod__")
  __rpow__ = _nmap_with_doc(_swapped_binop(operator.pow), "jax.Array.__rpow__")
  __rlshift__ = _nmap_with_doc(
      _swapped_binop(operator.lshift), "jax.Array.__rlshift__"
  )
  __rrshift__ = _nmap_with_doc(
      _swapped_binop(operator.rshift), "jax.Array.__rrshift__"
  )
  __rand__ = _nmap_with_doc(_swapped_binop(operator.and_), "jax.Array.__rand__")
  __ror__ = _nmap_with_doc(_swapped_binop(operator.or_), "jax.Array.__ror__")
  __rxor__ = _nmap_with_doc(_swapped_binop(operator.xor), "jax.Array.__rxor__")

  __abs__ = _nmap_with_doc(operator.abs, "jax.Array.__abs__")
  __neg__ = _nmap_with_doc(operator.neg, "jax.Array.__neg__")
  __pos__ = _nmap_with_doc(operator.pos, "jax.Array.__pos__")
  __invert__ = _nmap_with_doc(operator.inv, "jax.Array.__invert__")

  # Convenience wrappers: Scalar conversions.
  __bool__ = _wrap_scalar_conversion(bool)
  __complex__ = _wrap_scalar_conversion(complex)
  __int__ = _wrap_scalar_conversion(int)
  __float__ = _wrap_scalar_conversion(float)
  __index__ = _wrap_scalar_conversion(operator.index)

  # Convenience wrappers: np.ndarray / jax.Array methods.
  all = _wrap_array_method("all")
  any = _wrap_array_method("any")
  argmax = _wrap_array_method("argmax")
  argmin = _wrap_array_method("argmin")
  argpartition = _wrap_array_method("argpartition")
  argsort = _wrap_array_method("argsort")
  astype = _wrap_array_method("astype")
  choose = _wrap_array_method("choose")
  clip = _wrap_array_method("clip")
  compress = _wrap_array_method("compress")
  conj = _wrap_array_method("conj")
  conjugate = _wrap_array_method("conjugate")
  # copy not implemented
  cumprod = _wrap_array_method("cumprod")
  cumsum = _wrap_array_method("cumsum")
  diagonal = _wrap_array_method("diagonal")
  dot = _wrap_array_method("dot")
  flatten = _wrap_array_method("flatten")
  imag = _wrap_array_method("imag")
  item = _wrap_array_method("item")
  max = _wrap_array_method("max")
  mean = _wrap_array_method("mean")
  min = _wrap_array_method("min")
  # nbytes not implemented
  nonzero = _wrap_array_method("nonzero")
  prod = _wrap_array_method("prod")
  ptp = _wrap_array_method("ptp")
  ravel = _wrap_array_method("ravel")
  real = _wrap_array_method("real")
  repeat = _wrap_array_method("repeat")
  reshape = _wrap_array_method("reshape")
  round = _wrap_array_method("round")
  searchsorted = _wrap_array_method("searchsorted")
  sort = _wrap_array_method("sort")
  squeeze = _wrap_array_method("squeeze")
  std = _wrap_array_method("std")
  sum = _wrap_array_method("sum")
  swapaxes = _wrap_array_method("swapaxes")
  take = _wrap_array_method("take")
  # tobytes / tolist not implemented
  trace = _wrap_array_method("trace")
  transpose = _wrap_array_method("transpose")
  T = _wrap_array_method("T")
  mT = _wrap_array_method("mT")  # pylint: disable=invalid-name
  var = _wrap_array_method("var")
  view = _wrap_array_method("view")


@struct.pytree_dataclass
class NamedArray(NamedArrayBase, struct.Struct):
  r"""A multidimensional array with a combination of positional and named axes.

  Conceptually, ``NamedArray``\ s can have positional axes like an ordinary
  `jax.Array`, but can also have explicit named axes. Operations on
  ``NamedArray``\ s always act only on the positional axes, and are vectorized
  (or "lifted") over the named axes. To apply operations to a named axis, you
  can first `untag` that named axis as a positional axis, then apply the
  operation as normal. (This is intentional to avoid having to re-define
  separate "named" versions of every JAX or Numpy function.)

  Internally, a ``NamedArray`` stores its array data in the `data_array`
  attribute. The positional axes always appear as a prefix of the data array's
  shape; this means stacking NamedArrays along their first axis (e.g. axis=0) or
  iterating over this axis (e.g. with `jax.lax.scan`) will work correctly for
  ``NamedArray``\ s with a nonempty positional shape. However, to avoid
  unnecessary transpositions, some operations on a NamedArray will produce a
  `NamedArrayView` instead of a ``NamedArray``. (``NamedArrayView``\ s define
  the same methods as ``NamedArray``\ s but have a more complex data
  representation.)

  Operations on ``NamedArray``\ s generally involve constructing positional
  views
  of he axes you need to operate on:

  * To run primitive operations (like `jnp.sum` or `jax.nn.softmax`) on a
    ``NamedArray``, you can call `.untag` to mark specific axes as positional,
    then use `named_axes.nmap` (or wrapped instance methods like ``.sum``) to
    run the primitive positional operation over that locally-positional view.
    Any positional axes produced by the operation can be rebound to names using
    `tag`.

  * To slice an axis out of a ``NamedArray`` (e.g. as input to `jax.lax.scan`),
    you can first move the given axis names to the front of the array using
    ``.untag(...).with_positional_prefix()``, then do a `tree_map` over the
    internal `data_array` and slice the first axis (which is what `jax.lax.scan`
    does internally).

  * To stack ``NamedArray``\ s together along an axis (e.g. for the output of
    `jax.lax.scan`), you can just stack them normally, then use `tag` to
    give a name to the new axis. (If you want to stack `NamedArrayView`s,
    convert them to ``NamedArray``\ s first using `with_positional_prefix`.)

  Note that it's only safe to manipulate the prefix axes of `data_array` which
  do not have names. Any operations on the named axes should first assign
  positions using `untag`, and then either do the operation inside `nmap` or
  move those positions to the prefix of `data_array` using
  `with_positional_prefix`.

  The internal ordering of the named axes is part of the PyTree structure of a
  ``NamedArray`` or `NamedArrayView`, which means passing them through JAX
  operations sometimes requires re-ordering the named axes to ensure a
  consistent structure. To make one named array have the same PyTree structure
  as another, you can use ``first.order_like(second)`` or, for trees,

  ::

      jax.tree_util.tree_map(
          lambda a, b: a.order_like(b), tree1, tree2,
          is_leaf=pz.nx.is_namedarray,
      )

  You can also use `canonicalize` to reorder the internal axes in a canonical
  order; any named arrays with equivalent shapes will have the same PyTree
  structure after calling `canonicalize`.

  Attributes:
    named_axes: An ordered map from axis names to their lengths. The values must
      be a suffix of ``data_array.shape``, and the keys give names to each of
      the suffix dimensions of the array. Usually, this will be the same length
      as ``data_array.shape``.
    data_array: The underlying positional-indexed array.
  """

  named_axes: collections.OrderedDict[AxisName, int] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  data_array: jax.Array

  __eq__ = NamedArrayBase.__eq__

  @classmethod
  def wrap(cls, array: jax.typing.ArrayLike, *names: AxisName) -> NamedArray:
    """Wraps a positional array as a ``NamedArray``.

    Args:
      array: Array to wrap.
      *names: Optional names for the axes of the array. If provided, must be the
        same length as the array's shape. This is a convenience wrapper so that
        you can call ``wrap(array, "foo", "bar")`` instead of
        ``wrap(array).tag("foo", "bar")``.

    Returns:
      An equivalent ``NamedArray`` for the given array. If ``names`` is
      provided, the resulting array will have those names assigned to the
      corresponding axes. Otherwise, the resulting array will have a positional
      shape.
    """
    wrapped = NamedArray(
        named_axes=collections.OrderedDict(), data_array=jnp.asarray(array)
    )
    if names:
      return wrapped.tag(*names)
    else:
      return wrapped

  @property
  def dtype(self) -> np.dtype:
    return self.data_array.dtype

  def check_valid(self) -> None:
    if not hasattr(self.data_array, "shape") or not hasattr(
        self.data_array, "dtype"
    ):
      raise ValueError(
          "NamedArray.data_array must contain a jax or numpy array (or at least"
          f" something with a shape and dtype), not {type(self.data_array)}"
      )
    if not isinstance(self.named_axes, collections.OrderedDict) or not all(
        isinstance(size, int) for size in self.named_axes.values()
    ):
      raise ValueError(
          "NamedArray.named_axes must be an ordered dictionary of named"
          " axis shapes"
      )

    if any(isinstance(name, int) for name in self.named_axes.keys()):
      raise ValueError(
          "Integers are not allowed as axis names, to avoid confusion with"
          " positional axis indices."
      )

    true_suffix_shape = tuple(
        self.data_array.shape[
            len(self.data_array.shape) - len(self.named_axes) :
        ]
    )

    if true_suffix_shape != tuple(self.named_axes.values()):
      raise ValueError(
          "The axis sizes in `named_axes` must exactly match a suffix "
          " of the data array's shape, but"
          f" {tuple(self.named_axes.values())} was not a suffix of"
          f" {self.data_array.shape}"
      )

  @property
  def named_shape(self) -> Mapping[AxisName, int]:
    return dict(self.named_axes)

  @property
  def positional_shape(self) -> tuple[int, ...]:
    self.check_valid()
    positional_axis_count = len(self.data_array.shape) - len(self.named_axes)
    return self.data_array.shape[:positional_axis_count]

  def unwrap(self, *names) -> jax.Array:
    if names:
      if self.positional_shape:
        raise ValueError(
            "Cannot unwrap a NamedArray by providing an axis name ordering if"
            " it already has a positional shape. For advanced axis name"
            " manipulation, try using `untag` and `tag` directly."
        )
      name_bound = self.untag(*names)
      if name_bound.named_shape:
        raise ValueError(
            "When calling `unwrap` with axis names, a position must be given"
            f" for every axis name. Unassigned names: {name_bound.named_shape}"
        )
      return name_bound.unwrap()
    if self.named_axes:
      raise ValueError(
          "To unwrap a NamedArray with nonempty named shape, an ordering for"
          f" its named axes must be provided. Named shape: {self.named_axes}"
      )
    return self.data_array

  def with_positional_prefix(self) -> NamedArray:
    # Already has a positional prefix.
    return self

  def as_namedarrayview(self) -> NamedArrayView:
    self.check_valid()
    positional_axis_count = len(self.data_array.shape) - len(self.named_axes)

    data_axis_for_name = {
        name: index + positional_axis_count
        for index, name in enumerate(self.named_axes.keys())
    }
    data_axis_for_logical_axis = tuple(range(positional_axis_count))
    return NamedArrayView(
        data_shape=self.data_array.shape,
        data_axis_for_logical_axis=data_axis_for_logical_axis,
        data_axis_for_name=data_axis_for_name,
        data_array=self.data_array,
    )

  def untag(self, *axis_order: AxisName) -> NamedArray | NamedArrayView:
    return self.as_namedarrayview().untag(*axis_order)

  def tag(self, *names) -> NamedArray:
    return self.as_namedarrayview().tag(*names)


@struct.pytree_dataclass
class NamedArrayView(NamedArrayBase, struct.Struct):
  """A possibly-transposed view of an array with positional and named axes.

  This view identifies a particular set of axes in a data array as "virtual
  positional axes", which can be operated on by positional logic through
  `nmap`. Unlike `NamedArray`, the positional axes can be stored anywhere in the
  data array, not just as a prefix of its shape.

  Instances of ``NamedArrayView`` are generally constructed by calling `untag`
  on a NamedArray, or as the return value of a `nmap`-ed function. Views are
  then either passed to a positional operation using `nmap` or wrapped instance
  methods, converted back to a NamedArray by reassigning names with `tag`, or
  converted to a NamedArray while moving any remaining positional axes to the
  front with `with_positional_prefix`.

  Directly modifying the shape of ``data_array`` is not allowed, as it will
  break the invariants used in named axis tracking. If you need to slice axes of
  a named array, first make sure the logical positional axes are a prefix of
  ``data_array`` using `with_positional_prefix`.

  Attributes:
    data_shape: The required shape of the data array.
    data_axis_for_logical_axis: Maps the logical positional axes for this view
      to the true indices into ``data_array``'s shape.
    data_axis_for_name: Maps axis names to indices into ``data_array``'s shape.
    data_array: The underlying positional-indexed array.
  """

  data_shape: tuple[int, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  data_axis_for_logical_axis: tuple[int, ...] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  data_axis_for_name: dict[AxisName, int] = dataclasses.field(
      metadata={"pytree_node": False}
  )
  data_array: jax.Array

  __eq__ = NamedArrayBase.__eq__

  @property
  def dtype(self) -> np.dtype:
    return self.data_array.dtype

  def check_valid(self) -> None:
    # Data array has a shape.
    if not hasattr(self.data_array, "shape") or not hasattr(
        self.data_array, "dtype"
    ):
      raise ValueError(
          "NamedArrayView.data_array must contain a jax or numpy array (or at"
          " least something with a shape and dtype), not"
          f" {type(self.data_array)}"
      )
    # Data shape is valid.
    if self.data_shape != self.data_array.shape:
      raise ValueError(
          f"Expected data_array to have shape {self.data_shape}, but it has"
          f" shape {self.data_array.shape}. Modifying the shape of the data"
          " array of a NamedArrayView directly is not allowed; use `nmap`"
          " instead, or call `with_positional_prefix` if you need to"
          " manipulate the positional axes as prefix axes."
      )
    # Every axis appears exactly once.
    seen_axes = collections.Counter()
    seen_axes.update(self.data_axis_for_logical_axis)
    seen_axes.update(self.data_axis_for_name.values())
    if seen_axes != collections.Counter(range(len(self.data_shape))):
      raise ValueError(
          "Every axis index into `data_shape` must appear exactly once in"
          " either `data_axis_for_logical_axis` or `data_axis_for_name`."
      )
    # Check for bad names.
    if any(isinstance(name, int) for name in self.data_axis_for_name.keys()):
      raise ValueError(
          "Integers are not allowed as axis names, to avoid confusion with"
          " positional axis indices."
      )

  @property
  def named_shape(self) -> Mapping[AxisName, int]:
    return {
        name: self.data_shape[data_axis]
        for name, data_axis in self.data_axis_for_name.items()
    }

  @property
  def positional_shape(self) -> tuple[int, ...]:
    return tuple(
        self.data_shape[data_axis]
        for data_axis in self.data_axis_for_logical_axis
    )

  def unwrap(self, *names) -> jax.Array:
    self.check_valid()
    if names:
      if self.positional_shape:
        raise ValueError(
            "Cannot unwrap a NamedArrayView by providing an axis name ordering"
            " if it already has a positional shape. For advanced axis name"
            " manipulation, try using `untag` and `tag` directly."
        )
      name_bound = self.untag(*names)
      if name_bound.named_shape:
        raise ValueError(
            "When calling `unwrap` with axis names, a position must be given"
            f" for every axis name. Unassigned names: {name_bound.named_shape}"
        )
      return name_bound.unwrap()
    if self.named_shape:
      raise ValueError(
          "To unwrap a NamedArrayView with nonempty named shape, an ordering"
          " for its named axes must be provided. Named shape:"
          f" {self.named_shape}"
      )
    # with_positional_prefix will perform the necessary transpositions, we can
    # then simply unwrap the resulting NamedArray.
    return self.with_positional_prefix().unwrap()

  def with_positional_prefix(self) -> NamedArray:
    """Converts a view into a proper `NamedArray` by moving positional axes.

    The resulting `NamedArray` has the same named and positional shapes as this
    view, but the data array may be transposed so that all the positional axes
    are in the front. This makes it possible to manipulate those prefix axes
    safely using `jax.tree_util` or scan/map over them using JAX control flow
    primitives.

    Returns:
      An equivalent `NamedArray` for this view.
    """
    self.check_valid()
    transposition = []
    named_axes = collections.OrderedDict()
    # First all the positional axes
    for data_axis in self.data_axis_for_logical_axis:
      transposition.append(data_axis)
    # Then the named axes in sorted order (to try to avoid transposition)
    data_axes_and_names = sorted(
        (data_axis, name) for name, data_axis in self.data_axis_for_name.items()
    )
    for data_axis, name in data_axes_and_names:
      transposition.append(data_axis)
      named_axes[name] = self.data_shape[data_axis]

    if transposition == list(range(len(self.data_shape))):
      # No transposition needed
      return NamedArray(data_array=self.data_array, named_axes=named_axes)
    else:
      return NamedArray(
          data_array=self.data_array.transpose(transposition),
          named_axes=named_axes,
      )

  def as_namedarrayview(self) -> NamedArrayView:
    # Already a view.
    return self

  def untag(self, *axis_order: AxisName) -> NamedArray | NamedArrayView:
    self.check_valid()
    if self.data_axis_for_logical_axis:
      raise ValueError(
          "`untag` cannot be used to introduce positional axes for a"
          " NamedArray (or NamedArrayView) that already has positional axes."
          " Please assign names to the existing positional axes first using"
          " `tag`."
      )

    if not axis_order:
      # This array has no positional axes and none are requested, so we can
      # safely convert to NamedArray without device computations.
      return self.with_positional_prefix()

    requested_axis_set = set(axis_order)
    actual_axis_set = set(self.data_axis_for_name.keys())

    if len(requested_axis_set) != len(axis_order):
      raise ValueError("Repeats in `axis_order` are not allowed.")

    bad_names = requested_axis_set.difference(actual_axis_set)
    if bad_names:
      raise ValueError(
          f"Requested axis names {bad_names} are not present in the array."
      )

    # Build a view.
    data_axis_for_name = self.data_axis_for_name.copy()
    data_axis_for_logical_axis = []
    for name in axis_order:
      data_axis_for_logical_axis.append(data_axis_for_name[name])
      del data_axis_for_name[name]

    return NamedArrayView(
        data_shape=self.data_shape,
        data_axis_for_logical_axis=tuple(data_axis_for_logical_axis),
        data_axis_for_name=data_axis_for_name,
        data_array=self.data_array,
    )

  def tag(self, *names) -> NamedArray:
    self.check_valid()
    if len(names) != len(self.data_axis_for_logical_axis):
      raise ValueError(
          "There must be exactly as many names given to `tag` as there"
          f" are positional axes in the array, but got {names} for positional"
          f" shape {self.positional_shape}"
      )

    if any(isinstance(name, int) for name in names):
      raise ValueError(
          "Integers are not allowed as axis names, to avoid confusion with"
          " positional axis indices."
      )

    seen_axes = collections.Counter()
    seen_axes.update(self.data_axis_for_name.keys())
    seen_axes.update(names)
    repeated_names = [name for name, count in seen_axes.items() if count > 1]
    if repeated_names:
      raise ValueError(
          "Repeated axis names are not allowed; original names were"
          f" {tuple(self.data_axis_for_name.keys())} and new names passed to"
          f" tag were {names}; repeated: {repeated_names}"
      )

    names_by_index = {}
    for name, data_axis in self.data_axis_for_name.items():
      names_by_index[data_axis] = name

    for i, data_axis in enumerate(self.data_axis_for_logical_axis):
      names_by_index[data_axis] = names[i]

    return NamedArray(
        data_array=self.data_array,
        named_axes=collections.OrderedDict([
            (names_by_index[i], self.data_shape[i])
            for i in range(len(self.data_shape))
        ]),
    )


# Top-level alias for convenience.
wrap = NamedArray.wrap


def is_namedarray(value) -> typing.TypeGuard[NamedArrayBase]:
  """Returns True if this is a `NamedArray` or `NamedArrayView`."""
  return isinstance(value, NamedArrayBase)


def full(
    named_shape: Mapping[AxisName, int],
    fill_value: jax.typing.ArrayLike,
    dtype: np.DTypeLike | None = None,
) -> NamedArray:
  """Constructs a full named array with a given shape.

  Args:
    named_shape: Named shape for the result.
    fill_value: Value to fill the array with.
    dtype: Optional dtype for the result. If not provided, will be inferred from
      ``fill_value``.

  Returns:
    NamedArray with the given named shape, filled with ``fill_value``.
  """
  return NamedArray(
      named_axes=collections.OrderedDict(named_shape),
      data_array=jnp.full(tuple(named_shape.values()), fill_value, dtype=dtype),
  )


def zeros(
    named_shape: Mapping[AxisName, int],
    dtype: np.DTypeLike | None = None,
) -> NamedArray:
  """Constructs a named array of zeros with a given shape.

  Args:
    named_shape: Named shape for the result.
    dtype: Optional dtype for the result.

  Returns:
    NamedArray with the given named shape, filled with zeros.
  """
  return NamedArray(
      named_axes=collections.OrderedDict(named_shape),
      data_array=jnp.zeros(tuple(named_shape.values()), dtype),
  )


def ones(
    named_shape: Mapping[AxisName, int],
    dtype: np.DTypeLike | None = None,
) -> NamedArray:
  """Constructs a named array of ones with a given shape.

  Args:
    named_shape: Named shape for the result.
    dtype: Optional dtype for the result.

  Returns:
    NamedArray with the given named shape, filled with ones.
  """
  return NamedArray(
      named_axes=collections.OrderedDict(named_shape),
      data_array=jnp.ones(tuple(named_shape.values()), dtype),
  )


def arange(
    name: str,
    start: int,
    stop: int | None = None,
    step: int | None = None,
    dtype: jax.typing.DTypeLike | None = None,
) -> NamedArray:
  """Convenience function to create a range along a named axis.

  This is shorthand for ``wrap(jnp.arange(...)).tag(name)``.

  Args:
    name: Name for the resulting axis.
    start: Start of the range. If ``stop`` is not provided, this is instead
      interpreted as ``stop``, with ``start`` implicitly set to 0, following
      `jnp.arange`.
    stop: End of the range.
    step: Step size (defaults to 1).
    dtype: Optional dtype for the result.

  Returns:
    `NamedArray` with a single named axis containing a range of integers.
  """
  return wrap(jnp.arange(start, stop, step, dtype)).tag(name)


def random_split(
    key: jax.Array | NamedArrayBase,
    named_shape: Mapping[AxisName, int] | Sequence[tuple[AxisName, int]],
) -> NamedArray | NamedArrayView:
  """Splits a PRNG key into a `NamedArray` of PRNG keys with the given names.

  Args:
    key: PRNG key to split. Can also be a NamedArray of keys with disjoint names
      from ``named_shape``.
    named_shape: Named shape for the result. If an unordered mapping, the keys
      will be sorted before splitting. To avoid this (e.g. for unsortable keys)
      you can pass a `collections.OrderedDict` or a sequence of (name, size)
      tuples.

  Returns:
    `NamedArray` or `NamedArrayView` with the given named shape, filled with
    unique PRNG keys.
  """
  if not isinstance(key, NamedArrayBase):
    key = wrap(key)

  if isinstance(named_shape, collections.OrderedDict):
    names = named_shape.keys()
    sizes = named_shape.values()
  elif isinstance(named_shape, Sequence):
    names, sizes = zip(*named_shape)
  else:
    unsorted_keys = list(named_shape.keys())
    try:
      names = sorted(unsorted_keys)
    except Exception as exc:
      raise ValueError(
          "Unordered mappings must have sortable axis names when using"
          " `random_split`. If necessary, you can specify a particular ordering"
          " using a collections.OrderedDict or a tuple of (name, size) pairs."
      ) from exc
    sizes = [named_shape[name] for name in names]

  total_size = functools.reduce(operator.mul, sizes, 1)

  flat_split_keys = nmap(jax.random.split)(key, total_size)

  return flat_split_keys.reshape(
      tuple(sizes) + flat_split_keys.positional_shape[1:]
  ).tag_prefix(*names)


def concatenate(
    arrays: Sequence[NamedArrayBase], axis_name: AxisName
) -> NamedArray | NamedArrayView:
  """Concatenates a sequence of named arrays along a named axis.

  Args:
    arrays: Sequence of named arrays to concatenate, which must have the same
      shape except for the axis given by ``axis_name``.
    axis_name: Name of the axis along which to concatenate. Must be present in
      all arrays.

  Returns:
    A new concatenated array.
  """
  ndims = set(len(array.positional_shape) for array in arrays)
  if len(ndims) != 1:
    raise ValueError(
        "All arrays must have the same number of positional axes, but got"
        f" {ndims}"
    )
  (ndim,) = ndims

  orig_positional_axes = [TmpPosAxisMarker() for _ in range(ndim)]
  arrays_along_axis = [
      array.tag(*orig_positional_axes).untag(axis_name) for array in arrays
  ]
  concatenated = nmap(jnp.concatenate)(arrays_along_axis)
  return concatenated.tag(axis_name).untag(*orig_positional_axes)


def stack(
    arrays: Sequence[NamedArrayBase], axis_name: AxisName
) -> NamedArray | NamedArrayView:
  """Stacks a sequence of named arrays along a named axis.

  Args:
    arrays: Sequence of named arrays to stack, which must have the same shape.
    axis_name: Name of the axis along which to stack.

  Returns:
    A new stacked array.
  """
  ndims = set(len(array.positional_shape) for array in arrays)
  if len(ndims) != 1:
    raise ValueError(
        "All arrays must have the same number of positional axes, but got"
        f" {ndims}"
    )
  (ndim,) = ndims

  orig_positional_axes = [TmpPosAxisMarker() for _ in range(ndim)]
  arrays_along_axis = [array.tag(*orig_positional_axes) for array in arrays]
  concatenated = nmap(jnp.stack)(arrays_along_axis)
  return concatenated.tag(axis_name).untag(*orig_positional_axes)


def unstack(
    array: NamedArrayBase, axis_name: AxisName
) -> Sequence[NamedArray | NamedArrayView]:
  """Splits a named array across a given named axis.

  Args:
    array: The array to split.
    axis_name: Name of the axis along which to unstack.

  Returns:
    A sequence of slices across the given axis.
  """
  return [array[{axis_name: i}] for i in range(array.named_shape[axis_name])]


def order_like(value_tree: Any, reference_tree: Any):
  """Orders a tree of named arrays to match the structure of another tree.

  This function takes two PyTrees and makes each NamedArray in ``value_tree``
  have the same structure as the corresponding NamedArray in ``reference_tree``.
  This can be used when passing NamedArrays through JAX transformations that
  require identical PyTree structures.

  This is the equivalent of `NamedArrayBase.order_like` for trees of arrays.
  Leaves that are not NamedArrays or NamedArrayViews are left unchanged.

  Args:
    value_tree: The tree to order.
    reference_tree: The tree to match. Must have the same structure as
      ``value_tree`` except that NamedArrays may have differently-ordered axes.

  Returns:
    A tree with the same exact PyTree structure as ``reference_tree``, with
    array data from ``value_tree``.
  """

  def _fix(val, ref):
    if isinstance(val, NamedArrayBase):
      return val.order_like(ref)
    else:
      return val

  return jax.tree_util.tree_map(
      _fix, value_tree, reference_tree, is_leaf=is_namedarray
  )


def scan(
    f: Callable[[Any, Any], Any], axis: AxisName, init, xs=None, **scan_kwargs
) -> Any:
  """Scan a function over a named array axis while carrying along state.

  This function wraps `jax.lax.scan` to allow scanning over a named axis instead
  of over the leading axis of ``xs``. The inputs ``xs`` must contain
  `NamedArray` (or `NamedArrayView`) instances, and ``f`` should (usually) take
  and return named arrays also, with the difference that each value in ``xs``
  will be missing the named axis ``axis`` that is being scanned over. ``f`` will
  be called with slices of ``xs`` along the given named axis.

  When ``xs`` and the output of ``f`` are single ``NamedArray`` instances, the
  semantics of `scan` are given roughly by ::

    def scan(f, axis, init, xs):
      carry = init
      ys = []
      for i in range(xs.named_shape[axis]):
        x = xs[{axis: i}]
        carry, y = f(carry, x)
        ys.append(y)
      return carry, pz.nx.stack(ys, axis)

  This is a convenience function that is equivalent to calling `jax.lax.scan`
  in combination with the necessary `tag`, `untag`, and `with_positional_prefix`
  calls.

  Args:
    f: The function to scan. As in `jax.lax.scan`, this function should have
      signature ``c -> a -> (c, b)``, where ``c`` is the carry state, ``a`` is
      the current element of `xs`, and ``b`` is the result of the scan.
    axis: The name of the axis to scan over.
    init: The initial loop carry value of type ``c``, which can be any JAX
      PyTree. This value must have the same structure as the first element of
      the pair returned by ``f``.
    xs: The value or tree of values over which to scan. Must contain Penzai
      named arrays, each of which should have the named axis ``axis``.
    **scan_kwargs: Additional keyword arguments to pass to `jax.lax.scan`. In
      particular, if ``xs`` is None, this should include the keyword argument
      ``length`` to specify the length of the scan.

  Returns:
    A pair ``(final_carry, stacked_outputs)``, where ``final_carry`` is the
    final loop carry value, and ``stacked_outputs`` is a named array or tree
    of named arrays that represents the second outputs of ``f``, stacked over
    the named axis ``axis``.
  """

  # Untag the scan axis.
  def _untag(x):
    if isinstance(x, NamedArrayBase):
      return x.untag_prefix(axis).with_positional_prefix()
    else:
      raise ValueError("All leaves of `xs` must be Penzai named arrays.")

  xs_untagged = jax.tree_util.tree_map(_untag, xs, is_leaf=is_namedarray)

  # Run the function, ensuring that axis names are consistent for the carry,
  # and that outputs have positional prefixes.
  def wrapped_f(carry, x):
    new_carry, y = f(carry, x)
    new_carry = order_like(new_carry, carry)
    y = jax.tree_util.tree_map(
        lambda v: v.with_positional_prefix() if is_namedarray(v) else v,
        y,
        is_leaf=is_namedarray,
    )
    return new_carry, y

  # Run the scan, which will slice off the positional prefix from the inputs,
  # and add a positional prefix to the outputs.
  final_carry, ys_untagged = jax.lax.scan(
      wrapped_f, init, xs_untagged, **scan_kwargs
  )

  # Re-assign the scanned-over axis.
  def _retag(leaf):
    if isinstance(leaf, NamedArrayBase):
      return leaf.tag_prefix(axis)
    else:
      return wrap(leaf).tag_prefix(axis)

  ys = jax.tree_util.tree_map(_retag, ys_untagged, is_leaf=is_namedarray)
  return final_carry, ys
