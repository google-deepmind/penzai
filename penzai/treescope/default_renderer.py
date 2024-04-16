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

"""Configures the default renderer, and allows reconfiguring it dynamically."""

import ast
import contextlib
import functools
from typing import Any, Callable
import jax
from penzai.core import context
from penzai.core import selectors
from penzai.treescope import autovisualize
from penzai.treescope import canonical_aliases
from penzai.treescope import renderer
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import layout_algorithms
from penzai.treescope.foldable_representation import part_interface
from penzai.treescope.handlers import builtin_atom_handler
from penzai.treescope.handlers import builtin_structure_handler
from penzai.treescope.handlers import canonical_alias_postprocessor
from penzai.treescope.handlers import function_reflection_handlers
from penzai.treescope.handlers import generic_pytree_handler
from penzai.treescope.handlers import generic_repr_handler
from penzai.treescope.handlers import hardcoded_structure_handlers
from penzai.treescope.handlers import ndarray_handler
from penzai.treescope.handlers import repr_html_postprocessor
from penzai.treescope.handlers import shared_value_postprocessor
from penzai.treescope.handlers.penzai import data_effects_handlers
from penzai.treescope.handlers.penzai import layer_handler
from penzai.treescope.handlers.penzai import named_axes_handlers
from penzai.treescope.handlers.penzai import shapecheck_handlers
from penzai.treescope.handlers.penzai import struct_handler

Selection = selectors.Selection


handle_known_hardcoded_structures = (
    hardcoded_structure_handlers.HardcodedStructureHandler({
        ast.AST: (
            hardcoded_structure_handlers.HasFieldsInClassAttr(
                "_fields", render_subclasses=True
            )
        ),
        jax.ShapeDtypeStruct: (
            hardcoded_structure_handlers.HasFieldsInClassAttr("__slots__")
        ),
        jax.lax.Precision: hardcoded_structure_handlers.IsEnumLike(),
    })
)

active_renderer: context.ContextualValue[renderer.TreescopeRenderer] = (
    context.ContextualValue(
        module=__name__,
        qualname="active_renderer",
        initial_value=renderer.TreescopeRenderer(
            handlers=[
                # Named arrays.
                named_axes_handlers.handle_named_arrays,
                # Shapecheck array structures.
                shapecheck_handlers.handle_arraystructures,
                # Data effects objects.
                data_effects_handlers.handle_data_effects_objects,
                # Layers.
                layer_handler.handle_layers,
                # Other pz.Struct instances.
                struct_handler.handle_structs,
                # NDArrays.
                ndarray_handler.handle_ndarrays,
                # Reflection of functions and classes.
                function_reflection_handlers.handle_code_objects_with_reflection,
                # Numbers, strings, constants, enums, etc.
                builtin_atom_handler.handle_builtin_atoms,
                # Lists, dicts, tuples, dataclasses, namedtuples, etc.
                builtin_structure_handler.handle_builtin_structures,
                # Hardcoded simple types.
                handle_known_hardcoded_structures,
                # Dtype objects.
                ndarray_handler.handle_dtype_instances,
                # Fallback for unknown pytree types: Show repr and also the
                # PyTree children.
                generic_pytree_handler.handle_arbitrary_pytrees,
                # Fallback to ordinary `repr` for any other object.
                generic_repr_handler.handle_anything_with_repr,
            ],
            wrapper_hooks=[
                # Allow user-configurable visualizations.
                autovisualize.use_autovisualizer_if_present,
                # Show display objects inline.
                repr_html_postprocessor.append_repr_html_when_present,
                # Collapse well-known objects into aliases.
                canonical_alias_postprocessor.replace_with_canonical_aliases,
                # Annotate multiple references to the same mutable Python
                # object.
                shared_value_postprocessor.check_for_shared_values,
            ],
            context_builders=[
                # Set up a new context for each rendered object when rendering
                # shared values.
                shared_value_postprocessor.setup_shared_value_context,
                # Update canonical aliases to account for newly imported
                # modules before rendering.
                canonical_aliases.update_lazy_aliases,
            ],
        ),
    )
)
"""The default renderer to use when rendering a tree to HTML.

This determines the set of handlers and postprocessors to use when
rendering an object.

This can be overridden locally to reconfigure how nodes are rendered
with treescope. Users are free to set this to an arbitrary renderer of
their choice, and programs should not assume the renderer has a
particular form. Library functions can retrieve the current value of
this to render objects in a user-configurable way, and can optionally
make further adjustments using `TreescopeRenderer.extend_with`.
"""

active_expansion_strategy = context.ContextualValue[
    Callable[[part_interface.RenderableTreePart], None]
](
    module=__name__,
    qualname="active_expansion_strategy",
    initial_value=layout_algorithms.expand_for_balanced_layout,
)
"""The default expansion strategy to use when rendering a tree to HTML.

Expansion strategies are used to figure out how deeply to unfold an object
by default. They should operate by setting the expand states of the
foldable nodes inside the object.
"""


def using_expansion_strategy(
    max_height: int | None = 20,
    target_width: int = 60,
    relax_for_only_child: bool = True,
    recursive_expand_height: int = 7,
) -> contextlib.AbstractContextManager[None]:
  """Context manager that modifies the expansion strategy of Treescope.

  This is a convenience function for modifying the layout algorithm used by
  Treescope to determine how many nodes to expand by default. It returns a
  context manager within which the given balanced expansion strategy will be
  used.

  Args:
    max_height: Maximum number of lines for the rendering, which we should try
      not to exceed when expanding nodes. Can sometimes be exceeded for "only
      child" subtrees if ``relax_for_only_child`` is set. If None, then an
      arbitrary height is allowed, and only ``target_width`` will be used to
      determine expansion size.
    target_width: Target number of characters for the rendering. If a node is
      shorter than this width when collapsed, we will leave it on a single line
      instead of expanding it.
    relax_for_only_child: If True, we allow the height constraint to be violated
      once if there is only a single node that could possibly be expanded. In
      particular, we repeatedly expand nodes until we reach a node shorter than
      ``target_width`` or a node with two or more children. If we match or
      exceed the height constraint while doing this, we stop after expanding
      that node (but leave it expanded). If we haven't reached the constraint,
      we continue expanding under the stricter rules.
    recursive_expand_height: Whenever we mark a node as collapsed, we first
      recursively expand its children as if it was expanded with this as the max
      height, then collapse only the outermost node. This means it will render
      as a single line, but if expanded by the user, it will expand not just the
      clicked node but also some of its children. This is intended to support
      faster interactive exploration of large objects.

  Returns:
    Context manager in which the given strategy will be active.
  """
  return active_expansion_strategy.set_scoped(
      functools.partial(
          layout_algorithms.expand_for_balanced_layout,
          max_height=max_height,
          target_width=target_width,
          relax_height_constraint_for_only_child=relax_for_only_child,
          recursive_expand_height_for_collapsed_nodes=recursive_expand_height,
      )
  )


def build_foldable_representation(
    value: Any,
    ignore_exceptions: bool = False,
) -> part_interface.RenderableAndLineAnnotations:
  """Builds a foldable representation of an object using default configuration.

  Uses the default renderer and expansion strategy.

  Args:
    value: Value to render.
    ignore_exceptions: Whether to catch errors during rendering of subtrees and
      show a fallback for those subtrees, instead of failing the entire
      renderer. Best used in contexts where `value_to_foldable_html` is not
      being called directly, e.g. when registering this as a default
      pretty-printer.

  Returns:
    A text representation of the object.
  """
  foldable_ir_with_annotations = (
      active_renderer.get().to_foldable_representation(
          value, ignore_exceptions=ignore_exceptions
      )
  )
  # Expand the renderable ignoring its annotations.
  active_expansion_strategy.get()(foldable_ir_with_annotations.renderable)
  return foldable_ir_with_annotations


def render_to_text(
    value: Any,
    roundtrip_mode: bool = False,
    ignore_exceptions: bool = False,
) -> str:
  """Renders an object to text using the default renderer.

  Args:
    value: Value to render.
    roundtrip_mode: Whether to render in roundtrip mode.
    ignore_exceptions: Whether to catch errors during rendering of subtrees and
      show a fallback for those subtrees, instead of failing the entire
      renderer.

  Returns:
    A text representation of the object.
  """
  foldable_ir = basic_parts.build_full_line_with_annotations(
      build_foldable_representation(value, ignore_exceptions=ignore_exceptions)
  )
  return foldable_impl.render_to_text_as_root(foldable_ir, roundtrip_mode)


def render_to_html(
    value: Any,
    roundtrip_mode: bool = False,
    ignore_exceptions: bool = False,
) -> str:
  """Renders an object to HTML using the default renderer.

  Args:
    value: Value to render.
    roundtrip_mode: Whether to render in roundtrip mode.
    ignore_exceptions: Whether to catch errors during rendering of subtrees and
      show a fallback for those subtrees, instead of failing the entire
      renderer.

  Returns:
    HTML source code for the foldable representation of the object.
  """
  foldable_ir = basic_parts.build_full_line_with_annotations(
      build_foldable_representation(value, ignore_exceptions=ignore_exceptions)
  )
  return foldable_impl.render_to_html_as_root(foldable_ir, roundtrip_mode)
