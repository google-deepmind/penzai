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

"""Helpers to build frequently-used configurations of rendered parts."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from penzai.treescope import canonical_aliases
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import foldable_impl
from penzai.treescope.foldable_representation import part_interface


CSSStyleRule = part_interface.CSSStyleRule
JavaScriptDefn = part_interface.JavaScriptDefn
HtmlContextForSetup = part_interface.HtmlContextForSetup
RenderableTreePart = part_interface.RenderableTreePart
ExpandState = part_interface.ExpandState
FoldableTreeNode = part_interface.FoldableTreeNode

RenderableAndLineAnnotations = basic_parts.RenderableAndLineAnnotations


def build_copy_button(path: tuple[Any, ...] | None) -> RenderableTreePart:
  if path is None:
    return basic_parts.EmptyPart()
  else:
    return foldable_impl.StringCopyButton(
        annotation="Copy path: ",
        copy_string=(
            "(lambda root: root" + "".join(str(key) for key in path) + ")"
        ),
    )


def build_custom_foldable_tree_node(
    contents: RenderableTreePart,
    path: tuple[Any, ...] | None = None,
    label: RenderableTreePart = basic_parts.EmptyPart(),
    expand_state: part_interface.ExpandState = part_interface.ExpandState.WEAKLY_COLLAPSED,
) -> RenderableAndLineAnnotations:
  """Builds a custom foldable tree node with path buttons and hyperlink support.

  Args:
    contents: Contents of this foldable that should not open/close the custom
      foldable when clicked.
    path: Keypath to this node from the root. If provided, a copy-path button
      will be added at the end of the node.
    label: The beginning of the first line, which should allow opening/closing
      the custom foldable when clicked. Should not contain any other foldables.
    expand_state: Initial expand state for the foldable.

  Returns:
    A new renderable part, possibly with a copy button annotation, for use
    in part of a rendered treescope tree.
  """
  maybe_copy_button = build_copy_button(path)

  return RenderableAndLineAnnotations(
      renderable=foldable_impl.HyperlinkTarget(
          foldable_impl.FoldableTreeNodeImpl(
              label=label, contents=contents, expand_state=expand_state
          ),
          keypath=path,
      ),
      annotations=maybe_copy_button,
  )


def build_one_line_tree_node(
    line: RenderableAndLineAnnotations | RenderableTreePart | str,
    path: tuple[Any, ...] | None = None,
    background_color: str | None = None,
    background_pattern: str | None = None,
) -> RenderableAndLineAnnotations:
  """Builds a single-line tree node with path buttons and hyperlink support.

  Args:
    line: Contents of the line.
    path: Keypath to this node from the root. If provided, copy-path buttons
      will be added, and this node will be possible to target with hyperlinks.
    background_color: Optional background and border color for this node.
    background_pattern: Optional background pattern as a CSS "image". If
      provided, `background_color` must also be provided, and will be used as
      the border for the pattern.

  Returns:
    A new renderable part, possibly with a copy button annotation, for use
    in part of a rendered treescope tree.
  """
  maybe_copy_button = build_copy_button(path)

  if isinstance(line, RenderableAndLineAnnotations):
    line_primary = line.renderable
    annotations = basic_parts.Siblings.build(
        maybe_copy_button, line.annotations
    )
  elif isinstance(line, str):
    line_primary = basic_parts.Text(line)
    annotations = maybe_copy_button
  else:
    line_primary = line
    annotations = maybe_copy_button

  if background_pattern is not None:
    if background_color is None:
      raise ValueError(
          "background_color must be provided if background_pattern is"
      )
    line_primary = common_styles.WithBlockPattern(
        common_styles.PatternedSingleLineSpanGroup(line_primary),
        color=background_color,
        pattern=background_pattern,
    )
  elif background_color is not None and background_color != "transparent":
    line_primary = common_styles.WithBlockColor(
        common_styles.ColoredSingleLineSpanGroup(line_primary),
        color=background_color,
    )

  return RenderableAndLineAnnotations(
      renderable=foldable_impl.HyperlinkTarget(
          line_primary,
          keypath=path,
      ),
      annotations=annotations,
  )


def build_foldable_tree_node_from_children(
    prefix: RenderableTreePart | str,
    children: Sequence[RenderableAndLineAnnotations | RenderableTreePart | str],
    suffix: RenderableTreePart | str,
    comma_separated: bool = False,
    force_trailing_comma: bool = False,
    path: tuple[Any, ...] | None = None,
    background_color: str | None = None,
    background_pattern: str | None = None,
    first_line_annotation: RenderableTreePart | None = None,
) -> RenderableAndLineAnnotations:
  """Builds a foldable tree node with path buttons and hyperlink support.

  Args:
    prefix: Contents of the first line, before the children. Should not contain
      any other foldables. Usually ends with an opening paren/bracket, e.g.
      "SomeClass("
    children: Sequence of children of this node, which should each be rendered
      on their own line.
    suffix: Contents of the last line, after the children. Usually a closing
      paren/bracket for `prefix`.
    comma_separated: Whether to insert commas between children.
    force_trailing_comma: Whether to always insert a trailing comma after the
      last child.
    path: Keypath to this node from the root. If provided, copy-path buttons
      will be added, and this node will be possible to target with hyperlinks.
    background_color: Optional background and border color for this node.
    background_pattern: Optional background pattern as a CSS "image". If
      provided, `background_color` must also be provided, and will be used as
      the border for the pattern.
    first_line_annotation: An annotation for the first line of the node when it
      is expanded.

  Returns:
    A new renderable part, possibly with a copy button annotation, for use
    in part of a rendered treescope tree.
  """
  if not children:
    return build_one_line_tree_node(
        line=basic_parts.Siblings.build(prefix, suffix),
        path=path,
        background_color=background_color,
    )

  if path is None:
    maybe_copy_button = basic_parts.EmptyPart()
  else:
    maybe_copy_button = foldable_impl.StringCopyButton(
        annotation="Copy path: ",
        copy_string=(
            "(lambda root: root" + "".join(str(key) for key in path) + ")"
        ),
    )

  if isinstance(prefix, str):
    prefix = basic_parts.Text(prefix)

  if isinstance(suffix, str):
    suffix = basic_parts.Text(suffix)

  if background_pattern is not None:
    if background_color is None:
      raise ValueError(
          "background_color must be provided if background_pattern is"
      )

    def wrap_block(block):
      return common_styles.WithBlockPattern(
          block, color=background_color, pattern=background_pattern
      )

    wrap_topline = common_styles.PatternedTopLineSpanGroup
    wrap_bottomline = common_styles.PatternedBottomLineSpanGroup
    indented_child_class = common_styles.ColoredBorderIndentedChildren

  elif background_color is not None and background_color != "transparent":

    def wrap_block(block):
      return common_styles.WithBlockColor(block, color=background_color)

    wrap_topline = common_styles.ColoredTopLineSpanGroup
    wrap_bottomline = common_styles.ColoredBottomLineSpanGroup
    indented_child_class = common_styles.ColoredBorderIndentedChildren

  else:
    wrap_block = lambda rendering: rendering
    wrap_topline = lambda rendering: rendering
    wrap_bottomline = lambda rendering: rendering
    indented_child_class = basic_parts.IndentedChildren

  if first_line_annotation is not None:
    maybe_first_line_annotation = basic_parts.FoldCondition(
        expanded=first_line_annotation
    )
  else:
    maybe_first_line_annotation = basic_parts.EmptyPart()

  return RenderableAndLineAnnotations(
      renderable=wrap_block(
          foldable_impl.FoldableTreeNodeImpl(
              label=foldable_impl.HyperlinkTarget(
                  wrap_topline(prefix),
                  keypath=path,
              ),
              contents=basic_parts.Siblings.build(
                  maybe_copy_button,
                  maybe_first_line_annotation,
                  indented_child_class.build(
                      children,
                      comma_separated=comma_separated,
                      force_trailing_comma=force_trailing_comma,
                  ),
                  wrap_bottomline(suffix),
              ),
          )
      ),
      annotations=maybe_copy_button,
  )


def maybe_qualified_type_name(ty: type[Any]) -> RenderableTreePart:
  """Formats the name of a type so that it is qualified in roundtrip mode.

  Args:
    ty: A type object to render.

  Returns:
    A tree part that renders as a fully-qualified name in roundtrip mode, or as
    the simple class name otherwise. Module names will be either inferred from
    the type's definition or looked up using `canonical_aliases`.
  """
  class_name = ty.__name__

  alias = canonical_aliases.lookup_alias(
      ty, allow_outdated=True, allow_relative=True
  )
  if alias:
    access_path = str(alias)
  else:
    access_path = f"<unknown>.{class_name}"

  if access_path.endswith(class_name):
    return basic_parts.Siblings.build(
        basic_parts.RoundtripCondition(
            roundtrip=common_styles.QualifiedTypeNameSpanGroup(
                basic_parts.Text(access_path.removesuffix(class_name))
            )
        ),
        basic_parts.Text(class_name),
    )
  else:
    return basic_parts.RoundtripCondition(
        roundtrip=basic_parts.Text(access_path),
        not_roundtrip=basic_parts.Text(class_name),
    )


@dataclasses.dataclass(frozen=True)
class EmptyWithHeightGuess(basic_parts.BaseContentlessLeaf):
  """Helper class that reports a guess of its height."""

  fake_newlines: int

  def _compute_newlines_in_expanded_parent(self) -> int:
    return self.fake_newlines


def fake_placeholder_foldable(
    placeholder_content: RenderableTreePart, extra_newlines_guess: int
) -> FoldableTreeNode:
  """Builds a fake placeholder for deferred renderings.

  The main use for this is is to return as the placeholder for a deferred
  rendering, so that it can participate in layout decisions. The
  `get_expand_state` method can be used to infer whether the part was
  collapsed while deferred.

  Args:
    placeholder_content: The content of the placeholder to render.
    extra_newlines_guess: The number of fake newlines to pretend this object
      has.

  Returns:
    A foldable node that does not have any actual content, but pretends to
    contain the given number of newlines for layout decisions.
  """
  return foldable_impl.FoldableTreeNodeImpl(
      contents=basic_parts.siblings(
          placeholder_content,
          EmptyWithHeightGuess(fake_newlines=extra_newlines_guess),
      ),
      expand_state=ExpandState.WEAKLY_EXPANDED,
  )
