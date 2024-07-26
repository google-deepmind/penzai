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

"""Internal compatibility helpers for treescope migration.

Treescope has moved into a separate module `treescope`, with a new extension
API using the __treescope_repr__ method. This module configures Treescope to
also support the old __penzai_repr__ method, for backwards compatibility during
the transition period.
"""

from __future__ import annotations

import functools
from typing import Any

import treescope


def _active_renderer_extended_with_penzai_repr() -> (
    treescope.renderers.TreescopeRenderer
):
  """Extends the default renderer to support __penzai_repr__."""
  return treescope.active_renderer.get().extended_with(
      handlers=[
          functools.partial(
              treescope.handlers.handle_via_treescope_repr_method,
              method_name="__penzai_repr__",
          ),
      ]
  )


def display(
    value: Any,
    ignore_exceptions: bool = False,
    roundtrip_mode: bool = False,
):
  """Displays a value as an interactively foldable object.

  This is a compatibility wrapper for `treescope.display`, which also formats
  objects implementing the old `__penzai_repr__` method. Consider using
  `treescope.display` directly instead.

  Args:
    value: Value to fold.
    ignore_exceptions: Whether to catch errors during rendering of subtrees and
      show a fallback for those subtrees.
    roundtrip_mode: Whether to start in roundtrip mode.

  Raises:
    RuntimeError: If IPython is not available.
  """
  with treescope.active_renderer.set_scoped(
      _active_renderer_extended_with_penzai_repr()
  ):
    treescope.display(
        value,
        ignore_exceptions=ignore_exceptions,
        roundtrip_mode=roundtrip_mode,
    )


def show(*args, wrap: bool = False, space_separated: bool = True):
  """Shows a list of objects inline, like python print, but with rich display.

  This is a compatibility wrapper for `treescope.show`, which also formats
  objects implementing the old `__penzai_repr__` method. Consider using
  `treescope.show` directly instead.

  Args:
    *args: Values to show. Strings show as themselves, like python's print.
      Anything renderable by Treescope will show as it's treescope
      representation. Anything with a rich IPython representation will show as
      its IPython representation.
    wrap: Whether to wrap at the end of the line.
    space_separated: Whether to add single spaces between objects.

  Raises:
    RuntimeError: If IPython is not available.
  """
  with treescope.active_renderer.set_scoped(
      _active_renderer_extended_with_penzai_repr()
  ):
    treescope.show(*args, wrap=wrap, space_separated=space_separated)


def register_as_default(streaming: bool = True, compress_html: bool = True):
  """Registers treescope as the default IPython renderer.

  This is a compatibility wrapper for `treescope.register_as_default`, which
  also turns on support for rendering objects implementing the old
  `__penzai_repr__` method. Consider using `treescope.register_as_default`
  directly instead.

  Args:
    streaming: Whether to render in streaming mode, which immediately displays
      the structure of the output while computing more expensive leaf
      renderings. This is useful in interactive contexts, but can mess with
      other users of IPython's formatting because the final rendered HTML is
      empty.
    compress_html: Whether to zlib-compress (i.e. zip) treescope renderings to
      reduce their size when transmitted to the browser or saved into a
      notebook.

  Raises:
    RuntimeError: If IPython is not available.
  """
  treescope.register_as_default(
      streaming=streaming, compress_html=compress_html
  )
  treescope.active_renderer.set_globally(
      _active_renderer_extended_with_penzai_repr()
  )


def basic_interactive_setup(autovisualize_arrays: bool = True):
  """Sets up IPython for interactive use with Treescope.

  This is a compatibility wrapper for `treescope.basic_interactive_setup`, which
  also turns on support for rendering objects implementing the old
  `__penzai_repr__` method. Consider using `treescope.basic_interactive_setup`
  directly instead.

  Args:
    autovisualize_arrays: Whether to automatically visualize arrays.
  """
  treescope.basic_interactive_setup(autovisualize_arrays=autovisualize_arrays)
  treescope.active_renderer.set_globally(
      _active_renderer_extended_with_penzai_repr()
  )


def enable_interactive_context():
  """Compatibility function: No-op for consistency with old Treescope API.

  Originally, global overrides for Penzai/Treescope ContextualValue objects
  needed to be turned on using `enable_interactive_context`. This is no longer
  necessary, but old Penzai code may still call this function. This function
  is a no-op, for backwards compatibility. Eventually this will raise a warning
  and, later, be removed entirely.
  """
  pass


def disable_interactive_context():
  """Raises an error, since disabling is no longer supported.

  Originally, global overrides for Penzai/Treescope ContextualValue objects
  needed to be turned on using `enable_interactive_context`. Global overrides
  are now turned on by default, and this function is no longer supported.
  """
  raise RuntimeError("Disabling interactive context is no longer supported.")
