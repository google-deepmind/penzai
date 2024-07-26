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
"""Rewrites aliases in the pz namespace to point to their qualified versions."""

from typing import Any, Mapping

from sphinx import application
from treescope import canonical_aliases


def substitute_pz_in_autodoc_docstring(app, what, name, obj, options, lines):
  """Rewrites docstrings for objects defined in `penzai.pz` to add alias links."""
  del app, what, options
  if (
      name.startswith("penzai.pz")
      or name.startswith("penzai.experimental.v2.pz")
      or name.startswith("penzai.deprecated.v1.pz")
  ):
    if not lines:
      lines.append("")
    alias = canonical_aliases.lookup_alias(obj)
    if alias is not None:
      lines[0] = f"*Alias of* :obj:`{str(alias)}`: " + lines[0]


def setup(app: application.Sphinx) -> Mapping[str, Any]:
  app.connect("autodoc-process-docstring", substitute_pz_in_autodoc_docstring)
  return dict(version="0.1", parallel_read_safe=True)
