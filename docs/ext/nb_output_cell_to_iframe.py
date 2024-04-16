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
"""Wraps MyST-nb output cells in HTML templates.

This can be used to sandbox the environment of output cell renderings, which is
useful for Treescope renderings since they make heavy use of Javascript and CSS,
and can also improve responsiveness of the main page.

This transformation replaces output cells with ``<template>`` elements tagged
with the class ``cell_output_frame_src``. This can then be detected by the
``custom.js`` javascript, which will construct ``<iframe>`` elements with those
contents.
"""

from typing import Any, Mapping

from docutils import nodes
from sphinx import application
from sphinx.transforms import post_transforms


class OutputCellsToIframe(post_transforms.SphinxPostTransform):
  """Wraps output cell renderings in HTML templates."""

  default_priority = 199
  formats = ("html",)

  def run(self, **kwargs):
    for node in self.document.findall(nodes.container):
      if (
          "glued-cell-output" in node["classes"]
          or node.get("nb_element") == "cell_code_output"
      ):
        node.insert(
            0,
            nodes.raw(
                text='<template class="cell_output_frame_src">', format="html"
            ),
        )
        node.append(nodes.raw(text="</template>", format="html"))


def setup(app: application.Sphinx) -> Mapping[str, Any]:
  app.add_post_transform(OutputCellsToIframe)
  return dict(version="0.1", parallel_read_safe=True)
