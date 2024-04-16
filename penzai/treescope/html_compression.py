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

"""Helper function to compress HTML for saving and displaying.

Treescope HTML renderings are often much larger in size than the original `repr`
because they render objects using many nested HTML elements. However, most of
this structure is very repetitive.

This module contains utilities to compress HTML snippets into equivalent
JavaScript snippets using `zlib`, which is supported by Python and also by most
modern browsers.
"""

import base64
import html
import json
import uuid
import zlib
from penzai.treescope import html_escaping


def decompression_preamble() -> str:
  """Returns the preamble script to include before any compressed outputs."""
  # Emit javascript to deserialize and decompress. This script needs to:
  # - Wait until the element is visible, to avoid crashing the browser by
  #   decompressing too many things at once
  # - Convert the input string to a byte array
  # - Decompress and decode the byte array
  # - Reassemble the full string
  # - Wait for any previously-compressed renderings to finish
  # - Copy it into a new HTML element
  # - Replace itself with that new HTML element
  # - Rebuild all scripts inside the new HTML element so that they also execute,
  #   since scripts inserted by innerHTML don't execute by default.
  return html_escaping.without_repeated_whitespace("""
    <script>
    /* penzai.treescope rendering of a Python object (compressed) */
    (()=>{
    let observer;
    let lastStep = new Promise((resolve, reject) => {
      observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            resolve();
            observer.disconnect();
            return;
          }
        }
      }, {rootMargin: "1000px"});
    });
    window.treescope_decompress_enqueue = (encoded, destId) => {
      const previous = lastStep;
      const destElt = document.getElementById(destId);
      lastStep = (async () => {
        await previous;
        let blob = new Blob([
            Uint8Array.from(atob(encoded), (m) => m.codePointAt(0))
        ]);
        let reader = blob.stream().pipeThrough(
          new DecompressionStream("deflate")
        ).pipeThrough(
          new TextDecoderStream("utf-8")
        ).getReader();
        let parts = [];
        while (true) {
          let step = await reader.read();
          if (step.done) { break; }
          parts.push(step.value);
        }
        let newElt = document.createElement("div");
        newElt.innerHTML = parts.join("");
        destElt.parentNode.replaceChild(newElt, destElt);
        for (let oldScript of newElt.querySelectorAll("script")) {
            let newScript = document.createElement("script");
            newScript.type = oldScript.type;
            newScript.textContent = oldScript.textContent;
            oldScript.parentNode.replaceChild(newScript, oldScript);
        }
      })();
      requestAnimationFrame(() => {
        observer.observe(destElt);
      });
    }
    })();
    </script>
  """)


def compress_html(
    html_src: str,
    include_preamble: bool = True,
    loading_message: str | None = None,
) -> str:
  """Compresses HTML source to an equivalent compressed JavaScript <script> tag.

  This function compressed and then encodes the given HTML contents into an
  equivalent JavaScript source tag, which can then be output in place of the
  original HTML.

  Args:
    html_src: The HTML we want to compress.
    include_preamble: Whether to include the preamble script. Can be disabled if
      you plan to compress multiple HTML fragments and display them one at a
      time; to ensure scripts execute in the right order, only the first should
      include the preamble.
    loading_message: Message to display while the HTML is decompressing.

  Returns:
    An HTML snippet containing a <script> tag that will render the same as the
    given HTML when loaded in a modern browser.
  """
  # Compress the input string
  compressed = zlib.compress(html_src.encode("utf-8"), zlib.Z_BEST_COMPRESSION)
  # Serialize it
  serialized = json.dumps(base64.b64encode(compressed).decode("ascii"))
  # Emit javascript to deserialize and decompress it.
  if loading_message:
    src_template = (
        '<div id="__REPLACE_ME_WITH_UNIQUE_ID__"><script>'
        "window.treescope_decompress_enqueue("
        '__REPLACE_ME_WITH_SERIALIZED__, "__REPLACE_ME_WITH_UNIQUE_ID__");'
        '</script><span style="color: #aaaaaa; font-family: monospace">'
        f"{html.escape(loading_message)}</span></div>"
    )
  else:
    src_template = (
        '<div id="__REPLACE_ME_WITH_UNIQUE_ID__"><script>'
        "window.treescope_decompress_enqueue("
        '__REPLACE_ME_WITH_SERIALIZED__, "__REPLACE_ME_WITH_UNIQUE_ID__");'
        "</script></div>"
    )
  src = src_template.replace(
      "__REPLACE_ME_WITH_SERIALIZED__", serialized
  ).replace(
      "__REPLACE_ME_WITH_UNIQUE_ID__", "compress_html_" + uuid.uuid4().hex
  )
  if include_preamble:
    src = decompression_preamble() + src
  return src
