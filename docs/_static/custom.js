/**
 * Copyright 2024 The Penzai Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Move cell outputs into iframes to avoid JS/CSS conflicts and improve
  // responsiveness of the main page.
  const frameTpls = document.querySelectorAll('template.cell_output_frame_src');
  for (let frameTpl of frameTpls) {
    let frame = document.createElement('iframe');
    frame.classList.add('cell_output_frame');
    frame.sandbox = [
      'allow-downloads', 'allow-forms', 'allow-pointer-lock', 'allow-popups',
      'allow-same-origin', 'allow-scripts',
      'allow-storage-access-by-user-activation',
      'allow-popups-to-escape-sandbox'
    ].join(' ');
    frame.addEventListener("load", () => {
      frame.contentDocument.body.appendChild(frameTpl.content.cloneNode(true));
      frame.contentDocument.body.style.height = 'fit-content';
      frame.contentDocument.body.style.margin = '0';
      frame.contentDocument.body.style.padding = '0.5em 1ch 0.5em 1ch';
      const observer = new ResizeObserver(() => {
        const frameBounds = frame.getBoundingClientRect();
        const bounds = frame.contentDocument.body.getBoundingClientRect();
        if (frame.contentDocument.body.scrollWidth > frameBounds.width) {
          // Make room for the scrollbar.
          frame.style.height = `calc(1em + ${bounds.height}px)`;
        } else {
          frame.style.height = `${bounds.height}px`;
        }
      });
      observer.observe(frame.contentDocument.body);
    });
    frame.src = "about:blank";
    frameTpl.parentNode.replaceChild(frame, frameTpl);
  }

  // Add zero-width spaces to sidebar identifiers to improve word breaking.
  const sidebarNodes = document.querySelectorAll(
      '.bd-docs-nav .reference, .bd-docs-nav .reference *');
  for (let parent of sidebarNodes) {
    for (let elt of parent.childNodes) {
      if (elt instanceof Text) {
        elt.textContent =
            elt.textContent
                .split(/(?=_)|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])/g)
                .join('\u200b');
      }
    }
  }
});
