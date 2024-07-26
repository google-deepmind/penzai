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

"""Common aliases for treescope."""

# pylint: disable=g-multiple-import,g-importing-member,unused-import

from penzai.treescope._compatibility_setup import (
    display,
    register_as_default,
    basic_interactive_setup,
)
from treescope import (
    ArrayAutovisualizer,
    active_autovisualizer,
    active_expansion_strategy,
    active_renderer,
    Autovisualizer,
    ChildAutovisualizer,
    default_diverging_colormap,
    default_magic_autovisualizer,
    default_sequential_colormap,
    integer_digitbox,
    IPythonVisualization,
    register_autovisualize_magic,
    register_context_manager_magic,
    render_array_sharding,
    render_array,
    render_to_html,
    render_to_text,
    using_expansion_strategy,
)
from treescope.figures import (
    inline,
    indented,
    with_font_size,
    with_color,
    bolded,
    styled,
    text_on_color,
)

vocab_autovisualizer = ArrayAutovisualizer.for_tokenizer
