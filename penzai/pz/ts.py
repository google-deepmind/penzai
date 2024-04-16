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

from penzai.treescope.arrayviz.array_autovisualizer import (
    ArrayAutovisualizer,
)
from penzai.treescope.arrayviz.arrayviz import (
    default_diverging_colormap,
    default_sequential_colormap,
    integer_digitbox,
    render_array,
    text_on_color,
    render_array_sharding,
    render_sharded_shape,
)
from penzai.treescope.autovisualize import (
    Autovisualizer,
    ChildAutovisualizer,
    IPythonVisualization,
    CustomTreescopeVisualization,
    active_autovisualizer,
)
from penzai.treescope.copypaste_fallback import (
    NotRoundtrippable,
)
from penzai.treescope.default_renderer import (
    active_renderer,
    active_expansion_strategy,
    render_to_text,
    render_to_html,
    using_expansion_strategy,
)
from penzai.treescope.figures import (
    inline,
    indented,
    with_font_size,
    with_color,
    bolded,
    styled,
)
from penzai.treescope.treescope_ipython import (
    default_magic_autovisualizer,
    display,
    register_as_default,
    register_autovisualize_magic,
    register_context_manager_magic,
)

vocab_autovisualizer = ArrayAutovisualizer.for_tokenizer
