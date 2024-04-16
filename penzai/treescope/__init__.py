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

"""Treescope: Penzai's interactive HTML pretty-printer and array visualizer.

You can configure treescope as the default IPython pretty-printer using ::

  pz.ts.register_as_default()

  # Optional:
  pz.ts.register_autovisualize_magic()
  pz.enable_interactive_context()
  pz.ts.active_autovisualizer.set_interactive(pz.ts.ArrayAutovisualizer())

You can also pretty-print individual values using `pz.show` or `pz.ts.display`.
See also the :doc:`pz.ts </api/pz.ts>` module for convenient access to the most
commonly-used treescope functions and classes.
"""

from . import arrayviz
from . import autovisualize
from . import canonical_aliases
from . import copypaste_fallback
from . import default_renderer
from . import figures
from . import foldable_representation
from . import handlers
from . import html_compression
from . import html_escaping
from . import renderer
from . import selection_rendering
from . import treescope_ipython
