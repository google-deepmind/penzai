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

"""Patches the ipdb debugger to enable rich output (e.g. with treescope).

``displayhook`` is the method that is called to print outputs in PDB to the
console. This isn't a documented attribute on `pdb.PDB` but overriding it
appears to work correctly.

This patch is fairly unergonomic, since it directly rewrites methods on the
IPython ``Pdb`` class. However, modifying this displayhook is otherwise fairly
complex, since the debugger class is hardcoded in various places to enable
IPython magics like ``%debug``. This also has the advantage of automatically
patching all subclasses of IPython's debugger as well (unless they specifically
override the displayhook).

Note that this will only use `treescope` if treescope has been registered
as the default IPython display hook, using `treescope.register_as_default()`
"""

import IPython.core.debugger
import IPython.display


def displayhook_with_ipython(self, obj):
  """Custom displayhook to display results with IPython."""
  del self
  # reproduce the behavior of the standard displayhook, not printing None
  if obj is not None:
    IPython.display.display(obj)


old_displayhook = None


def patch_ipdb(force: bool = False):
  """Patches the IPython debugger to use `IPython.display.display`."""
  global old_displayhook
  if old_displayhook is not None and not force:
    raise ValueError(
        "IPython.core.debugger.Pdb.displayhook has already been patched!"
    )
  old_displayhook = IPython.core.debugger.Pdb.displayhook
  IPython.core.debugger.Pdb.displayhook = displayhook_with_ipython


def unpatch_ipdb():
  """Unpatches the IPython debugger so that it uses its original displayhook."""
  global old_displayhook
  assert old_displayhook is not None
  assert IPython.core.debugger.Pdb.displayhook is displayhook_with_ipython
  IPython.core.debugger.Pdb.displayhook = old_displayhook
  old_displayhook = None
