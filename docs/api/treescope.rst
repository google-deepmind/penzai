(Moved!) Treescope: Penzai's interactive pretty-printer
=========================================================

.. module:: penzai.treescope

Treescope, Penzai's interactive pretty-printer, has moved to an independent
package `treescope`.

The subpackage `penzai.treescope` contains compatibility stubs to ensure that
packages using `penzai.treescope` continue to work. We recommend that new users
instead use `treescope` directly.

If using the Penzai compatibility version of Treescope, you can configure
Treescope as the default IPython pretty-printer using ::

  pz.ts.basic_interactive_setup()

or, for more control: ::

  pz.ts.register_as_default()
  pz.ts.register_autovisualize_magic()
  pz.enable_interactive_context()
  pz.ts.active_autovisualizer.set_interactive(pz.ts.ArrayAutovisualizer())

You can also pretty-print individual values using `pz.show` or `pz.ts.display`.
