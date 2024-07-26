penzai.treescope (Moved!)
=========================================================

.. module:: penzai.treescope

Treescope, Penzai's interactive pretty-printer, has moved to an independent
package `treescope`. See the
`Treescope documentation <https://treescope.readthedocs.io/en/stable/>`_
for more information.

The subpackage `penzai.treescope` contains compatibility stubs to ensure that
libraries that use `penzai.treescope` to render custom types continue to work.
We recommend that new users use `treescope` directly.

One exception is if you want to use Treescope to render custom types that were
configured using the old ``__penzai_repr__`` extension method and which have
not yet been migrated to use the ``__treescope_repr__`` method. In this case,
you can enable Treescope using the legacy Penzai API, using ::

  pz.ts.basic_interactive_setup()

or, for more control: ::

  pz.ts.register_as_default()
  pz.ts.register_autovisualize_magic()
  pz.ts.active_autovisualizer.set_globally(pz.ts.ArrayAutovisualizer())

You can also pretty-print individual values using `pz.show` or `pz.ts.display`.
