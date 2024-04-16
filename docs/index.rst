:github_url: https://github.com/google-deepmind/penzai/tree/main/docs

======
Penzai
======

.. epigraph::

  **盆 ("pen", tray) 栽 ("zai", planting)** - *an ancient Chinese art of forming
  trees and landscapes in miniature, also called penjing and an ancestor of the
  Japanese art of bonsai.*

Penzai is a JAX library for writing models as legible, functional pytree data
structures, with tools that make it easy to visualize, edit, and analyze them
both before and after they are trained.

It is structured as a collection of modular tools, designed together but each
useable independently:

* `penzai.nn` (``pz.nn``): A declarative combinator-based neural network
  library and an alternative to other neural network libraries like Flax, Haiku,
  Keras, or Equinox, which exposes the full structure of your model's
  forward pass in the model pytree. This means you can see everything your model
  does by pretty printing it, and inject new runtime logic with `jax.tree_util`.
  Like Equinox, there's no magic: models are just callable pytrees under the
  hood.

* `penzai.treescope` (``pz.ts``): A superpowered interactive Python
  pretty-printer, which works as a drop-in replacement for the ordinary
  IPython/Colab renderer. It's designed to help understand Penzai models and
  other deeply-nested JAX pytrees, with built-in support for visualizing
  arbitrary-dimensional NDArrays.

* `penzai.core.selectors` (``pz.select``): A pytree swiss-army-knife,
  generalizing JAX's ``.at[...].set(...)`` syntax to arbitrary type-driven
  pytree traversals, and making it easy to do complex rewrites or
  on-the-fly patching of Penzai models and other data structures.

* `penzai.core.named_axes` (``pz.nx``): A lightweight named axis system which
  lifts ordinary JAX functions to vectorize over named axes, and allows you to
  seamlessly switch between named and positional programming styles without
  having to learn a new array API.

* `penzai.data_effects` (``pz.de``): An opt-in system for side arguments, random
  numbers, and state variables that is built on pytree traversal and puts you
  in control, without getting in the way of writing or using your model.


With Penzai, your neural networks could look like this:

.. container:: glued-cell-output

  .. glue:any:: penzai_teaser
    :doc: _include/_glue_figures.ipynb

This is an interactive visualization; try clicking the `▶` buttons to expand
layers and look at their parameters! (You can also hold shift while scrolling to
scroll horizontally instead of vertically.)


Getting Started
---------------

If you haven't already installed JAX, you should do that first, since the
installation process depends on your platform. You can find instructions in the
`JAX documentation <https://jax.readthedocs.io/en/latest/installation.html>`_.
Afterward, you can install Penzai using

::

  pip install penzai


and import it using

::

  import penzai
  from penzai import pz

(:obj:`penzai.pz` is an *alias namespace*, which makes it easier to reference
common Penzai objects.)

When working in an Colab or IPython notebook, we recommend also configuring
Penzai as the default pretty printer, and enabling some utilities for
interactive use::

  pz.ts.register_as_default()
  pz.ts.register_autovisualize_magic()
  pz.enable_interactive_context()


You can then automatically visualize arrays by using the ``%%autovisualize``
IPython cell magic. Alternatively, you can turn on array autovisualization
by default using ::

  pz.ts.active_autovisualizer.set_interactive(pz.ts.ArrayAutovisualizer())


We recommend starting with the
:doc:`"How to Think in Penzai" <notebooks/how_to_think_in_penzai>`
notebook, which gives a high-level overview of how to think about and use Penzai
models. Afterward, you coould:

* Take a look at one of the example notebooks to see how you can use Penzai to visualize and modify pretrained models.
* Or, read through the guides in the left sidebar to learn more about each of Penzai's components.


.. toctree::
  :hidden:
  :caption: Guides

  notebooks/how_to_think_in_penzai
  notebooks/treescope_prettyprinting
  notebooks/treescope_arrayviz
  notebooks/selectors
  notebooks/named_axes
  notebooks/data_effects
  notebooks/jitting_and_sharding

.. toctree::
  :hidden:
  :caption: Example Notebooks

  notebooks/lora_from_scratch
  notebooks/induction_heads
  notebooks/gemma_from_scratch

.. toctree::
   :hidden:
   :caption: API Reference
   :maxdepth: 2

   api/pz
   _autosummary/penzai.core
   _autosummary/penzai.nn
   _autosummary/penzai.data_effects
   _autosummary/penzai.treescope

.. toctree::
   :hidden:
   :caption: Extras
   :maxdepth: 2

   _autosummary/penzai.toolshed
   _autosummary/penzai.example_models

License
-------

Penzai is licensed under the `Apache 2.0 License <https://github.com/google-deepmind/penzai/blob/main/LICENSE>`_.
