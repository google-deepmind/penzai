:github_url: https://github.com/google-deepmind/penzai/tree/main/docs

======
Penzai
======

  **盆 ("pen", tray) 栽 ("zai", planting)** - *an ancient Chinese art of forming
  trees and landscapes in miniature, also called penjing and an ancestor of the
  Japanese art of bonsai.*

Penzai is a JAX library for writing models as legible, functional pytree data
structures, along with tools for visualizing, modifying, and analyzing them.
Penzai focuses on **making it easy to do stuff with models after they have been
trained**, making it a great choice for research involving reverse-engineering
or ablating model components, inspecting and probing internal activations,
performing model surgery, debugging architectures, and more. (But if you just
want to build and train a model, you can do that too!)

With Penzai, your neural networks could look like this:

.. container:: glued-cell-output

  .. glue:any:: penzai_teaser
    :doc: _include/_glue_figures.ipynb

This is an interactive visualization; try clicking the `▶` buttons to expand
layers and look at their parameters! (You can also hold shift while scrolling to
scroll horizontally instead of vertically.)

Penzai is structured as a collection of modular tools, designed together but each
useable independently:

* A superpowered interactive Python pretty-printer:

  * `Treescope <https://treescope.readthedocs.io/en/stable/>`_ (``pz.ts``):
    A drop-in replacement for the ordinary IPython/Colab renderer, originally
    a part of Penzai but now available as a standalone package. It's designed to
    help understand Penzai models and other deeply-nested JAX pytrees, with
    built-in support for visualizing arbitrary-dimensional NDArrays.

* A set of JAX tree and array manipulation utilities:

  * `penzai.core.selectors` (``pz.select``): A pytree swiss-army-knife,
    generalizing JAX's ``.at[...].set(...)`` syntax to arbitrary type-driven
    pytree traversals, and making it easy to do complex rewrites or
    on-the-fly patching of Penzai models and other data structures.

  * `penzai.core.named_axes` (``pz.nx``): A lightweight named axis system which
    lifts ordinary JAX functions to vectorize over named axes, and allows you to
    seamlessly switch between named and positional programming styles without
    having to learn a new array API.

* A declarative combinator-based neural network library, where models are
  represented as easy-to-modify data structures:

  * `penzai.nn` (``pz.nn``): An alternative to other neural network libraries like
    Flax, Haiku, Keras, or Equinox, which exposes the full structure of your model's
    forward pass using declarative combinators. Like Equinox, models are
    represented as JAX PyTrees, which means you can see everything your model
    does by pretty printing it, and inject new runtime logic with `jax.tree_util`.
    However, `penzai.nn` models may also contain mutable variables at the leaves
    of the tree, allowing them to keep track of mutable state and parameter
    sharing.

* A modular implementation of common Transformer architectures:

  * `penzai.models.transformer`: A reference Transformer implementation that
    can load the pre-trained weights for the Gemma, Llama, Mistral, and
    GPT-NeoX / Pythia architectures. Built using modular components and named
    axes, to support research into interpretability, model surgery, and
    training dynamics:

These components are described in more detail in the guides in the left
sidebar.

.. important::
  Penzai 0.2 includes a number of breaking changes to the neural network API.
  These changes are intended to simplify common workflows
  by introducing first-class support for mutable state and parameter sharing
  and removing unnecessary boilerplate. You can read about the differences
  between the old "V1" API and the current "V2" API in the
  :doc:`"Changes in the V2 API" <guides/v2_differences>` overview.

  If you are currently using the V1 API and have not yet converted to the V2
  system, you can instead keep the old behavior by importing from the
  `penzai.deprecated.v1` submodule, e.g. ::

    from penzai.deprecated.v1 import pz
    from penzai.deprecated.v1.example_models import simple_mlp



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
Treescope (Penzai's companion pretty-printer) as the default pretty printer, and
enabling some utilities for interactive use::

  import treescope
  treescope.basic_interactive_setup(autovisualize_arrays=True)

Here's how you could initialize and visualize a simple neural network::

  from penzai.models import simple_mlp
  mlp = simple_mlp.MLP.from_config(
      name="mlp",
      init_base_rng=jax.random.key(0),
      feature_sizes=[8, 32, 32, 8]
  )

  # Models and arrays are visualized automatically when you output them from a
  # Colab/IPython notebook cell:
  mlp

Here's how you could capture and extract the activations after the elementwise
nonlinearities::

  @pz.pytree_dataclass
  class AppendIntermediate(pz.nn.Layer):
    saved: pz.StateVariable[list[Any]]
    def __call__(self, x: Any, **unused_side_inputs) -> Any:
      self.saved.value = self.saved.value + [x]
      return x

  var = pz.StateVariable(value=[], label="my_intermediates")

  # Make a copy of the model that saves its activations:
  saving_model = (
      pz.select(mlp)
      .at_instances_of(pz.nn.Elementwise)
      .insert_after(AppendIntermediate(var))
  )

  output = saving_model(pz.nx.ones({"features": 8}))
  intermediates = var.value


To learn more about how to build and manipulate neural networks with Penzai,
we recommend starting with the
:doc:`"How to Think in Penzai" tutorial <notebooks/how_to_think_in_penzai>`,
which gives a high-level overview of how to think about and use Penzai
models. Afterward, you could:

* Take a look at one of the example notebooks to see how you can use Penzai to
  visualize and modify pretrained models.
* Or, read through the guides in the left sidebar to learn more about each of
  Penzai's components.

.. toctree::
  :hidden:
  :caption: Neural Networks

  notebooks/how_to_think_in_penzai
  guides/howto_reference
  notebooks/jitting_and_sharding
  notebooks/lora_from_scratch
  notebooks/induction_heads
  guides/v2_differences

.. toctree::
  :hidden:
  :caption: JAX Tools

  notebooks/selectors
  notebooks/named_axes

.. toctree::
  :hidden:
  :caption: See Also

  Pretty-Printing With Treescope <https://treescope.readthedocs.io/en/stable/notebooks/pretty_printing.html>
  Array Visualization With Treescope <https://treescope.readthedocs.io/en/stable/notebooks/array_visualization.html>

.. toctree::
   :hidden:
   :caption: API Reference
   :maxdepth: 2

   api/pz
   _autosummary/penzai.core
   _autosummary/penzai.nn
   _autosummary/penzai.models
   _autosummary/penzai.toolshed
   api/treescope
   api/penzai.deprecated.v1


License
-------

Penzai is licensed under the `Apache 2.0 License <https://github.com/google-deepmind/penzai/blob/main/LICENSE>`_.
