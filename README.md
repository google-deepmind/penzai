# Penzai

> **盆 ("pen", tray) 栽 ("zai", planting)** - *an ancient Chinese art of forming
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

![Screenshot of the Gemma model in Penzai](docs/_static/readme_teaser.png)

Penzai is structured as a collection of modular tools, designed together but
each useable independently:


* A superpowered interactive Python pretty-printer:

  * `penzai.treescope` (``pz.ts``): A drop-in replacement for the ordinary
  IPython/Colab renderer. It's designed to help understand Penzai models and
  other deeply-nested JAX pytrees, with built-in support for visualizing
  arbitrary-dimensional NDArrays.

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
    forward pass in the model pytree. This means you can see everything your model
    does by pretty printing it, and inject new runtime logic with `jax.tree_util`.
    Like Equinox, there's no magic: models are just callable pytrees under the
    hood.

  * `penzai.data_effects` (``pz.de``): An opt-in system for side arguments, random
    numbers, and state variables that is built on pytree traversal and puts you
    in control, without getting in the way of writing or using your model.

  * **(NEW)** `penzai.experimental.v2`: An improved version of `penzai.nn` with
    less boilerplate, including first-class support for mutable state and
    parameter sharing.

* An implementation of the Gemma open-weights model using modular components and
  named axes, built to enable interpretability and model surgery research.

  * **(NEW)** The V2 version also supports Llama, Mistral, and GPT-NeoX / Pythia
    models!

Documentation on Penzai can be found at
[https://penzai.readthedocs.io](https://penzai.readthedocs.io).

> [!IMPORTANT]
> Penzai currently has two versions of its neural network API: the original
> "V1" API, and a new "V2" API located in `penzai.experimental.v2`.
>
> The V2 API aims to be simpler and more flexible, by introducing first-class
> support for mutable state and parameter sharing, and removing unnecessary
> boilerplate. It also includes a more flexible transformer implementation with
> support for more pretrained model variants. You can read about the
> differences between the two APIs in the
> ["Changes in the V2 API"](v2_differences) overview.
>
> We plan to stabilize the V2 API and move it out of experimental in release
> ``0.2.0``, replacing the V1 API. If you wish to keep the V1 behavior, we
> recommend pinning the ``0.1.x`` release series (e.g. ``penzai>=0.1,<0.2``)
> to avoid breaking changes.

[v2_differences]: https://penzai.readthedocs.io/en/stable/guides/v2_differences.html


## Getting Started

If you haven't already installed JAX, you should do that first, since the
installation process depends on your platform. You can find instructions in the
[JAX documentation](https://jax.readthedocs.io/en/latest/installation.html).
Afterward, you can install Penzai using

```python
pip install penzai
```

and import it using

```python
import penzai
from penzai import pz
```

(`penzai.pz` is an *alias namespace*, which makes it easier to reference
common Penzai objects.)

When working in an Colab or IPython notebook, we recommend also configuring
Penzai as the default pretty printer, and enabling some utilities for
interactive use:

```python
pz.ts.register_as_default()
pz.ts.register_autovisualize_magic()
pz.enable_interactive_context()

# Optional: enables automatic array visualization
pz.ts.active_autovisualizer.set_interactive(pz.ts.ArrayAutovisualizer())
```

Here's how you could initialize and visualize a simple neural network:

```python
from penzai.example_models import simple_mlp
mlp = pz.nn.initialize_parameters(
    simple_mlp.MLP.from_config([8, 32, 32, 8]),
    jax.random.key(42),
)

# Models and arrays are visualized automatically when you output them from a
# Colab/IPython notebook cell:
mlp
```

Here's how you could capture and extract the activations after the elementwise
nonlinearities:

```python
mlp_with_captured_activations = pz.de.CollectingSideOutputs.handling(
    pz.select(mlp)
    .at_instances_of(pz.nn.Elementwise)
    .insert_after(pz.de.TellIntermediate())
)

output, intermediates = mlp_with_captured_activations(
  pz.nx.ones({"features": 8})
)
```

To learn more about how to build and manipulate neural networks with Penzai,
we recommend starting with the "How to Think in Penzai" tutorial ([V1 API version][how_to_think_1], [V2 API version][how_to_think_2]), or one
of the other tutorials in the [Penzai documentation][].

[how_to_think_1]: https://penzai.readthedocs.io/en/stable/notebooks/how_to_think_in_penzai.html
[how_to_think_2]: https://penzai.readthedocs.io/en/stable/notebooks/v2_how_to_think_in_penzai.html
[Penzai documentation]: https://penzai.readthedocs.io


---

*This is not an officially supported Google product.*
