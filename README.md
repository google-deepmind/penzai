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

Penzai is structured as a collection of modular tools, designed together but
each useable independently:

* `penzai.nn` (`pz.nn`): A declarative combinator-based neural network
  library and an alternative to other neural network libraries like Flax, Haiku,
  Keras, or Equinox, which exposes the full structure of your model's
  forward pass in the model pytree. This means you can see everything your model
  does by pretty printing it, and inject new runtime logic with `jax.tree_util`.
  Like Equinox, there's no magic: models are just callable pytrees under the
  hood.

* `penzai.treescope` (`pz.ts`): A superpowered interactive Python
  pretty-printer, which works as a drop-in replacement for the ordinary
  IPython/Colab renderer. It's designed to help understand Penzai models and
  other deeply-nested JAX pytrees, with built-in support for visualizing
  arbitrary-dimensional NDArrays.

* `penzai.core.selectors` (`pz.select`): A pytree swiss-army-knife,
  generalizing JAX's `.at[...].set(...)` syntax to arbitrary type-driven
  pytree traversals, and making it easy to do complex rewrites or
  on-the-fly patching of Penzai models and other data structures.

* `penzai.core.named_axes` (`pz.nx`): A lightweight named axis system which
  lifts ordinary JAX functions to vectorize over named axes, and allows you to
  seamlessly switch between named and positional programming styles without
  having to learn a new array API.

* `penzai.data_effects` (`pz.de`): An opt-in system for side arguments, random
  numbers, and state variables that is built on pytree traversal and puts you
  in control, without getting in the way of writing or using your model.

Documentation on Penzai can be found at
[https://penzai.readthedocs.io](https://penzai.readthedocs.io).


## Getting Started

If you haven't already installed JAX, you should do that first, since the
installation process depends on your platform. You can find instructions in the
[JAX documentation](https://jax.readthedocs.io/en/latest/installation.html).
Afterward, you can install Penzai using

```
pip install penzai
```

and import it using

```
import penzai
from penzai import pz
```

(`penzai.pz` is an *alias namespace*, which makes it easier to reference
common Penzai objects.)

When working in an Colab or IPython notebook, we recommend also configuring
Penzai as the default pretty printer, and enabling some utilities for
interactive use:

```
pz.ts.register_as_default()
pz.ts.register_autovisualize_magic()
pz.enable_interactive_context()

# Optional: enables automatic array visualization
pz.ts.active_autovisualizer.set_interactive(pz.ts.ArrayAutovisualizer())
```

Here's how you could initialize and visualize a simple neural network:

```
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

```
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
we recommend starting with the ["How to Think in Penzai" tutorial][], or one
of the other tutorials in the [Penzai documentation][].

["How to Think in Penzai" tutorial]: https://penzai.readthedocs.io/en/stable/notebooks/how_to_think_in_penzai.html
[Penzai documentation]: https://penzai.readthedocs.io


---

*This is not an officially supported Google product.*
