# Changes in the V2 API

Penzai includes two neural network APIs:

- The initial design (V1), implemented in `penzai.deprecated.v1.nn` and `penzai.deprecated.v1.data_effects` and used in `penzai.deprecated.v1.example_models`.
- A newer simpified design (V2), now available in `penzai.nn` and used in `penzai.models`, which changes how parameters, state, and side effects work to simplify the user experience and remove boilerplate.

This document explains the major changes in the V2 API, relative to the V1 API. In short:

- Parameters and state variables are now represented by *mutable* `Parameter` and `StateVariable` objects, with ordinary Python shared-reference semantics.
  - Model layers are still immutable JAX PyTree nodes, but their leaves may now be `Parameter` or `StateVariable` instances instead of JAX arrays.
  - Penzai's helper functions can be used to manipulate variables and call models purely functionally as needed.
  - This removes the need for effect handler boilerplate for state and parameter sharing.
- Side inputs should be passed through models as keyword arguments to each layer's `__call__`, instead of being injected as attributes.
  - The signature of `Layer.__call__` is changing from `__call__(self, arg, /)` to `__call__(self, arg, /, **kwargs)`.
  - Layers are expected to ignore side inputs that they do not recognize.
- Parameter initialization is more direct and less verbose.
  - Models will always be initialized eagerly, without a separate `pz.nn.initialize_parameters` step.
  - Parameter sharing will "just work", because shared parameters are represented by multiple copies of the same `Parameter` object.
  - Signatures of the `from_config` classmethod will change from `from_config(cls, **config_kwargs)` to `from_config(cls, name: str, init_base_rng: jax.Array | None, **config_kwargs)`.
- The data-effect system is no longer used.
  - Parameter sharing, state, and side outputs will instead use `Parameter` and `StateVariable`.
  - Side inputs should be passed as keyword arguments.
- The built-in Transformer implementation also supports loading Llama, Mistral, and GPT-NeoX / Pythia models.
  - This implementation is in `penzai.experimental.v2.models.transformer`, and shares the same high-level interface across all transformer variants.

With Penzai release v0.2.0, `penzai.nn` now uses the V2 API, and the V1 API has moved to `penzai.deprecated.v1.nn`.
(This is a **breaking change** to Penzai's existing model implementations.)

This document is intended for users who are already familiar with the old v1 API. If you haven't used the v1 API at all, you may wish to skip this document and instead read ["How to Think in Penzai"](../notebooks/how_to_think_in_penzai.ipynb), which gives a self-contained introduction to the new system.

## Background

In the original design, Penzai represented models as PyTrees of arrays, inspired by Equinox. This simplified the process of passing them through JAX transformations, since JAX already understands how to traverse PyTrees and their data. In particular, Penzai models could be passed through JAX transformations at the top level.

However, there are some features which are difficult to express in an immutable PyTree. For instance, we may want to use the same parameter value in multiple layers (shared parameters), or collect mutable state. Penzai has a system, `penzai.deprecated.v1.data_effects`, designed to support this, which works by temporarily replacing certain sentinel nodes in the PyTree structure (effect references) with mutable Python objects (effect implementations).

To preserve a "functional" top-level interface, Penzai previously required invariants to be maintained across models that use these features:
- All effects must be children of a handler block, which "handles" them.
  - This handler block is responsible for replacing the effect references with the mutable Python implementations.
- Every parameter must appear in the model tree exactly once.
  - If a parameter is not shared, it can be directly inlined.
  - But if a parameter is shared, it must be replaced with a "lookup" effect, and have the actual value of the parameter be owned by some outer handler object.

While this design simplifies passing Penzai models through JAX transformations, this design also has a number of drawbacks:
- Any model with parameter sharing has to be explicitly configured to use Penzai's side-input effects.
  - This complicates the process of initializing parameters.
  - It also makes it hard to visualize shared parameters, since they live "somewhere else" in the model tree.
  - "Model surgery" on models with shared parameters is complex, because it requires explicitly un-binding and re-binding the effect handlers
  - Any user that wants to use a model with shared parameters has to learn about the effect handlers and maintain their invariants. This makes it harder to get started with Penzai.
- Similarly, any model with state needs to be configured using Penzai's state effect handlers.
- Even with `data_effects`, models cannot easily use JAX transformations internally.
  - For instance, there is no current way to support wrapping a single block in `jax.jit` or `jax.remat`, because that block may have effects in it due to some outer handler.
  - This is a blocker for supporting more general and powerful transformations inside Penzai models.

## Changes

### Parameters and state variables becoming mutable, shareable variable objects

In the v1 API, parameters were ordinary PyTree nodes, and state variables were managed by sentinel PyTree nodes in combination with the `penzai.deprecated.v1.data_effects` handlers.
In the v2 API, parameters and state variables are instead represented as "variable" objects, which are mutable Python objects embedded inside the model data structure. There are two types of variable:

- `Parameter`: A parameter in a model, which can be shared between multiple models or multiple parts of a model, and is updated using an optimizer (but not inside the forward pass).
- `StateVariable`: A mutable variable that is intended to be modified inside the forward pass.

Multiple references to the same variable are allowed, and share their values following ordinary Python semantics.

Since they are mutable, variable objects are not directly passable through most JAX transformations. Instead, Penzai provides utilities for identifying and extracting all variables in a model and running the model logic in a functional way. To support this, all variable objects must have a unique *label*, which can either be specified manually or generated automatically.

Note that model layers themselves will remain immutable PyTree nodes; the `Parameter` or `StateVariable` objects will be PyTree leaves. This strikes a balance between mutability and functional manipulation, and ensures that model structures can be easily copied and modified without unexpected side effects.

### Eager parameter initialization and sharing-by-default

In the v1 API, parameter initialization was lazy, with parameters configured with `UninitializedParameter` instances, renamed with `pz.nn.add_parameter_prefix`, possibly shared with `pz.nn.mark_shareable` / `pz.nn.attach_shared_parameters`, and then finally initialized at the top level with `pz.nn.initialize_parameters`.

In the v2 API, parameter initialization is eager, and `Parameter` instances are shared by reference whenever they appears in multiple places in the model.

To enable this, the `from_config` methods of most layers must be modified to take two additional arguments:
- `name`: The name for this layer, used as a prefix for all parameters in this layer,
- `init_base_rng`: A JAX PRNGKey that will be used to initialize all parameters in this layer.

An example of how initializers could change to support the new pattern:

```diff
 @pz.pytree_dataclass(has_implicitly_inherited_fields=True)
 class MLP(pz.nn.Sequential):
   """Sequence of Affine layers."""

   @classmethod
   def from_config(
       cls,
+      name: str,
+      init_base_rng: jax.Array | None,
       feature_sizes: list[int],
       activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu,
       feature_axis: str = "features",
   ) -> MLP:
     assert len(feature_sizes) >= 2
     children = []
     for i, (feats_in, feats_out) in enumerate(
         zip(feature_sizes[:-1], feature_sizes[1:])
     ):
       if i:
         children.append(pz.nn.Elementwise(activation_fn))
       children.append(
-          pz.nn.add_parameter_prefix(
-              f"Affine_{i}",
-              pz.nn.Affine.from_config(
-                  input_axes={feature_axis: feats_in},
-                  output_axes={feature_axis: feats_out},
-              ),
+          pz.nn.Affine.from_config(
+              name=f"{name}/Affine_{i}",
+              init_base_rng=init_base_rng,
+              input_axes={feature_axis: feats_in},
+              output_axes={feature_axis: feats_out},
           )
       )
     return cls(sublayers=children)
```

```diff
 @struct.pytree_dataclass
 class EmbeddingTable(struct.Struct):
   embeddings: parameters.ParameterLike[named_axes.NamedArray]
   vocabulary_axis: str = dataclasses.field(metadata={"pytree_node": False})

   @classmethod
   def from_config(
       cls,
+      name: str,
+      init_base_rng: jax.Array | None,
       vocab_size: int,
       embedding_axes: dict[str, int],
       vocabulary_axis: str = "vocabulary",
       initializer: linear_and_affine.LinearOperatorWeightInitializer = ...,
       dtype: np.typing.DTypeLike = np.float32,
   ) -> EmbeddingTable:
     if vocabulary_axis in embedding_axes:
       raise ValueError(
           f"`vocabulary_axis` {vocabulary_axis} should not appear in"
           f"`embedding_axes` {embedding_axes}"
       )

     return cls(
-        embeddings=parameters.UninitializedParameter(
-            initializer=functools.partial(
-                initializer,
-                input_axes={},
-                output_axes=embedding_axes,
-                parallel_axes={vocabulary_axis: vocab_size},
-                convolution_spatial_axes={},
-                dtype=dtype,
-            ),
-            name="embeddings",
+        embeddings=parameters.make_parameter(
+            f"{name}.embeddings",
+            init_base_rng,
+            initializer,
+            input_axes={},
+            output_axes=embedding_axes,
+            parallel_axes={vocabulary_axis: vocab_size},
+            convolution_spatial_axes={},
+            dtype=dtype,
         ),
         vocabulary_axis=vocabulary_axis,
     )
```

To share parameters between layers, the same layer can simply be used twice. This will insert two references to the same `Parameter` object, which will share their state automatically.

### Simpler side inputs as keyword arguments

Some Penzai layers need access to "side inputs" that do not come directly from their previous layer (e.g. the `ApplyAttentionMask` layer needs to know what attention mask to use). In the v1 API, this was possible using the side input effect in `penzai.deprecated.v1.data_effects`, but this requires a fair amount of boilerplate to use. Much of this boilerplate involves handler IDs and bound effect references, which are used to ensure that there are no conflicts between different inputs.

The v2 API replaces this with a simpler keyword-argument system. The signature of `Layer` is now

```python
class Layer(pz.Struct, abc.ABC):
  @abc.abstractmethod
  def __call__(self, argument: Any, /, **side_inputs) -> Any:
    ...
```
where `**side_inputs` is a collection of arbitrary side inputs. Importantly, each `Layer` should ignore all side inputs it does not recognize. Combinator layers like `Sequential` can then simply forward all side inputs to all of their children.


### Deprecation of `data_effects`

In the v2 API, the functionality originally provided by `data_effects` is instead enabled by variables, keyword argument side inputs, or a combination of these. Given this, the original `data_effects` system is deprecated and no longer recommended for use.


## Migration Guide

### Imports

The V2 API has been moved to the top-level namespace, which means that importing from `penzai.nn` (or using the `penzai.pz` aliases) will refer to the new V2 API components. To simplify migration, the original versions can still be accessed through the `penzai.deprecated.v1` namespace:

```python
# Old V1 API:
from penzai.deprecated.v1 import pz
from penzai.deprecated.v1.example_models import simple_mlp
import penzai.deprecated.v1.toolshed

# New V2 API:
from penzai import pz
from penzai.models import simple_mlp
import penzai.toolshed
```


### Model initialization

As the user of a model, you should provide the initialization PRNGKey as the `init_base_key` argument instead of using a separate `pz.nn.initialize_parameters` call:

```python
# Old
pz.nn.initialize_parameters(
  simple_mlp.MLP.from_config(feature_sizes=[2, 32, 32, 2]),
  jax.random.key(10),
)

# New
mlp = simple_mlp.MLP.from_config(
    name="mlp",
    init_base_rng=jax.random.key(10),
    feature_sizes=[2, 32, 32, 2],
)
```

As a model implementer, you will need to change the signature of your `from_config` method to plumb through the new arguments, as shown in the "Eager parameter initialization and sharing-by-default" section above. Uses of `pz.nn.mark_shareable` and `pz.nn.attach_shared_parameters` can simply be removed, since they are no longer needed.

If you would like to build a model *without* initializing its parameters, you can call `from_config` with `init_base_rng=None`. This will insert placeholder objects in place of each parameter.

### Mutable state and random numbers

Using models with mutable state will no longer require using effect handlers, and should "just work". However, you should ensure that the mutable state is kept inside a `StateVariable` instance. For instance, a simple counter could be implemented as:

```python
@pz.pytree_dataclass
class StateIncrementLayer(pz.nn.Layer):
  state: pz.StateVariable[int]

  def __call__(self, x, **unused_side_inputs):
    # Mutate the `value` attribute of the variable:
    self.state.value = self.state.value + 1
    return x

inc_layer = StateIncrementLayer(pz.StateVariable(value=0))
my_model = pz.nn.Sequential([
  ..., inc_layer, ...
])

_ = my_model(...)
assert inc_layer.state.value == 1
```

Similarly, random number generations will no longer require effect handlers. However, you will need to pass a stateful `RandomStream` as a keyword argument:

```python
# Build a model that needs random numbers.
mlp = simple_mlp.DropoutMLP.from_config(
    name="mlp",
    init_base_rng=jax.random.key(0),
    feature_sizes=[8, 16, 32, 32],
    drop_rate=0.2,
)
# Call with an RNG side input.
result = mlp(
   input_features,
   rng=pz.RandomStream.from_base_key(jax.random.key(0))
)
```

### Capturing intermediate values

Capturing intermediate values can be done easily in the new system by storing those intermediate values in `StateVariable`s, without needing to use effect handlers.

Instead of this pattern from the V1 design

```python
# Old
model_with_collector = pz.de.CollectingSideOutputs.handling(
    pz.select(model)
    .at_instances_of(SomeLayer)
    .insert_after(pz.de.TellIntermediate())
)
_, intermediates = model_with_collector(inputs)
```

you could instead do something like

```python
@pz.pytree_dataclass
class AppendIntermediate(pz.nn.Layer):
  saved: pz.StateVariable[list[Any]]

  def __call__(self, x):
    self.saved.value = self.saved.value + [x]
    return x

intermediates_cell = pz.StateVariable([])
model_saving_intermediates = (
    pz.select(model)
    .at_instances_of(SomeLayer)
    .insert_after(AppendIntermediate(intermediates_cell))
)

_ = model_saving_intermediates(inputs)

intermediates = intermediates_cell.value
```

or use the built-in helper layer `save_intermediates.SaveIntermediate`:

```python
# New
from penzai.toolshed import save_intermediates

model_saving_intermediates = (
    pz.select(model)
    .at_instances_of(SomeLayer)
    .insert_after(save_intermediates.SaveIntermediate())
)

_ = model_saving_intermediates(inputs)

intermediates = [
    saver.cell.value for saver in (
        pz.select(model_saving_intermediates)
        .at_instances_of(save_intermediates.SaveIntermediate)
        .get_sequence()
    )
]
```


### JIT compilation and functional transformations

Models with `Parameter`s or `StateVariable`s must be preprocessed before they can be passed through `jax.jit`, because variable objects are not JAX PyTrees or array types.

The simplest approach is to replace `jax.jit` with `pz.variable_jit`, which wraps `jax.jit` so that it correctly updates variable values. `pz.variable_jit` should be a drop-in replacement for `jax.jit` and allows variables to be contained in any of the function arguments.

For more control, you can also "unbind" the variables and manipulate them using a functional interface. For instance:

```python
# Extract all variables:
model_without_vars, all_vars = pz.unbind_variables(model)

# Freeze cell states, obtaining a JAX PyTree of cell values
frozen_vars = [var.freeze() for var in all_vars]

# Call the model in a functional style and get updated states (safe to run under
# jax.jit or any other function transformation):
output, new_vars = model_without_vars.call_with_local_vars(
  input, frozen_vars
)

# (Optional) Update the original vars:
for k, var in vars.items:
  var.value = new_vars[k].value
```


### Loading pretrained transformers

The V2 API includes a new transformer implementation with support for additional transformer variants. If you are using the v1 Gemma model, you will need to change how you load it:

```python
# Old
from penzai.deprecated.v1.example_models import gemma
model = gemma.model_core.GemmaTransformer.from_pretrained(flax_params_dict)
# (model is an instance of GemmaTransformer)

# New
from penzai.models.transformer import variants
model = variants.gemma.gemma_from_pretrained_checkpoint(flax_params_dict)
# (model is an instance of TransformerLM)
```

Additionally, the types of various model components have changed to become more generic (e.g. `TransformerFeedForward` instead of `GemmaFeedForward`).
