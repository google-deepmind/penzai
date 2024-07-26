
# Guide - Common Tasks

This notebook is a guide to accomplishing a variety of tasks with Penzai, using the new V2 NN API.

For this guide, we assume you have imported the `pz` alias namespace:
```
from penzai import pz
```

## Visualization

The Treescope pretty-printer has moved to a separate package. This is a short overview of how to use it; see the [Treescope documentation](https://treescope.readthedocs.io/en/stable/) for more details.

### Setting up pretty-printing
When using Penzai in IPython notebooks, it's recommended to set up Treescope as the default pretty-printer and turn on array autovisualization. You can do this by running

```
import treescope
treescope.basic_interactive_setup(autovisualize_arrays=True)
```

Alternatively, you can instead pretty-print specific objects using `treescope.show(...)`.

### Manually vizualizing arrays
Automatic visualization truncates large arrays to avoid showing visualizations that are too large. You can manually render specific arrays using `treescope.render_array`, which will show the full array data, and also allows you to customize axis labels or add annotations for particular values.

----------------------------
## Using Named Axes

This is a short overview of the named array system. For more details, see the [named array tutorial](../notebooks/named_axes.ipynb).

### Converting between named arrays and JAX arrays

To convert a JAX array to a Penzai named array, you can use `pz.nx.wrap` to wrap it, and then the `tag` method to assign names to each axis:

```
pz.nx.wrap(jnp.ones([3, 4, 5])).tag("foo", "bar", "baz")
```

For convenience, you can also pass the axis names to `wrap`:

```
pz.nx.wrap(jnp.ones([3, 4, 5]), "foo", "bar", "baz")
```

To convert a named array back to a JAX array, you can `untag` the axis names, and then call `unwrap`:

```
my_named_array.untag("bar", "foo", "baz").unwrap()
```

Alternatively,  you can pass the axis names directly to `unwrap`:

```
my_named_array.unwrap("bar", "foo", "baz")
```

The order of axis names in `untag` (or `unwrap`) will determine the order of the positional axes in the result.

### Inspecting array shapes

Penzai named arrays have two shapes:

- The `.positional_shape` attribute is the sequence of dimension sizes for the positional axes in the array.
- The `.named_shape` attribute is a dictionary mapping axis names to their dimension sizes.

When you call `wrap` on a JAX array, it will initially have a `.positional_shape` that matches the JAX array, and an empty named shape. After you `tag` the axes with names, they will be removed from `.positional_shape` and instead appear in `.named_shape`. Note that each axis is in *either* the positional shape *or* the named shape, never both.

It's possible for named arrays to have "mixed" shapes. For instance, you can call `.untag` with only a subset of axis names, which will move a subset of axes from the `.named_shape` to the `.positional_shape`.

### "Lifting" positional operations

To run operations on named arrays, you can use `pz.nx.nmap`, which maps a function over a collection of named arrays. Similar to `jax.vmap`, this vectorizes the function over particular axes. However, unlike `vmap`, `pz.nx.nmap` infers the axes to map over using the axis names:
- Every axis in the `.named_shape` of each input argument is vectorized over. Axes with the same name will be matched together, and none of them will be visible inside the function.
- Every axis in the `.positional_shape` of each input argument is kept. Inside the function, the JAX tracer will have this as its shape.

This means that you can apply most positional operations to specific axes of named arrays by:

- running `.untag` to move those axes to the `.positional_shape`
- calling `pz.nx.nmap(some_positional_op)(...args...)`
- running `.tag` to move the resulting axes from the  `.positional_shape` back to the `.named_shape`.

### Fixing PyTree mismatch errors

Sometimes, you may run into JAX PyTree issues when manipulating `NamedArray`s, because internally each `NamedArray` implicitly stores the named axes in a specific order.

One way to fix this is to call `canonicalize` on each `NamedArray`, which ensures that the named axes are in sorted order.

Another option is to call `current_array.order_like(target_array)`, or `pz.nx.order_like(current_tree_of_arrays, target_tree_of_arrays)`. This will ensure that the ordering of the "current" NamedArray(s) matches the ordering of the "target" NamedArray(s), and thus have the same structure according to JAX.

----------------------------

## Building Models

### Initializing models or layers from scratch

Models and layers can be initialized from scratch by calling their constructor classmethod, usually called `from_config`, and passing a name and a JAX PRNG key. For instance, to initialize a MLP model, you can call

```
mlp = simple_mlp.MLP.from_config(
    name="mlp",
    init_base_rng=jax.random.key(10),
    feature_sizes=[2, 32, 32, 2],
)
```

You can also pass `init_base_rng=None`, which will build the model without initializing the parameters. Parameters will instead be represented as `ParameterSlot` nodes, indicating a missing parameter.


### Implementing new models out of existing components

Penzai's neural network components are based on *combinators*, which allow you to build more complex model logic out of simple components. Combinators include:

- `pz.nn.Sequential`, which runs layers in sequence,
- `pz.nn.Residual`, which runs its child layers and then adds their output to its input,
- `pz.nn.BranchAndAddTogether` and `pz.nn.BranchAndMultiplyTogether`, which allow outputs of different components to be combined additively or multiplicatively,
- `pz.nn.Attention`, which routes inputs between query, key, value, masking, and output computations.

The primitive components include `pz.nn.Affine`, `pz.nn.Linear`, `pz.nn.AddBias`, `pz.nn.ApplyAttentionMask`, `pz.nn.Elementwise`, `pz.nn.Softmax`, and similar components.

When implementing a new model architecture, the preferred approach is to do so by combining Penzai's existing primitives using combinators. For instance, this might look like:

```python
def build_my_model(
    name: str,
    init_base_rng: jax.Array | None,
    # ... any other arguments ...
):
  # Initialize all model components and return them, e.g.:
  return pz.nn.Sequential([
      pz.nn.Affine.from_config(
          # Extend the name:
          name=f"{name}/Affine_0",
          # Pass along the initialization key (no need to split it)
          init_base_rng=init_base_rng,
          # Configure the layer (for example)
          input_axes={"features": 8},
          output_axes={"features": 8},
      ),
      # ... Add more layers as needed ...
  ])
```

If you are building a re-usable or semantically-meaningful model component, you may want to disambiguate it by defining a custom layer class. A common pattern is to subclass `Sequential`, and add a new method `from_config` that initializes it, e.g.

```python
@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class MyCustomLayer(pz.nn.Sequential):
  @classmethod
  def from_config(cls, name: str, init_base_rng: jax.Array | None, ...):
    ...
```

As an example, `pz.nn.Affine` and `pz.nn.LayerNorm` are both subclasses of `pz.nn.Sequential`, built out of simpler components. (This convention makes it easier to inspect and intervene on parts of models after they are built.)

### Defining custom layer logic

If you want to implement an operation that isn't already included in Penzai, you can do so by directly subclassing `pz.nn.Layer`, and defining its attributes, its `__call__` logic, and how to initialize it.

Every `pz.nn.Layer` must be a Python dataclass and a JAX PyTree, and should have the `@pz.pytree_dataclass` decorator. It should also have a type-annotated list of fields, indicating the attributes that each instance will store. By default, each field will be a JAX PyTree child, which is appropriate for attributes that contain other layers, arrays, parameters, or state variables. You can also annotate a field using `dataclasses.field(metadata={"pytree_node": False})` to indicate that it should not be treated as a PyTree child by JAX; this is appropriate for "static" metadata such as axis names or array shapes.

The `__call__` method should always have signature `__call__(self, argument, /, **side_inputs)`. Here `argument` is the main input from the previous layer, and `side_inputs` contains additional values that the layer may or may not need. Layers should ignore side inputs that they do not use.

Finally, most layers should implement a builder method that randomly initializes any parameters. This is usually called `from_config`, and should accept `name` and `init_base_rng` arguments. Parameters can then be instantiated using `pz.nn.make_parameter`, which takes care of splitting the `init_base_rng` and constructing a new parameter object.

An example `Layer` implementation would be:

```
@pz.pytree_dataclass
class SimpleLinear(pz.nn.Layer):
  # Parameters are annotated as `ParameterLike` to allow swapping them out after
  # initialization.
  kernel: pz.nn.ParameterLike[pz.nx.NamedArray]

  # Non-Pytree fields (which are not arraylike) should be annotated as such to
  # tell JAX not to try to convert them:
  features_axis: str = dataclasses.field(metadata={"pytree_node": False})

  def __call__(
    self, x: pz.nx.NamedArray, /, **unused_side_inputs
  ) -> pz.nx.NamedArray:
    pos_x = x.untag(self.features_axis)
    pos_kernel = self.kernel.value.untag("out_features", "in_features")
    pos_y = pz.nx.nmap(jnp.dot)(pos_kernel, pos_x)
    return pos_y.tag(self.features_axis)

  @classmethod
  def from_config(
      cls,
      name: str,
      init_base_rng: jax.Array | None,
      in_features: int,
      out_features: int,
      features_axis: str = "features",
  ) -> "SimpleLinear":
    """Constructs a linear layer from configuration arguments."""
    def _initializer(key):
      arr = jax.nn.initializers.xavier_normal()(
          key, (out_features, in_features)
      )
      return pz.nx.wrap(arr).tag("out_features", "in_features")

    return cls(
        kernel=pz.nn.make_parameter(
            name=f"{name}.kernel",
            init_base_rng=init_base_rng,
            initializer=_initializer,
        ),
        features_axis=features_axis,
    )
```

You can read more about Penzai's conventions for layers in ["How to Think in Penzai"](../notebooks/how_to_think_in_penzai.ipynb), or see more examples in `penzai.nn`.

----------------------------

## Loading Pretrained Models

### Loading Gemma

Penzai's Gemma implementation includes a conversion utility that converts the ["Flax" model weights from Kaggle](https://www.kaggle.com/models/google/gemma) into the correct form. You can load it using:

```python
import kagglehub
import orbax.checkpoint
from penzai.models.transformer import variants

weights_dir = kagglehub.model_download('google/gemma/Flax/7b')
ckpt_path = os.path.join(weights_dir, '7b')

checkpointer = orbax.checkpoint.PyTreeCheckpointer()
flax_params_dict = checkpointer.restore(ckpt_path)
model = variants.gemma.gemma_from_pretrained_checkpoint(flax_params_dict)
```

### Loading Llama, Mistral, or GPT-NeoX / Pythia

Penzai also includes re-implementations of the architectures used by [Llama](https://llama.meta.com/), [Mistral](https://mistral.ai/), and the [GPT-NeoX](https://www.eleuther.ai/artifacts/gpt-neox-20b) family of models, including the [Pythia](https://github.com/EleutherAI/pythia) model scaling suite. To load these models into Penzai, you can first load the weights using the HuggingFace `transformers` library, then convert them to Penzai:

```python
import transformers
from penzai.models.transformer import variants

# To load a Llama model:
hf_model = transformers.LlamaForCausalLM.from_pretrained(...)
pz_model = variants.llama.llama_from_huggingface_model(hf_model)

# To load a Mistral model:
hf_model = transformers.MistralForCausalLM.from_pretrained(...)
pz_model = variants.mistral.mistral_from_huggingface_model(hf_model)

# To load a GPT-NeoX / Pythia model:
hf_model = transformers.GPTNeoXForCausalLM.from_pretrained(...)
pz_model = variants.gpt_neox.gpt_neox_from_huggingface_model(hf_model)
```

### Freezing pretrained model weights

When working with pretrained models, you may wish to freeze their parameters so that they are not updated by optimizers, and so that the model can be passed through JAX transformations more easily. You can do this by passing the model to `pz.nn.freeze_params`.

(Note that freezing parameters also breaks any shared parameter relationships in the model, replacing them with immutable copies of the current parameter values. If you would like to take derivatives with respect to changes in parameters, or compare parameters at different checkpoints, it may be easier to keep parameters in their original non-frozen form.)

----------------------------

## Using Models and Variables

### Running the model forward pass

To run the forward pass of a model, you can simply call it like a function. By convention, Penzai models and layers can be called with a single argument, representing the input to the first layer in the model, along with an arbitrary number of side inputs as keyword arguments, which get propagated to any child layers that need them.

For instance, to run a transformer, you might use

```
results = model(
  tokens, # Positional input, passed to the first layer.
  token_positions=token_positions, # Side input used by ApplyRoPE layers.
  attn_mask=attn_mask, # Side input used by ApplyAttentionMask layers.
)
```

The set of side inputs that you need to pass depends on the layers in the model and the arguments they expect.

### Extracting and freezing model parameters

To extract model parameters, you can use `pz.unbind_params`, which extracts and deduplicates the parameters in a model PyTree. `pz.unbind_params` returns two outputs, an "unbound" model (with each `Parameter` replaced with a `ParameterSlot` placeholder), and a list of (mutable) `Parameter` objects.

```
unbound_model, params = pz.unbind_params(model)
```

After extracting parameters, you may also want to freeze them, which produces an immutable snapshot of the current value of each parameter (as an instance of `ParameterValue`). You can do this by calling `.freeze()` on each parameter, by using `pz.freeze_params` to freeze all parameters in a collection, or by using `pz.unbind_params(model, freeze=True)`. Frozen parameters are ordinary JAX PyTrees, making them safe to use across JAX transformation boundaries.

Both mutable `Parameter` instances and frozen `ParameterValue` instances can be substituted back into a model with `ParameterSlot`s `pz.bind_variables`. A common pattern is to unbind and freeze `Parameter`s before a JAX transformation, and then re-bind their frozen values inside the function being transformed.

### Taking gradients with respect to model parameters

As a special case of the above pattern, you can take gradients with respect to parameters using something like:

```python

def my_loss(params, unbound_model):
  rebound_model = pz.bind_variables(unbound_model, params)
  result = rebound_model(...)  # call it with some arguments
  loss = # (... compute the loss ...)
  return loss

unbound_model, frozen_params = pz.unbind_params(model, freeze=True)
grads = jax.grad(my_loss, argnums=0)(frozen_params, unbound_model)
```

You can similarly compute forward-mode derivatives using something like

```python

def my_func(params, unbound_model):
  rebound_model = pz.bind_variables(unbound_model, params)
  return rebound_model(...)  # call it with some arguments

unbound_model, frozen_params = pz.unbind_params(model, freeze=True)

# Build your input perturbations somehow
perturbations = jax.tree_util.tree_map(some_func, frozen_params)
tangents = jax.jvp(my_loss)((frozen_params, unbound_model), (perturbations, None))
```

### Extracting and manipulating state variables

Some Penzai layers keep track of mutable `pz.StateVariable` instances and update them when called. For instance, some layers that capture intermediate values store them in  `StateVariable`s, and Transformer key-value caches are also stored in `StateVariable`s.

Outside of JAX transformations, you can usually just mutate state variables normally. However, running stateful operations inside JAX transformations can require some care. Additionally, it's sometimes useful to take a snapshot of the state of all variables in a model.

When working with a model that uses state variables, you can unbind the state variables using `pz.unbind_state_vars`, and optionally freeze them using `pz.freeze_state_vars` (or unbind with `freeze=True`), similar to the corresponding methods for `Parameter`s. This allows you to extract an immutable view of the model state that is safe to manipulate in JAX, e.g. via

```
stateless_model, frozen_states = pz.unbind_state_vars(model, freeze=True)
```

Every subclass of `Layer` implements the method `stateless_call`, which takes frozen state variables as input and returns updated state variables as output:

```
y, new_state = stateless_model.stateless_call(frozen_states, x, **side_inputs)
```

Internally, `stateless_call` is implemented by using `frozen_var.unfreeze_as_copy()` to make temporary mutable copies of each state variable, and then binding them back to the model using `pz.bind_variables`. `unfreeze_as_copy` can also be used directly to implement more complex transformations of models.

(Note: If your model does not use state variables, then using `stateless_call` is usually unnecessary; you can directly use JAX function transformations without worrying about state.)

### Running models with stochastic components

When a model has stochastic components, you should usually use a `pz.RandomStream`. Each `RandomStream` keeps track of an internal offset (in a `pz.StateVariable`) and uses it to generate unique JAX PRNGKeys each time `.next_key()` is called.

Random streams are usually passed as side inputs, e.g.

```
results = stochastic_model(
  input_array,
  random_stream=pz.RandomStream.from_base_key(jax.random.key(42))
)
```

However, you can also store `RandomStream`s as layer attributes, which will ensure different random numbers are generated each time the model is called.


----------------------------
## Modifying and Patching Models

### Modifying model layers by position or by type

In Penzai, model modifications are generally performed by using `pz.select` to make a modified copy of the original model (but sharing the same parameters). This involves "selecting" the part of the model you want to modify, then applying a modification, similar to the `.at[...].set(...)` syntax for modifying JAX arrays.

You can select model sublayers by passing a path to them, by using their type, or some combination, e.g.:

```
# Modify the layer at .body.sublayers[2]:
pz.select(model).at(lambda model: model.body.sublayers[2]).apply(some_func)

# Modify all instances of pz.nn.Linear:
pz.select(model).at_instances_of(pz.nn.Linear).apply(some_func)

# Modify linear layers inside query heads:
(
    pz.select(model)
    .at_instances_of(pz.nn.Attention)
    .at(lambda attn: attn.input_to_query)
    .at_instances_of(pz.nn.Linear)
    .apply(some_func)
)
```

Here `some_func` can be any function that accepts a selected subtree of the model and returns a new subtree to replace it with.

In addition to `.apply`, you can also use `.insert_before` or `.insert_after` to splice new layers into a model. See the [Selectors tutorial](../notebooks/selectors.ipynb) for more details on working with selectors.

### Capturing or modifying intermediate activations

To manipulate intermediate activations in a model, you can insert a new layer with the appropriate effects. For instance, to save intermediate values to a list, you could use something like

```python
# Define a layer that implements the intervention.
@pz.pytree_dataclass
class AppendIntermediate(pz.nn.Layer):
  saved: pz.StateVariable[list[Any]]

  def __call__(self, x, **_side_inputs):
    self.saved.value = self.saved.value + [x]
    return x

# Make a copy of the model that includes the intervention.
intermediates_cell = pz.StateVariable([])
modified_model = (
    pz.select(model)
    .at_instances_of(SomeLayer)
    .insert_after(AppendIntermediate(intermediates_cell))
)

# Call the model, then retrieve the saved intermediate.
_ = modified_model(inputs)
intermediates = intermediates_cell.value
```

To modify the intermediate, you can instead return a modified value in the `__call__` for the modified layer.

Note that interventions in Penzai models always involve changing the structure of copies of the model. You can visualize the intervention by pretty-printing the modified model.

(`penzai.toolshed.save_intermediates` also includes a built-in layer you can use to save intermediates at particular places in a model, if you don't want to implement it from scratch.)

### Isolating small components of models

Penzai provides a utility `call_and_extract_submodel` to capture the activations flowing into and out of a particular layer in a model, defined in `penzai.toolshed.isolate_submodel`. This can help with investigation and debugging of parts of models.

To use it, first select the particular layer you want to extract using `pz.select`. Then, call `call_and_extract_submodel`, passing the model and the input. This will produce an `IsolatedSubmodel` data structure which captures the model, its inputs, its saved outputs, and the states of any `StateVariable`s used by it. These can then be used to re-run the submodel in a controlled setting and debug or intervene on its behavior.

### Removing parts of models

You can remove parts of models entirely by first selecting them using `pz.select` and then calling `.remove_from_parent()`, which will produce a copy of the model with the selected parts removed. (This only works if the selected parts were elements of a list, e.g. if they were sublayers in a `pz.nn.Sequential` instance.)

This can be useful for simplifying or ablating model components, or for removing operations like dropout that are no longer needed after training.


### Linearizing layers

It is sometimes useful to replace layers with a linear approximation around a particular input. Penzai includes a utility for this called `LinearizeAndAdjust` in `penzai.toolshed.model_rewiring`. `LinearizeAndAdjust` can be used like

```
patched_model = (
  pz.select(model).at(some_path)
  .apply(
      lambda original_layer: LinearizeAndAdjust(
          linearize_around=<some layer that computes the linearization point>,
          evaluate_at=pz.nn.Identity(),
          target=original_layer,
      )
  )
)
```

where `target` is the layer to linearize, `linearize_around` computes the input that the layer should be linearized at (e.g. by modifying its input activation or returning a constant), and `evaluate_at` computes the input that the linear approximation should be evaluated at (usually the same as the original input, but can also be different).


### Customizing attention masks in `TransformerLM`

By default, most `TransformerLM` architecture variants are specialized to causal attention masks, using the `pz.nn.ApplyCausalAttentionMask` layer (or sometimes `pz.nn.ApplyCausalSlidingWindowAttentionMask`). These layers use the token positions input to build a causal attention mask and apply it to the attention logits.

If you would like to customize the attention mask computation, you can swap out these layers for `pz.nn.ApplyExplicitAttentionMask` layers, using something like

```
explicit_attn_model = (
  pz.select(model)
  .at_instances_of(
    pz.nn.ApplyCausalAttentionMask
    | pz.nn.ApplyCausalSlidingWindowAttentionMask
  )
  .apply(lambda old: pz.nn.ApplyExplicitAttentionMask(
    mask_input_name="attn_mask",
    masked_out_value=old.masked_out_value,
  ))
)
```

This will create a copy of the model that expects a side input called `attn_mask`, and uses it to mask the inputs. You can call it using something like

```
# tokens should have named shape {..., "seq": n_seq}
# token_positions should have named shape {..., "seq": n_seq}
# attn_mask should be a boolean array with named shape
#   {..., "seq": n_seq, "kv_seq": n_seq}
token_logits = explicit_attn_model(
  tokens, token_positions=token_positions, attn_mask=attn_mask
)
```

For more control, you can also define your own layer and insert it in place of the attention masking logic.


### Reducing backward-pass memory usage using gradient checkpointing

By default, when computing gradients through a model, JAX will save all of the intermediate values produced by the computation. This can sometimes lead to out-of-memory errors.

To prevent this, you can enable gradient checkpointing, which tells JAX to recompute some intermediate values during the backward pass. In Penzai models, you can enable gradient checkpointing using the `Checkpointed` combinator from `penzai.toolshed.gradient_checkpointing`, which adapts the `jax.checkpoint` function transformation to support Penzai's variables. For instance, to prevent saving intermediate values inside each attention layer, you can use something like

```
checkpointed_model = (
    pz.select(model)
    .at_instances_of(pz.nn.Attention)  # for example
    .apply(gradient_checkpointing.Checkpointed)
)
```

`Checkpointed` can be wrapped around any `pz.nn.Layer` in the model to add gradient checkpointing. It is also itself a Penzai layer, and can be modified and pretty-printed just like any other layer.


----------------------------
## Training and Fine-Tuning Models

### Training model parameters

Penzai provides a basic `StatefulTrainer` object in `penzai.toolshed.basic_training`, which is designed to allow you to quickly set up a training loop for a loss function. This trainer object will update the values of `Parameter` variables in the model automatically, similar to e.g. PyTorch optimizers.

Penzai models are compatible with any JAX training loop, however. If you prefer a purely functional training loop to a stateful one, you can use `pz.unbind_params(model, freeze=True)` to obtain a deduplicated PyTree of model parameters, then update these parameters using your preferred JAX PyTree-compatible optimizer or training loop implementation. (This is how `StatefulTrainer` is implemented internally, except that it also updates the value of the original stateful `Parameter` variables after each training iteration.)

### Low-rank adaptation

To finetune a model using low-rank adaptation, you can use the function `loraify_linears_in_selection` defined in `penzai.toolshed.lora`. First, you will likely want to freeze the existing parameters, using `pz.freeze_params`. Then, you can use `pz.select` to identify the parts of the model that should be adapted, and then use `loraify_linears_in_selection` to insert new `LowRankAdapter` layers with new parameters.

`loraify_linears_in_selection` returns a modified copy of the model, where each of the `pz.nn.Linear` layers in the original model has been replaced with a `lora.LowRankAdapter` with the same signature. As with any other Penzai model transformation, you can visualize the new model structure by pretty-printing the new model copy.

### Checkpointing model parameters

Penzai does not include any Penzai-specific checkpointing utilities. However, Penzai is compatible with any PyTree-based JAX checkpointing system, as long as you first extract the parameters using `pz.unbind_parameters(model, freeze=True)`. One checkpointing library that is currently well supported in JAX is [`orbax.checkpoint`](https://orbax.readthedocs.io/en/latest/orbax_checkpoint_101.html).
