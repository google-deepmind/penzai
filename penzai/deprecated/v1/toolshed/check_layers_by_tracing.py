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

"""Utility to check that a layer has been configured correctly by tracing it.

This module contains a utility for checking that a layer or model has been set
up correctly and will execute properly when run, under the assumption that
the top-level model or layer has implemented the ``input_structure`` method. It
does this by running the model under `jax.eval_shape` and passing in a dummy
input that matches the expected structure, with arbitrary sizes for unspecified
dimension variables.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from penzai.deprecated.v1 import pz


def _generate_primes() -> Iterable[int]:
  yield from (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
  offsets_under_30 = [1, 7, 11, 13, 17, 19, 23, 29]
  primes_over_5 = [7, 11, 13, 17, 19, 23, 29]
  for i in itertools.count(start=1):
    for j in offsets_under_30:
      n = i * 30 + j
      if not any(n % p == 0 for p in primes_over_5):
        yield n
        primes_over_5.append(n)


def _generate_with_sensible_dtype(structure: pz.chk.ArraySpec):
  """Constructs a zeros array with a reasonable dtype and a given structure."""
  if structure.dtype == jax.dtypes.prng_key:
    new_dtype = jax.eval_shape(jax.random.key, 0).dtype
  else:
    concrete_default_dtypes = [np.float32, np.int32, np.bool_, np.complex64]
    for dtype in concrete_default_dtypes:
      if jnp.issubdtype(dtype, structure.dtype):
        new_dtype = dtype
        break
    else:
      new_dtype = jax.dtypes.result_type(structure.dtype)
  adj_structure = pz.chk.ArraySpec(
      shape=structure.shape, named_shape=structure.named_shape, dtype=new_dtype
  ).into_pytree()
  return jax.tree_util.tree_map(jnp.zeros_like, adj_structure)


def check_layer(
    layer: pz.Layer,
    argument: Any | None = None,
    initialize: bool = True,
) -> Any:
  """Checks that a layer has been configured correctly by tracing it.

  This function runs the layer under `jax.eval_shape` and passes in a dummy
  input that matches the expected structure, with arbitrary sizes for
  unspecified dimension variables. It then checks that the output shape matches
  the expected output shape.

  Note that not all layers may be able to fully encode their preconditions
  in their ``input_structure``; in this case it may be necessary to provide a
  dummy argument that matches the expected input structure.

  Args:
    layer: The layer to check.
    argument: An optional argument to pass to the layer. If not provided, a
      dummy argument will be created based on the layer's input structure. Can
      contain jax.ShapeDtypeStruct leaves.
    initialize: Whether to initialize any uninitialized parameters in the model.

  Returns:
    The traced output structure from the call, as a PyTree of
    jax.ShapeDtypeStruct leaves.

  Raises:
    ValueError: If the layer's input structure contains ANY but no dummy
      argument was provided.

    Note that the layer may also raise other exceptions if misconfigured.
  """

  def go(layer, argument):
    if initialize:
      layer = pz.nn.initialize_parameters(layer, jax.random.key(0))

    if argument is None:
      input_structure = layer.input_structure()
      if (
          not pz.select(input_structure)
          .at_instances_of(pz.chk.Wildcard)
          .is_empty()
      ):
        raise ValueError(
            "Cannot synthesize an input for a layer whose input structure"
            " contains ANY. Please provide an explicit dummy argument."
        )
      dimension_vars = pz.chk.get_dimension_variables(input_structure)
      new_subst = pz.chk.DimensionVariableSubstitution({}, {}, {})
      primes = _generate_primes()
      for _ in range(3):
        # Skip the small primes
        next(primes)
      for var in dimension_vars.size_variables:
        new_subst.size_variables[var] = next(primes)
      for var in dimension_vars.sequence_variables:
        new_subst.sequence_variables[var] = (next(primes), next(primes))
      for var in dimension_vars.mapping_variables:
        n1 = next(primes)
        n2 = next(primes)
        new_subst.mapping_variables[var] = {
            f"__trace_check_{n1}__": n1,
            f"__trace_check_{n2}__": n2,
        }
      argument_structure = pz.chk.full_substitute_dimension_variables(
          input_structure, new_subst
      )
      argument = (
          pz.select(argument_structure)
          .at_instances_of(pz.chk.ArraySpec)
          .apply(_generate_with_sensible_dtype)
      )

    matches = pz.chk.check_structure(argument, layer.input_structure())
    result = layer(argument)
    pz.chk.check_structure(result, layer.output_structure(), known_vars=matches)
    return result

  return jax.eval_shape(go, layer, argument)
