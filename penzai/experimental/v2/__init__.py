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

"""Redirector module for Penzai V2.

Penzai's V2 neural network API was originally defined in
`penzai.experimental.v2`. However, with the release of Penzai V2, the API was
moved to the top-level `penzai` namespace. This module is a redirector to the
new location.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "Accessing Penzai's V2 API from `penzai.experimental.v2` is"
    " deprecated. As of the 0.2.0 release, the V2 API is now available in"
    " the top-level `penzai` namespace. Please use `penzai` directly"
    " instead."
)


def _make_redirector(stub_module_name: str, target_module_name: str | None):
  """Builds a redirector module for the given target module."""

  # pylint: disable=g-import-not-at-top
  import importlib
  import importlib.machinery
  import importlib.util
  import sys
  # pylint: enable=g-import-not-at-top

  spec = importlib.machinery.ModuleSpec(name=stub_module_name, loader=None)
  mod = importlib.util.module_from_spec(spec)
  mod.__doc__ = f"""Redirector module for {stub_module_name}."""

  if target_module_name is not None:

    def redirecting_getattr(name: str):
      if name.startswith("__"):
        raise AttributeError(
            f"module {repr(stub_module_name)} has no attribute '{name}'"
        )

      reference_module = importlib.import_module(target_module_name)
      return getattr(reference_module, name)

    def redirecting_dir():
      reference_module = importlib.import_module(target_module_name)
      return dir(reference_module)

    mod.__getattr__ = redirecting_getattr
    mod.__dir__ = redirecting_dir

  stub_parent_name, mod_name = stub_module_name.rsplit(".", 1)
  setattr(sys.modules[stub_parent_name], mod_name, mod)

  sys.modules[stub_module_name] = mod


_make_redirector("penzai.experimental.v2.core", None)
_make_redirector(
    "penzai.experimental.v2.core.auto_order_types",
    "penzai.core.auto_order_types",
)
_make_redirector(
    "penzai.experimental.v2.core.random_stream",
    "penzai.core.random_stream",
)
_make_redirector(
    "penzai.experimental.v2.core.variables",
    "penzai.core.variables",
)

_make_redirector("penzai.experimental.v2.models", None)
_make_redirector(
    "penzai.experimental.v2.models.simple_mlp",
    "penzai.models.simple_mlp",
)

_make_redirector("penzai.experimental.v2.models.transformer", None)
_make_redirector(
    "penzai.experimental.v2.models.transformer.model_parts",
    "penzai.models.transformer.model_parts",
)
_make_redirector(
    "penzai.experimental.v2.models.transformer.sampling_mode",
    "penzai.models.transformer.sampling_mode",
)
_make_redirector(
    "penzai.experimental.v2.models.transformer.simple_decoding_loop",
    "penzai.models.transformer.simple_decoding_loop",
)

_make_redirector("penzai.experimental.v2.models.transformer.variants", None)
_make_redirector(
    "penzai.experimental.v2.models.transformer.variants.",
    "penzai.models.transformer.variants.",
)
_make_redirector(
    "penzai.experimental.v2.models.transformer.variants.gemma",
    "penzai.models.transformer.variants.gemma",
)
_make_redirector(
    "penzai.experimental.v2.models.transformer.variants.gpt_neox",
    "penzai.models.transformer.variants.gpt_neox",
)
_make_redirector(
    "penzai.experimental.v2.models.transformer.variants.llamalike_common",
    "penzai.models.transformer.variants.llamalike_common",
)
_make_redirector(
    "penzai.experimental.v2.models.transformer.variants.llama",
    "penzai.models.transformer.variants.llama",
)
_make_redirector(
    "penzai.experimental.v2.models.transformer.variants.mistral",
    "penzai.models.transformer.variants.mistral",
)

_make_redirector("penzai.experimental.v2.nn", None)
_make_redirector(
    "penzai.experimental.v2.nn.attention",
    "penzai.nn.attention",
)
_make_redirector(
    "penzai.experimental.v2.nn.basic_ops",
    "penzai.nn.basic_ops",
)
_make_redirector(
    "penzai.experimental.v2.nn.combinators",
    "penzai.nn.combinators",
)
_make_redirector(
    "penzai.experimental.v2.nn.dropout",
    "penzai.nn.dropout",
)
_make_redirector(
    "penzai.experimental.v2.nn.embeddings",
    "penzai.nn.embeddings",
)
_make_redirector(
    "penzai.experimental.v2.nn.grouping",
    "penzai.nn.grouping",
)
_make_redirector(
    "penzai.experimental.v2.nn.layer",
    "penzai.nn.layer",
)
_make_redirector(
    "penzai.experimental.v2.nn.layer_stack",
    "penzai.nn.layer_stack",
)
_make_redirector(
    "penzai.experimental.v2.nn.linear_and_affine",
    "penzai.nn.linear_and_affine",
)
_make_redirector(
    "penzai.experimental.v2.nn.parameters",
    "penzai.nn.parameters",
)
_make_redirector(
    "penzai.experimental.v2.nn.standardization",
    "penzai.nn.standardization",
)

_make_redirector("penzai.experimental.v2.pz", "penzai.pz")
_make_redirector("penzai.experimental.v2.pz.nn", "penzai.pz.nn")
_make_redirector("penzai.experimental.v2.pz.ts", "penzai.pz.ts")

_make_redirector("penzai.experimental.v2.toolshed", None)
_make_redirector(
    "penzai.experimental.v2.toolshed.basic_training",
    "penzai.toolshed.basic_training",
)
_make_redirector(
    "penzai.experimental.v2.toolshed.gradient_checkpointing",
    "penzai.toolshed.gradient_checkpointing",
)
_make_redirector(
    "penzai.experimental.v2.toolshed.isolate_submodel",
    "penzai.toolshed.isolate_submodel",
)
_make_redirector(
    "penzai.experimental.v2.toolshed.jit_wrapper",
    "penzai.toolshed.jit_wrapper",
)
_make_redirector(
    "penzai.experimental.v2.toolshed.lora",
    "penzai.toolshed.lora",
)
_make_redirector(
    "penzai.experimental.v2.toolshed.model_rewiring",
    "penzai.toolshed.model_rewiring",
)
_make_redirector(
    "penzai.experimental.v2.toolshed.save_intermediates",
    "penzai.toolshed.save_intermediates",
)
_make_redirector(
    "penzai.experimental.v2.toolshed.sharding_util",
    "penzai.toolshed.sharding_util",
)
_make_redirector(
    "penzai.experimental.v2.toolshed.unflaxify",
    "penzai.toolshed.unflaxify",
)
