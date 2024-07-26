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

"""Tests for standardization layers."""

from absl.testing import absltest
import jax
from penzai import pz


class StandardizationTest(absltest.TestCase):

  def test_standardize_one(self):
    layer = pz.nn.Standardize(across="foo")
    result = layer(pz.nx.ones({"batch": 3, "foo": 4}))
    pz.chk.check_structure(
        result, pz.chk.ArraySpec(named_shape={"batch": 3, "foo": 4})
    )

  def test_standardize_multi(self):
    layer = pz.nn.Standardize(across=("foo", "bar"))
    result = layer(pz.nx.ones({"batch": 3, "foo": 4, "bar": 2}))
    pz.chk.check_structure(
        result, pz.chk.ArraySpec(named_shape={"batch": 3, "foo": 4, "bar": 2})
    )

  def test_layernorm(self):
    layer = pz.nn.LayerNorm.from_config(
        name="test",
        init_base_rng=jax.random.key(2),
        across_axes={"foo": 4, "bar": 2},
    )
    result = layer(pz.nx.ones({"batch": 3, "foo": 4, "bar": 2}))
    pz.chk.check_structure(
        result, pz.chk.ArraySpec(named_shape={"batch": 3, "foo": 4, "bar": 2})
    )


if __name__ == "__main__":
  absltest.main()
