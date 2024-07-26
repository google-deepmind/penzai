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

"""Tests for basic ops."""

from absl.testing import absltest
import jax
from penzai import pz


class BasicOpsTest(absltest.TestCase):

  def test_elementwise(self):
    layer = pz.nn.Elementwise(jax.nn.relu)
    result = layer(pz.nx.ones({"foo": 3}))
    pz.chk.check_structure(result, pz.chk.ArraySpec(named_shape={"foo": 3}))

  def test_softmax(self):
    layer = pz.nn.Softmax(("foo", "bar"))
    result = layer(pz.nx.ones({"foo": 3, "bar": 4, "baz": 5}))
    pz.chk.check_structure(
        result, pz.chk.ArraySpec(named_shape={"foo": 3, "bar": 4, "baz": 5})
    )


if __name__ == "__main__":
  absltest.main()
