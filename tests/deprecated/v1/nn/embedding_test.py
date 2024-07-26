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

"""Tests for embedding layer."""

from absl.testing import absltest
from penzai.deprecated.v1 import pz
from penzai.deprecated.v1.toolshed import check_layers_by_tracing


class EmbeddingTest(absltest.TestCase):

  def test_embedding_lookup(self):
    layer = pz.nn.EmbeddingLookup(
        pz.nn.EmbeddingTable.from_config(
            vocab_size=17, embedding_axes={"foo": 5, "bar": 7}
        )
    )
    check_layers_by_tracing.check_layer(layer)

  def test_embedding_decode(self):
    layer = pz.nn.EmbeddingDecode(
        pz.nn.EmbeddingTable.from_config(
            vocab_size=17, embedding_axes={"foo": 5, "bar": 7}
        )
    )
    check_layers_by_tracing.check_layer(layer)


if __name__ == "__main__":
  absltest.main()
