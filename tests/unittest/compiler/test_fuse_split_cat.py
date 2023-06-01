#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import unittest

import torch
from aitemplate.compiler import compile_model, ops

from aitemplate.compiler.base import Tensor
from aitemplate.compiler.public import IntImm

from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor, graph_has_op

class FuseSplitCatTestCase(unittest.TestCase):
    def test_fuse_split_cat_rearrange(self):
        dtype = "float16"
        M = IntImm(512)
        N = IntImm(512)

        input_1 = Tensor(
            shape=[M, N],
            name="input_1",
            is_input=True,
        )
        split_2 = ops.split()(
            input_1, int(M.value()/2), 0
        )
        concatenate_3 = ops.concatenate()(split_2[::-1], 0)

        # Set outputs
        concatenate_3._attrs["name"] = "output_0"
        concatenate_3._attrs["is_output"] = True
        # Compile
        model = compile_model(
            concatenate_3,
            detect_target(),
            "./tmp",
            self._testMethodName
        )
        # Check that split was removed
        self.assertFalse(graph_has_op(model.debug_sorted_graph, "split"))
        # Run
        input_1 = get_random_torch_tensor((M.value(), N.value()), dtype=dtype)
        # Compare
        split_pt = torch.split(input_1, int(M.value()/2), 0)
        y_pt = torch.cat(
            [split_pt[1], split_pt[0]],
            0,
        )
        y_ait = torch.empty_like(y_pt)
        model.run_with_tensors(
            {"input_1": input_1},
            [y_ait],
        )
        torch.testing.assert_close(y_ait, y_pt, atol=0, rtol=0)

    def test_fuse_split_cat_dim1(self):
        dtype = "float16"
        M = IntImm(512)
        N = IntImm(512)

        input_1 = Tensor(
            shape=[M, N],
            name="input_1",
            is_input=True,
        )
        split_2 = ops.split()(
            input_1, int(N.value()/2), 1
        )
        concatenate_3 = ops.concatenate()(split_2[::-1], 1)

        # Set outputs
        concatenate_3._attrs["name"] = "output_0"
        concatenate_3._attrs["is_output"] = True
        # Compile
        model = compile_model(
            concatenate_3,
            detect_target(),
            "./tmp",
            self._testMethodName
        )
        # Check that split was removed
        self.assertFalse(graph_has_op(model.debug_sorted_graph, "split"))
        # Run
        input_1 = get_random_torch_tensor((M.value(), N.value()), dtype=dtype)
        # Compare
        split_pt = torch.split(input_1, int(N.value()/2), 1)
        y_pt = torch.cat(
            split_pt[::-1],
            1,
        )
        y_ait = torch.empty_like(y_pt)
        model.run_with_tensors(
            {"input_1": input_1},
            [y_ait],
        )
        torch.testing.assert_close(y_ait, y_pt, atol=0, rtol=0)


    def test_fuse_split_cat_different_dims(self):
        """The case of splitting and then catting on different dims is not
        expected to be optimized currently."""
        dtype = "float16"
        M = IntImm(512)
        N = IntImm(512)

        input_1 = Tensor(
            shape=[M, N],
            name="input_1",
            is_input=True,
        )
        split_2 = ops.split()(
            input_1, int(M.value()/2), 0
        )
        concatenate_3 = ops.concatenate()(split_2[::-1], 1)

        # Set outputs
        concatenate_3._attrs["name"] = "output_0"
        concatenate_3._attrs["is_output"] = True
        # Compile
        model = compile_model(
            concatenate_3,
            detect_target(),
            "./tmp",
            self._testMethodName
        )
        # Run
        input_1 = get_random_torch_tensor((M.value(), N.value()), dtype=dtype)
        # Compare
        split_pt = torch.split(input_1, int(M.value()/2), 0)
        y_pt = torch.cat(
            split_pt[::-1],
            1,
        )
        y_ait = torch.empty_like(y_pt)
        model.run_with_tensors(
            {"input_1": input_1},
            [y_ait],
        )
        torch.testing.assert_close(y_ait, y_pt, atol=0, rtol=0)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
