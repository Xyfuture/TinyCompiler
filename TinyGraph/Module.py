from typing import Tuple, List

import numpy as np

from TinyGraph.Graph import MicroNode, MicroGraph, MicroOp
from TinyGraph.DSL import DepTensor


class DepModule:
    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        pass


class DepConv2d(DepModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def forward(self, input_tensor: np.ndarray):
        pass


class TransferOp(MicroOp):
    def __init__(self,src_core_id:int, dst_core_id:int, data_size:int):
        super().__init__()
        self.src_core_id = src_core_id
        self.dst_core_id = dst_core_id
        self.data_size = data_size

        self.output_dep_tensor = DepTensor((self.data_size,))

        self.output_dep_tensor.tensor_op.fill(self)
        self.output_dep_tensor.tensor_position.fill(self.dst_core_id)

    def code_gen(self):
        pass


class MatVecMulOp(MicroOp):
    def __init__(self, core_id: int, input_size: int, output_size: int,src_vec_op_list:List[MicroOp]):
        super().__init__()

        self.core_id = core_id
        self.src_vec_op_list = src_vec_op_list

        self.input_size = input_size
        self.output_size = output_size

        self.output_dep_tensor = DepTensor((self.output_size,))
        self.output_dep_tensor.tensor_op.fill(self)
        self.output_dep_tensor.tensor_position.fill(self.core_id)

    def code_gen(self):
        pass
