from typing import Tuple

import numpy as np

from TinyGraph.DSL import MatrixVar, DepTensor


class DepModule:
    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        pass


class DepConv2d(DepModule):
    def __init__(self, core_id: int, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 padding: int = 0, bias: bool = True):
        super().__init__()
        self.core_id = core_id

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.weight_matrix_shape = (in_channels * kernel_size[0] * kernel_size[1], out_channels)

        self.xbar_matrix = MatrixVar(self.weight_matrix_shape, )

    def forward(self, input_tensor: DepTensor, *args, **kwargs):
        pass

class DepMaxpool2d(DepModule):
    def __init__(self, kernel_size: Tuple[int, int], stride: Tuple[int,int],padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input_tensor:DepTensor):
        pass


class DepLinear(DepModule):
    def __init__(self,in_features: int, out_features: int,bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias



    def forward(self, input_tensor: DepTensor, *args, **kwargs):
        pass

class DepElementAdd(DepModule):
    def __init__(self):
        super().__init__()
        pass



