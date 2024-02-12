import copy
from typing import Tuple, Dict

import numpy as np

from TinyGraph.DSL import MatrixVar, DepTensor
from TinyGraph.Kernel import _conv2d_kernel, _maxpool2d_kernel, _matrix_vec_mul_kernel, _add_kernel, _relu_kernel


class DepModule:
    def __init__(self):
        self._module_dict: Dict[DepModule, None] = {}

    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, DepModule):
            self._module_dict.setdefault(value)
        super().__setattr__(name, value)

    def _add_module_dict(self, module):
        self._module_dict.setdefault(module)

    def mapping(self):
        for module in self._module_dict:
            module.mapping()


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

        self.weight_matrix_shape = (in_channels * kernel_size[0] * kernel_size[1], out_channels)

        self.weight_matrix = MatrixVar(self.weight_matrix_shape, )
        self.weight_matrix.dummy_mapping()

    def forward(self, input_tensor: DepTensor, *args, **kwargs):
        output_tensor = _conv2d_kernel(input_tensor, self.weight_matrix,
                                       self.in_channels, self.out_channels,
                                       self.kernel_size, self.stride, self.padding)
        return output_tensor

    def mapping(self):
        # do something

        super().mapping()


class DepMaxpool2d(DepModule):
    def __init__(self, kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input_tensor: DepTensor):
        local_input_tensor = copy.deepcopy(input_tensor)

        output_tensor = _maxpool2d_kernel(local_input_tensor,
                                          self.kernel_size,
                                          self.stride,
                                          self.padding)

        return output_tensor


class DepLinear(DepModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.weight_matrix_shape = (in_features, out_features)
        self.weight_matrix = MatrixVar(self.weight_matrix_shape, )
        self.weight_matrix.dummy_mapping()

    def forward(self, input_tensor: DepTensor, *args, **kwargs):
        local_input_tensor = copy.deepcopy(input_tensor)

        output_tensor = _matrix_vec_mul_kernel(local_input_tensor, self.weight_matrix)

        return output_tensor

    def mapping(self):
        # do something here
        super().mapping()


class DepElementAdd(DepModule):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, *args, **kwargs):
        in_a = copy.deepcopy(a)
        in_b = copy.deepcopy(b)

        output_tensor = _add_kernel(in_a, in_b)
        return output_tensor


class DepReLU(DepModule):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, *args, **kwargs):
        local_input_tensor = copy.deepcopy(input_tensor)
        output_tensor = _relu_kernel(local_input_tensor)
        return output_tensor


class DepSequential(DepModule):
    def __init__(self, *args):
        super().__init__()
        self.layers = args
        for layer in self.layers:
            if isinstance(layer, DepModule):
                self._add_module_dict(layer)

    def forward(self, input_tensor: DepTensor):
        for layer in self.layers:
            input_tensor = layer(input_tensor)
        return input_tensor
