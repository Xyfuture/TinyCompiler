import copy
from collections import OrderedDict
from typing import Tuple, Dict

import numpy as np

from TinyGraph.DSL import MatrixVar, DepTensor
from TinyGraph.Kernel import _conv2d_kernel, _maxpool2d_kernel, _matrix_vec_mul_kernel, _add_kernel, _relu_kernel


class DepModule:
    id_counter = {}
    base_name = "DepModule"

    def __init__(self):
        self._module_dict: OrderedDict[str, DepModule] = OrderedDict()

        self.module_id = self.id_counter.get(self.__class__, 1)
        self.id_counter[self.__class__] = self.module_id + 1

    @property
    def module_name(self):
        return f"{self.base_name}_{self.module_id}"

    def __str__(self):
        inner_str = ""
        for k, v in self._module_dict.items():
            inner_str += f"self.{k} : {v}\n"
        if str:
            s = (f'{self.module_name} : [\n'
                 f'{inner_str}]')
        else:
            s = f'{self.module_name}'
        return s

    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, DepModule):
            self._module_dict[name] = value
        super().__setattr__(name, value)

    def _add_module_dict(self, name, module):
        self._module_dict[name] = module

    def mapping(self):
        for module in self._module_dict.values():
            module.mapping()


def report_mapping_status(network: DepModule):
    s = ""
    for module in network._module_dict.values():
        if isinstance(module, DepConv2d):
            s += (f'Module Name: {module.module_name}\n'
                  f'{module.report_mapping()}\n\n')
        elif isinstance(module, DepLinear):
            s += (f'Module Name: {module.module_name}:\n'
                  f'{module.report_mapping()}\n\n')
        else:
            s += report_mapping_status(module)
    return s


def string_with_tab(s: str) -> str:
    return '\n'.join('\t' + line for line in s.splitlines())


class DepConv2d(DepModule):
    base_name = "Conv2d"

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

        self.weight_matrix = MatrixVar(self.weight_matrix_shape, self)
        # self.weight_matrix.dummy_mapping()

    def forward(self, input_tensor: DepTensor, *args, **kwargs):
        output_tensor = _conv2d_kernel(input_tensor, self.weight_matrix,
                                       self.in_channels, self.out_channels,
                                       self.kernel_size, self.stride, self.padding)
        return output_tensor

    def mapping(self):
        # do something
        self.weight_matrix.mapping()
        super().mapping()

    def report_mapping(self) -> str:
        s = (f"Module Info:\n"
             f"\tIn Channels: {self.in_channels}\n"
             f"\tOut Channels: {self.out_channels}"
             f"\tKernel Size: {self.kernel_size}\n")

        s += (f"Mapping Info:\n"
              f"{string_with_tab(self.weight_matrix.report_mapping())}")

        return s


class DepMaxpool2d(DepModule):
    base_name = "MaxPool2d"

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
    base_name = "Linear"

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.weight_matrix_shape = (in_features, out_features)
        self.weight_matrix = MatrixVar(self.weight_matrix_shape, self)
        # self.weight_matrix.dummy_mapping()

    def forward(self, input_tensor: DepTensor, *args, **kwargs):
        local_input_tensor = copy.deepcopy(input_tensor)

        output_tensor = _matrix_vec_mul_kernel(local_input_tensor, self.weight_matrix)

        return output_tensor

    def mapping(self):
        # do something here
        self.weight_matrix.mapping()
        super().mapping()

    def report_mapping(self):
        s = (f"Module Info:\n"
             f"\tIn Features: {self.in_features}\n"
             f"\tOut Features: {self.out_features}\n")

        s += (f"Mapping Info:\n"
              f"{string_with_tab(self.weight_matrix.report_mapping())}")

        return s


class DepElementAdd(DepModule):
    base_name = "ElementAdd"

    def __init__(self):
        super().__init__()

    def forward(self, a, b, *args, **kwargs):
        in_a = copy.deepcopy(a)
        in_b = copy.deepcopy(b)

        output_tensor = _add_kernel(in_a, in_b)
        return output_tensor


class DepReLU(DepModule):
    base_name = "ReLU"

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, *args, **kwargs):
        local_input_tensor = copy.deepcopy(input_tensor)
        output_tensor = _relu_kernel(local_input_tensor)
        return output_tensor


class DepSequential(DepModule):
    base_name = "Sequential"

    def __init__(self, *args):
        super().__init__()
        self.layers = args
        for index, layer in enumerate(self.layers):
            if isinstance(layer, DepModule):
                self._add_module_dict(str(index), layer)

    def forward(self, input_tensor: DepTensor):
        for layer in self.layers:
            input_tensor = layer(input_tensor)
        return input_tensor
