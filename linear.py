import torch
import torch.nn as nn
from .module import *
from .utils import *


class linear(module):
    def __init__(self,in_features, out_features, bias=True):
        super(linear, self).__init__()

        self.linear = nn.Linear(in_features,out_features,bias)

        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features
        self.bias = self.linear.bias

        self.input_shape = None
        self.output_shape = None

    def forward(self, input_tensors, relation_tensors:vtensor):
        self.in_tensors.append(relation_tensors)
        self.pre_modules[relation_tensors] = relation_tensors.pre_module
        relation_tensors.post_module = self

        self.input_shape = tuple(input_tensors.shape)
        output_tensor = self.linear(input_tensors)
        self.output_shape = tuple(output_tensor.shape)

        output_vtensor = gen_vtensor(output_tensor)
        output_vtensor.pre_module = self
        self.out_tensors.append(output_vtensor)

        return output_tensor,output_vtensor

    def code_gen(self):
        pass