from TinyNet.old_wrapper.module import module
import torch.nn as nn
from TinyNet.old_wrapper.tensor import vtensor
from TinyNet.old_wrapper.utils import gen_vtensor


class conv2d(module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True,activation_func='relu'):
        super(conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,groups=groups,bias=bias)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.groups = self.conv.groups
        self.bias = self.conv.bias
        self.activation_func = activation_func

        self.input_shape = []
        self.output_shape = []
    def forward(self, input_tensors, relation_tensors:vtensor):
        # record information

        self.in_tensors.append(relation_tensors)
        self.pre_modules[relation_tensors]=relation_tensors.pre_module
        relation_tensors.post_module = self

        self.input_shape = tuple(input_tensors.shape)
        self.input_shape = [self.input_shape[0],self.input_shape[2],self.input_shape[3],self.input_shape[1]]
        output_tensor = self.conv(input_tensors)
        self.output_shape = tuple(output_tensor.shape)
        self.output_shape = [self.output_shape[0],self.output_shape[2],self.output_shape[3],self.output_shape[1]]
        # NCHW to NHWC

        out_vtensor = gen_vtensor(output_tensor)
        out_vtensor.pre_module = self
        self.out_tensors.append(out_vtensor)

        return output_tensor,out_vtensor


    def code_gen_forward(self):
        pass
