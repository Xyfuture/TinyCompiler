import torch
import torch.nn as nn



from TinyNet.modules.convpipe import ConvPipeLayer
from TinyNet.wrapper.base import module


class Conv(module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True,activation_func='relu'):
        super(Conv, self).__init__()

        self.conv_module = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                                     bias=bias)
        self.activation_module = None
        if activation_func == 'relu':
            self.activation_module = nn.ReLU()
        elif activation_func == 'sigmoid':
            self.activation_module = nn.Sigmoid()
        elif activation_func == 'tanh':
            self.activation_module = nn.Tanh()


        self.conv_args = [
            'in_channels', 'out_channels', 'kernel_size', 'stride',
            'padding', 'groups', 'bias', 'activation_func'
        ]
        self.conv_config = {}

        self.activation_func = activation_func
        self.conv_config['activation_func'] = self.activation_func
        self.bias = bias
        self.conv_config['bias']= self.bias

        for arg in self.conv_args:
            if arg not in ['activation_func','bias']:
                item = self.conv_module.__getattribute__(arg)
                self.conv_config[arg] = item
                self.__setattr__(arg,item)

        self.misc_config = {
            'mat_bitwidth':1,'act_bitwidth':1
        }

        self.input_shape = []
        self.output_shape = []
        self.conv_layer:ConvLayer = None


    def torch_forward(self, input_tensors:torch.Tensor):
        # NCHW to HWC
        tmp_in_shape = list(input_tensors.shape)
        self.input_shape = [tmp_in_shape[2],tmp_in_shape[3],tmp_in_shape[1]]
        output_tensor = self.conv_module(input_tensors)
        tmp_out_shape = list(output_tensor.shape)
        self.output_shape = [tmp_out_shape[2],tmp_in_shape[3],tmp_in_shape[1]]

        if self.activation_module:
            output_tensor = self.activation_module(output_tensor)

        return output_tensor


    def pim_forward(self,in_ten):
        self.allocate()
        out_ten = self.conv_layer.forward(in_ten)

        return out_ten

    def allocate(self):
        self.conv_layer = ConvPipeLayer(self.conv_config,self.input_shape,self.output_shape,self.misc_config)

