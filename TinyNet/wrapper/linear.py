
import torch.nn as nn

from TinyNet.modules.linear import LinearLayer
from TinyNet.wrapper.base import module


class Linear(module):
    def __init__(self,in_features,out_features,bias=True,activation_func='relu'):
        super(Linear, self).__init__()

        self.linear_module = nn.Linear(in_features,out_features,bias)
        self.activation_module = None
        if activation_func == 'relu':
            self.activation_module = nn.ReLU()
        elif activation_func == 'sigmoid':
            self.activation_module = nn.Sigmoid()
        elif activation_func == 'tanh':
            self.activation_module = nn.Tanh()


        self.linear_args = ['in_features','out_features','bias','activation_func']
        self.activation_func = activation_func

        self.linear_config = {}
        self.linear_config['activation_func'] = self.activation_func
        self.bias = bias
        self.linear_config['bias'] = bias


        for arg in self.linear_args:
            if arg not in ['activation_func','bias']:
                item = self.linear_module.__getattribute__(arg)
                self.linear_config[arg] = item
                self.__setattr__(arg,item)

        self.misc_config = {
            'mat_bitwidth':1,'act_bitwidth':1
        }

        self.input_shape = []
        self.output_shape = []
        self.linear_layer:LinearLayer = None


    def torch_forward(self, input_tensors):
        tmp_in_shape = list(input_tensors.shape)

        if len(tmp_in_shape) == 4: # CONV output
            input_tensors.view(-1,self.in_features)
            self.input_shape = [tmp_in_shape[2],tmp_in_shape[3],tmp_in_shape[1]]
        elif len(tmp_in_shape) == 2:
            self.input_shape = [1,1,self.in_features]
        else:
            raise "error"

        output_tensor = self.linear_module(input_tensors)

        tmp_out_shape = list(output_tensor.shape)
        if len(tmp_out_shape) == 2:
            self.output_shape=[1,1,self.out_features]
        else:
            raise "error"

        if self.activation_module:
            output_tensor = self.activation_module(output_tensor)

        return output_tensor


    def pim_forward(self,in_ten):
        self.allocate()
        out_ten = self.linear_layer.forward(in_ten) # in_ten 可能是个function
        # out_ten 同理，可能是tensor，也可能是function
        return out_ten

    def allocate(self):
        self.linear_layer = LinearLayer(self.linear_config,self.input_shape,self.output_shape,self.misc_config)
