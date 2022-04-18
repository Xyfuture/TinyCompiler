import torch.nn as nn

from TinyNet.modules.pooling import MaxPoolingLayer
from TinyNet.wrapper.base import module



class MaxPooling(module):
    def __init__(self,kernel_size,stride):
        super(MaxPooling, self).__init__()

        self.maxpooling_module = nn.MaxPool2d(kernel_size,stride)

        self.pooling_args = ['kernel_size','stride']
        self.pooling_config = {}
        # self.kernel_size = kernel_size
        # self.stride = stride
        #
        for arg in self.pooling_args:
            item = self.maxpooling_module.__getattribute__(arg)
            self.pooling_config[arg] = item
            self.__setattr__(arg,item)

        self.misc_config ={
            'bitwidth':1
        }


        self.input_shape = []
        self.output_shape = []
        self.maxpooling_layer:MaxPoolingLayer = None



    def torch_forward(self, input_tensors):
        tmp_in_shape = list(input_tensors.shape)

        self.input_shape = [tmp_in_shape[2],tmp_in_shape[3],tmp_in_shape[1]]

        output_tensor = self.maxpooling_module(input_tensors)
        tmp_out_shape = list(output_tensor)

        self.output_shape = [tmp_out_shape[2],tmp_out_shape[3],tmp_in_shape[1]]

        return output_tensor


    def pim_forward(self,in_ten):
        self.allocate()
        out_ten = self.maxpooling_layer.forward(in_ten)

        return out_ten

    def allocate(self):
        self.maxpooling_layer = MaxPoolingLayer(self.pooling_config,self.input_shape,self.output_shape,self.misc_config)