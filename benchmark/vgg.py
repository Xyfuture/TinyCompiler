from  collections import OrderedDict

import torch
import torch.nn as nn

from TinyDSL.DataType.tensor import TensorVar
from TinyDSL.HwResource.core import core_allocator
from TinyNet.wrapper.conv import Conv
from TinyNet.wrapper.linear import Linear
from TinyNet.wrapper.pooling import MaxPooling
from TinyNet.wrapper.base import module

class Vgg8:
    def __init__(self):
        super(Vgg8, self).__init__()
        self.conv1 = Conv(in_channels=3,out_channels=128,kernel_size=3,padding=1,activation_func='relu')
        self.conv2 = Conv(in_channels=128,out_channels=128,kernel_size=3,padding=1,activation_func='relu')
        self.max_pool1= MaxPooling(kernel_size=2,stride=2)
        self.conv3 = Conv(in_channels=128,out_channels=256,kernel_size=3,padding=1,activation_func='relu')
        self.conv4 = Conv(in_channels=256,out_channels=256,kernel_size=3,padding=1,activation_func='relu')
        self.max_pool2 = MaxPooling(kernel_size=2,stride=2)
        self.conv5 = Conv(in_channels=256,out_channels=512,kernel_size=3,padding=1,activation_func='relu')
        self.conv6 = Conv(in_channels=512,out_channels=512,kernel_size=3,padding=1,activation_func='relu')
        self.max_pool3 = MaxPooling(kernel_size=2,stride=2)
        self.conv7 = Conv(in_channels=512,out_channels=1024,kernel_size=3,padding=0,activation_func='relu')
        self.max_pool4 = MaxPooling(kernel_size=2,stride=2)
        self.linear = Linear(in_features=1024,out_features=10)

        self.module_list = [
            # self.conv1

            self.conv1,self.conv2,self.max_pool1,
            self.conv3,self.conv4,self.max_pool2,
            self.conv5,self.conv6,self.max_pool3,
            self.conv7,self.max_pool4,self.linear
        ]


    def forward(self,torch_tensor,pim_tensor):
        ret = (torch_tensor,pim_tensor)

        for cnt,layer in enumerate(self.module_list):
            print(cnt)
            ret = layer(*ret)

if __name__ == "__main__":
    net = Vgg8()

    torch_tensor = torch.randn([1,3,32,32])
    core_id = core_allocator.get_core()
    core = core_allocator.access_core(core_id)

    pim_tensor = TensorVar([32,32,3],core_id,1)

    net.forward(torch_tensor,pim_tensor)

    for i,c in enumerate(core_allocator.allocated_cores()):
        print('{}:{}'.format(i,len(c.inst_buffer.inst_list)))

    # core = core_allocator.access_core(50)
    # core.inst_buffer.print_asm()





