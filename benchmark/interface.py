import os,sys
import torch

from TinyDSL.DataType.tensor import TensorVar
from TinyDSL.HwResource.core import core_allocator


from TinyNet.wrapper.convpipe import Conv
from TinyNet.wrapper.linearpipe import Linear
from TinyNet.wrapper.pooling import MaxPooling


class net_gen():
    def __init__(self,net,dir_path='./binary/'):
        self.net = net
        self.dir_path = dir_path


    def inst_gen(self,tensor_shape): # CHW

        allocate_list = []

        torch_tensor = torch.randn(tensor_shape)
        for i,stage in enumerate(self.net.stage_list):


            # tmp_core_allocator = Core_allocator(core_config)
            tmp_core_id = core_allocator.get_core()

            if len(tensor_shape) == 4:
                pim_tensor = TensorVar([tensor_shape[2],tensor_shape[3],tensor_shape[1]],tmp_core_id,1)
            elif len(tensor_shape) == 2:
                pim_tensor = TensorVar([1,1,tensor_shape[1]],tmp_core_id,1)


            ret = (torch_tensor,pim_tensor)
            for cnt,layer in enumerate(stage):
                ret = layer(*ret)

            # 存储为dict
            cur_dir = self.dir_path + 'stage_{}'.format(i)
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)

            for core_id in core_allocator.allocated_core_id():
                if core_id not in allocate_list:
                    allocate_list.append(core_id)
                    core = core_allocator.access_core(core_id)
                    core.inst_buffer.save_dict(cur_dir+'\\{}.pkl'.format(core_id))
                    core.inst_buffer.save_asm(cur_dir+'\\{}.txt'.format(core_id))

            torch_tensor,_ = ret
            tensor_shape = list(torch_tensor.shape)


class vgg8:
    def __init__(self):
        super(vgg8, self).__init__()
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

        self.stage_list = [
            [self.conv1],[self.conv2,self.max_pool1],[self.conv3],
            [self.conv4,self.max_pool2],[self.conv5],[self.conv6,self.max_pool3],
            [self.conv7,self.max_pool4],[self.linear]
        ]


class lenet:
    def __init__(self):

        self.conv1 = Conv(in_channels=3,out_channels=6,kernel_size=5,activation_func='relu')
        self.max_pool1 = MaxPooling(kernel_size=2,stride=2)

        self.conv2 = Conv(in_channels=6,out_channels=16,kernel_size=5,activation_func='relu')
        self.max_pool2 = MaxPooling(kernel_size=2,stride=2)

        self.conv3 = Conv(in_channels=16,out_channels=120,kernel_size=5,activation_func='relu')

        self.linear1 = Linear(in_features=120,out_features=84,activation_func='relu')

        self.linear2 = Linear(in_features=84,out_features=10)

        self.stage_list = [
            [self.conv1,self.max_pool1],[self.conv2,self.max_pool2],
            [self.conv3],[self.linear1],[self.linear2]
        ]





if __name__ == "__main__":
    tensor_shape = [1,3,32,32]
    net = vgg8()
    gen = net_gen(net,'./binary/vgg8/')
    gen.inst_gen(tensor_shape)
