from TinyNet.wrapper.module import module
import torch.nn as nn
from TinyNet.wrapper.tensor import vtensor
from TinyNet.wrapper.utils import gen_vtensor


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

    def allocate(self):
        # 分配资源，matrix activation 和 输出
        rows = self.kernel_size[0]*self.kernel_size[1]*self.in_channels # 矩阵行的形状
        columns = self.out_channels
        matrix_shape = (rows,columns)
        mat = matrix(matrix_shape,bitwidth=1)
        act_shape = self.input_shape[1:] # HWC
        act = conv_activation(act_shape,1,self.kernel_size,self.padding,self.stride)

        # 分配 mat 的核
        core_cnt = mat.core_cnt
        core_list = []
        core_id_list = []
        for i in range(core_cnt):
            id,c = core_allocator.get_core()
            core_id_list.append(id)
            core_list.append(c)

        mat.map_to_core(core_id_list)
        act.bind_allocate_mem(core_id_list)

        self.__setattr__('core_cnt',core_cnt)
        self.__setattr__('core_list',core_list)
        self.__setattr__('core_id_list',core_id_list)
        self.__setattr__('mat',mat)
        self.__setattr__('act',act)



    def code_gen(self):
        '''
        首先完成通讯，将前一层的数据传输到该层的各个核上
        然后在核上进行一次次的计算，顺便完成激活函数的操作
        计算完的结果收集到特定的几个核上，通过vtensor发送到其他的核上
        '''
        pass

    def receive_pre_layer_activation(self):
        pass

    def conv_compute_gen(self):
        pass
