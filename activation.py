from TinyNet.old_wrapper.tensor import *
from collections import OrderedDict


# 认怂 直接将padding考虑在内
class conv_activation:
    def __init__(self, shape, bitwidth, kernel_size=(3, 3), padding=(0, 0), stride=(1, 1), **kwargs):
        self.shape = shape
        self.bitwidth = bitwidth
        self.kernel_size = kernel_size
        self.padding =padding
        self.stride = stride

        self.shape = [self.shape[i]+self.padding[i]*2 for i in range(2)] + [self.shape[2]]# padding 考虑在内

        self.length = 1
        for i in self.shape:
            self.length *= i
        self.size = self.length*self.bitwidth

        self.core_cnt = 0
        self.core_id_list = []
        self.core_dict = OrderedDict()
        # 通过core_id 进行访问
        self.tensor_dict = OrderedDict()


    def bind_allocate_mem(self,core_id_list):
        self.core_cnt = len(core_id_list)
        self.core_id_list = core_id_list
        for i in self.core_id_list:
            self.core_dict[i] = core_allocator.access_core(i)
        for i in self.core_id_list :
            self.tensor_dict[i] = tensor(i, self.shape, self.bitwidth, location="heap")


    # 限制只能一个batch
    def im2col(self,section_info:dict):
        """
        :param section_info: 一个dict，对应所有核需要的区间
        :return:
        """
        height,width,channel = self.shape
        stride_height,stride_width = self.stride
        kernel_height,kernel_width = self.kernel_size

        for i in range(0,height,stride_height):
            for j in range(0,width,stride_width):
                tmp_mem_dict = {}
                final_mem_dict = {}
                for core_id,section in section_info.items():
                    t = self.tensor_dict[core_id]
                    tmp_tensor = t[i:i+kernel_height,j:j+kernel_width,:]
                    start = section.start
                    stop = section.stop
                    tmp_size = (start-stop) * self.bitwidth
                    tmp_mem = self.core_dict[core_id].mem_allocator.get_stack_mem(tmp_size,self.bitwidth)
                    final_tmp_mem = tmp_tensor.vectorize(section,tmp_mem)
                    tmp_mem_dict[core_id] = tmp_mem
                    final_tmp_mem[core_id] = final_mem_dict
                yield final_mem_dict
                for k,v in tmp_mem_dict.items():
                    self.core_dict[k].mem_allocator.release_stack_mem(v)







