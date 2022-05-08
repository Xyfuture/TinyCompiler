import torch

from TinyDSL.DataType.tensor import TensorVar
from TinyDSL.HwResource.core import core_allocator
from TinyNet.wrapper.convpipe import Conv
from TinyNet.wrapper.linear import Linear


class TestLayer:
    def __init__(self):
        self.conv = Conv(in_channels=3,out_channels=128,kernel_size=3,padding=1,activation_func='relu')

    def forward(self,torch_tensor,pim_tensor):
        ret = torch_tensor,pim_tensor

        ret = self.conv(*ret)
        return ret



if __name__ == "__main__":
    net = TestLayer()

    torch_tensor = torch.randn(1,3,32,32)
    core_id = core_allocator.get_core()
    core = core_allocator.access_core(core_id)

    pim_tensor = TensorVar([32,32,3],core_id,1)

    net.forward(torch_tensor,pim_tensor)
    # core_allocator.access_core(1).inst_buffer.print_dict()


    cnt = 0
    for i,c in enumerate(core_allocator.allocated_cores()):
        # c.inst_buffer.print_asm()
        c.inst_buffer.save_dict('E:\code\TinyCompiler\\benchmark\\{}.pkl'.format(i))
    #     for inst in c.inst_buffer.dump_dict():
    #         if inst['op'] == 'gemv':
    #             cnt += 1
    # print(cnt)
    core_allocator.access_core(1).inst_buffer.print_asm()