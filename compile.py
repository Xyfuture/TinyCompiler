import os,sys
import torch

from TinyDSL.DataType.tensor import TensorVar
from TinyDSL.HwResource.core import core_allocator

from Examples.vgg8 import vgg8
from Examples.vgg16 import vgg16
from Examples.auto_encoder import auto_encoder
from Examples.lenet import lenet

import argparse


parser = argparse.ArgumentParser(description='Tiny Compiler for Neural Network on PIM')

parser.add_argument('--net', type=str, default='auto_encoder',help="Current Support vgg8 vgg16 auto_encoder lenet")
parser.add_argument('--output', type=str, default='./inst_output/', help="Output Directory path")

args = parser.parse_args()
output_path = args.output if args.output.endswith('/') else args.output + '/'


class net_gen():
    def __init__(self,net,dir_path='./binary/'):
        self.net = net
        self.dir_path = dir_path


    def inst_gen(self,tensor_shape): # CHW

        allocate_list = []

        torch_tensor = torch.randn(tensor_shape)
        for i,stage in enumerate(self.net.stage_list):
            if stage[0] == 'view':
                torch_tensor = torch_tensor.view(torch_tensor.size(0),-1)
            tensor_shape = list(torch_tensor.shape)
            # tmp_core_allocator = Core_allocator(core_config)
            tmp_core_id = core_allocator.get_core()

            if len(tensor_shape) == 4:
                pim_tensor = TensorVar([tensor_shape[2],tensor_shape[3],tensor_shape[1]],tmp_core_id,1)
            elif len(tensor_shape) == 2:
                pim_tensor = TensorVar([1,1,tensor_shape[1]],tmp_core_id,1)


            ret = (torch_tensor,pim_tensor)
            for cnt,layer in enumerate(stage):
                if layer == 'view':
                    continue
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

            print(f"Stage:{i} Finished")

            torch_tensor,_ = ret



if __name__ == "__main__":

    # tensor_shape = [1,3,32,32]
    # net = vgg16()

    # net = auto_encoder()
    # tensor_shape = [1,4096]
    # gen = net_gen(net,'./binary/auto_encoder/')
    # gen.inst_gen(tensor_shape)

    print("Compile Start:\n"
          f"Net: {args.net}\n"
          f"Output Path: {output_path}\n")

    net = auto_encoder()
    tensor_shape = [1, 4096]

    if args.net == 'auto_encoder':
        net = auto_encoder()
        tensor_shape = [1, 4096]
    elif args.net == 'vgg8':
        net = vgg8()
        tensor_shape = [1,3,32,32]
    elif args.net == 'vgg16':
        net = vgg16()
        tensor_shape= [1, 3, 32, 32]
    elif args.net == 'lenet':
        net=lenet()
        tensor_shape = [1,3,32,32]
    else:
        print("Error Net Type")

    gen = net_gen(net,output_path)
    gen.inst_gen(tensor_shape)
    print("\nCompile Finished!\n"
          f"Binary and ASM file Write to {output_path}")

    os.system("pause")