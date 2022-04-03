import torch

from inst import instruction


class ActivationFunc:
    def __init__(self,act_func):
        self.act_func = act_func

    def __call__(self, vec):
        def gen(result_vec):
            assert result_vec.core_id == vec.core_id
            inst.rd = result_vec.get_addr_reg() # 回写地址

            vec.core.inst_buffer.append(inst)

        inst = instruction(instruction.VRELU,rs1=vec.get_addr_reg())
        if self.act_func == 'relu':
            inst.op = instruction.VRELU
        elif self.act_func == 'sigmoid':
            inst.op = instruction.VSIGMOID
        elif self.act_func == 'tanh':
            inst.op = instruction.VTANH

        return gen


def get_act_func(func_name):
    return ActivationFunc(func_name)

