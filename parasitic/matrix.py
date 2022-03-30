import torch
from ..core import core_allocator
from reg import *

class MatrixVar:
    def __init__(self,mat_shape,core_id,packet_id,meu_id_list,bitwidth,**kwargs):
        self.mat_shape = mat_shape
        self.core_id = core_id
        self.core = core_allocator.access_core(core_id)
        self.packet_id = packet_id
        self.meu_id_list = meu_id_list
        self.bitwidth = bitwidth

        self.rows,self.columns = self.mat_shape

        self.addr_reg = RegVar(core_id,imm=self.packet_id)
        self.length_reg = RegVar(core_id,imm=self.columns)


    def __mul__(self, vec):
        def gen(dest_vec):

            self.core.inst_buffer.append(instruction(instruction.GEMV,rd=self.get_addr_reg(),rs1=vec.get_addr_reg(),rs2=vec.get_length_reg(),bitwidth=vec.bitwidth))
            self.core.inst_buffer.append(instruction(instruction.GVR, rd=self.get_addr_reg(), rs1=dest_vec.get_addr_reg(), rs2=self.get_length_reg(), ))

        return gen


    def get_addr_reg(self):
        return self.addr_reg.reg_id

    def get_length_reg(self):
        return self.length_reg.reg_id

# matrix group