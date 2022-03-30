# import torch
from inst import instruction
from mem import mem_entry
from .reg import RegVar
from ..core import core_allocator
from .vector import VectorVar


# 创建实例必定申请内存
class TensorVar:
    def __init__(self,ten_shape,core_id,bitwidth,**kwargs):
        self.ten_shape = ten_shape
        self.dim = len(self.ten_shape)
        self.core_id = core_id
        self.core = core_allocator.access_core(self.core_id)
        self.bitwidth = bitwidth

        self.length = 1
        for i in self.ten_shape:
            self.length *= i

        if 'location' in kwargs:
            self.location = kwargs['location']
        else :
            self.location = 'stack'

        if 'mem' in kwargs:
            self.mem = kwargs['mem']
            self.mem_owner = False
        else:
            self.mem = self.core.mem_allocator.get_mem(self.length*self.bitwidth,self.location)
            self.mem_owner = True

        self.logic_offset = [0 for i in range(self.dim)]
        for i in range(self.dim):
            tmp = 1
            for j in range(self.dim-i-1,0,-1):
                tmp *= self.ten_shape[j]
            self.logic_offset[i] = tmp

        self.mem_offset = [i*self.bitwidth for i in self.logic_offset]

        if 'logic_offset' in kwargs:
            self.logic_offset = kwargs['logic_offset']
        if 'mem_offset' in kwargs:
            self.mem_offset = kwargs['mem_offset']


    def get_addr_reg(self):
        if 'addr_reg' in self.__dict__ :
            return self.addr_reg.reg_id
        tmp = RegVar(self.core_id,imm=self.mem.addr)
        self.__setattr__('addr_reg',tmp)
        return self.addr_reg.reg_id

    def get_length_reg(self):
        if 'addr_reg' in self.__dict__ :
            return self.addr_reg.reg_id
        tmp = RegVar(self.core_id,imm=self.vec_shape)
        self.__setattr__('addr_reg',tmp)
        return self.addr_reg.reg_id

    def get_vec(self,indice,length): # indice 全部是int型即可
        assert len(indice) == self.dim
        vec_mem_offset = 0
        for i,cur in enumerate(indice):
            vec_mem_offset += cur*self.mem_offset[i]
        vec_mem_offset += self.mem.addr # offset + start_posi
        new_mem = mem_entry(self.core_id,vec_mem_offset,length*self.bitwidth,self.bitwidth)
        return VectorVar(length,self.core_id,self.bitwidth,mem=new_mem)

    def copy(self,src_ten):
        assert self.mem.size == src_ten.mem.size
        if self.core_id == src_ten.core_id:
            inst = instruction(instruction.VMV,rd=self.get_addr_reg(),rs1=src_ten.get_addr_reg(),rs2=self.get_length_reg())
            self.core.inst_buffer.append(inst)
        else:
            send_inst = instruction(instruction.SEND,rs1=src_ten.get_addr_reg(),rs2=src_ten.get_length_reg(),imm=self.core_id)
            recv_inst = instruction(instruction.RECV,rs1=self.get_addr_reg(),rs2=self.get_length_reg(),imm=src_ten.core_id)

            self.core.inst_buffer.append(recv_inst)
            src_ten.core.inst_buffer.append(send_inst)



    def __del__(self):
        pass