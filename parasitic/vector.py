import torch
from ..core import core_allocator
from reg import *

'''
vector 最重要的区别是存有一个addr_reg和一个length_reg，这使得这个万一直接支持参与指令的运算
'''
class VectorVar:
    def __init__(self,vec_shape,core_id,bitwidth,**kwargs):
        # shape 就是length，只是为了统一名字
        self.vec_shape = vec_shape
        self.core_id = core_id
        self.core = core_allocator.access_core(self.core_id)
        self.bitwidth = bitwidth
        if 'mem' in kwargs:
            self.mem = kwargs['mem']
            self.mem_owner = False
        else :
            self.mem = self.core.mem_allocator.get_stack_mem(self.vec_shape*self.bitwidth)
            self.mem_owner = True


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

    def copy(self,src_vec): # 两个vec 之间直接复制
        assert self.bitwidth*self.vec_shape == src_vec.bitwidth * src_vec.shape

        inst = instruction(instruction.VMV,rd=self.get_addr_reg(),rs1=src_vec.get_addr_reg(),rs2=self.get_length_reg())
        self.core.inst_buffer.append(inst)


    def assign(self,func): # 用于算数运算时进行赋值，参数是个函数，主要用于回写地址啥的
        func(self) # 调用传过来的函数


    def __getitem__(self, item): # 提出部分来用于复制等操作 or 赋值给寄存器等等
        if isinstance(item,slice):
            start = item.start
            stop = item.stop
            assert stop<=self.vec_shape
            new_mem_addr = self.mem.addr + start*self.bitwidth
            new_vec_shape = stop-start
            new_mem = mem_entry(self.core_id,new_mem_addr,new_vec_shape*self.bitwidth,self.bitwidth)
            return VectorVar(new_vec_shape,self.core_id,self.bitwidth,mem=new_mem)
        elif isinstance(item,int):
            pass


    def __del__(self):
        pass