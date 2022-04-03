from TinyDSL.HwResource.core import *
from TinyDSL.HwResource.inst import *

class RegVar:
    def __init__(self,core_id,**kwargs):
        self.core_id = core_id
        self.core = core_allocator.access_core(self.core_id)
        self.reg_id = self.core.reg_allocator.get_reg()
        if 'imm' in kwargs:
            self.imm_write(kwargs['imm'])
        if 'reg_compute_result' in kwargs:
            self.reg_compute_write(kwargs['reg_compute_result'])

    def imm_write(self,value):
        self.core.inst_buffer.append(instruction(instruction.LDI,rd=self.reg_id,imm=value))

    def mem_write(self):
        pass

    def reg_compute_write(self,func): # scalar 计算的结果，输入一个函数（RegVar变量算数运算返回的）
        func(self)

    def assign(self,item):
        if callable(item):
            item(self)
        elif isinstance(item,int):
            self.imm_write(item)
        elif isinstance(item,RegVar):
            self.copy(item)

    def copy(self,src_reg):
        assert self.core_id == src_reg.core_id
        assert self.reg_id != src_reg.reg_id

        inst = instruction(instruction.SADD,rd=self.reg_id,rs1=src_reg.reg_id,rs2=0)
        self.core.inst_buffer.append(inst)


    def __del__(self):
        pass

    def __add__(self, other):
        assert self.core_id == other.core_id
        if isinstance(other,RegVar):
            inst = instruction(instruction.SADD,rs1=self.reg_id,rs2=other.reg_id)
        elif isinstance(other,RegVar):
            inst = instruction(instruction.SADDI,rs1=self.reg_id,imm=other)

        def gen(result_reg):
            assert self.core_id == result_reg.core_id
            inst.rd = result_reg.reg_id
            self.core.inst_buffer.append(inst)

        return gen

    def __radd__(self, other):
        return self.__add__(other)


    def __mul__(self, other):
        pass

    def __divmod__(self, other):
        pass

