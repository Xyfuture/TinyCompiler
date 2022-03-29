from ..core import *
from ..inst import *
class RegVar:
    def __init__(self,core_id,**kwargs):
        self.core_id = core_id
        self.core = core_allocator.access_core(self.core_id)
        self.reg_id = self.core.reg_allocator.get_reg()
        if 'imm' in kwargs:
            self.imm_write(kwargs['imm'])

    def imm_write(self,value):
        self.core.inst_buffer.append(instruction(instruction.LDI,imm=value))

    def mem_write(self):
        pass

    def __del__(self):
        pass

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __divmod__(self, other):
        pass

