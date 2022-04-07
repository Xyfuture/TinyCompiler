from TinyDSL.Utils.base import *
from TinyDSL.HwResource.config import core_config

class Reg_allocator:
    # 64bit reg with count of 32
    def __init__(self):
        # 0号寄存器的值永远是0
        self.reg_bitmap = bitmap(core_config.reg_cnt-1)

    def get_reg(self):
        return self.reg_bitmap.get_free()+1

    def release_reg(self,id):
        self.reg_bitmap.free(id-1)
