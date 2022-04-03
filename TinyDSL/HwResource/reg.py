from TinyDSL.Utils.base import *
from TinyDSL.HwResource.config import core_config

class Reg_allocator:
    # 64bit reg with count of 32
    def __init__(self):
        self.reg_bitmap = bitmap(core_config.reg_cnt)

    def get_reg(self):
        return self.reg_bitmap.get_free()

    def release_reg(self,id):
        self.reg_bitmap.free(id)
