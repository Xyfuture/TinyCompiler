from .mem import *
from .tensor import *
from .base import *
from .mem import *
from .config import *

class core:
    def __init__(self,core_id,cfg:core_cfg,**kwargs):

        self.core_id = core_id
        self.inst_buffer = linkList()
        self.cfg = cfg
        self.meu_cnt = self.cfg.meu_cnt

        self.mem_allocator = Mem_allocator(self.cfg.omu_size,self.core_id)
        self.reg_allocator = Reg_allocator()
        self.meu_state = bitmap(self.cfg.meu_cnt)




class Core_allocator:
    pass




class meu:
    '''
    just record infomation of sub matrix
    not gen code
    '''
    def __init__(self,core_id,meu_id,matrix,posi,**kwargs):
        self.core_id = core_id
        self.meu_id = meu_id
        self.matrix = matrix
        self.posi = posi
