from TinyDSL.HwResource.inst import InstBuffer
from TinyDSL.HwResource.mem import mem_entry, Mem_allocator
from TinyDSL.HwResource.reg import Reg_allocator
from TinyDSL.HwResource.config import core_config, core_cfg
from TinyDSL.Utils.base import *

class core:
    def __init__(self,core_id,**kwargs):

        self.core_id = core_id
        self.inst_buffer = InstBuffer()
        self.cfg = core_config
        self.meu_cnt = self.cfg.meu_cnt

        self.mem_allocator = Mem_allocator(self.cfg.omu_size,self.core_id)
        self.reg_allocator = Reg_allocator()

        self.meu_state = bitmap(self.meu_cnt)
        self.meu_list = [meu(core_id,i) for i in range(self.meu_cnt)]

    def get_meu(self,cnt=1):
        meu_id_list = self.meu_state.get_free(cnt)
        if cnt == 1:
            return self.meu_list[meu_id_list]
        else:
            tmp_meu_list = [self.meu_list[i] for i in meu_id_list]
            return tmp_meu_list



class Core_allocator:
    def __init__(self,cfg_core:core_cfg):
        self.core_cnt = 64

        self.cfg = cfg_core
        self.core_list = [core(i) for i in range(self.core_cnt)]
        self.core_allocate_state = bitmap(self.core_cnt,"free","free")

    def get_core(self):
        tmp = self.core_allocate_state.get_free(1,"allocated")
        # self.core_allocate_state[tmp] = "allocated"
        return tmp

    def release_core(self,core_id):
        self.core_allocate_state[core_id] = "unused"

    def access_core(self,core_id):
        return self.core_list[core_id]

    def query_core_state(self,core_id):
        return self.core_allocate_state[core_id]

    def __getitem__(self, item):
        return self.core_list[item]

class meu:
    '''
    just record infomation of sub matrix
    not gen code
    '''
    def __init__(self,core_id,meu_id,**kwargs):
        self.core_id = core_id
        self.meu_id = meu_id
        self.matrix = None
        self.posi = None
        self.cfg = core_config
        self.shape = None
        self.group = None


class meu_group:
    def __init__(self):
        pass



core_allocator = Core_allocator(core_config)

