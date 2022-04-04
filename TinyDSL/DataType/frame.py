import gc
from collections import OrderedDict

from TinyDSL.HwResource.core import core_allocator


class FrameStackCore:
    def __init__(self,core_id):
        self.core_id = core_id
        self.core = core_allocator.access_core(self.core_id)

        self.entry_state = OrderedDict()
        self.entry_mem = OrderedDict()

    def release(self,entry_id):
        assert self.entry_state.get(entry_id)
        self.entry_state[entry_id] = False
        self.cleanup()


    def insert(self,entry_id,mem):
        self.entry_state[entry_id] = True
        self.entry_mem[entry_id] = mem

    def cleanup(self):
        gc.collect()
        tmp_gc_list = []

        for k in reversed(self.entry_state):
            if not self.entry_state[k]:
                tmp_gc_list.append(k)
            else:
                for k in tmp_gc_list:
                    del self.entry_state[k]
                    self.core.mem_allocator.release_stack_mem(self.entry_mem[k])
                    del self.entry_mem[k]
                return
    def __del__(self):
        for k,v in self.entry_state.items():
            print(k,v)



class frameStack:
    def __init__(self):
        self.frame_core_dict = {}

    def __getitem__(self, item):
        if self.frame_core_dict.get(item):
            return self.frame_core_dict[item]
        else :
            self.frame_core_dict[item] = FrameStackCore(item)
            return self.frame_core_dict[item]



FrameStack = frameStack()
