import gc
from collections import OrderedDict

from TinyDSL.HwResource.core import core_allocator


class FrameStackCore:
    def __init__(self,core_id):
        self.core_id = core_id
        self.core = core_allocator.access_core(self.core_id)

        self.entry_state = OrderedDict()
        self.entry_mem = OrderedDict()
        self.resurrection = OrderedDict()

    def release(self,entry_id,vec_ten):
        assert self.entry_state.get(entry_id)
        # print('in release:',entry_id)
        self.entry_state[entry_id] = False
        self.resurrection[entry_id] = vec_ten
        self.cleanup()


    def checker(self):
        if len(self.entry_mem)!=len(self.core.mem_allocator.stack_entry):
            print('not equal')
        assert len(self.entry_mem) == len(self.core.mem_allocator.stack_entry)

        for i,k in enumerate(self.entry_mem):
            assert self.entry_mem[k] is self.core.mem_allocator.stack_entry[i]

    def insert(self,entry_id,mem):
        # gc.collect()
        if entry_id in self.entry_state:
            print('wrong')
        self.entry_state[entry_id] = True
        self.entry_mem[entry_id] = mem
        # print('current:',entry_id)
        self.checker()

    def cleanup(self):
        self.checker()
        # gc.collect()
        tmp_gc_list = []
        # print('clearn up')
        # for k in reversed(self.entry_state):
        #     print(k,self.entry_state[k])
        for k in reversed(self.entry_state):
            if not self.entry_state[k]:
                tmp_gc_list.append(k)
            else:
                break
        for k in tmp_gc_list:
            # print('release:',k)
            del self.entry_state[k]
            self.core.mem_allocator.release_stack_mem(self.entry_mem[k])
            del self.entry_mem[k]
            del self.resurrection[k]

    def __del__(self):
        for k,v in self.entry_state.items():
            print(k,v)



class FrameStack:
    def __init__(self):
        self.frame_core_dict = {}

    def __getitem__(self, item):
        if self.frame_core_dict.get(item):
            return self.frame_core_dict[item]
        else :
            self.frame_core_dict[item] = FrameStackCore(item)
            return self.frame_core_dict[item]



frame_stack = FrameStack()
