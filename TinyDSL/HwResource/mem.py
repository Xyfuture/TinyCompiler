# from core import core_allocator

class mem_entry:
    def __init__(self, core_id,addr, size, bitwidth, **kwargs):
        self.addr = addr
        self.size = size
        self.bitwidth = bitwidth
        self.core_id = core_id

        self.data_type = None
        self.location = None
        if 'data_type' in kwargs:
            self.data_type = kwargs['data_type']
        if 'location' in kwargs:
            self.location = kwargs['location']

class vmem_entry(mem_entry):
    def __init__(self,addr, size, bitwidth, **kwargs):
        super(vmem_entry, self).__init__(-1,addr, size, bitwidth, **kwargs)

# every core has its own mem allocator
class Mem_allocator:
    def __init__(self, max_size, core_id):
        self.stack_mem_addr = 0
        self.heap_mem_addr = 1e8

        self.stack_entry = []
        self.heap_entry = []

        self.max_size = max_size
        self.core_id = core_id

        self.stack_used = 0
        self.heap_used = 0

    def get_mem(self,size,bitwidth,location='stack',**kwargs):
        if location == 'stack':
            return self.get_stack_mem(size,bitwidth,**kwargs)
        elif location == 'heap':
            return self.get_heap_mem(size,bitwidth,**kwargs)
        return None

    def get_heap_mem(self, size, bitwidth, **kwargs):
        kwargs['location'] = 'heap'
        entry = mem_entry(self.core_id,self.heap_mem_addr, size, bitwidth, **kwargs)

        self.heap_mem_addr += size
        self.heap_used += size
        self.check_size()

        self.heap_entry.append(entry)
        return entry

    def get_stack_mem(self, size, bitwidth, **kwargs):
        kwargs['location'] = 'stack'
        entry = mem_entry(self.core_id,self.stack_mem_addr, size, bitwidth, **kwargs)

        self.stack_mem_addr += size
        self.stack_used += size
        self.check_size()

        self.stack_entry.append(entry)
        return entry

    # heap will not be released
    def release_stack_mem(self, entry=None):
        stack_top = self.stack_entry.pop()
        if entry:
            assert stack_top is entry, "ERROR: stack top unmatch"
        self.stack_mem_addr -= stack_top.size
        self.stack_used -= stack_top.size

        return stack_top

    @property
    def mem_used(self):
        return self.stack_used + self.heap_used

    def check_size(self):
        assert self.mem_used <= self.max_size, "ERROR: mem size overflow"



