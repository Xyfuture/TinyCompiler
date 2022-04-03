from typing import List

from TinyDSL.HwResource.core import core_allocator
from TinyDSL.HwResource.core import *
from TinyDSL.HwResource.mem import *
import math




class vtensor:
    def __init__(self,shape,bitwidth,name='vtensor'):
        self.shape = shape
        self.bitwidth = bitwidth
        self.name = name
        self.length = 1
        for i in self.shape:
            self.length *= i
        self.size = self.length * self.bitwidth

        self._pre_module = None
        self._post_module = None

        self.mem_array: List[mem_entry] = []
        # 存储一些列的mem_entry，需要保证是连续的，不允许出现复制的情况（同一块内存同时在两个core中存在）


    @property
    def pre_module(self):
        return self._pre_module

    @pre_module.setter
    def pre_module(self,pre):
        self._pre_module = pre

    @property
    def post_module(self):
        assert self._post_module, "ERROR: _post_module not defined"
        return self._post_module

    @post_module.setter
    def post_module(self,post):
        self._post_module = post



# unused currently
class tensor:
    def __init__(self,core_id,shape,bitwidth,location,**kwargs):
        self.core_id = core_id
        self.core = core_allocator.access_core(self.core_id)
        self.shape = shape
        self.dim = len(shape)
        self.bitwidth =bitwidth
        self.location = location

        self.sliced = False
        if 'sliced' in kwargs:
            self.sliced = kwargs['sliced']

        self.logic_offset = [0 for i in range(self.dim)]
        for i in range(self.dim):
            tmp = 1
            for j in range(self.dim-i-1,0,-1):
                tmp *= self.shape[j]
            self.logic_offset[i] = tmp
        if 'logic_offset' in kwargs:
            self.logic_offset = kwargs['logic_offset']

        # real mem offset
        self.offset =[0 for i in range(self.dim)]
        for i in range(self.dim):
            tmp = self.bitwidth
            for j in range(self.dim-i-1,0,-1):
                tmp *= self.shape[j]
            self.offset[i] = tmp
        if 'offset' in kwargs:
            self.offset = kwargs['offset']


        self.start_offset = 0
        if 'start_offset' in kwargs:
            self.start_offset = kwargs['start_offset']

        self.length = 1
        for i in self.shape:
            self.length *= i

        if 'mem' in kwargs:
            self.mem = kwargs['mem']
        else:
            mem_size = self.length * self.bitwidth
            self.mem = self.core.mem_allocator.get_mem(mem_size,self.bitwidth,self.location)

    def get_addr(self,posi):
        return self.mem.addr + self.get_logic_addr(posi)

    def get_logic_addr(self,posi):
        tmp = self.start_offset
        for i in range(self.dim):
            tmp += posi[i]*self.offset[i]
        return tmp

    def revectorize(self,nvec_shape):
        start = 0
        stop = nvec_shape
        size = nvec_shape*self.bitwidth
        for i in range(math.ceil(self.length / nvec_shape)):
            if stop >self.length:
                stop = self.length
            tmp_mem = self.core.mem_allocator.get_stack_mem(size,self.bitwidth)
            yield self.vectorize(slice(start,stop),tmp_mem)
            self.core.mem_allocator.release_stack_mem(tmp_mem)
            start += nvec_shape
            stop += nvec_shape

    def vectorize(self, section:slice, tmp_mem:mem_entry=None):
        '''
        分两种情况，一种是原生的tensor，另一种是经过划分的tensor
        :param section:
        :param tmp_mem:
        :return:
        '''
        def get_posi(s):
            tmp = s
            posi = [0 for i in range(self.dim)]
            for i in range(self.dim):
                posi[i] = tmp // self.logic_offset[i]
                tmp = tmp%self.logic_offset[i]
            return posi

        start = section.start
        stop = section.stop
        size = (section.stop - section.start) * self.bitwidth
        start_posi = get_posi(start)
        stop_posi = get_posi(stop)

        if not self.sliced:
            cur_start_addr = self.get_addr(get_posi(start))
            tmp_mem = mem_entry(self.core_id,cur_start_addr,size,self.bitwidth,location=self.location)
            return tmp_mem

        # 前面 dim-1维都是相同的
        if start_posi[:self.dim-2] == stop_posi[:self.dim-2]:
            cur_start_addr = self.get_addr(get_posi(start))
            tmp_mem = mem_entry(self.core_id,cur_start_addr,size,self.bitwidth,location=self.location)
            return tmp_mem

        assert tmp_mem, "No tmp mem"
        tmp_mem_addr = tmp_mem.addr
        written_size = 0
        cur_start_addr = self.get_addr(start_posi)
        sec_start = start[self.dim-1]
        while start_posi != stop_posi:
            cur_write_size = (self.shape[self.dim-1] - sec_start) * self.bitwidth
            if written_size+cur_write_size > size:
                cur_write_size = size - written_size
            self.core.inst_buffer.append("mv #{} #{} ${} ${}".format(cur_start_addr,tmp_mem_addr,cur_write_size,1))

            start += self.shape[self.dim-1] - sec_start
            sec_start = 0
            written_size += cur_write_size
            tmp_mem_addr += cur_write_size
            start_posi = get_posi(start)
            cur_start_addr = self.get_addr(start_posi)

        return tmp_mem


    def __getitem__(self, item):
        assert len(item)==self.dim , "dim not equal"
        #check
        for i,s in enumerate(item):
            assert s.start>=0 and s.stop <=self.shape[i] , "dim not equal"

        nshape = [s.stop-s.start for s in item]
        nstart_offset = 0
        for i,s in enumerate(item):
            nstart_offset += s.start*self.offset[i]

        tmp = tensor(self.core_id,nshape,self.bitwidth,self.location,start_offset=nstart_offset,offset=self.offset,sliced=True)
        return tmp

    def view(self,nshape):
        # 使用相同的内存
        # 这个是有错的
        tmp_len = 1
        for i in nshape:
            tmp_len *= i

        if tmp_len<0:
            tmp_len *=-1
            conduct = self.length // tmp_len
            tmp_shape = []
            for i in nshape:
                if i != -1:
                    tmp_shape.append(i)
                else :
                    tmp_shape.append(conduct)
            nshape = tuple(tmp_shape)
        new_tensor = tensor(self.core_id,nshape,self.bitwidth,self.location,mem=self.mem)