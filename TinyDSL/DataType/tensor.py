# import torch
import copy

from TinyDSL.HwResource.mem import mem_entry
from TinyDSL.DataType.reg import RegVar
from TinyDSL.HwResource.core import core_allocator
from TinyDSL.DataType.vector import VectorVar
from TinyDSL.DataType.frame import frame_stack

# 创建实例必定申请内存
class TensorVar:
    def __init__(self,ten_shape,core_id,bitwidth,**kwargs):
        self.ten_shape = ten_shape
        self.dim = len(self.ten_shape)
        self.core_id = core_id
        self.core = core_allocator.access_core(self.core_id)
        self.bitwidth = bitwidth

        self.length = 1
        for i in self.ten_shape:
            self.length *= i

        self.location = 'stack' # kwargs will check below
        if 'location' in kwargs:
            self.location = kwargs['location']

        if 'mem' in kwargs:
            self.mem = kwargs['mem']
            self.mem_owner = False
        else:
            self.mem = self.core.mem_allocator.get_mem(self.length*self.bitwidth,self.bitwidth,self.location)
            self.mem_owner = True

        if self.mem_owner and self.location == 'stack':
            frame_stack[self.core_id].insert(id(self), self.mem)


        # 对于完全连续的TensorVar设置各类偏移信息
        self.logic_offset = [0 for i in range(self.dim)]
        for i in range(self.dim):
            tmp = 1
            for j in range(self.dim-i-1,0,-1):
                tmp *= self.ten_shape[j]
            self.logic_offset[i] = tmp

        self.mem_offset = [i*self.bitwidth for i in self.logic_offset]
        self.continue_dim = 0 # 在这个及之后的维度都是连续的,这个对应的是数组的标号，有+-1的情况需注意
        self.continue_length = self.length # 这个是内存连续的长度
        self.sliced = False
        self.start_mem_offset = 0 # 起始位置相对于self。mem.addr的偏移情况


        # 对于不连续的情况 读取参数
        para = ['sliced','continue_dim','continue_length','logic_offset','mem_offset','start_mem_offset']

        for p in para:
            if p in kwargs:
                self.__setattr__(p,kwargs[p])

    # 对于sliced的来说可能不太好用，因为地址不是连续的
    def get_addr_reg(self):
        if 'addr_reg' in self.__dict__ :
            return self.addr_reg.reg_id
        tmp = RegVar(self.core_id,imm=self.mem.addr+self.start_mem_offset) # 原始位置加上偏移
        self.__setattr__('addr_reg',tmp)
        return self.addr_reg.reg_id

    def get_length_reg(self):
        if 'length_reg' in self.__dict__ :
            return self.length_reg.reg_id
        tmp = RegVar(self.core_id,imm=self.length)
        self.__setattr__('length_reg',tmp)
        return self.length_reg.reg_id


    def get_vec(self,indice,length): # indice 全部是int型即可
        # 这个length是可能会超长的！！！ 暂时先不检查这个东西了吧，全凭自己操作
        assert len(indice) == self.dim
        vec_mem_offset = self.start_mem_offset # 加入起始的偏移
        for i,cur in enumerate(indice):
            vec_mem_offset += cur*self.mem_offset[i]
        vec_mem_offset += self.mem.addr # offset + start_posi
        new_mem = mem_entry(self.core_id,vec_mem_offset,length*self.bitwidth,self.bitwidth)
        return VectorVar(length,self.core_id,self.bitwidth,mem=new_mem)

    def get_vec_offset(self,offset,length): # 使用一维的绝对偏移获得位置
        def offset_to_indice(offset):
            indice = [0 for i in range(self.dim)]
            for i in range(self.dim):
                indice[i] = offset // self.logic_offset[i]
                offset = offset%self.logic_offset[i]
            return indice
        indice = offset_to_indice(offset)
        return self.get_vec(indice,length)

        # vec_mem_offset = offset*self.bitwidth
        # vec_mem_offset += self.mem.addr
        # new_mem = mem_entry(self.core_id,vec_mem_offset,length*self.bitwidth,self.bitwidth)
        # return VectorVar(length,self.core_id,self.bitwidth,mem=new_mem)

    # 可以考虑加一个更底层的memcpy函数，可以放在类外
    def copy(self,src_ten):
        assert self.ten_shape == src_ten.ten_shape # 形状不同不可直接赋值

        c_dim = max(self.continue_dim,src_ten.continue_dim)
        c_length = min(self.continue_length,src_ten.continue_length)

        assert self.length % c_length == 0

        # 使用vector进行通信
        for l in range(0,self.length,c_length):
            self.get_vec_offset(l,c_length).copy(src_ten.get_vec_offset(l,c_length))

        # if self.core_id == src_ten.core_id:
        #
        #     inst = instruction(instruction.VMV,rd=self.get_addr_reg(),rs1=src_ten.get_addr_reg(),rs2=self.get_length_reg())
        #     self.core.inst_buffer.append(inst)
        # else:
        #     send_inst = instruction(instruction.SEND,rs1=src_ten.get_addr_reg(),rs2=src_ten.get_length_reg(),imm=self.core_id)
        #     recv_inst = instruction(instruction.RECV,rs1=self.get_addr_reg(),rs2=self.get_length_reg(),imm=src_ten.core_id)
        #
        #     self.core.inst_buffer.append(recv_inst)
        #     src_ten.core.inst_buffer.append(send_inst)

    def assign(self,item):
        if callable(item):
            item(self) # 这里出现的函数不是加减乘除返回的函数，应该是concat之类的操作返回的函数
        elif isinstance(item,TensorVar):
            self.copy(item)

    def __getitem__(self, item):
        assert len(item) == self.dim
        nlist = []
        for i,s in enumerate(item):
            if isinstance(s,int):
                s = slice(s,s+1)
            elif isinstance(s,slice):
                if not s.start:
                    s.start = 0
                if not s.stop:
                    s.stop = self.ten_shape[i]
            nlist.append(s)
            assert s.start>=0 and s.stop <=self.ten_shape[i] , "dim not equal"

        item = nlist

        nshape = [s.stop - s.start for s in item]
        nstart_offset = self.start_mem_offset
        for i, s in enumerate(item):
            nstart_offset += s.start * self.mem_offset[i]

        reverse_item = copy.deepcopy(item)
        reverse_item.reverse()
        c_dim = 0
        for i,s in enumerate(reverse_item):
            if s.stop-s.start != self.ten_shape[self.dim-i-1]:
                c_dim = self.dim-i-1
        c_length = 1
        for i in range(c_dim,self.dim):
            c_length *= self.ten_shape[i] # 连续的话就和原本的一致了

        # 对于原本就分割过的tensor，还需要判断是否超了原先tensor的长度
        if c_dim > self.continue_dim :
            c_dim = self.continue_dim
            c_length = self.continue_length


        tmp = TensorVar(nshape,self.core_id,self.bitwidth,
                        sliced=True,continue_dim=c_dim,continue_length=c_length,location=self.location,
                        mem_offset=self.mem_offset,start_mem_offset=nstart_offset,mem=self.mem)

        return tmp



    def __del__(self):
        try:
            if self.location == 'stack':
                if self.mem_owner:
                    frame_stack[self.core_id].release(id(self))
        except Exception:
            import traceback,sys
            traceback.print_exc()
