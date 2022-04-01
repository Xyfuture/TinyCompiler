import torch
from ..core import core_allocator
from reg import *

'''
vector 最重要的区别是存有一个addr_reg和一个length_reg，这使得这个万一直接支持参与指令的运算
'''
class VectorVar:

    VVSET_BITWIDTH = {} # 两个dict，core_id:value
    VVSET_LENGTH = {}

    def __init__(self,vec_shape,core_id,bitwidth,**kwargs):
        # shape 就是length，只是为了统一名字
        self.vec_shape = vec_shape
        self.core_id = core_id
        self.core = core_allocator.access_core(self.core_id)
        self.bitwidth = bitwidth

        self.location = 'stack'
        if 'location' in kwargs:
            self.location = kwargs['location']

        if 'mem' in kwargs:
            self.mem = kwargs['mem']
            self.mem_owner = False
        else :
            self.mem = self.core.mem_allocator.get_mem(self.vec_shape*self.bitwidth,self.bitwidth,location=self.location)
            self.mem_owner = True




    def get_addr_reg(self):
        if 'addr_reg' in self.__dict__ :
            return self.addr_reg.reg_id
        tmp = RegVar(self.core_id,imm=self.mem.addr)
        self.__setattr__('addr_reg',tmp)
        return self.addr_reg.reg_id

    def get_length_reg(self):
        if 'length_reg' in self.__dict__ :
            return self.length_reg.reg_id
        tmp = RegVar(self.core_id,imm=self.vec_shape)
        self.__setattr__('length_reg',tmp)
        return self.length_reg.reg_id

    def check_vvset(self):
        assert VectorVar.VVSET_LENGTH == self.vec_shape , "vvset:length not equal"
        assert VectorVar.VVSET_BITWIDTH == self.bitwidth , "vvset:bitwidth not equal"

    def copy(self,src_vec): # 两个vec 之间直接复制
        # assert self.bitwidth*self.vec_shape == src_vec.bitwidth * src_vec.shape
        assert self.mem.size == src_vec.mem.size
        if (self.core_id == src_vec.core_id):
            assert self.mem != src_vec.mem # 本身是同一块就不用复制了
            inst = instruction(instruction.VMV,rd=self.get_addr_reg(),rs1=src_vec.get_addr_reg(),rs2=self.get_length_reg())
            self.core.inst_buffer.append(inst)
        else:
            # 跨核之间的通信
            send_inst = instruction(instruction.SEND,rs1=src_vec.get_addr_reg(),rs2=src_vec.get_length_reg(),imm=self.core_id)
            recv_inst = instruction(instruction.RECV,rs1=self.get_addr_reg(),rs2=self.get_length_reg(),imm=src_vec.core_id)

            self.core.inst_buffer.append(recv_inst)
            src_vec.core.inst_buffer.append(send_inst)

    def assign(self,item):
        # 相当于等于号，可能会传进一个函数，或者一个VectorVar用于复制
        # 传入一个寄存器或者立即数都用于初始化赋值
        if callable(item):
            item(self)
        elif isinstance(item,int):
            inst=instruction(instruction.STI,rd=self.get_addr_reg(),rs1=self.get_length_reg(),imm=item)
            self.core.inst_buffer.append(inst)
        elif isinstance(item,RegVar):
            inst=instruction(instruction.ST,rd=self.get_addr_reg(),rs1=item.reg_id,rs2=self.get_length_reg())
            self.core.inst_buffer.append(inst)
        elif isinstance(item,VectorVar):
            self.copy(item)



    def __getitem__(self, item): # 提出部分来用于复制等操作 or 赋值给寄存器等等
        if isinstance(item,slice):
            start = item.start
            stop = item.stop
            assert stop<=self.vec_shape
            new_mem_addr = self.mem.addr + start*self.bitwidth
            new_vec_shape = stop-start
            new_mem = mem_entry(self.core_id,new_mem_addr,new_vec_shape*self.bitwidth,self.bitwidth)
            return VectorVar(new_vec_shape,self.core_id,self.bitwidth,mem=new_mem)
        elif isinstance(item,int):
            #  for register
            pass

    def __add__(self, other):
        assert self.core_id == other.core_id

        inst = instruction(instruction.VVADD,rs1=self.get_addr_reg(),rs2=other.get_addr_reg())
        def gen(result_vec):
            # check vvest length and bitwidth
            self.check_vvset()
            other.check_vvset()
            result_vec.check_vvset()

            inst.rd = result_vec.get_addr_reg()
            self.core.inst_buffer.append(inst)
        return gen

    def __radd__(self, other):
        return self.__add__(other)

    def __rshift__(self, other):
        pass



    def __del__(self):
        pass



class VectorSet:
    # 类本身不占有资源，不用释放资源
    def __init__(self,core_id,bitwidth,length):
        self.core_id = core_id
        self.core = core_allocator.access_core(self.core_id)
        self.bitwidth = bitwidth
        self.length = length

        self.old_bitwidth = -1
        self.old_length = -1

    def __enter__(self):
        self.set()
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.reset()
        # 不返回true就会正常抛出异常
        pass

    def set(self):

        if self.bitwidth == VectorVar.VVSET_BITWIDTH.get(self.core_id) and self.length == VectorVar.VVSET_LENGTH.get(self.core_id):
            return
        if self.core_id in VectorVar.VVSET_LENGTH:
            self.old_bitwidth = VectorVar.VVSET_BITWIDTH[self.core_id]
            self.old_length = VectorVar.VVSET_LENGTH[self.core_id]

        VectorVar.VVSET_BITWIDTH[self.core_id] = self.bitwidth
        VectorVar.VVSET_LENGTH[self.core_id] = self.length

        tmp = RegVar(self.core_id, imm=self.length)
        inst = instruction(instruction.VVSET,rd=tmp.reg_id,bitwidth=self.bitwidth)
        self.core.inst_buffer.append(inst)

    def reset(self): # may not use
        VectorVar.VVSET_BITWIDTH[self.core_id] = self.old_bitwidth
        VectorVar.VVSET_LENGTH[self.core_id] = self.old_length

        tmp = RegVar(self.core_id, imm=self.old_length)
        inst = instruction(instruction.VVSET,rd=tmp.reg_id,bitwidth=self.bitwidth)
        self.core.inst_buffer.append(inst)

