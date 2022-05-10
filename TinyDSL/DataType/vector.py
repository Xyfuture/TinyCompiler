import sys,os

from TinyDSL.DataType.reg import RegVar
from TinyDSL.HwResource.core import core_allocator
from TinyDSL.HwResource.inst import instruction
from TinyDSL.HwResource.mem import mem_entry
from TinyDSL.DataType.frame import frame_stack

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

        if self.vec_shape < 0:
            print('here')


        self.location = 'stack'
        if 'location' in kwargs:
            self.location = kwargs['location']

        if 'mem' in kwargs:
            self.mem = kwargs['mem']
            self.mem_owner = False
        else :
            self.mem = self.core.mem_allocator.get_mem(self.vec_shape*self.bitwidth,self.bitwidth,location=self.location)
            self.mem_owner = True

        if self.mem_owner and self.location == 'stack':
            frame_stack[self.core_id].insert(id(self), self.mem)


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
        assert VectorVar.VVSET_LENGTH[self.core_id] == self.vec_shape , "vvset:length not equal"
        assert VectorVar.VVSET_BITWIDTH[self.core_id] == self.bitwidth , "vvset:bitwidth not equal"

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

    def __del__(self):
        # 只有在内存是栈内存，且内存属于自己的情况下才会删除
        try:
            if self.location == 'stack':
                if self.mem_owner:
                    frame_stack[self.core_id].release(id(self),self)
        except Exception:
            import traceback,sys
            traceback.print_exc()

    #  为了vector-vector指令设计两个用于指令生成的函数
    def vv_inst_check_set(self,other,inst_op):
        # 保证在同一个核上进行运算
        assert self.core_id == other.core_id

        # 检查是否能够只能vv操作
        self.check_vvset()
        other.check_vvset()

        inst = instruction(inst_op,rs1=self.get_addr_reg(),rs2=other.get_addr_reg())
        return inst

    def vv_inst_set_result(self,inst,result_vec):
        # 检查核id和vvset
        assert self.core_id == result_vec.core_id
        result_vec.check_vvset()

        inst.rd = result_vec.get_addr_reg()
        self.core.inst_buffer.append(inst)


    def __add__(self, other):

        def gen(result_vec):
            inst = self.vv_inst_check_set(other,instruction.VVADD)
            self.vv_inst_set_result(inst,result_vec)

        return gen

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        def gen(result_vec):
            inst = self.vv_inst_check_set(other, instruction.VVSUB)
            self.vv_inst_set_result(inst,result_vec)

        return gen

    def __rsub__(self, other):
        return self.__sub__(other)

    def __lshift__(self, other):

        def gen(result_vec):
            inst = self.vv_inst_check_set(other,instruction.VVSLL)

            assert self.core_id == result_vec.core_id
            inst.rd = result_vec.get_addr_reg()
            inst.bitwidth = result_vec.bitwidth()

            self.core.inst_buffer.append(inst)
        return gen

    def __rshift__(self, other):

        def gen(result_vec):
            inst = self.vv_inst_check_set(other, instruction.VVSRL)
            # 这里稍微有点却别，因为这个的数据宽度是单独设计的
            # self.vv_inst_set_result(inst,result_vec)
            assert self.core_id == result_vec.core_id
            inst.rd = result_vec.get_addr_reg()
            inst.bitwidth = result_vec.bitwidth

            self.core.inst_buffer.append(inst)

        return gen

    def __and__(self, other):

        def gen(result_vec):
            inst = self.vv_inst_check_set(other, instruction.VVAND)
            self.vv_inst_set_result(inst,result_vec)

        return gen

    def __mul__(self, other):

        def gen(result_vec):
            inst = self.vv_inst_check_set(other,instruction.VVMUL)
            self.vv_inst_set_result(inst,result_vec)

        return gen


    # 激活函数
    def activation_func(self,func_name='relu'):
        self.check_vvset()

        func_op = instruction.VRELU
        if func_name == 'relu':
            func_op = instruction.VRELU
        elif func_name == 'sigmoid':
            func_op = instruction.VSIGMOID
        elif func_name == 'tanh':
            func_op = instruction.VTANH

        inst = instruction(func_op, rs1=self.get_addr_reg())
        def gen(result_vec):
            self.vv_inst_set_result(inst,result_vec) # 恰好操作相同，借用一下

        return gen




    @classmethod
    def gtm(cls, vec_a, vec_b):
        assert vec_a.core_id == vec_b.core_id
        vec_a.check_vvset()
        vec_b.check_vvset()

        core = vec_a.core
        def gen(result_vec):
            assert result_vec.core_id == vec_a.core_id
            result_vec.check_vvset()
            # 避免临时变量被gc
            inst = instruction(instruction.VVGTM, rs1=vec_a.get_addr_reg(),
                               rs2=vec_b.get_addr_reg())
            inst.rd = result_vec.get_addr_reg()
            core.inst_buffer.append(inst)
        return gen






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
        # print(exc_type,exc_val,exc_tb)
        return False # 不在这里面异常，直接把异常抛出去

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

    # def __del__(self):
    #     print('del')