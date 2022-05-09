import copy
import pickle
from TinyDSL.HwResource.isa import FUNCT5_MAP, FUNCT_LIST_MAP,OPCODE_MAP
from TinyDSL.Utils.base import linkList

class instruction:
    GEMV = "gemv"
    GVR = "gvr"

    VMV = "vmv"
    VVADD = "vvadd"
    VVSET = "vvset"
    VVSRL = 'vvsrl'
    VVSLL = 'vvsll'
    VRELU = 'vrelu'
    VSIGMOID = 'vsigmoid'
    VTANH = 'vtanh'
    VVGTM = 'vvgtm'
    VVAND = 'vvand'
    VVSUB = 'vvsub'

    SEND = 'send'
    RECV = 'recv'
    ST = 'st'
    STI = 'sti'
    LDI = "ldi"

    SADD = 'sadd'
    SADDI = 'saddi'

    RD_RS1_RS2 = ['vvadd','vvsub','vvmul','vvgtm','vvgt','vveq','vvand','vvor','vvsrl','vvsra','vvdmul','vmv',
                  'sadd','ssub','smul','sdiv',
                  'st','bind','gemv','gvr']

    RD_RS1 = ['vinvt','vrelu','vsigmoid','vtanh',
              'saddi','ssubi','smuli',
              'ld','sti']

    RD = ['vvset','vrandg','unbind','ldi']

    RS1_RS2 = ['send','recv']

    IMM = ['vmv','saddi','ssubi','smuli','send','recv','sti','bind','ldi'] # unchecked

    BITWIDTH = ['vvset','vvsll','vvsra','bind','gemv']

    def __init__(self,op='none',**kwargs):
        self.op = op
        default_value = {'rd':0,'rs1':0,'rs2':0,'bitwidth':0,'imm':0}
        for k,v in default_value.items():
            if k in kwargs:
                self.__setattr__(k,kwargs[k])
            else :
                self.__setattr__(k,v)
        self.funct5 = 0

    def __str__(self):
        tmp = ''
        for k,v in self.__dict__.items():
            tmp += k+':'+str(v)+'  '
        return tmp

    def dump_binary(self):
        return  BinaryDump(self).dump()

    def dump_asm(self):
        _str = self.op

        reg_str = ''
        if self.op in self.RD_RS1_RS2:
            reg_str = ' rd:'+str(self.rd)+' rs1:'+str(self.rs1)+' rs2:'+str(self.rs2)
        elif self.op in self.RD_RS1:
            reg_str = ' rd:' + str(self.rd) + ' rs1:' + str(self.rs1)
        elif self.op in self.RD:
            reg_str = ' rd:' + str(self.rd)
        elif self.op in self.RS1_RS2:
            reg_str = ' rs1:' + str(self.rs1) + ' rs2:'+str(self.rs2)

        imm_str = ''
        if self.op in self.IMM:
            imm_str = ' imm:'+str(self.imm)

        bitwidth_str = ''
        if self.op in self.BITWIDTH:
            bitwidth_str = ' bitwidth:'+str(self.bitwidth)

        _str += reg_str+imm_str+bitwidth_str
        return _str

    def dump_dict(self):
        default_value = {'op':'none' ,'rd':0,'rs1':0,'rs2':0,'bitwidth':0,'imm':0}

        for k in default_value:
            if k in self.__dict__:
                default_value[k] = self.__dict__[k]
        return default_value


class InstBuffer():
    def __init__(self):
        self.inst_list = []

    def dump_binary(self):
        for i in self.inst_list:
            yield i.dump_binary()

    def dump_asm(self):
        for i in self.inst_list:
            yield i.dump_asm()

    def dump_dict(self):
        for i in self.inst_list:
            yield i.dump_dict()

    def print_dict(self):
        for _str in self.dump_dict():
            print(_str)


    def print_asm(self):
        for _str in self.dump_asm():
            print(_str)

    def print_binary(self):
        for _str in self.dump_binary():
            print(_str)

    def append(self,value,**kwargs):
        self.inst_list.append(value)

    def save_dict(self,path='./tmp.pkl'):
        inst_list =[i for i in self.dump_dict()]
        with open(path,'wb') as f:
            pickle.dump(inst_list,f)

    def save_asm(self,path='./tmp.txt'):
        with open(path,'w') as f:
            for cur in self.inst_list:
                f.write(cur.dump_asm()+'\n')


# class InstBuffer(linkList):
#     def __init__(self):
#         super(InstBuffer, self).__init__()
#
#     def dump_binary(self):
#         cur = self.head.next
#         while cur is not self.tail:
#             yield cur.value.dump_binary()
#             cur = cur.next
#
#     def dump_asm(self):
#         cur = self.head.next
#         while cur is not self.tail:
#             yield cur.value.dump_asm()
#             cur = cur.next
#
#     def dump_dict(self):
#         cur = self.head.next
#         while cur is not self.tail:
#             yield cur.value.dump_dict()
#             cur = cur.next
#
#     def print_dict(self):
#         for _str in self.dump_dict():
#             print(_str)
#
#
#     def print_asm(self):
#         for _str in self.dump_asm():
#             print(_str)
#
#     def print_binary(self):
#         for _str in self.dump_binary():
#             print(_str)
#
#     def append(self,value,**kwargs):
#         super(InstBuffer,self).append(value,**kwargs)
#
#     def save_dict(self,path='./tmp.pkl'):
#         inst_list = self.dump_dict()
#         with open(path,'wb') as f:
#             pickle.dump(inst_list,f)


class BinaryInst:
    # 注意一个问题，高低位反转的问题，我们正常看一个数组是从0到31，首先出现的数是低位，但对于verilog而言是31到0，首先出现的数是高位
    def __init__(self):
        self.inst_array =['0' for _ in range(32)]

    def __setitem__(self, key, value):
        if isinstance(key,int):
            assert isinstance(value,str) and value in ['1','0']
            self.inst_array[key] = value
        elif isinstance(key,slice):
            start = key.start
            stop = key.stop

            assert start >=0 and stop<=32
            assert stop - start == len(value)

            # 注意这里是反转的
            for i,k in enumerate(reversed(value)):
                assert k in ['0','1']
                self.inst_array[start+i] = k

    def __getitem__(self, item):
        if isinstance(item,int):
            return self.inst_array[item]
        elif isinstance(item,slice):
            _str = ''
            # 反转一下 有点别扭，slice给的是小到大，但反的数据是大位到小位
            for i in reversed(self.inst_array[item]):
                _str += i
            return _str

    def dump(self):
        _str = ''
        # 反转一下，第一个数是最高位
        for i in reversed(self.inst_array):
            _str += i
        return _str



class BinaryDump:
    def __init__(self,inst:instruction):
        self.inst = inst
        self.binary = BinaryInst()

    def dump(self):
        func_list = FUNCT_LIST_MAP[self.inst.op]
        self.opcode_dump()
        for f in func_list:
            tmp_func = self.__getattribute__(f)
            tmp_func()
        return self.binary.dump()

    def opcode_dump(self):
        self.binary[0:7] = OPCODE_MAP[self.inst.op]

    def rd_dump(self):
        assert 0<= self.inst.rd <= 31
        self.binary[7:12] = '{:05b}'.format(self.inst.rd)

    def rs1_dump(self):
        assert 0 <= self.inst.rs1 <= 31
        self.binary[12:17] = '{:05b}'.format(self.inst.rs1)

    def rs2_dump(self):
        assert 0 <= self.inst.rs2 <= 31
        self.binary[17:22] = '{:05b}'.format(self.inst.rs2)

    def funct5_dump(self):
        self.binary[22:27] = FUNCT5_MAP[self.inst.op]

    def bitwidth_dump(self):
        # 暂时是表示的几个byte，也可以转换为2^x的表示方式
        self.binary[29:32] = '{:03b}'.format(self.inst.bitwidth)


    # 现在所有的立即数都是没有check过的，因此可能会出现溢出的问题，需要注意
    def imm_v_dump(self):
        self.binary[22:32] = '{:010b}'.format(self.inst.imm)

    def imm_s_dump(self):
        self.binary[17:32] = '{:015b}'.format(self.inst.imm)

    def imm_d_dump(self):
        self.binary[22:32] = '{:010b}'.format(self.inst.imm)

    def imm_l_dump(self):
        self.binary[12:32] = '{:020b}'.format(self.inst.imm)

    def imm_m_dump(self):
        self.binary[22:29] = '{:07b}'.format(self.inst.imm)

