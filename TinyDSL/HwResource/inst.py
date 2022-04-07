import copy

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

    RD_RS1_RS2 = ['vvadd','vvsub','vvmul','vvgtm','vvgt','vveq','vvand','vvor','vvsll''vvsra','vvdmul','vmv'
                  'sadd','ssub','smul','sdiv',
                  'st','bind','gemv','gvr']

    RD_RS1 = ['vinvt','vrelu','vsigmoid','vtanh',
              'saddi','ssubi','smuli',
              'ld','sti']

    RD = ['vvset','vrandg','unbind','ldi']

    RS1_RS2 = ['send','recv']

    IMM = ['vmv','saddi','ssubi','smuli','send','recv','sti','bind','ldi'] # unchecked

    BITWIDTH = ['vvset','vvsll','vvsra','bind','gemv']

    def __init__(self,op=None,**kwargs):
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
            reg_str = ' rd:'+str(self.rd)+' rs1:'+str(self.rs1)+'rs2:'+str(self.rs2)
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





class instBuffer(linkList):
    def __init__(self):
        super(instBuffer, self).__init__()

    def dump_binary(self):
        cur = self.head.next
        while cur is not self.tail:
            print(cur.value.dump_binary())
            cur = cur.next

    def dump_asm(self):
        cur = self.head.next
        while cur is not self.tail:
            print(cur.value.dump_asm())
            cur = cur.next


class BinaryInst:
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

            for i,k in enumerate(value):
                assert k in ['0','1']
                self.inst_array[start+i] = k

    def __getitem__(self, item):
        if isinstance(item,int):
            return self.inst_array[item]
        elif isinstance(item,slice):
            _str = ''
            for i in self.inst_array[item]:
                _str += i
            return _str

    def dump(self):
        _str = ''
        for i in self.inst_array:
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

