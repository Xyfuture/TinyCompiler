import copy

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
        pass

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
            bitwidth_str = ' bitwidth'+str(self.bitwidth)

        _str += reg_str+imm_str+bitwidth_str
        return _str





class instBuffer(linkList):
    def __init__(self):
        super(instBuffer, self).__init__()

    def dump_binary(self):
        pass

    def dump_asm(self):
        cur = self.head.next
        while cur is not self.tail:
            print(cur.value.dump_asm())
            cur = cur.next

