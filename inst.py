import copy


class instruction:
    GEMV = "gemv"
    GVR = "gvr"

    VMV = "vmv"
    VVADD = "vvadd"

    SEND = 'send'
    RECV = 'recv'
    ST = 'st'
    STI = 'sti'
    LDI = "ldi"

    SADD = 'sadd'
    SADDI = 'saddi'

    def __init__(self,op=None,**kwargs):
        self.op = op
        default_value = {'rd':0,'rs1':0,'rs2':0,'bitwidth':0,'imm':0}
        for k,v in default_value.items():
            if k in kwargs:
                self.__setattr__(k,kwargs[k])
            else :
                self.__setattr__(k,v)
        self.funct5 = 0