# 主要涉及ISA的定义和二进制(文本形式)指令的转换

# from TinyDSL.HwResource.inst import instruction

FUNCT5_MAP = {
    'vvadd':'00001',
    'vvsub':'00010',
    'vvmul':'00011',
    'vvgtm':'10001',
    'vvgt':'10010',
    'vveq':'10011',
    'vvand':'10100',
    'vvor':'10101',

    'vvsll':'00001',
    'vvsra':'00010',

    'vrelu':'00001',
    'vsigmoid':'00010',
    'vtanh':'00011',

    'sadd':'00001',
    'ssub':'00010',
    'smul':'00011',
    'sidv':'00100'
}

OPCODE_MAP ={
    "vvset":'0010001',
    "vvadd":'0010010',
    "vvsub":'0010010',
    "vvmul":'0010010',
    "vvgtm":'0010010',
    "vvgt":'0010010',
    "vveq":'0010010',
    "vvand":'0010010',
    "vvor":'0010010',
    "vvsll":'0010011',
    "vvsra":'0010011',
    "vvdmul":'0010100',
    "vinvt":'0010101',
    "vrandg":'0010111',
    "vrelu":'0011000',
    "vsigmoid":'0011000',
    "vtanh":'0011000',
    "vmv":'0011001',
    "sadd":'0100001',
    "ssub":'0100001',
    "smul":'0100001',
    "sdiv":'0100001',
    "saddi":'0100010',
    "ssubi":'0100011',
    "smuli":'0100100',
    "send":'0110001',
    "recv":'0110010',
    "ld":'0110011',
    "st":'0110100',
    "sti":'0110101',
    "ldi":'0110110',
    "bind":'1000001',
    "unbind":'1000010',
    "gemv":'1000011',
    "gvr":'1000100'
}


FUNCT_LIST_MAP = {
    "vvset":['rd_dump','bitwidth_dump'],
    "vvadd":['rd_dump','rs1_dump','rs2_dump','funct5_dump'],
    "vvsub":['rd_dump','rs1_dump','rs2_dump','funct5_dump'],
    "vvmul":['rd_dump','rs1_dump','rs2_dump','funct5_dump'],
    "vvgtm":['rd_dump','rs1_dump','rs2_dump','funct5_dump'],
    "vvgt":['rd_dump','rs1_dump','rs2_dump','funct5_dump'],
    "vveq":['rd_dump','rs1_dump','rs2_dump','funct5_dump'],
    "vvand":['rd_dump','rs1_dump','rs2_dump','funct5_dump'],
    "vvor":['rd_dump','rs1_dump','rs2_dump','funct5_dump'],
    "vvsll":['rd_dump','rs1_dump','rs2_dump','funct5_dump','bitwidth_dump'],
    "vvsra":['rd_dump','rs1_dump','rs2_dump','funct5_dump','bitwidth_dump'],
    "vvdmul":['rd_dump','rs1_dump','rs2_dump'],
    "vinvt":['rd_dump','rs1_dump'],
    "vrandg":['rd_dump'],
    "vrelu":['rd_dump','rs1_dump','funct5_dump'],
    "vsigmoid":['rd_dump','rs1_dump','funct5_dump'],
    "vtanh":['rd_dump','rs1_dump','funct5_dump'],
    "vmv":['rd_dump','rs1_dump','rs2_dump','imm_v_dump'],
    "sadd":['rd_dump','rs1_dump','rs2_dump','funct5_dump'],
    "ssub":['rd_dump','rs1_dump','rs2_dump','funct5_dump'],
    "smul":['rd_dump','rs1_dump','rs2_dump','funct5_dump'],
    "sdiv":['rd_dump','rs1_dump','rs2_dump','funct5_dump'],
    "saddi":['rd_dump','rs1_dump','imm_s_dump'],
    "ssubi":['rd_dump','rs1_dump','imm_s_dump'],
    "smuli":['rd_dump','rs1_dump','imm_s_dump'],
    "send":['rs2_dump','rs1_dump','imm_d_dump'],
    "recv":['rs2_dump','rs1_dump','imm_d_dump'],
    "ld":['rd_dump','rs1_dump'],
    "st":['rd_dump','rs1_dump','rs2_dump'],
    "sti":['rd_dump','rs1_dump'],
    "ldi":['rd_dump','imm_l_dump'],
    "bind":['rd_dump','rs1_dump','rs2_dump','imm_m_dump','bitwidth_dump'],
    "unbind":['rd_dump'],
    "gemv":['rd_dump','rs1_dump','rs2_dump','bitwidth_dump'],
    "gvr":['rd_dump','rs1_dump','rs2_dump']
    }







