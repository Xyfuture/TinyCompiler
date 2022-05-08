import math

class core_cfg:
    def __init__(self):
        self.meu_cell_bit = 2
        self.meu_rows = 512
        self.meu_columns = 512

        self.meu_cnt = 8
        self.omu_size = 1024*1024*2

        self.reg_cnt = 32
        # self.core_cnt = 128


    def get_config(self):
        pass

# 临时使用，之后需要改一下
core_config = core_cfg()