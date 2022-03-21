import torch
import math
from .config import *
from core import *

class matrix:
    '''
    matrix just 2d-tensor
    virtual concept
    '''
    def __init__(self,shape,bitwidth,**kwargs):
        self.shape = shape
        self.bitwidth = bitwidth

        self.rows,self.colums = self.shape


    def simple_split(self,cfg:core_cfg,map_method='row'):
        '''
        simple split strategy
        :return:
        '''

        ele_per_column = cfg.meu_rows
        ele_per_row = (cfg.meu_columns*cfg.meu_cell_bit)//self.bitwidth

        if map_method == 'row':
            ele_core_column = cfg.meu_rows
            ele_core_row = ele_per_row * cfg.meu_cnt
        elif map_method == 'column':
            ele_core_column = ele_per_column * cfg.meu_cnt
            ele_core_row = ele_per_row

        # todo core allocator



class core_matrix:
    '''
    physical
    direct map to core
    generate final code
    '''

    def __init__(self,matrix,posi,shape,core_id,map_method,**kwargs):
        self.matrix = matrix
        self.posi = posi # tuple stores positions in virtual matrix
        self.shape = shape # sub matrix shape
        self.core_id = core_id
        self.map_method = map_method


    def map_to_core(self):
        pass


