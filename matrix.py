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
        
        self.sub_matrix_array = []


    def simple_split(self,cfg:core_cfg,map_method='row'):
        '''
        simple split strategy
        :return:
        '''

        setattr(self,'map_method',map_method)
        meu_ele_rows = cfg.meu_rows # 有多少行
        meu_ele_columns = (cfg.meu_columns*cfg.meu_cell_bit)//self.bitwidth # 有多少列

        if map_method == 'row':
            core_ele_rows = cfg.meu_rows # 一整个核有多少行
            core_ele_columns = meu_ele_columns * cfg.meu_cnt # 一个核有多少列
        elif map_method == 'column':
            core_ele_rows = meu_ele_rows * cfg.meu_cnt
            core_ele_columns = meu_ele_columns


        if map_method == 'row':
            for i in range(math.ceil(self.rows/core_ele_rows)):
                tmp = []
                for j in range(math.ceil(self.colums/core_ele_columns)):
                    sub_matrix_shape = [meu_ele_rows,meu_ele_columns]
                    if (i+1)*core_ele_rows > self.rows:
                        sub_matrix_shape[0] = self.rows - i*core_ele_rows
                    if (j+1)*core_ele_columns > self.colums:
                        sub_matrix_shape[1] = self.colums - i*core_ele_columns
                    core_id,c = core_allocator.get_core()
                    cm = core_matrix(self,(i*core_ele_rows,j*core_ele_columns),sub_matrix_shape,core_id,map_method,core_cfg)
                    cm.map_to_core()
                    tmp.append(cm)
                self.sub_matrix_array.append(tmp)

        # todo core allocator
        


class core_matrix:
    '''
    physical
    direct map to core
    generate final code
    '''

    def __init__(self,matrix,posi,shape,core_id,map_method,cfg:core_cfg,**kwargs):
        self.matrix = matrix
        self.posi = posi # tuple stores positions in virtual matrix
        self.shape = shape # sub matrix shape
        self.core_id = core_id
        self.core = core_allocator[core_id]
        self.map_method = map_method

        self.rows,self.columns = self.shape
        self.cfg = cfg
        self.meu_ele_rows = cfg.meu_rows
        self.meu_ele_columns = (cfg.meu_columns*cfg.meu_cell_bit)//matrix.bitwidth
        self.meu_array = []

    def map_to_core(self):
        if self.map_method == 'row':
            assert self.rows <= self.meu_ele_rows
            meu_cnt = math.ceil(self.columns/self.meu_ele_columns)
            for i in range(meu_cnt):
                meu_shape = [self.rows,self.meu_ele_columns]
                if (i+1)*self.meu_ele_columns > self.columns:
                    meu_shape[1] = self.columns - i*self.meu_ele_columns
                tmp_meu_posi = (self.posi[0],self.posi[1]+i*self.meu_ele_columns)
                tmp_meu = self.core.get_meu()
                tmp_meu.map_to_matrix(self.matrix,tmp_meu_posi,meu_shape)
                self.meu_array.append(tmp_meu)

    def gemv_gen(self):
        pass


