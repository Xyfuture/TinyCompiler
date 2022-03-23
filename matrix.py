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
        self.map_method = 'row'
        if 'map_method' in kwargs:
            self.map_method = kwargs['map_method']
        self.cfg = core_allocator.cfg
        self.get_core_cnt()
        self.sub_matrix_array = []

    def get_core_cnt(self):
        cfg = self.cfg
        self.meu_ele_rows = cfg.meu_rows  # 有多少行
        self.meu_ele_columns = (cfg.meu_columns * cfg.meu_cell_bit) // self.bitwidth  # 有多少列
        if self.map_method == 'row':
            self.core_ele_rows = cfg.meu_rows  # 一整个核有多少行
            self.core_ele_columns = self.meu_ele_columns * cfg.meu_cnt  # 一个核有多少列
        self.core_cnt = (math.ceil(self.rows/self.core_ele_rows),math.ceil(self.colums/self.core_ele_columns))

    def map_to_core(self,core_id_list):
        '''
        simple split strategy
        :return:
        '''
        if self.map_method == 'row':
            for i in range(math.ceil(self.rows/self.core_ele_rows)):
                tmp = []
                for j in range(math.ceil(self.colums/self.core_ele_columns)):
                    sub_matrix_shape = [self.core_ele_rows,self.core_ele_columns]
                    if (i+1)*self.core_ele_rows > self.rows:
                        sub_matrix_shape[0] = self.rows - i*self.core_ele_rows
                    if (j+1)*self.core_ele_columns > self.colums:
                        sub_matrix_shape[1] = self.colums - i*self.core_ele_columns
                    cm = core_matrix(self,(i*self.core_ele_rows,j*self.core_ele_columns),sub_matrix_shape,core_id_list[i][j],self.map_method)
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

    def __init__(self,matrix,posi,shape,core_id,map_method,**kwargs):
        self.matrix = matrix
        self.posi = posi # tuple stores positions in virtual matrix
        self.shape = shape # sub matrix shape
        self.core_id = core_id
        self.core = core_allocator[core_id]
        self.map_method = map_method

        self.rows,self.columns = self.shape
        self.cfg = core_allocator.cfg
        self.meu_ele_rows = self.cfg.meu_rows
        self.meu_ele_columns = (self.cfg.meu_columns*self.cfg.meu_cell_bit)//matrix.bitwidth
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


