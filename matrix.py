import torch
import math
from .config import *
from .core import *
from .utils import *


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
        self.find_core_map_pattern()

    def find_core_map_pattern(self):
        def get_core_cnt (pattern):
            a,b = self.get_core_layout(pattern)
            return a*b

        self.meu_ele_rows = self.cfg.meu_rows
        self.meu_ele_columns = (self.cfg.meu_columns*self.cfg.meu_cell_bit)//self.bitwidth

        candiate_patterns = number_decompose(self.cfg.meu_cnt)
        best_core_cnt = 100000
        best_pattern = candiate_patterns[0]
        for pattern in candiate_patterns:
            if get_core_cnt(pattern) < best_core_cnt:
                best_core_cnt = get_core_cnt(pattern)
                best_pattern = pattern

        self.map_pattern = best_pattern # 一个core内部meu的map方式
        self.core_cnt = best_core_cnt
        self.core_layout = self.get_core_layout(self.map_pattern) # 整个矩阵对应的核的map方式
        self.sub_matrix_array = [[] for i in range(self.core_layout[1])] # 构建一个和core_list一致的list存储core对应的sub_matrix


    def get_core_layout(self,pattern):
        pattern_rows = self.meu_ele_rows * pattern[0]
        pattern_columns = self.meu_ele_columns * pattern[1]
        # 核在各个方向上的数量
        core_rows = math.ceil(self.rows / pattern_rows)
        core_columns = math.ceil(self.colums / pattern_columns)

        return (core_rows, core_columns)

    # def get_core_cnt(self):
    #     cfg = self.cfg
    #     self.meu_ele_rows = cfg.meu_rows  # 有多少行
    #     self.meu_ele_columns = (cfg.meu_columns * cfg.meu_cell_bit) // self.bitwidth  # 有多少列
    #     if self.map_method == 'row':
    #         self.core_ele_rows = cfg.meu_rows  # 一整个核有多少行
    #         self.core_ele_columns = self.meu_ele_columns * cfg.meu_cnt  # 一个核有多少列
    #     self.core_cnt = (math.ceil(self.rows/self.core_ele_rows),math.ceil(self.colums/self.core_ele_columns))

    def map_to_core(self,core_id_list):
        core_rows,core_columns = self.core_layout

        for i in range(core_rows):
            for j in range(core_columns):
                tmp_rows,tmp_columns = self.map_pattern # 仅meu的情况
                tmp_rows *= self.meu_ele_rows
                tmp_columns *= self.meu_ele_columns

                if (i+1)*tmp_rows > self.rows:
                    tmp_rows = self.rows - i*tmp_rows
                if (j+1)*tmp_columns > self.colums:
                    tmp_columns = self.colums - j*tmp_columns

                tmp_shape = (tmp_rows,tmp_columns)

                tmp_core_id = core_id_list[ i*core_rows + j]
                tmp_sub_matrix = sub_matrix_core(self,(i,j),tmp_shape,tmp_core_id,self.map_pattern)
                tmp_sub_matrix.map_to_meu()

                self.sub_matrix_array[i].append(tmp_sub_matrix)

    def gemv_gen(self):
        pass
        

# posi 这个参数可能是用不到的
# 一个sub_matrix_core对应一个core
class sub_matrix_core(meu_group):
    '''
    physical
    direct map to core
    generate final code
    '''

    def __init__(self,matrix,posi,shape,core_id,map_pattern,**kwargs):
        super(sub_matrix_core, self).__init__()

        self.matrix = matrix
        self.posi = posi # tuple stores positions in virtual matrix
        self.shape = shape # sub matrix shape
        self.core_id = core_id
        self.core = core_allocator[core_id]
        self.map_pattern = map_pattern

        self.rows,self.columns = self.shape
        self.cfg = core_allocator.cfg
        self.meu_ele_rows = self.cfg.meu_rows
        self.meu_ele_columns = (self.cfg.meu_columns*self.cfg.meu_cell_bit)//matrix.bitwidth

        # 实际使用到的meu的pattern,map_pattern 可能出现空着的情况
        self.used_pattern = (math.ceil(self.rows/self.meu_ele_rows),math.ceil(self.columns/self.meu_ele_columns))
        self.meu_array = [[] for i in range (self.used_pattern[1])]

    def map_to_meu(self):
        pattern_rows,patter_columns = self.used_pattern

        for i in range(pattern_rows):
            for j in range(patter_columns):
                tmp_rows,tmp_columns = self.meu_ele_rows,self.meu_ele_columns
                if (i+1)*tmp_rows > self.rows:
                    tmp_rows = self.rows - i*tmp_rows
                if (j+1)*tmp_columns > self.columns:
                    tmp_columns = self.columns - j*tmp_columns
                tmp_shape = (tmp_rows,tmp_columns)

                tmp_meu = self.core.get_meu()
                tmp_meu.map_to_matrix(self,tmp_shape)

                self.meu_array[i].append(tmp_meu)

    def gemv_gen(self):
        pass


