import torch
from ...core import core_allocator
from ..tensor import TensorVar
from ..matrix import MatrixVar
from ..vector import VectorVar


class ConvLayer:
    def __init__(self,conv_config:dict,input_shape,output_shape):
        conv_args = ['in_channels', 'out_channels', 'kernel_size', 'stride',
                     'padding', 'group', 'bias', 'activation_func']
        for arg in conv_args:
            self.__setattr__(arg, conv_config[arg])

        self.input_shape = input_shape # HWC
        self.output_shape = output_shape
        assert len(self.input_shape)==3 , "HWC not NHWC"

        self.in_act_pad_shape = [self.input_shape[i]+self.padding[i] for i in range(2)]+[self.input_shape[2]]
        self.out_act_shape = self.output_shape

        self.core_config = core_allocator.cfg

    def allocate(self):
        pass



    def forward(self,act_ten):
        # 分为三个阶段 首先是接收来自上一层的数据，然后是进行卷积运算，在计算actfunc，最后送到下一层

        pass



# 单个核对应的权重和input
class ConvCore:
    def __init__(self,conv_config:dict,misc_config:dict,aggregate=False):
        conv_args = ['in_channels','out_channels','kernel_size','stride',
                     'padding','group','bias','activation_func']
        for arg in conv_args:
            self.__setattr__(arg,conv_config[arg])

        self.kernel_height,self.kernel_width = self.kernel_size


        # meu_layout 是一个二元组，描述meu 的分布，有几行，每一行有几个meu
        # mat_shape 是真正映射到一个core上矩阵的形状。具体映射到meu上还需要进一步的分割
        # posi 是该core在大矩阵中的相对位置，是一个二元组，第一个表示在行上的起始位置（tuple），第二个是列上的
        # posi 设置为slice的list
        misc_args = ['in_act_pad_shape','out_act_shape','meu_layout','mat_shape',
                     'act_bitwidth','mat_bitwidth','core_config','posi']
        for arg in misc_args:
            self.__setattr__(arg,misc_args)

        self.aggregate = aggregate # 是否是最后的聚集节点

        self.core_id = core_allocator.get_core()
        self.core = core_allocator.access_core(self.core_id)

        self.allocate()

    def allocate(self):
        # 分配input 和 output feature map 的内存
        self.in_act_ten = TensorVar(self.in_act_pad_shape,self.core_id,self.act_bitwidth,loaction='heap')
        if self.aggregate:
            self.out_act_ten = TensorVar(self.out_act_shape,self.core_id,self.act_bitwidth,location='heap')
        # 分配weight matrix 对应的meu

        self.rows,self.columns = self.mat_shape

        self.meu_ele_rows = self.core_config.meu_rows
        self.meu_ele_columns = (self.core_config.meu_columns * self.core_config.meu_cell_bit)//self.mat_bitwidth

        self.meu_line_rows = self.meu_ele_rows # 多个meu组成一行，可以并行，称为meu_line
        self.meu_line_columns = self.meu_ele_columns * self.meu_layout[1]

        self.meu_map = {} # packet_id to meu_id_list
        self.meu_line_list = [] # MatrixVar ，每一个就是一行，可能有多行
        self.meu_line_posi = []

        for i in range(self.meu_layout[0]):
            tmp_meu_list = self.core.get_meu(self.meu_layout[1])
            self.meu_map[i] = tmp_meu_list # 记录packet_id 到meu 的映射

            # 计算MatrixVar形状
            tmp_posi = slice(i*self.meu_line_rows,(i+1)*self.meu_line_rows)
            tmp_mat_shape = [self.meu_line_rows,self.meu_line_columns]
            if (i+1)*self.meu_line_rows>self.rows:
                tmp_mat_shape[0] = self.rows - i*self.meu_line_rows
                tmp_posi.stop = self.rows
            if self.meu_line_columns > self.columns:
                tmp_mat_shape[1] = self.columns

            tmp_mat = MatrixVar(tmp_mat_shape,self.core_id,i,tmp_meu_list,self.mat_bitwidth)
            self.meu_line_list.append(tmp_mat)
            self.meu_line_posi.append(tmp_posi)

        # 这里还没有考虑量化，bias，act function的资源资源分配

    def im2col(self):
        pass


    def compute(self,i,j,pre=None):
        '''
        计算一个卷积窗口的情况，pre是上一部分送来的结果
        '''
        window_length = self.kernel_height * self.kernel_width * self.out_channels # 整个卷积窗口的大小
        wc_length = self.kernel_width*self.out_channels # 一个窗口中act连续的大小

        im2col_vec = VectorVar(window_length,self.core_id,self.act_bitwidth)
        for c in range(self.kernel_height):
            im2col_vec[c * wc_length:(c + 1) * wc_length].copy(self.in_act_ten.get_vec([c,j,0],wc_length))
        act_part = im2col_vec[self.posi[0]] # 节选出当前core需要的一个窗口中的部分

        result_vec = VectorVar(self.columns,self.core_id,4) # 计算得到的最终结果
        result_vec.assign(0)
        tmp_vec = VectorVar(self.columns,self.core_id,4)
        for c,cur_mat in enumerate(self.meu_line_list):
            cur_act_part = act_part[self.meu_line_posi[c]]
            tmp_vec.assign(cur_mat*cur_act_part)
            result_vec.assign(result_vec+tmp_vec)

        return result_vec





        pass



