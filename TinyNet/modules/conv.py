from TinyNet.functions.act_func import get_act_func
from TinyDSL.DataType.tensor import TensorVar
from TinyDSL.DataType.matrix import MatrixVar
from TinyDSL.DataType.vector import VectorVar,VectorSet
from TinyDSL.Utils.utils import *
from TinyDSL.HwResource.config import core_config
from TinyDSL.HwResource.core import core_allocator



class ConvLayer:
    def __init__(self,conv_config:dict,input_shape,output_shape,misc_config):
        conv_args = ['in_channels', 'out_channels', 'kernel_size', 'stride',
                     'padding', 'group', 'bias', 'activation_func']
        self.conv_args = conv_args
        for arg in conv_args:
            self.__setattr__(arg, conv_config[arg])

        misc_args = ['mat_bitwidth','act_bitwidth']
        for arg in misc_args:
            self.__setattr__(arg,misc_config[arg])


        self.input_shape = input_shape # HWC
        self.output_shape = output_shape
        assert len(self.input_shape)==3 , "HWC not NHWC"

        self.in_act_pad_shape = [self.input_shape[i]+self.padding[i] for i in range(2)]+[self.input_shape[2]]
        self.out_act_shape = self.output_shape

        self.core_config = core_config
        self.meu_ele_rows = self.core_config.meu_rows
        self.meu_ele_columns = (self.core_config.meu_columns*self.core_config.meu_cell_bit)//(self.mat_bitwidth*8)

        # 计算整个权重矩阵的形状
        self.weight_mat_shape = [self.kernel_size[0]*self.kernel_size[1]*self.in_channels,self.out_channels]
        self.rows,self.columns = self.weight_mat_shape

        self.find_core_meu_layout()
        self.allocate()



    def allocate(self):
        # 创建convcore 并分配资源
        self.conv_core_array = [[] for i in range(self.core_layout[0])]

        for i,row_list in enumerate(self.conv_core_array):
            for j in range(self.core_layout[1]):
                misc_config={'in_act_pad_shape':self.in_act_pad_shape,'out_act_shape':self.out_act_shape,'meu_layout':self.meu_layout,
                             'act_bitwidth':self.act_bitwidth,'mat_bitwidth':self.mat_bitwidth,'core_config':self.core_config}
                tmp_posi = [[i*self.core_mat_rows,(i+1)*self.core_mat_rows],
                            [j*self.core_mat_columns,(j+1)*self.core_mat_columns]]
                tmp_core_mat_shape = [self.core_mat_rows,self.core_mat_columns]
                if (i+1)*self.core_mat_rows > self.rows:
                    tmp_core_mat_shape[0] = self.rows - i*self.core_mat_rows
                    tmp_posi[0][1] = self.rows
                if (j+1)*self.core_mat_columns > self.columns:
                    tmp_core_mat_shape[1] = self.core_mat_columns - j*self.core_mat_columns
                    tmp_posi[1][1] = self.columns

                tmp_posi = [slice(tmp_posi[i][0],tmp_posi[i][1]) for i in range(2)]

                misc_config['posi']=tmp_posi
                misc_config['out_act_shape'] = [self.output_shape[0],self.output_shape[1],tmp_core_mat_shape[1]]
                # 上面这个是实际的输出保存的形状，core_mat_rows/columns 都是比较理想的情况
                misc_config['mat_shape'] = tmp_core_mat_shape

                aggregate = False
                if j == self.core_layout[1]-1:
                    aggregate = True
                tmp_core_mat = ConvCore(self.conv_config,misc_config,aggregate)
                row_list.append(tmp_core_mat)





    def find_core_meu_layout(self):
        def get_core_layout(meu_layout):
            core_mat_rows = self.meu_ele_rows * meu_layout[0]
            core_mat_columns = self.meu_ele_columns * meu_layout[1]

            core_rows = math.ceil(self.rows/core_mat_rows)
            core_columns = math.ceil(self.columns/core_mat_columns)
            return (core_rows,core_columns)

        def get_core_cnt(meu_layout):
            r,c = get_core_layout(meu_layout)
            return r*c

        candidate_meu_layout = number_decompose(self.core_config.meu_cnt)

        best_core_cnt = get_core_cnt(candidate_meu_layout[0])
        best_meu_layout = candidate_meu_layout[0]
        for meu_layout in candidate_meu_layout:
            if get_core_cnt(meu_layout) < best_core_cnt:
                best_core_cnt = get_core_cnt(meu_layout)
                best_meu_layout = meu_layout

        self.meu_layout = best_meu_layout
        self.used_core_cnt = best_core_cnt
        self.core_layout = get_core_layout(self.meu_layout)

        #每个核代表的小矩阵的行数和列数，都是理想情况下的，实际的可能有出入
        self.core_mat_rows = self.meu_ele_rows*self.meu_layout[0]
        self.core_mat_columns = self.meu_ele_columns*self.meu_layout[1]
        self.core_mat_shape = (self.core_mat_rows,self.core_mat_columns)



    def recv_act(self, pre_act):
        for row_list in self.conv_core_array:
            for cur_core in row_list:
                cur_core.recv_act(pre_act)



    def compute(self):
        # 注意最后的收集节点
        for i in range(0,self.in_act_pad_shape[0]-self.kernel_size[0]+1,self.stride[0]): # 暂时还没有验证这个范围对不对
            for j in range(0,self.in_act_pad_shape[1]-self.kernel_size[1]+1,self.stride[1]):
                tmp_vec_list = [None for _ in range(self.core_layout[1])]
                for cur_conv_core_list in self.conv_core_array:
                    for t,cur_conv_core in enumerate(cur_conv_core_list):
                        tmp_vec = cur_conv_core.compute(i,j,tmp_vec_list[t])
                        tmp_vec_list[t] = tmp_vec

    def send_act(self):
        pass

    def forward(self,pre_act):
        # 分为三个阶段 首先是接收来自上一层的数据，然后是进行卷积运算，在计算actfunc，最后送到下一层
        # pre_act 可能是function也可能是普通的TensorVar，只要能被assign接收即可
        self.recv_act(pre_act)
        self.compute()
        return self.send_act()




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
            self.__setattr__(arg,misc_config[arg])

        self.aggregate = aggregate # 是否是最后的聚集节点

        self.core_id = core_allocator.get_core()
        self.core = core_allocator.access_core(self.core_id)

        self.allocate()

    def allocate(self):
        # 分配input 和 output feature map 的内存
        self.in_act_ten = TensorVar(self.in_act_pad_shape,self.core_id,self.act_bitwidth,loaction='heap')
        if self.aggregate:
            self.shift_vec = VectorVar(self.columns,self.core_id,4,location='heap')
            # 这里这个输出tensor应该是已经shift完的，就是原始act-bitwidth
            self.out_act_ten = TensorVar(self.out_act_shape,self.core_id,self.act_bitwidth,location='heap')

        # 分配weight matrix 对应的meu

        self.rows,self.columns = self.mat_shape

        self.meu_ele_rows = self.core_config.meu_rows
        self.meu_ele_columns = (self.core_config.meu_columns * self.core_config.meu_cell_bit)//(self.mat_bitwidth*8)

        self.meu_line_rows = self.meu_ele_rows # 多个meu组成一行，可以并行，称为meu_line
        self.meu_line_columns = self.meu_ele_columns * self.meu_layout[1]

        self.meu_map = {} # packet_id to meu_id_list
        self.meu_line_list = [] # MatrixVar ，每一个就是一行，可能有多行
        self.meu_line_posi = []

        for i in range(self.meu_layout[0]):
            tmp_meu_list = self.core.get_meu(self.meu_layout[1])
            self.meu_map[i] = tmp_meu_list # 记录packet_id 到meu 的映射

            # 计算MatrixVar形状,slice 是不可变的
            tmp_posi = [i*self.meu_line_rows,(i+1)*self.meu_line_rows]
            tmp_mat_shape = [self.meu_line_rows,self.meu_line_columns]
            if (i+1)*self.meu_line_rows>self.rows:
                tmp_mat_shape[0] = self.rows - i*self.meu_line_rows
                tmp_posi[1] = self.rows
            if self.meu_line_columns > self.columns:
                tmp_mat_shape[1] = self.columns

            tmp_posi = slice(tmp_posi[0],tmp_posi[1])


            tmp_mat = MatrixVar(tmp_mat_shape,self.core_id,i,tmp_meu_list,self.mat_bitwidth)
            self.meu_line_list.append(tmp_mat)
            self.meu_line_posi.append(tmp_posi)

        # 这里还没有考虑量化，bias，act function的资源资源分配

    def recv_act(self,pre_act_func):
        assert callable(pre_act_func)
        self.in_act_ten.assign(pre_act_func)

    def compute(self,i,j,pre=None):
        '''
        计算一个卷积窗口的情况，pre是上一部分送来的结果
        目前还没有设置 vvset的相关信息
        '''
        vv_set_act_bit = VectorSet(self.core_id,self.act_bitwidth,self.columns)
        vv_set_4bit = VectorSet(self.core_id,4,self.columns) # 向量相加等操作时的bitwidth和length

        window_length = self.kernel_height * self.kernel_width * self.out_channels # 整个卷积窗口的大小
        wc_length = self.kernel_width*self.out_channels # 一个窗口中act连续的大小

        im2col_vec = VectorVar(window_length,self.core_id,self.act_bitwidth)

        with vv_set_act_bit:
            for c in range(self.kernel_height):
                im2col_vec[c * wc_length:(c + 1) * wc_length].copy(self.in_act_ten.get_vec([i+c,j,0],wc_length))
        act_part = im2col_vec[self.posi[0]] # 节选出当前core需要的一个窗口中的部分

        result_vec = VectorVar(self.columns,self.core_id,4) # 计算得到的最终结果
        result_vec.assign(0)
        tmp_vec = VectorVar(self.columns,self.core_id,4)

        with vv_set_4bit:
            for c,cur_mat in enumerate(self.meu_line_list):
                cur_act_part = act_part[self.meu_line_posi[c]]
                tmp_vec.assign(cur_mat*cur_act_part)
                result_vec.assign(result_vec+tmp_vec)
            # 与之前的结果相加
            if pre:
                result_vec.assign(result_vec+pre)

        # 收集节点操作
        # 加入激活函数的部分，收集节点负责这件事
        if self.aggregate:
            with vv_set_4bit:
                shifted_vec = VectorVar(self.columns,self.core_id,self.act_bitwidth)
                shifted_vec.assign(result_vec>>self.shift_vec)

            # relu函数激活
            with vv_set_act_bit:
                func = get_act_func(self.activation_func)
                shifted_vec.assign(func(shifted_vec)) # relu 还没有实现

            # 这里给出的(i,j)都是相对于input的偏移，但是对于output，我们希望得到的是相对这首歌偏移
            with vv_set_act_bit:
                out_i = i//self.stride[0]
                out_j = j//self.stride[1]

                self.out_act_ten.get_vec([out_i,out_j,0],self.columns).assign(shifted_vec)

        return result_vec

    def send_act(self):
        assert self.aggregate

        return self.out_act_ten


