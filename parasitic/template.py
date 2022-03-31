import torch
from .matrix import *
from .vector import *
from .tensor import *


def conv(mat,act,kernel,out_channels,stride):
    new_act = torch.tensor(5,3)
    kernel_height,kernel_width = kernel
    height,width,channels = act.shape
    for i in range(0,height,stride[0]):
        for j in range(0,width,stride[1]):
            im2col = act[i:i+kernel_height,j:j+kernel_width,:].vectorize()
            ans = mat*im2col
            new_act[i,j,:] = ans


# def conv_v2 (mat,act,kernel,out_channels,stride):
#     new_act = torch.tensor((5,3,2))
#     kernel_height, kernel_width = kernel
#     height, width, channels = act
#
#     i = reg()
#     j = reg()
#     for i in i.range(0,height,stride[0]):
#         for j in j.range(0,width,stride[1]):
#             im2col[12:24] = act([],128)

def conv_v3 (mat:MatrixVar,act:TensorVar,kernel_size,out_channels,stride,output_shape):
    core_id = 0
    bitwidth = 1
    # HWC no batch
    new_act = TensorVar(output_shape,core_id,bitwidth)
    kernel_height, kernel_width = kernel_size
    height, width, in_channels = act.ten_shape

    windows_length = kernel_height*kernel_width*out_channels
    wc_len = kernel_width*in_channels

    for i in range(0,height,stride[0]):
        for j in range(0,width,stride[1]):
            im2col_vec = VectorVar(windows_length,core_id,bitwidth)
            for t in range(kernel_height):
                im2col_vec[t*wc_len:(t+1)*wc_len].copy(act.get_vec([i+t,j,0],wc_len))
            new_act.get_vec([i,j,0],out_channels).assign(mat*im2col_vec)

def concat(ten_list:List[TensorVar],dim):
    def check(a,b):
        assert a == dims
        assert b == dims
        assert a.bitwidth == b.bitwidth

        for i in range(dims):
            if i != dims:
                assert a.ten_shape[i] == b.ten_shape[i]
    # 检查维度的匹配情况
    sample = ten_list[0]
    dims = sample.dim

    for i in range(len(ten_list)-1):
        check(ten_list[i],ten_list[i+1])

    # 计算偏移信息
    outer_cnt = 1
    for d,i in enumerate(sample.ten_shape):
        if d<dim:
            outer_cnt *= i
    # 每个tensor的偏移信息
    inner_len_list = []
    for t in ten_list:
        tmp_len = 1
        for d,i in enumerate(t.ten_shape):
            if d >= dim:
                tmp_len *= i
        inner_len_list.append(tmp_len)

    step_sum = 0
    for i in inner_len_list:
        step_sum += i


    # 使用copy的话似乎就不用区分核内或者核外的传输了，因为copy已经做了这件事
    def gen(result_ten:TensorVar):
        for i in range(outer_cnt):
            cur_sum = 0
            for t,l in zip(ten_list,inner_len_list):
                result_ten.get_vec_offset(i*step_sum+cur_sum,l).copy(
                    t.get_vec_offset(i*l,l)
                )
                cur_sum += l
    return gen


def concat_v2(ten_list:List[TensorVar],dim):
    def check(a,b):
        assert a == dims
        assert b == dims
        assert a.bitwidth == b.bitwidth

        for i in range(dims):
            if i != dims:
                assert a.ten_shape[i] == b.ten_shape[i]
    # 检查维度的匹配情况
    sample = ten_list[0]
    dims = sample.dim

    for i in range(len(ten_list)-1):
        check(ten_list[i],ten_list[i+1])


    def gen(result_ten:TensorVar):
        pass


    return gen

