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