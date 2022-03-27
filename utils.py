import torch
import torch.nn
from .tensor import *
import math


def gen_vtensor(t:torch.Tensor,bitwidth=1,name='vtensor'):
    t_shape = tuple(t.shape)
    tmp = vtensor(t_shape,bitwidth,name)
    return tmp


def number_decompose(num):
    sq = math.ceil(math.sqrt(num))
    decompose = []
    for i in range(1,sq+1):
        if num % i == 0:
            decompose.append((i,num//i))
            decompose.append((num//i,i))
    return decompose
