import torch
from TinyNet.wrapper.tensor import vtensor


def gen_vtensor(t:torch.Tensor,bitwidth=1,name='vtensor'):
    t_shape = tuple(t.shape)
    tmp = vtensor(t_shape,bitwidth,name)
    return tmp