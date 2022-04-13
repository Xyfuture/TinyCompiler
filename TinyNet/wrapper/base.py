import torch
import torch.nn as nn
from abc import abstractmethod, ABCMeta
import collections


# assume one modules only generate one tensor to next

class module(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def torch_forward(self, input_tensors):
        pass

    @abstractmethod
    def pim_forward(self,in_ten):
        pass

    def __call__(self, *args, **kwargs):
        torch_tensor,pim_tensor = args
        return self.torch_forward(torch_tensor),self.pim_forward(pim_tensor)
