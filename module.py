import torch
import torch.nn as nn
from abc import abstractmethod, ABCMeta
from .tensor import *
import collections


# assume one module only generate one tensor to next

class module(metaclass=ABCMeta):
    def __init__(self):
        self.pre_modules = collections.OrderedDict()
        self.post_modules = collections.OrderedDict()

        self.in_tensors = []
        self.out_tensors = []

    @abstractmethod
    def forward(self, input_tensors, relation_tensors):
        # input_tensor : torch.Tensor for real compute
        # relation_tensor just record information
        pass

    def backward(self, relation_tensor: vtensor):
        assert relation_tensor in self.out_tensors, "ERROR: relation tensor not in pre module"
        assert relation_tensor not in self.post_modules, "ERROR: relation tensor has been processed"

        self.post_modules[relation_tensor] = relation_tensor.post_module

        for t in self.in_tensors:
            self.pre_modules[t].backward(t)

    def __call__(self, *args, **kwargs):
        return self.forward(*args)
