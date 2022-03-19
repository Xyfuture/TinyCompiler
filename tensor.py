import torch
import torch.nn as nn
from .mem import *

class vtensor():
    def __init__(self,shape,bitwidth,name='vtensor'):
        self.shape = shape
        self.bitwidth = bitwidth
        self.name = name
        self.size = 1
        for i in self.shape:
            self.size *= i
        self.length =self.size
        self.size *= self.bitwidth

        self._pre_module = None
        self._post_module = None

    @property
    def pre_module(self):
        assert self._pre_module ,"ERROR: _pre_module not defined"
        return self._pre_module

    @pre_module.setter
    def pre_module(self,pre):
        self._pre_module = pre

    @property
    def post_module(self):
        assert self._post_module, "ERROR: _post_module not defined"
        return self._post_module

    @post_module.setter
    def post_module(self,post):
        self._post_module = post




class tensor():
    pass