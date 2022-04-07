


from TinyDSL.DataType.tensor import TensorVar
from TinyDSL.DataType.vector import VectorVar, VectorSet
from TinyDSL.HwResource.config import core_config
from TinyDSL.HwResource.core import core_allocator


class MaxPoolingLayer:
    def __init__(self,pooling_config:dict,input_shape,output_shape,misc_config):
        pooling_args = ['kernel_size', 'stride'] # pad = 0 and dilation =0
        for arg in pooling_args:
            self.__setattr__(arg,pooling_config[arg])

        self.input_shape = input_shape # HWC no Batch
        self.output_shape = output_shape

        self.in_height,self.in_width,self.in_channels = self.input_shape
        self.out_height,self.out_width,self.out_channels = self.output_shape

        misc_args = ['bitwidth']
        for arg in misc_args:
            self.__setattr__(arg,misc_config[arg])

        self.core_config = core_config

        # runtime allocation not static
        self.core_id = None
        self.core = None
        self.out_ten:VectorVar = None


    def forward(self,in_ten:TensorVar):
        self.allocate(in_ten.core_id)
        self.compute(in_ten)
        return self.out_ten


    def allocate(self,core_id ):
        self.core_id = core_id
        self.core = core_allocator.access_core(self.core_id)

        # 分配到堆上吧
        self.out_ten = TensorVar(self.output_shape,self.core_id,self.bitwidth,location='heap')

    def compute(self,in_ten:TensorVar):
        vvset = VectorSet(self.core_id,self.bitwidth,self.in_channels)
        for h_o,h_i in enumerate(range(0,self.in_height,self.stride)):
            for w_o,w_i in enumerate(range(0,self.in_width,self.stride)):
                # 首先将第一行复制到out_ten之中
                self.out_ten[h_o, w_o, :].copy(in_ten[h_i,w_i,:])
                for posi_offset in range(1,self.kernel_size**2):
                    h_off = posi_offset // self.kernel_size
                    w_off = posi_offset % self.kernel_size

                    self.out_ten[h_o,w_o,:].assign(VectorVar.gtm(self.out_ten[h_o,w_o,:],in_ten[h_i+h_off,w_i+w_off,:]))


