


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

        # static allocation
        self.core_id = core_allocator.get_core()
        self.core = core_allocator.access_core(self.core_id)

        self.allocate()


    def forward(self,in_ten):
        self.compute(in_ten)
        return self.out_ten


    def allocate(self):
        # 分配到堆上吧
        self.in_ten = TensorVar(self.input_shape,self.core_id,self.bitwidth,location='heap')
        self.out_ten = TensorVar(self.output_shape,self.core_id,self.bitwidth,location='heap')

    def compute(self,in_ten:TensorVar):
        vvset = VectorSet(self.core_id,self.bitwidth,self.in_channels)
        self.in_ten.assign(in_ten)

        for h_o,h_i in enumerate(range(0,self.in_height-self.kernel_size+1,self.stride)):
            for w_o,w_i in enumerate(range(0,self.in_width-self.kernel_size+1,self.stride)):
                # 目前的范围应该啊是对的
                # 首先将第一行复制到out_ten之中
                self.out_ten[h_o, w_o, :].copy(self.in_ten[h_i,w_i,:])
                tmp_result = self.out_ten[h_o, w_o, :].get_vec_offset(0, self.in_channels)

                for posi_offset in range(1,self.kernel_size**2):
                    h_off = posi_offset // self.kernel_size
                    w_off = posi_offset % self.kernel_size
                    with vvset:
                        tmp_result.assign(
                            VectorVar.gtm(tmp_result,
                            self.in_ten[h_i + h_off, w_i + w_off, :].get_vec_offset(0, self.in_channels))
                        )

                        # self.out_ten[h_o,w_o,:].get_vec_offset(0,self.in_channels).assign(
                        #     VectorVar.gtm(self.out_ten[h_o,w_o,:].get_vec_offset(0,self.in_channels),
                        #     in_ten[h_i+h_off,w_i+w_off,:].get_vec_offset(0,self.in_channels))
                        # )






if __name__ == "__main__":
    pooling_config = {
        "kernel_size":3,"stride":2
    }

    misc_config = {
        'bitwidth':1
    }

    input_shape = (28,28,64)
    output_shape = (13,13,64)

    core_id = core_allocator.get_core()
    core = core_allocator.access_core(core_id)
    input_tensor = TensorVar(input_shape,core_id,1)

    pool_layer = MaxPoolingLayer(pooling_config,input_shape,output_shape,misc_config)
    pool_layer.forward(input_tensor)
    pool_layer.core.inst_buffer.print_asm()
