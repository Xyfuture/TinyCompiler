from TinyDSL.DataType.tensor import TensorVar
from TinyDSL.DataType.vector import VectorVar
from TinyDSL.HwResource.core import core_allocator


class ConvPipe():
    def __init__(self,conv_config:dict,misc_config:dict):
        conv_args = ['in_channels','out_channels','kernel_size','stride',
                     'padding','groups','bias','activation_func']
        misc_args = ['in_act_shape','in_act_pad_shape','out_act_shape','meu_layout','mat_shape',
                     'act_bitwidth','mat_bitwidth','core_config','posi']

        for arg in conv_args:
            self.__setattr__(arg,conv_config[arg])

        for arg in misc_args:
            self.__setattr__(arg,misc_config[arg])


        self.core_id = core_allocator.get_core()
        self.core = core_allocator.access_core(self.core_id)

        self.kernel_height, self.kernel_width = self.kernel_size
        self.rows, self.columns = self.mat_shape




    def allocate(self):

        self.in_act_ten = TensorVar(self.in_act_pad_shape,self.core_id,self.act_bitwidth,location='heap')
        self.shift_vec = VectorVar(self.columns,self.core_id,4,location='heap')
        self.out_act = TensorVar(self.out_act_shape,self.core_id,self.act_bitwidth,location='heap')

        self.meu_ele_rows = self.core_config.meu_rows
        self.meu_ele_columns = (self.core_config.meu_cloumns*self.core_config.meu_cell_bit)//(self.mat_bitwidth*8)
