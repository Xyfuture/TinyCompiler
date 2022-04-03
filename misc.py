from TinyNet.wrapper.module import *
from TinyDSL.Utils.utils import *


# transmit
class concat(module):
    def __init__(self,in_cnt,dim=0):
        super(concat, self).__init__()

        self.in_cnt = in_cnt
        self.dim = dim

        self.input_shape = []
        self.output_shape = None



    def forward(self, input_tensors, relation_tensors):
        assert len(input_tensors) == self.in_cnt , "ERROR: unmatch tensor cnt"
        assert len(relation_tensors) == self.in_cnt, "ERROR: unmatch vtensor cnt"
        self.input_shape = [tuple(i.shape) for i in input_tensors]

        for t in relation_tensors:
            self.pre_modules[t] = t.pre_module
            t.post_module = self
            self.in_tensors.append(t)

        input_tensors = tuple(input_tensors)
        output_tensor = torch.cat(input_tensors,dim=self.dim)

        self.output_shape = tuple(output_tensor.shape)

        output_vtensor = gen_vtensor(output_tensor)
        output_vtensor.pre_module = self
        self.out_tensors.append(output_vtensor)

        return output_tensor,output_vtensor



