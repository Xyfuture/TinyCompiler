from typing import List, Optional, Tuple

from TinyGraph.Graph import MicroOp


class AddOp(MicroOp):
    def __init__(self, core_id: int, vector_size: int):
        super().__init__()
        self.core_id = core_id
        self.vector_size = vector_size

    def code_gen(self):
        pass


class TransferOp(MicroOp):
    def __init__(self, src_core_id: int, dst_core_id: int, data_size: int):
        super().__init__()
        self.src_core_id = src_core_id
        self.dst_core_id = dst_core_id
        self.data_size = data_size

    def code_gen(self):
        pass


class MatVecMulOp(MicroOp):
    def __init__(self, core_id: int, xbar_group_id: int, input_size: int, output_size: int,
                 src_vec_op_list: Optional[List[MicroOp]] = None):
        super().__init__()

        self.core_id = core_id
        self.xbar_group_id = xbar_group_id
        self.src_vec_op_list = src_vec_op_list

        self.input_size = input_size
        self.output_size = output_size

    def code_gen(self):
        pass


class MaxPool2dOp(MicroOp):
    def __init__(self, core_id: int, kernel_size: Tuple[int, int], vector_size: int):
        super().__init__()
        self.core_id = core_id

        self.kernel_size = kernel_size
        self.vector_size = vector_size

    def code_gen(self):
        pass


class PadOp(MicroOp):
    def __init__(self, core_id: int):
        super().__init__()

        self.core_id = core_id

    def code_gen(self):
        pass
