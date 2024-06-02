from typing import Tuple

from TinyGraph.ConductArray import ConductArray
from TinyGraph.DSL import DepTensor
from TinyGraph.Graph import MicroGraph
from TinyGraph.MicroOps import RawInputOp


# (1,) reduce_dim_size = 1024
# 只支持batch = 1 的情况
def create_input_tensor(input_shape: Tuple[int, ...], reduce_dim_size: int) -> DepTensor:
    input_tensor = DepTensor(input_shape, reduce_dim_size,
                             ConductArray.full(input_shape, None),
                             ConductArray.full(input_shape, -1))  # -1 represent offchip dram

    for idx in input_tensor.index():
        micro_op = RawInputOp(reduce_dim_size)
        MicroGraph.current_graph.create_node([], micro_op)
        input_tensor.tensor_op[idx] = micro_op
        # position 不需要设置

    return input_tensor
