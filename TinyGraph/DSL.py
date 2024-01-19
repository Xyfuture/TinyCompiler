from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from TinyGraph.Graph import MicroNode, MicroOp, MicroGraph
from TinyGraph.Module import TransferOp, MatVecMulOp


class XbarGroupVar:
    def __init__(self, xbar_group_shape: Tuple[int, int], core_id: int):
        self.core_id: int = core_id

        self.xbar_group_id = 0
        self.xbar_array_count: int = 0

        self.xbar_group_shape: Tuple[int, int] = xbar_group_shape

    def mul_vector(self, src_vector: DepTensor):
        pass


class MatrixVar:
    def __init__(self, matrix_shape: Tuple[int, int]):
        self.matrix_shape = matrix_shape

        self.xbar_group_array = np.zeros(1, dtype=object)

        pass

    def mapping(self):
        pass

    def mul_vector(self, src_vector: DepTensor):
        # transfer the vector to the xbar core
        # 先把横着的都算完,然后在算纵向的

        pass

        # # make all vector data in the core
        # for index, position in np.ndenumerate(src_vector.tensor_position):
        #     if position != self.core_id:
        #         src_vec_op: MicroOp = src_vector.tensor_op[index]
        #         trans_op = TransferOp(position, self.core_id, 1)
        #
        #         input_nodes = [src_vec_op.node]
        #         MicroGraph.current_graph.create_node(input_nodes, trans_op)
        #
        #         src_vector[index] = (trans_op, self.core_id)
        #
        # # 传输结束,乘法运算,暂时先不处理reshape的问题
        # src_vec_op_list = []
        # op: MicroOp
        # for op in np.nditer(src_vector.tensor_op):
        #     src_vec_op_list.append(op)
        # mat_vec_mul_op = MatVecMulOp(self.core_id, src_vector.shape[0], self.matrix_shape[1], src_vec_op_list)
        #
        # input_nodes = [op.node for op in src_vec_op_list]
        # MicroGraph.current_graph.create_node(input_nodes, mat_vec_mul_op)
        #
        # output_dep_tensor = DepTensor((self.matrix_shape[1],), np.full(self.matrix_shape[1], mat_vec_mul_op),
        #                               np.full(self.matrix_shape[1], self.core_id))
        #
        # return output_dep_tensor


class DepTensor:
    """
    tensor 中记录 该tensor 来自于哪一个micro node
    主要是方便数据传输
    只用于构建图, 不存在于图中

    应该要压缩一个维度,虽然叫tensor,但应该主要是1维或者2维的,自由度不是很高
    """

    def __init__(self, tensor_shape: Tuple[int, ...], reduced_dim_size: int,
                 tensor_op: Optional[np.ndarray] = None, tensor_position: Optional[np.ndarray] = None):
        self.tensor_shape = tensor_shape
        self.reduced_dim_size = reduced_dim_size

        if tensor_op:
            assert tensor_op.shape == tensor_shape
            self.tensor_op = tensor_op
        else:
            self.tensor_op = np.full(self.shape, None, dtype=object)
        if tensor_position:
            assert tensor_position.shape == tensor_shape
            self.tensor_position = tensor_position
        else:
            self.tensor_position = np.full(self.shape, 0)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.tensor_shape

    def __getitem__(self, item):
        tensor_op = self.tensor_op[item]
        tensor_position = self.tensor_position[item]
        return DepTensor(tensor_op.shape, self.reduced_dim_size, tensor_op, tensor_position)

    def __setitem__(self, item, value):
        if isinstance(value, DepTensor):
            assert self.reduced_dim_size == value.reduced_dim_size

            self.tensor_op[item] = value.tensor_op
            self.tensor_position[item] = value.tensor_position
        elif isinstance(value, tuple):
            self.tensor_op[item] = value[0]
            self.tensor_position[item] = value[1]

    def flat(self):
        return zip(self.tensor_op.flat, self.tensor_position.flat)

    def move_to(self,core_id:int):
        pass


