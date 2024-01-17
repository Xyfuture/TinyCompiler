from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from TinyGraph.Graph import MicroNode, MicroOp, MicroGraph
from TinyGraph.Module import TransferOp, MatVecMulOp


class XbarMatrix:
    def __init__(self, matrix_shape: Tuple[int, int], core_id: int):
        self.matrix_shape = matrix_shape
        self.core_id = core_id

        pass

    def mul_vector(self, src_vector: DepTensor):
        # make all vector data in the core
        for index, position in np.ndenumerate(src_vector.tensor_position):
            if position != self.core_id:
                src_vec_op: MicroOp = src_vector.tensor_op[index]
                trans_op = TransferOp(position, self.core_id, 1)

                input_nodes = [src_vec_op.node]
                MicroGraph.current_graph.create_node(input_nodes, trans_op)

                src_vector[index] = (trans_op, self.core_id)

        # 传输结束,乘法运算,暂时先不处理reshape的问题
        src_vec_op_list = []
        op: MicroOp
        for op in np.nditer(src_vector.tensor_op):
            src_vec_op_list.append(op)
        mat_vec_mul_op = MatVecMulOp(self.core_id, src_vector.shape[0], self.matrix_shape[1], src_vec_op_list)

        input_nodes = [op.node for op in src_vec_op_list]
        MicroGraph.current_graph.create_node(input_nodes, mat_vec_mul_op)

        return


class DepTensor:
    """
    tensor 中记录 该tensor 来自于哪一个micro node
    主要是方便数据传输
    只用于构建图, 不存在于图中

    """

    def __init__(self, tensor_shape: Tuple[int, ...],
                 tensor_op: Optional[np.ndarray] = None, tensor_position: Optional[np.ndarray] = None):
        self.tensor_shape = tensor_shape
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
        return DepTensor(tensor_op.shape, tensor_op, tensor_position)

    def __setitem__(self, item, value):
        if isinstance(value, DepTensor):
            self.tensor_op[item] = value.tensor_op
            self.tensor_position[item] = value.tensor_position
        elif isinstance(value, tuple):
            self.tensor_op[item] = value[0]
            self.tensor_position[item] = value[1]

    def flat(self):
        return zip(self.tensor_op.flat, self.tensor_position.flat)
