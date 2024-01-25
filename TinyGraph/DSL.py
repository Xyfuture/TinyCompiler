from __future__ import annotations

from typing import Tuple, Optional, List

import numpy as np

from TinyGraph.ConductArray import ConductArray
from TinyGraph.Graph import MicroGraph
from TinyGraph.Machine import Core
from TinyGraph.Ops import TransferOp, PadOp


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

        # 不是二维,只能是一维的
        self.xbar_group_array: List[XbarGroupVar] = []

        pass

    def mapping(self):
        pass

    def dummy_mapping(self):
        core = Core()

        self.xbar_group_array.append(XbarGroupVar(self.matrix_shape, core.core_id))

    def mul_vector(self, src_vector: DepTensor):
        # transfer the vector to the xbar core

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

    def __init__(self, tensor_shape: Tuple[int, ...], reduced_dim_size: int = 1,
                 tensor_op: Optional[ConductArray] = None, tensor_position: Optional[ConductArray] = None):

        # tensor_shape expect to be tuple
        # may be int will be passed in
        self.reduced_dim_size = reduced_dim_size

        if tensor_op is not None:
            self.tensor_op = tensor_op
        else:
            self.tensor_op = ConductArray.full(tensor_shape, None)
        if tensor_position is not None:
            self.tensor_position = tensor_position
        else:
            self.tensor_position = ConductArray.full(tensor_shape, 0)

        self.tensor_shape = self.tensor_position.shape

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.tensor_shape

    def decorate_item(self, item):
        if isinstance(item, tuple) and not any([isinstance(x, slice) for x in item]):
            item = list(item)
            item[-1] = slice(item[-1], item[-1] + 1)
            item = tuple(item)
        elif isinstance(item, int):
            item = (slice(item, item + 1))

        return item

    def __getitem__(self, item):
        # 总是返回一个DepTensor,不会直接返回值
        item = self.decorate_item(item)
        tensor_op = self.tensor_op[item]
        tensor_position = self.tensor_position[item]
        return DepTensor(tensor_op.shape, self.reduced_dim_size, tensor_op, tensor_position)

    def __setitem__(self, item, value):
        # 总是对一个slice进行赋值
        # conduct array 针对区间赋值处理更好，能识别value为array的情况
        item = self.decorate_item(item)
        if isinstance(value, DepTensor):
            assert self.reduced_dim_size == value.reduced_dim_size

            self.tensor_op[item] = value.tensor_op
            self.tensor_position[item] = value.tensor_position
        elif isinstance(value, tuple):
            self.tensor_op[item] = value[0]
            self.tensor_position[item] = value[1]

    def flat(self):
        return zip(self.tensor_op.flat(), self.tensor_position.flat())

    def move_to(self, core_id: int):
        for index, position in self.tensor_position.enum():
            if position == core_id:
                continue
            else:
                input_op =  self.tensor_op[index]
                trans_op = TransferOp(position, core_id, self.reduced_dim_size,input_op)
                input_nodes = [input_op.node]

                MicroGraph.current_graph.create_node(input_nodes, trans_op)

                self.tensor_op[index] = trans_op
                self.tensor_position[index] = core_id
        return self

    def reshape(self, new_shape: Tuple[int, ...]):
        self.tensor_op = self.tensor_op.reshape(new_shape)
        self.tensor_position = self.tensor_position.reshape(new_shape)

        self.tensor_shape = self.tensor_op.shape

        return self

    @staticmethod
    def pad(input_tensor: DepTensor, pad_width: int):
        # 默认是四周的映射模式
        if pad_width:
            pad_op = PadOp(-1)
            MicroGraph.current_graph.create_node([], pad_op)

            tensor_op = ConductArray.pad(input_tensor.tensor_op, pad_width,pad_op)
            tensor_position = ConductArray.pad(input_tensor.tensor_position, pad_width,-1)
            shape = tensor_op.shape
            output = DepTensor(shape, input_tensor.reduced_dim_size, tensor_op, tensor_position)
        else:
            output = input_tensor
        return output

    # def __copy__(self):
    #     new_tensor = DepTensor(
    #         self.tensor_shape, self.reduced_dim_size,
    #         self.tensor_op.copy(),
    #         self.tensor_position.copy()
    #     )
    #
    #     return new_tensor
