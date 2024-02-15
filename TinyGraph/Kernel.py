import copy
import operator
from functools import reduce
from math import floor
from typing import Tuple, List, Dict

import numpy as np

from TinyGraph.ConductArray import ConductArray
from TinyGraph.Graph import MicroGraph, MicroOp, MicroNode

from TinyGraph.DSL import MatrixVar, XbarGroupVar, DepTensor
from TinyGraph.MicroOps import AddOp, TransferOp, MatVecMulOp, MaxPool2dOp, ReLUOp


def _make_data_to_core_kernel(src: DepTensor, core_id: int) -> DepTensor:
    for index, position in src.tensor_position.enum():
        if position == core_id:
            continue
        else:
            src_op = src.tensor_op[index]
            trans_op = TransferOp(position, core_id, src.reduced_dim_size, src_op)
            input_node = src_op.node
            trans_node = MicroGraph.current_graph.create_node([], trans_op)
            input_node.insert_node_after_with(trans_node)

            src[index] = (trans_op, core_id)

    return src


def _add_same_core_kernel(in_1: DepTensor, in_2: DepTensor) -> DepTensor:
    # 假设数据都在一个core上 进行传输
    out_dep_tensor = DepTensor(in_1.shape, in_1.reduced_dim_size)

    for index in np.ndindex(in_1.tensor_position.shape):
        core_id = in_1.tensor_position[index]
        dim_size = in_1.reduced_dim_size
        src_op = [in_1.tensor_op[index], in_2.tensor_op[index]]
        add_op = AddOp(core_id, dim_size, src_op[0], src_op[1])

        input_nodes = [in_1.tensor_op[index].node, in_2.tensor_op[index].node]

        MicroGraph.current_graph.create_node(input_nodes, add_op)

        out_dep_tensor.tensor_op[index] = add_op
        out_dep_tensor.tensor_position[index] = core_id

    return out_dep_tensor


def _add_kernel(in_1: DepTensor, in_2: DepTensor) -> DepTensor:
    dst_core_id = in_1.tensor_position.flat()[0]

    output_dep_tensor = DepTensor(in_1.tensor_shape, in_1.reduced_dim_size)

    for index, position in in_1.tensor_position.enum():
        tmp_1 = _make_data_to_core_kernel(in_1[index], dst_core_id)
        tmp_2 = _make_data_to_core_kernel(in_2[index], dst_core_id)

        tmp_3 = _add_same_core_kernel(tmp_1, tmp_2)

        output_dep_tensor[index] = _make_data_to_core_kernel(tmp_3, dst_core_id)

    return output_dep_tensor


def _sum_kernel(input_dep_tensors: List[DepTensor]) -> DepTensor:
    tmp = input_dep_tensors[0]

    for i in range(len(input_dep_tensors) - 1):
        tmp = _add_kernel(tmp, input_dep_tensors[i + 1])

    return tmp


def _xbar_vec_mul_kernel(input_vec: DepTensor, xbar_matrix: XbarGroupVar) -> DepTensor:
    core_id = xbar_matrix.core_id
    input_vec.move_to(core_id)

    vec_size = reduce(operator.mul, input_vec.tensor_shape)
    vec_size = vec_size * input_vec.reduced_dim_size

    # check input size and matrix rows
    assert vec_size == xbar_matrix.xbar_group_shape[0]

    input_ops:List[MicroOp] = []
    for op in input_vec.tensor_op.flat():
        if op:
            input_ops.append(op)

    input_nodes = [op.node for op in input_ops]

    mat_vec_mul_op = MatVecMulOp(core_id, xbar_matrix.xbar_group_id, vec_size, xbar_matrix.xbar_group_shape[1],
                                 input_ops)
    MicroGraph.current_graph.create_node(input_nodes, mat_vec_mul_op)

    output_tensor = DepTensor((1,), xbar_matrix.xbar_group_shape[1],
                              ConductArray.full((1,), mat_vec_mul_op),
                              ConductArray.full((1,), core_id))
    return output_tensor


def _conv2d_kernel(input_feature_map: DepTensor,
                   weight_matrix: MatrixVar,
                   in_channels: int, out_channels: int,
                   kernel_size: Tuple[int, int],
                   stride: Tuple[int, int] = (1, 1),
                   padding: int = 0,
                   ) -> DepTensor:
    def get_xbar_group_input(window_tensor: DepTensor, start_index: int, xbar_group: XbarGroupVar) -> DepTensor:
        interval = window_tensor.reduced_dim_size

        start = floor(start_index / interval)
        end = floor(start_index + xbar_group.xbar_group_shape[0] / interval)

        return window_tensor.reshape((-1,))[start:end + 1]

    # check
    assert in_channels == input_feature_map.reduced_dim_size \
           and in_channels * kernel_size[0] * kernel_size[1] == weight_matrix.matrix_shape[0]
    assert out_channels == weight_matrix.matrix_shape[1]

    # pad_input_op = np.pad(input_feature_map.tensor_op, ((padding, padding), (padding, padding)))
    # pad_input_position = np.pad(input_feature_map.tensor_position, ((padding, padding), (padding, padding)))
    # pad_shape = pad_input_op.shape
    #
    # pad_input_tensor = DepTensor(pad_shape, input_feature_map.reduced_dim_size, pad_input_op, pad_input_position)

    pad_input_tensor = DepTensor.pad(input_feature_map, padding)
    pad_shape = pad_input_tensor.shape

    output_shape = tuple(
        (pad_shape[i] - kernel_size[i]) // stride[i] + 1 for i in range(len(kernel_size))
    )

    rows, cols = output_shape

    output_feature_map = DepTensor(output_shape, out_channels, )

    xbar_rows = len(weight_matrix.xbar_group_array)
    core_input_tensor_map: Dict[int, DepTensor] = {}

    for xbar_group in weight_matrix.xbar_group_array:
        if xbar_group.core_id not in core_input_tensor_map:
            # core_input_tensor_map[xbar_group.core_id] = copy.copy(pad_input_tensor)
            core_input_tensor_map[xbar_group.core_id] = copy.deepcopy(pad_input_tensor)

    with MicroGraph.current_graph.use_sequential_node_dep():
        for i in range(rows):
            for j in range(cols):
                # 一次窗口
                # 先把横着的都算完,然后在算纵向的
                partial_sum_list: List[DepTensor] = []
                start_index = 0
                for xbar_i in range(xbar_rows):
                    # 计算这次需要的窗口
                    # different core use different window
                    core_id = weight_matrix.xbar_group_array[xbar_i].core_id
                    current_input_window = core_input_tensor_map[core_id][
                                           i * stride[0]:i * stride[0] + kernel_size[0],
                                           j * stride[1]:j * stride[1] + kernel_size[1]
                                           ]

                    # TODO get partial window

                    current_input = get_xbar_group_input(current_input_window, start_index,
                                                         weight_matrix.xbar_group_array[xbar_i])

                    partial_sum_list.append(
                        _xbar_vec_mul_kernel(
                            current_input,
                            weight_matrix.xbar_group_array[xbar_i]
                        )
                    )

                    start_index += weight_matrix.xbar_group_array[xbar_i].xbar_group_shape[0]

                current_output = _sum_kernel(partial_sum_list)
                output_feature_map[i, j] = current_output

    return output_feature_map


def _maxpool2d_kernel(input_feature_map: DepTensor,
                      kernel_size: Tuple[int, int],
                      stride: Tuple[int, int],
                      padding: int) -> DepTensor:
    # inplace 进行pooling 操作
    # 核选择第一个元素所在的核
    core_id: int = input_feature_map.tensor_position[0, 0]
    vector_size = input_feature_map.reduced_dim_size

    # pad_input_op = np.pad(input_feature_map.tensor_op, ((padding, padding), (padding, padding)))
    # pad_input_position = np.pad(input_feature_map.tensor_position, ((padding, padding), (padding, padding)))
    # pad_shape = pad_input_op.shape
    #
    # pad_input_tensor = DepTensor(pad_shape, input_feature_map.reduced_dim_size, pad_input_op, pad_input_position)

    pad_input_tensor = DepTensor.pad(input_feature_map, padding)
    pad_shape = pad_input_tensor.shape

    output_shape = tuple(
        (pad_shape[i] - kernel_size[i]) // stride[i] + 1 for i in range(len(kernel_size))
    )

    rows, cols = output_shape

    output_feature_map = DepTensor(tuple(output_shape), input_feature_map.reduced_dim_size, )

    with MicroGraph.current_graph.use_sequential_node_dep():
        for i in range(rows):
            for j in range(cols):
                # 首先传输到同一个核上来
                current_input_window = pad_input_tensor[
                                       i * stride[0]:i * stride[0] + kernel_size[0],
                                       j * stride[1]:j * stride[1] + kernel_size[1]
                                       ].move_to(core_id)
                input_ops = [op for op in current_input_window.tensor_op.flat()]

                maxpool2d_op = MaxPool2dOp(core_id, kernel_size, vector_size,input_ops)

                input_nodes = [op.node for op in input_ops]
                MicroGraph.current_graph.create_node(input_nodes, maxpool2d_op)

                output_feature_map[i, j] = (maxpool2d_op, core_id)

    return output_feature_map


def _matrix_vec_mul_kernel(input_tensor: DepTensor,
                           weight_matrix: MatrixVar) -> DepTensor:
    # 可以和conv 结合起来写
    interval = input_tensor.reduced_dim_size

    with MicroGraph.current_graph.use_sequential_node_dep():
        processed_index = 0
        partial_sum_list: List[DepTensor] = []

        for xbar_i in range(len(weight_matrix.xbar_group_array)):
            core_id = weight_matrix.xbar_group_array[xbar_i].core_id

            input_start_index = floor(processed_index / interval)
            input_end_index = floor((processed_index + weight_matrix.xbar_group_array[xbar_i].xbar_group_shape[0])
                                    / interval) + 1

            partial_sum_list.append(
                _xbar_vec_mul_kernel(
                    input_tensor[input_start_index:input_end_index],
                    weight_matrix.xbar_group_array[xbar_i]
                )
            )

            processed_index += weight_matrix.xbar_group_array[xbar_i].xbar_group_shape[0]

        current_output = _sum_kernel(partial_sum_list)

    return current_output


def _relu_kernel(input_tensor: DepTensor) -> DepTensor:
    output_tensor = DepTensor(input_tensor.shape, input_tensor.reduced_dim_size)

    for index, position in input_tensor.tensor_position.enum():
        pre_op:MicroOp = input_tensor.tensor_op[index]
        relu_op = ReLUOp(position,pre_op)
        pre_node: MicroNode = pre_op.node
        MicroGraph.current_graph.create_node([pre_node], relu_op)

        output_tensor[index] = (relu_op, position)

    return output_tensor
