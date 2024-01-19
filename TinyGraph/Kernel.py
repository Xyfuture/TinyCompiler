import copy
from typing import Tuple, List, Dict

import numpy as np
from TinyGraph.Graph import MicroGraph, MicroOp

from TinyGraph.DSL import DepTensor, MatrixVar, XbarGroupVar
from TinyGraph.Module import TransferOp, AddOp


def _make_data_to_core_kernel(src: DepTensor, core_id: int) -> DepTensor:
    for index, position in np.ndenumerate(src.tensor_position):
        if position == core_id:
            continue
        else:
            trans_op = TransferOp(position, core_id, src.reduced_dim_size)
            input_nodes = [src.tensor_op[index].node]
            MicroGraph.current_graph.create_node(input_nodes, trans_op)

            src[index] = (trans_op, core_id)

    return src


def _add_same_core_kernel(in_1: DepTensor, in_2: DepTensor) -> DepTensor:
    # 假设数据都在一个core上 进行传输
    out_dep_tensor = DepTensor(in_1.shape, in_1.reduced_dim_size)

    for index in np.ndindex(in_1.tensor_position.shape):
        core_id = in_1.tensor_position[index]
        dim_size = in_1.reduced_dim_size
        add_op = AddOp(core_id, dim_size)
        input_nodes = [in_1.tensor_op[index].node, in_2.tensor_op[index].node]

        MicroGraph.current_graph.create_node(input_nodes, add_op)

        out_dep_tensor.tensor_op[index] = add_op
        out_dep_tensor.tensor_position[index] = core_id

    return out_dep_tensor


def _add_kernel(in_1: DepTensor, in_2: DepTensor) -> DepTensor:
    dst_core_id = in_1.tensor_position[0]

    output_dep_tensor = DepTensor(in_1.tensor_shape, in_1.reduced_dim_size)

    for index, position in np.ndenumerate(in_1.tensor_position):
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
    pass


def _conv2d_kernel(input_feature_map: DepTensor,
                   weight_matrix: MatrixVar,
                   in_channels: int, out_channels: int,
                   kernel_size: Tuple[int, int],
                   stride: Tuple[int, int] = (1, 1),
                   padding: int = 0,
                   ) -> DepTensor:
    # check
    assert in_channels == input_feature_map.reduced_dim_size and in_channels == weight_matrix.matrix_shape[0]
    assert out_channels == weight_matrix.matrix_shape[1]

    pad_input_op = np.pad(input_feature_map.tensor_op, ((padding, padding), (padding, padding)))
    pad_input_position = np.pad(input_feature_map.tensor_position, ((padding, padding), (padding, padding)))

    pad_shape = pad_input_op.shape
    output_shape = (
        (pad_shape[i] - kernel_size[i]) // stride[i] + 1 for i in range(len(kernel_size))
    )

    rows, cols = output_shape

    output_feature_map = DepTensor(tuple(output_shape), out_channels, )

    core_input_tensor_map: Dict[int, DepTensor] = {}
    xbar_group: XbarGroupVar
    for index, xbar_group in np.ndenumerate(weight_matrix.xbar_group_array):
        if xbar_group.core_id not in core_input_tensor_map:
            core_input_tensor_map[xbar_group.core_id] = copy.deepcopy(input_feature_map)

    with MicroGraph.current_graph.use_sequential_node_dep():
        for i in range(rows):
            for j in range(cols):
                pass

    return output_feature_map


def _maxpooling2d_kernel() -> DepTensor:
    pass
