from typing import Optional, List, Callable, Tuple

from TinyGraph.ConductArray import ConductArray
from TinyGraph.DSL import DepTensor
from TinyGraph.Graph import MicroGraph, topo_sort
from TinyGraph.Machine import Chip
from TinyGraph.MicroOps import RawInputOp
from TinyGraph.Module import DepModule


def graph_compile(chip: Chip, graph: MicroGraph, network: DepModule, input_shape: Tuple[int, int, int],
                  graph_passes: Optional[List[Callable]] = None):
    """
    完整的执行流程
    设置 current chip 和 current graph

    首先初始化一个input tensor 设定其相关的位置
    完成对network中对chip的mapping 工作
    network forward 实现对micro graph的构建
    对 micro graph 跑一些 pass

    对micro graph 跑拓扑序
    由micro node lower 到 每个核上的 machine op
    由 每个核上的 machine op lower 到 实际上每个和的 指令

    """

    # 设置 current chip和graph
    Chip.current_chip = chip
    MicroGraph.current_graph = graph

    # 设置 input tensor
    d, h, w = input_shape

    input_tensor_shape = (h, w)
    input_tensor = DepTensor(input_shape,d,
                             ConductArray.full(input_tensor_shape,None),
                             ConductArray.full(input_tensor_shape,-1))
    # 设置初始的 input op, RawInputOp
    for i in range(h):
        for j in range(w):
            micro_op = RawInputOp(d)
            graph.create_node([],micro_op)
            input_tensor.tensor_op[i,j] = micro_op

    # 进行 mapping 工作
    network.mapping()

    # 运行 inference 得到 计算图
    output_tensor:DepTensor = network(input_tensor)

    # 运行一些pass
    for graph_pass in graph_passes:
        graph_pass(graph)

    # 获得拓扑序
    topo_node_list = topo_sort(graph)

    # micro node  lower to machine op in each core
    # TODO 可以将这个功能整合到 micro graph 中
    for node in topo_node_list:
        node.micro_op.machine_op_gen()

    # machine op lower to final inst
    chip.lower_to_inst()

    print('pass')



