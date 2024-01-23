from TinyGraph.DSL import DepTensor
from TinyGraph.Graph import MicroGraph, topo_sort
from TinyGraph.Module import DepConv2d
import numpy as np

from TinyGraph.Ops import PadOp, MatVecMulOp, TransferOp, AddOp

graph = MicroGraph()
MicroGraph.current_graph = graph

conv2d = DepConv2d(0, in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)

pad_op = PadOp(0)
MicroGraph.current_graph.create_node([], pad_op)

input_tensor = DepTensor((32, 32), 3,
                         np.full((32, 32), pad_op, dtype=object),
                         np.full((32, 32), 0))

output_tensor = conv2d.forward(input_tensor)

print(len(graph.nodes))
node_list = topo_sort(graph)

print(len(node_list))

count_map = {
    "mat_vec_mul": 0,
    "transfer": 0,
    "add": 0,
    "pad": 0
}

for node in node_list:
    if isinstance(node.micro_op, MatVecMulOp):
        count_map["mat_vec_mul"] += 1
    elif isinstance(node.micro_op, TransferOp):
        count_map["transfer"] += 1
    elif isinstance(node.micro_op, AddOp):
        count_map['add'] += 1
    elif isinstance(node.micro_op, PadOp):
        count_map['pad'] += 1

for k, v in count_map.items():
    print(f'{k} count: {v}')
