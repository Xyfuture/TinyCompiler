from TinyGraph.ConductArray import ConductArray
from TinyGraph.DSL import DepTensor
from TinyGraph.Graph import MicroGraph, topo_sort
from TinyGraph.Kernel import _add_kernel
from TinyGraph.Module import DepConv2d
import numpy as np

from TinyGraph.Ops import PadOp, MatVecMulOp, TransferOp, AddOp

graph = MicroGraph()
MicroGraph.current_graph = graph

conv2d_1 = DepConv2d(0, in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
conv2d_2 = DepConv2d(0,in_channels=32,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1)

pad_op = PadOp(0)
MicroGraph.current_graph.create_node([], pad_op)

input_tensor = DepTensor((32, 32), 3,
                         ConductArray.full((32, 32), pad_op),
                         ConductArray.full((32, 32), 0))

output_tensor = conv2d_1.forward(input_tensor)
output_tensor_2 = conv2d_2.forward(output_tensor)

output_tensor_3 = _add_kernel(output_tensor,output_tensor_2)

print(output_tensor_2.shape)

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