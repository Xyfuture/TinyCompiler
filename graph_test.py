from TinyGraph.DSL import DepTensor
from TinyGraph.Graph import MicroGraph, topo_sort
from TinyGraph.Module import DepConv2d
import numpy as np

from TinyGraph.Ops import PadOp

graph = MicroGraph()
MicroGraph.current_graph = graph


conv2d = DepConv2d(0,in_channels=3,out_channels=32,kernel_size=(3,3),stride=(1,1),padding=1)


pad_op = PadOp(0)
MicroGraph.current_graph.create_node([],pad_op)

input_tensor = DepTensor((32,32),3,
                         np.full((32,32),pad_op,dtype=object),
                         np.full((32,32), 0))

output_tensor = conv2d.forward(input_tensor)

print(len(graph.nodes))
node_list = topo_sort(graph)

print(len(node_list))