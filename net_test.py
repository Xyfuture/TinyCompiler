from TinyGraph.ConductArray import ConductArray
from TinyGraph.DSL import DepTensor
from TinyGraph.Graph import MicroGraph, MicroNode, topo_sort
from TinyGraph.Machine import Chip, ChipConfig
from TinyGraph.MicroOps import RawInputOp, pad_to_core
from TinyGraph.Module import DepSequential, DepModule, DepConv2d, DepLinear
from resnet18 import resnet18

chip_config = ChipConfig()
chip = Chip(chip_config)
Chip.current_chip = chip

input_shape = (32, 32)
input_tensor = DepTensor(input_shape, 3,
                         ConductArray.full(input_shape, None),
                         ConductArray.full(input_shape, -1))

graph = MicroGraph()
MicroGraph.current_graph = graph

for i in range(input_shape[0]):
    for j in range(input_shape[1]):
        micro_op = RawInputOp(3)
        graph.create_node([], micro_op)
        input_tensor.tensor_op[i, j] = micro_op

net = resnet18()

net.mapping()
output_tensor = net(input_tensor)

# 跑一些 pass

pad_to_core(graph)

# 拓扑序

topo_node_list = topo_sort(graph)

# lower to machine op

# 输出 micro op的信息
# print('start')
# for node in topo_node_list:
#     print(node.micro_op.full_info())
# print('end')

for node in topo_node_list:
    node.micro_op.machine_op_gen()


def get_mapping_status(network: DepModule):
    s = ""
    for module in network._module_dict.values():
        if isinstance(module, DepConv2d):
            s += f'{module.module_name} : {module.weight_matrix}\n'
        elif isinstance(module, DepLinear):
            s += f'{module.module_name} : {module.weight_matrix}\n'
        else:
            s += get_mapping_status(module)
    return s


print(get_mapping_status(net))

# lower to inst
chip.inst_code_gen()

# for index,machine_op in enumerate(chip.core_array[0].machine_op_list):
#     print(f"{index} : {machine_op}")
# print('all machine ops')

chip.dump_output_to_file("./inst/")


usage,used_xbar_cnt,total_xbar_cnt = chip.get_xbar_usage()
print(f'usage: {usage}')

print("pass")
