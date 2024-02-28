from TinyGraph.Graph import MicroGraph
from TinyGraph.Machine import ChipConfig, Chip
from TinyGraph.MicroOps import pad_to_core
from graph_compile import graph_compile
from resnet18 import resnet18

chip_config = ChipConfig()
chip = Chip(chip_config)
graph = MicroGraph()
network = resnet18()

input_shape = (3, 32, 32)
graph_passes = [
    pad_to_core,
]

graph_compile(chip, graph, network, input_shape, graph_passes)


chip.dump_inst_to_file('./inst/')