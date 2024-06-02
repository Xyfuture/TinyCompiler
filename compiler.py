import argparse
from TinyGraph.ConductArray import ConductArray
from TinyGraph.DSL import DepTensor
from TinyGraph.Graph import MicroGraph, topo_sort
from TinyGraph.Machine import Chip, ChipConfig
from TinyGraph.MicroOps import RawInputOp, pad_to_core
from TinyGraph.Module import DepModule, DepConv2d, DepLinear
from Networks.resnet18 import resnet18
import json

from utils import create_input_tensor

parser = argparse.ArgumentParser(description='Compiler for CIM')

parser.add_argument('--mapping', '-m', type=str, default='performance',
                    help='Mapping Strategy : performance or utilization')
parser.add_argument('--config', '-c', type=str, default='example/resnet_config.json', help='Configuration File Path')
parser.add_argument('--trace', '-t', type=str, default='', help='Trace File Path')
parser.add_argument('--network', '-n', type=str, default='resnet', help='Network Name')
parser.add_argument('--verbose', '-v', type=bool, action='store_true', help='Report All Module Results')

args = parser.parse_args()


def resnet_run():
    net = resnet18()
    net.mapping()

    input_tensor = create_input_tensor((32, 32), 32)
    output_tensor = net(input_tensor)
    return output_tensor


def autoencoder_run():
    pass


if __name__ == '__main__':

    graph = MicroGraph()
    MicroGraph.current_graph = graph

    with open(args.config) as f:
        config_dict = json.load(f)
        chip_config = ChipConfig(**config_dict)

    chip = Chip(chip_config)
    Chip.current_chip = chip

    if args.network == 'resnet':
        _ = resnet_run()
    elif args.network == 'autoencoder':
        pass
    else:
        raise "Set your own network here"

    # run some optimization pass
    pad_to_core(graph)

    # lowering
    graph.lower_to_machine_op()
    chip.lower_to_inst()

    # write to file
    chip.dump_output_to_file(args.trace)
