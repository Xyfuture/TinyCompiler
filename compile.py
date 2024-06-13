import argparse
import pickle

from Networks.auto_encoder import AutoEncoder, get_autoencoder
from Networks.yolo import get_yolovgg
from TinyGraph.ConductArray import ConductArray
from TinyGraph.DSL import DepTensor
from TinyGraph.Graph import MicroGraph, topo_sort
from TinyGraph.Machine import Chip, ChipConfig
from TinyGraph.MicroOps import RawInputOp, pad_to_core, find_right_input_op
from TinyGraph.Module import DepModule, DepConv2d, DepLinear, report_mapping_status
from Networks.resnet18 import resnet18, get_resnet18
import json

from utils import create_input_tensor

parser = argparse.ArgumentParser(description='Compiler for CIM')

# 通过config 指定 mapping 方式
parser.add_argument('--strategy', '-s', type=str, default='performance',
                    help='Mapping Strategy : performance or utilization')
parser.add_argument('--target', '-t', type=str, default='trace', help="Output Target: trace or mapping")
parser.add_argument('--mapping', '-m', type=str, default=None, help='Mapping File Path')
parser.add_argument('--output', '-o', type=str, default=None, help='Output File Path: Trace or Mapping')
parser.add_argument('--config', '-c', type=str, default='example/resnet_config.json', help='Configuration File Path')
# parser.add_argument('--trace', '-t', type=str, default='example/resnet_trace.json', help='Trace File Path')
parser.add_argument('--network', '-n', type=str, default='resnet', help='Network Name')
parser.add_argument('--verbose', '-v', action='store_true', help='Report All Module Results')

args = parser.parse_args()

if __name__ == '__main__':

    split_line = "=" * 70
    print(f'{split_line}\n'
          f'{"Compiler Start".center(70)}\n'
          f'{split_line}\n')

    graph = MicroGraph()
    MicroGraph.current_graph = graph

    with open(args.config) as f:
        config_dict = json.load(f)
        chip_config = ChipConfig(**config_dict)
        if __name__ == '__main__':
            assert args.mapping in ['performance', 'utilization'], "Unsupported Mapping Strategy"
        chip_config.mapping_strategy = args.mapping

    chip = Chip(chip_config)
    Chip.current_chip = chip

    if args.network == 'resnet':
        net, input_tensor = get_resnet18()
    elif args.network == 'autoencoder':
        net, input_tensor = get_autoencoder()
    elif args.network == 'yolo':
        net, input_tensor = get_yolovgg()
    else:
        raise "Set your own network here"

    print(f'{split_line}\n'
          f'{"Network Mapping".center(70)}\n'
          f'{split_line}\n')

    if args.target == 'mapping':
        net.mapping()

        # 输出mapping的结果
        if args.verbose:
            split_line = "=" * 70
            print(f'{split_line}\n'
                  f'{"Mapping Results".center(70)}\n'
                  f'{split_line}\n')
            # 输出Mapping
            print(report_mapping_status(net))

        with open(args.output, 'wb') as f:
            pickle.dump(net, f)
    elif args.target == 'trace':
        if args.mapping is None:
            # 进行mapping， 根据 config 中的strategy方式进行映射
            net.mapping()
        else:
            with open(args.mapping, 'rb') as f:
                net = pickle.load(f)
        # 输出mapping的结果
        if args.verbose:
            split_line = "=" * 70
            print(f'{split_line}\n'
                  f'{"Mapping Results".center(70)}\n'
                  f'{split_line}\n')
            # 输出Mapping
            print(report_mapping_status(net))

        print(f'{split_line}\n'
              f'{"Code Generation".center(70)}\n'
              f'{split_line}\n')
        # 进行编译流程
        output_tensor = net(input_tensor)

        # run some optimization pass
        pad_to_core(graph)
        find_right_input_op(graph)

        # lowering
        graph.lower_to_machine_op()
        chip.lower_to_inst()

        # write to file
        chip.dump_output_to_file(args.output)

        print(f"File Write To {args.output}")
