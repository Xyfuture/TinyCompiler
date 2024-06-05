import argparse

from Networks.auto_encoder import AutoEncoder, get_autoencoder
from TinyGraph.ConductArray import ConductArray
from TinyGraph.DSL import DepTensor
from TinyGraph.Graph import MicroGraph, topo_sort
from TinyGraph.Machine import Chip, ChipConfig
from TinyGraph.MicroOps import RawInputOp, pad_to_core
from TinyGraph.Module import DepModule, DepConv2d, DepLinear, report_mapping_status
from Networks.resnet18 import resnet18, get_resnet18
import json

from utils import create_input_tensor

parser = argparse.ArgumentParser(description='Compiler for CIM')

# 通过config 指定 mapping 方式
parser.add_argument('--mapping', '-m', type=str, default='performance',help='Mapping Strategy : performance or utilization')
parser.add_argument('--config', '-c', type=str, default='example/resnet_config.json', help='Configuration File Path')
parser.add_argument('--trace', '-t', type=str, default='example/resnet_trace.json', help='Trace File Path')
parser.add_argument('--network', '-n', type=str, default='resnet', help='Network Name')
parser.add_argument('--verbose', '-v', action='store_false', help='Report All Module Results')

args = parser.parse_args()


# def resnet_run():
#     net = resnet18()
#     net.mapping()
#
#     input_tensor = create_input_tensor((32, 32), 3)
#     output_tensor = net(input_tensor)
#     return net
#
#
# def autoencoder_run():
#     # TODO 测试
#     net = AutoEncoder()
#     net.mapping()
#
#     input_tensor = create_input_tensor((1,),1024)
#     output_tensor = net(input_tensor)
#     return net


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

    if args.network == 'resnet':
        net,input_tensor = get_resnet18()
    elif args.network == 'autoencoder':
        net,input_tensor = get_autoencoder()
    else:
        raise "Set your own network here"

    print(f'{split_line}\n'
          f'{"Network Mapping".center(70)}\n'
          f'{split_line}\n')

    # 进行mapping， 根据 config 中的mapping方式进行映射
    net.mapping()



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

    # lowering
    graph.lower_to_machine_op()
    chip.lower_to_inst()

    # write to file
    chip.dump_output_to_file(args.trace)

    print(f"File Write To {args.trace}")
