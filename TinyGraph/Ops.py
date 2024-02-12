from typing import List, Optional, Tuple

from TinyGraph.Graph import MicroOp, MicroGraph
from TinyGraph.Machine import Core


class AddOp(MicroOp):
    def __init__(self, core_id: int, vector_size: int, src_op_1: MicroOp, src_op_2: MicroOp):
        super().__init__()
        self.core_id = core_id
        self.vector_size = vector_size

        self.input_ops = [src_op_1, src_op_2]

    def code_gen(self):
        pass

    def dummy_code_gen(self):
        core = Core.get_core_by_id(self.core_id)
        if core:
            core.dummy_inst.append(self.full_info())

    def full_info(self):
        return f"AddOp-{self.op_id} #src1: {self.input_ops[0]} #src2:{self.input_ops[1]}"

    def __repr__(self):
        return f"AddOp-{self.op_id}"


class TransferOp(MicroOp):
    def __init__(self, src_core_id: int, dst_core_id: int, data_size: int, src_op: MicroOp):
        super().__init__()
        self.src_core_id = src_core_id
        self.dst_core_id = dst_core_id
        self.data_size = data_size

        self.src_op = src_op

    def code_gen(self):
        pass

    def dummy_code_gen(self):
        src_core = Core.get_core_by_id(self.src_core_id)
        if src_core:
            src_core.dummy_inst.append(
                f"TransferOp-{self.op_id} #src:{self.src_op} #dst: core id {self.dst_core_id}"
            )
        dst_core = Core.get_core_by_id(self.dst_core_id)
        if dst_core:
            dst_core.dummy_inst.append(
                f"TransferOp-{self.op_id} #src: core id {self.src_core_id}  op {self.src_op}"
            )

    def full_info(self):
        return f"TransferOp-{self.op_id} #src: core id {self.src_core_id} op {self.src_op} #dst: core id {self.dst_core_id}"

    def __repr__(self):
        return f"TransferOp-{self.op_id}"


class MatVecMulOp(MicroOp):
    def __init__(self, core_id: int, xbar_group_id: int, input_size: int, output_size: int,
                 src_vec_op_list: Optional[List[MicroOp]] = None):
        super().__init__()

        self.core_id = core_id
        self.xbar_group_id = xbar_group_id
        self.src_vec_op_list = src_vec_op_list

        self.input_size = input_size
        self.output_size = output_size

    def code_gen(self):
        pass

    def dummy_code_gen(self):
        core = Core.get_core_by_id(self.core_id)
        if core:
            core.dummy_inst.append(self.full_info())

    def __repr__(self):
        return f"MatVecMulOp-{self.op_id}"

    def full_info(self):
        return f"MatVecMulOp-{self.op_id} #src: {self.src_vec_op_list}"


class MaxPool2dOp(MicroOp):
    def __init__(self, core_id: int, kernel_size: Tuple[int, int], vector_size: int,
                 src_op_list: List[MicroOp]):
        super().__init__()
        self.core_id = core_id

        self.kernel_size = kernel_size
        self.vector_size = vector_size

        self.src_op_list = src_op_list

    def code_gen(self):
        pass

    def dummy_code_gen(self):
        core = Core.get_core_by_id(self.core_id)
        if core:
            core.dummy_inst.append(self.full_info())

    def full_info(self):
        return f"MaxPool2dOp-{self.op_id} #src: {self.src_op_list}"

    def __repr__(self):
        return f"MaxPool2dOp-{self.op_id}"


class PadOp(MicroOp):
    def __init__(self, core_id: int):
        super().__init__()

        self.core_id = core_id

    def code_gen(self):
        pass

    def dummy_code_gen(self):
        core = Core.get_core_by_id(self.core_id)
        if core:
            core.dummy_inst.append(
                self.full_info()
            )

    def full_info(self):
        return f"PadOp-{self.core_id} core id:{self.core_id}"

    def __repr__(self):
        return f"PadOp-{self.op_id}"


class ReLUOp(MicroOp):
    def __init__(self, core_id: int, src_op: MicroOp):
        super().__init__()
        self.core_id = core_id

        self.src_op = src_op

    def code_gen(self):
        pass

    def dummy_code_gen(self):
        core = Core.get_core_by_id(self.core_id)
        if core:
            core.dummy_inst.append(self.full_info())

    def full_info(self):
        return f"ReLUOp-{self.op_id} #src: {self.src_op}"

    def __repr__(self):
        return f"ReLUOp-{self.op_id}"


class RootOp(MicroOp):
    def __init__(self):
        super().__init__()


def transfer_fusion(graph: MicroGraph):
    for node in graph.nodes:
        if isinstance(node.micro_op, TransferOp):
            if len(node._output_nodes) == 1:
                # 执行fusion 操作
                new_input_nodes = []
                use_node = next(iter(node._output_nodes))

                for input_node in use_node._input_nodes:
                    if input_node is node:
                        continue
                    new_input_nodes.append(input_node)

                node.add_input_nodes(new_input_nodes)

    return graph
