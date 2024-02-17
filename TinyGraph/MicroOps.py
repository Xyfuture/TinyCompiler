from typing import List, Optional, Tuple, Dict

from TinyGraph.Graph import MicroOp, MicroGraph, MicroNode
from TinyGraph.Machine import Core, Chip
from TinyGraph.MachineOps import MachineVectorOp, AddressManager, SharedAddressManager, MachineTransferOp, MachineOp, \
    MachineRearrangeOp, MachineMatrixOp


class AddOp(MicroOp):
    def __init__(self, core_id: int, vector_size: int, src_ops: List[MicroOp], shr_manager_id: int = 0):
        super().__init__(core_id, vector_size, src_ops, shr_manager_id)
        self.core_id = core_id

    def machine_op_gen(self):
        core = Core.get_core_by_id(self.core_id)
        input_machine_ops = [op.output_machine_op for op in self.src_ops]

        output_manager = self.get_core_address_manager()

        self.output_machine_op = MachineVectorOp(self.core_id, 'vadd', input_machine_ops, output_manager)
        core.machine_op_list.append(self.output_machine_op)

    def dummy_code_gen(self):
        core = Core.get_core_by_id(self.core_id)
        if core:
            core.dummy_inst.append(self.full_info())

    def full_info(self):
        return f"AddOp-{self.op_id} #src1: {self.src_ops[0]} #src2:{self.src_ops[1]}"

    def __repr__(self):
        return f"AddOp-{self.op_id}"


class TransferOp(MicroOp):
    def __init__(self, src_core_id: int, dst_core_id: int, vector_size: int, src_op: MicroOp, shr_manager_id: int = 0):
        super().__init__(dst_core_id, vector_size, [src_op], shr_manager_id)
        self.src_core_id = src_core_id
        self.dst_core_id = dst_core_id

        # self.src_op = src_op

    def machine_op_gen(self):
        def inter_core():
            # 需要从两个core上下手
            # 涉及到global buffer的内存读取

            src_core = Core.get_core_by_id(self.src_core_id)
            dst_core = Core.get_core_by_id(self.dst_core_id)

            # src core part
            # 得改成send/recv的模式 这里直接send吧
            # TODO 目前两个src id dst id是互相不知道的
            src_machine_op = MachineTransferOp(self.src_core_id, 'send', [self.src_ops[0].output_machine_op], None)
            src_core.machine_op_list.append(src_machine_op)

            output_manager = self.get_core_address_manager()
            dst_machine_op = MachineTransferOp(self.dst_core_id, 'recv', [], output_manager)
            self.output_machine_op = dst_machine_op
            dst_core.machine_op_list.append(dst_machine_op)

        def dram_core():
            # src_core_id = -1
            dst_core = Core.get_core_by_id(self.dst_core_id)

            # TODO 对模拟器的改动 添加 dram to local的操作
            output_manager = self.get_core_address_manager()
            dst_machine_op = MachineTransferOp(self.dst_core_id, 'dram_to_local', [self.src_ops[0].output_machine_op],
                                               output_manager)
            self.output_machine_op = dst_machine_op
            dst_core.machine_op_list.append(dst_machine_op)

        if self.src_core_id == -1:
            dram_core()
        else:
            inter_core()

    def dummy_code_gen(self):
        src_core = Core.get_core_by_id(self.src_core_id)
        if src_core:
            src_core.dummy_inst.append(
                f"TransferOp-{self.op_id} #src:{self.src_ops[0]} #dst: core id {self.dst_core_id}"
            )
        dst_core = Core.get_core_by_id(self.dst_core_id)
        if dst_core:
            dst_core.dummy_inst.append(
                f"TransferOp-{self.op_id} #src: core id {self.src_core_id}  op {self.src_ops[0]}"
            )

    def full_info(self):
        return f"TransferOp-{self.op_id} #src: core id {self.src_core_id} op {self.src_ops[0]} #dst: core id {self.dst_core_id}"

    def __repr__(self):
        return f"TransferOp-{self.op_id}"


class MatVecMulOp(MicroOp):
    def __init__(self, core_id: int, xbar_group_id: int, input_size: int, output_size: int,
                 src_op_list: List[MicroOp], start_offset, end_offset, shr_manager_id: int = 0):
        super().__init__(core_id, output_size, src_op_list, shr_manager_id)

        self.core_id = core_id
        self.xbar_group_id = xbar_group_id

        self.input_size = input_size
        self.output_size = output_size

        self.start_offset = start_offset
        self.end_offset = end_offset
        # TODO 细化offset到 machine op中去

    def machine_op_gen(self):
        core = Core.get_core_by_id(self.core_id)

        # TODO 修改 模拟器 改变其 读写内存的方式,不在以 pe num 为准
        # TODO 支持对应位置的访存方式

        # 输入是一系列 vector , 需要先reshape到一个vector
        input_machine_ops = [op.output_machine_op for op in self.src_ops]
        reshape_output_manager = AddressManager(self.input_size, core.memory_allocator)  # TODO 可能也需要修改这一个
        reshape_machine_op = MachineRearrangeOp(self.core_id, input_machine_ops, reshape_output_manager)
        core.machine_op_list.append(reshape_machine_op)

        # 执行矩阵操作
        output_manager = self.get_core_address_manager()
        matrix_machine_op = MachineMatrixOp(self.core_id, [reshape_machine_op], output_manager,
                                            self.start_offset, self.output_size)
        self.output_machine_op = matrix_machine_op
        core.machine_op_list.append(matrix_machine_op)

    def dummy_code_gen(self):
        core = Core.get_core_by_id(self.core_id)
        if core:
            core.dummy_inst.append(self.full_info())

    def __repr__(self):
        return f"MatVecMulOp-{self.op_id}"

    def full_info(self):
        return f"MatVecMulOp-{self.op_id} #src: {self.src_ops}"


class MaxPool2dOp(MicroOp):
    def __init__(self, core_id: int, kernel_size: Tuple[int, int], vector_size: int,
                 src_op_list: List[MicroOp], shr_manager_id: int = 0):
        super().__init__(core_id, vector_size, src_op_list, shr_manager_id)
        self.core_id = core_id

        self.kernel_size = kernel_size

    def machine_op_gen(self):
        core = Core.get_core_by_id(self.core_id)
        assert self.kernel_size[0] * self.kernel_size[1] == len(self.src_ops)
        prev_machine_op = self.src_ops[0].output_machine_op
        for i in range(1, len(self.src_ops)):
            cur_micro_op = self.src_ops[1]

            input_machine_ops = [prev_machine_op, cur_micro_op.output_machine_op]

            output_manager = self.get_core_address_manager()

            self.output_machine_op = MachineVectorOp(self.core_id, 'vmax', input_machine_ops, output_manager)
            core.machine_op_list.append(self.output_machine_op)

            prev_machine_op = self.output_machine_op

    def dummy_code_gen(self):
        core = Core.get_core_by_id(self.core_id)
        if core:
            core.dummy_inst.append(self.full_info())

    def full_info(self):
        return f"MaxPool2dOp-{self.op_id} #src: {self.src_ops}"

    def __repr__(self):
        return f"MaxPool2dOp-{self.op_id}"


class PadOp(MicroOp):
    def __init__(self, core_id: int, vector_size: int, shr_manager_id: int = 0):
        super().__init__(core_id, vector_size, [], shr_manager_id)
        self.core_id = core_id

    def machine_op_gen(self):
        # TODO 修改这个 因为不知道core id  可以通过添加Pass的方式实现core id的读取
        core = Core.get_core_by_id(self.core_id)
        output_manager = AddressManager(self.vector_size, core.memory_allocator)
        self.output_machine_op = MachineTransferOp(self.core_id, 'local_clr', [], output_manager)
        core.machine_op_list.append(self.output_machine_op)

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


class RawInputOp(MicroOp):
    # 片外内存中的图像输入
    def __init__(self, vector_size: int):
        super().__init__(-1, vector_size, [], False)

    def machine_op_gen(self):
        # 仅仅进行片外内存的申请
        # 直接执行lower to inst 操作
        output_manager = AddressManager(self.vector_size, Chip.current_chip.dram_allocator)

        self.output_machine_op = MachineOp(-1, 'raw_input', [], output_manager)

        self.output_machine_op.lower_to_inst()  # 拿到一个dram的地址


class ReLUOp(MicroOp):
    def __init__(self, core_id: int, vector_size: int, src_op: MicroOp, shr_manager_id: int = 0):
        super().__init__(core_id, vector_size, [src_op], shr_manager_id)
        self.core_id = core_id

    def machine_op_gen(self):
        core = Core.get_core_by_id(self.core_id)
        src_machine_op = self.src_ops[0].output_machine_op
        input_ops = [src_machine_op]

        output_manager = self.get_core_address_manager()

        self.output_machine_op = MachineVectorOp(self.core_id, 'relu', input_ops, output_manager)
        core.machine_op_list.append(self.output_machine_op)

    def dummy_code_gen(self):
        core = Core.get_core_by_id(self.core_id)
        if core:
            core.dummy_inst.append(self.full_info())

    def full_info(self):
        return f"ReLUOp-{self.op_id} #src: {self.src_ops[0]}"

    def __repr__(self):
        return f"ReLUOp-{self.op_id}"


class RootOp(MicroOp):
    def __init__(self):
        super().__init__()


# optimization pass
def transfer_fusion(graph: MicroGraph):
    # TODO 这个是不是有问题来 忘了?
    # TODO 其他的一个问题,对micro node进行修改之后,对于micro op的input list 也应该进行修改
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


def pad_to_core(graph: MicroGraph):
    # 让原本没有分配 core id 的pad支持分配 core id
    for node in graph.nodes:
        micro_op = node.micro_op
        if isinstance(micro_op, PadOp) and micro_op.core_id == -1:
            # 未经过分配的 pad
            user_core_map: Dict[int, PadOp] = {}
            user_node: MicroNode
            for user_node in node._output_nodes:
                user_micro_op = user_node.micro_op
                user_core_id = user_micro_op.core_id

                if user_core_id in user_core_map:
                    new_pad_op = user_core_map[user_core_id]
                    # TODO 大改一下micro op 的结构了
                    # 现在micro op 不支持 对 src 进行修改 这是个问题

                else:
                    pass
