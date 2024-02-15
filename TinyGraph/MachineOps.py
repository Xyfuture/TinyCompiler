from __future__ import annotations

from typing import Dict, Tuple

from TinyGraph.Machine import MemoryAllocator


class AddressManager:
    def __init__(self):
        pass


class MachineOp:
    # 直接在machine op 中构建依赖关系
    def __init__(self, core_id: int, op_name: str, input_ops: Dict[str, MachineOp]):
        self.core_id = core_id
        self.op_name = op_name

        self._user_count: int = 0  # 依赖控制转到address manager中进行操作

        self.input_ops: Dict[str, MachineOp] = input_ops
        for k, input_op in self.input_ops.items():
            input_op._user_count += 1

    def release_memory(self):
        # 调用 address manager 释放内存
        pass

    def before_code_gen(self):
        pass

    def after_code_gen(self):
        # 释放依赖关系
        for k,input_op in self.input_ops.items():
            input_op._user_count -= 1
            if input_op._user_count == 0:
                input_op.release_memory()

    def code_gen(self):
        # vec len 的问题
        pass

    def lower_to_inst(self):
        self.before_code_gen()
        inst = self.code_gen()
        self.after_code_gen()
        return inst


class MachineVectorOp(MachineOp):
    pass


class MachineMatrixOp(MachineOp):
    pass


class MachineTransferOp(MachineOp):
    pass
