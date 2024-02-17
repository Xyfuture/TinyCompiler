from __future__ import annotations

from typing import Dict, Tuple, Optional, OrderedDict, List

from TinyGraph.Machine import MemoryAllocator


class AddressManager:
    def __init__(self, size: int, memory_allocator: MemoryAllocator):
        self.size: int = size
        self.addr: Optional[int] = None
        self.memory_allocator: MemoryAllocator = memory_allocator

    def malloc(self):
        if self.addr is None:
            self.addr = self.memory_allocator.malloc(self.size)
        return self.addr

    def free(self):
        self.memory_allocator.free(self.addr)

    @property
    def address_info(self):
        return self.addr, self.size


class SharedAddressManager(AddressManager):
    id_counter: int = 0
    id_map: Dict[int, SharedAddressManager] = {}

    def __init__(self, size: int, memory_allocator: MemoryAllocator):
        super().__init__(size, memory_allocator)
        self.reuse_times = 0

    def malloc(self):
        self.reuse_times += 1
        return super().malloc()

    def free(self):
        if self.reuse_times == 0:
            assert False
        self.reuse_times -= 1
        if self.reuse_times == 0:
            self.memory_allocator.free(self.addr)

    @classmethod
    def get_shr_addr_manager_by_id(cls, id: int) -> Optional[SharedAddressManager]:
        if id in cls.id_map:
            return cls.id_map[id]
        return None

    @classmethod
    def get_new_shr_addr_manager_id(cls):
        cls.id_counter += 1
        return cls.id_counter

    @classmethod
    def get_shr_addr_manager(cls, id: int, size: int, memory_allocator: MemoryAllocator) -> SharedAddressManager:
        if id in cls.id_map:
            return cls.id_map[id]
        else:
            shr_addr_manager = SharedAddressManager(size, memory_allocator)
            cls.id_map[id] = shr_addr_manager
            return shr_addr_manager


class MachineOp:
    # 直接在machine op 中构建依赖关系
    # 一条machine op 只能有一个输出, 但是可能生成多条指令
    def __init__(self, core_id: int, op_name: str,
                 input_ops: List[MachineOp], output_manager: Optional[AddressManager]):
        self.core_id = core_id
        self.op_name = op_name

        self._user_count: int = 0  # 依赖控制转到address manager中进行操作

        self.input_ops: List[MachineOp] = input_ops  # key 这里应该是src1 or src2 之类的
        for input_op in self.input_ops:
            input_op._user_count += 1

        self.output_manager = output_manager  # 用于管理输出的内存地址

    def release_memory(self):
        # 调用 address manager 释放内存
        self.output_manager.free()

    def before_code_gen(self):
        self.output_manager.malloc()

    def after_code_gen(self):
        # 释放依赖关系
        for input_op in self.input_ops:
            input_op._user_count -= 1
            if input_op._user_count == 0:
                input_op.release_memory()

    def code_gen(self) -> Dict:
        # vec len 的问题
        return {}

    def lower_to_inst(self):
        self.before_code_gen()
        inst = self.code_gen()
        self.after_code_gen()
        return inst


class MachineVectorOp(MachineOp):
    def __init__(self, core_id: int, vector_op_name: str, input_ops: List[MachineOp],
                 output_manager: Optional[AddressManager]):
        super().__init__(core_id, 'vector ' + vector_op_name, input_ops, output_manager)
        pass


class MachineMatrixOp(MachineOp):
    def __init__(self, core_id: int, input_ops: List[MachineOp], output_manager: Optional[AddressManager],
                 start_offset:int, output_size:int ):
        super().__init__(core_id, 'gemv', input_ops, output_manager)

        self.start_offset = start_offset
        self.output_size = output_size

    def code_gen(self) -> Dict:
        pass


class MachineTransferOp(MachineOp):
    def __init__(self, core_id: int, transfer_op_name: str, input_ops: List[MachineOp],
                 output_manager: Optional[AddressManager]):
        super().__init__(core_id, 'transfer ' + transfer_op_name, input_ops, output_manager)
        # 这个会有一些特殊

    def code_gen(self) -> Dict:
        # 插入一个同步指令
        # 插入一个 global memory的传输指令
        pass


class MachineRearrangeOp(MachineOp):
    # 重新排布内存布局  类似于 im2col的操作
    def __init__(self, core_id: int, input_ops: List[MachineOp], output_manager: Optional[AddressManager]):
        super().__init__(core_id, 're_arrange', input_ops, output_manager)

    def code_gen(self) -> Dict:
        pass
