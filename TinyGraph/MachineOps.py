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
        if self.output_manager:
            self.output_manager.free()

    def before_code_gen(self):
        if self.output_manager:
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

    def __repr__(self):
        return "MachineOp"

class MachineVectorOp(MachineOp):
    def __init__(self, core_id: int, vector_op_name: str, input_ops: List[MachineOp],
                 output_manager: Optional[AddressManager], **kwargs):
        super().__init__(core_id, 'vector ' + vector_op_name, input_ops, output_manager)
        self.vector_op_name = vector_op_name

        self.kwargs = kwargs  # 存储 imm

    def code_gen(self) -> Dict:
        for src_op in self.input_ops:
            assert src_op.output_manager.size == self.output_manager.size

        inst = {"op": self.vector_op_name, 'dst_addr': self.output_manager.addr, 'len': self.output_manager.size}

        for i, src_op in enumerate(self.input_ops):
            inst[f'src{i}_addr'] = src_op.output_manager.addr

        for k, v in self.kwargs.items():
            inst[k] = v

        return inst

    def __repr__(self):
        return "VectorOp"


class MachineMatrixOp(MachineOp):
    def __init__(self, core_id: int, input_ops: List[MachineOp], output_manager: Optional[AddressManager],
                 input_size: int, output_size: int, group_id: int, xbar_cnt: int):
        super().__init__(core_id, 'gemv', input_ops, output_manager)

        self.input_size = input_size
        self.output_size = output_size

        self.group_id = group_id
        self.xbar_cnt = xbar_cnt

    def code_gen(self) -> Dict:
        assert self.output_manager.size == self.output_size
        assert len(self.input_ops) == 1
        assert self.input_size == self.input_ops[0].output_manager.size

        inst = {'op': 'gemv', 'dst_addr': self.output_manager.addr,
                'input_len': self.input_size, 'output_len': self.output_manager.size,
                'group_id': self.group_id, 'xbar_cnt': self.xbar_cnt,
                'src1_addr': self.input_ops[0].output_manager.addr}

        return inst

    def __repr__(self):
        return "MatrixOp"


class MachineTransferOp(MachineOp):
    def __init__(self, core_id: int, transfer_op_name: str, input_ops: List[MachineOp],
                 output_manager: Optional[AddressManager],
                 dst_core_id: Optional[int] = None, src_core_id: Optional[int] = None):
        super().__init__(core_id, 'transfer ' + transfer_op_name, input_ops, output_manager)
        # 这个会有一些特殊

        self.transfer_op_name = transfer_op_name

        self.dst_core_id = dst_core_id
        self.src_core_id = src_core_id

    def code_gen(self) -> Dict:
        # 直接使用send recv 进行数据传输
        # dram_to_local local_to_dram
        # send recv
        # dram_clr local_clr

        inst = {'op': self.transfer_op_name}

        if self.transfer_op_name == 'send':
            assert len(self.input_ops) == 1
            inst['src1_addr'] = self.input_ops[0].output_manager.addr
            inst['imm'] = self.dst_core_id  # core
            inst['len'] = self.input_ops[0].output_manager.size
        elif self.transfer_op_name == 'recv':
            assert len(self.input_ops) == 0
            inst['dst_addr'] = self.output_manager.addr
            inst['imm'] = self.src_core_id
            inst['len'] = self.output_manager.size
        elif self.transfer_op_name == 'local_clr':
            inst['dst_addr'] = self.output_manager.addr
            inst['len'] = self.output_manager.size
        elif self.transfer_op_name == 'dram_to_local':
            inst['dst_addr'] = self.output_manager.addr
            inst['src1_addr'] = self.input_ops[0].output_manager.addr
            inst['len'] = self.input_ops[0].output_manager.size
        else:
            assert False

        return inst

    def __repr__(self):
        return "TransferOp"


class MachineRearrangeOp(MachineOp):
    # 重新排布内存布局  类似于 im2col的操作
    def __init__(self, core_id: int, input_ops: List[MachineOp], output_manager: Optional[AddressManager],
                 start_offset: int, end_offset: int):
        super().__init__(core_id, 're_arrange', input_ops, output_manager)

        self.start_offset = start_offset
        self.end_offset = end_offset

    def code_gen(self) -> List[Dict]:
        # 堆上一堆 local cpy 可以实现数据的重新布局
        # 注意偏移的情况
        inst_list = []
        if len(self.input_ops) == 1:
            # 一条指令的情况可以特殊处理
            cur_input_op = self.input_ops[0]
            start_address = cur_input_op.output_manager.addr + self.start_offset
            size = self.end_offset - self.start_offset
            assert size == self.output_manager.size
            inst_list.append(
                {'op': 'local_cpy', 'dst_addr': self.output_manager.addr,
                 'src1_addr': start_address, 'len': self.output_manager.size}
            )
        else:
            dst_addr = self.output_manager.addr

            start_input_op = self.input_ops[0]
            start_address = start_input_op.output_manager.addr + self.start_offset
            start_len = start_input_op.output_manager.size - self.start_offset
            assert start_len > 0
            inst_list.append(
                {'op': 'local_cpy', 'dst_addr': dst_addr,
                 'src1_addr': start_address, 'len': start_len}
            )
            dst_addr += start_len

            # 处理中间的情况
            for i in range(1, len(self.input_ops) - 1):
                cur_input_op = self.input_ops[i]
                inst_list.append(
                    {'op': 'local_cpy', 'dst_addr': dst_addr,
                     'src1_addr': cur_input_op.output_manager.addr, 'len': cur_input_op.output_manager.size}
                )
                dst_addr += cur_input_op.output_manager.size

            # 处理最后的情况
            end_input_op = self.input_ops[-1]
            inst_list.append(
                {'op': 'local_cpy', 'dst_addr': dst_addr,
                 'src1_addr': end_input_op.output_manager.addr, 'len': self.end_offset}
            )
            dst_addr += self.end_offset

            assert dst_addr == self.output_manager.addr + self.output_manager.size

        return inst_list

    def __repr__(self):
        return "RearrangeOp"
