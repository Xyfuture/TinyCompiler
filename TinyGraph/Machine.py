from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel

from TinyGraph.MachineOps import MachineOp


class CoreConfig(BaseModel):
    xbar_cell_bit: int = 2
    xbar_size: Tuple[int, int] = (128, 128)
    xbar_cnt: int = 8
    local_buffer_size: int = 64 * 1024  # 64KByte


class Core:
    id_counter: int = 0
    id_map: Dict[int, Core] = {}

    def __init__(self, core_config: CoreConfig = CoreConfig()):
        Core.id_counter += 1
        self.core_id = Core.id_counter
        Core.id_map[self.core_id] = self

        self.core_config = core_config
        self.memory_allocator = MemoryAllocator(self.core_config.local_buffer_size)  # 设置内存分配器

        self.machine_op_list: List[MachineOp] = []
        self.dummy_inst = []

        self.xbar_mapping: Dict[int, int] = {}  # key: xbar id value: group id
        self.unused_xbar_cnt = self.core_config.xbar_cnt
        self.group_cnt = 0

    def assign_group(self, xbar_cnt: int):
        if self.unused_xbar_cnt < xbar_cnt:
            assert False
        self.unused_xbar_cnt -= xbar_cnt
        self.group_cnt += 1
        group_id = self.group_cnt
        for i in range(len(self.xbar_mapping), len(self.xbar_mapping) + xbar_cnt):
            self.xbar_mapping[i] = group_id
        return group_id

    @classmethod
    def get_core_by_id(cls, core_id: int):
        if core_id >= 1:
            return cls.id_map[core_id]
        return None


class ChipConfig(BaseModel):
    core_cnt: int = 16
    global_buffer_size: int = 1204 * 1024  # 1MByte
    dram_size: int = 1024 * 1024 * 1024
    core_config: CoreConfig = CoreConfig()


class Chip:
    current_chip: Optional[Chip] = None

    # 基于 chip 去做资源分配
    def __init__(self, chip_config: ChipConfig = ChipConfig()):
        self.chip_config = chip_config

        self.core_array: List[Core] = []
        for i in range(self.chip_config.core_cnt):
            self.core_array.append(Core(self.chip_config.core_config))

        self.dram_allocator = MemoryAllocator(self.chip_config.dram_size)

    def get_unmapped_core(self):
        for core in self.core_array:
            if core.unused_xbar_cnt == core.core_config.xbar_cnt:
                return core
        return None

    # 如何分配新的核


class _Node:
    def __init__(self, payload=None):
        self.prev: _Node = self
        self.next: _Node = self

        self.payload = payload

    def prepend(self, x: _Node):
        # 可能需要先删除x之前的连接情况
        p = self.prev
        p.next, x.prev = x, p
        x.next, self.prev = self, x

    def append(self, x: _Node):
        self.next.prepend(x)

    def remove(self):
        p, n = self.prev, self.next
        p.next, n.prev = n, p


class _LinkedList:
    def __init__(self):
        self.root: _Node = _Node()
        self.node_cnt: int = 0

    def insert_after(self, position: _Node, new_node: _Node):
        position.append(new_node)
        self.node_cnt += 1

    def insert_before(self, position: _Node, new_node: _Node):
        position.prepend(new_node)
        self.node_cnt += 1

    def append(self, new_node: _Node):
        self.root.prepend(new_node)
        self.node_cnt += 1

    def remove(self, node: _Node):
        node.remove()
        self.node_cnt -= 1

    def __len__(self):
        return self.node_cnt

    def __iter__(self):
        cur = self.root.next
        while cur is not self.root:
            yield cur
            cur = cur.next

    def __reversed__(self):
        cur = self.root.prev
        while cur is not self.root:
            yield cur
            cur = cur.prev


class MemoryBlock:
    def __init__(self, start_addr: int, end_addr: int, allocated: bool = False):
        self.start_addr: int = start_addr
        self.end_addr: int = end_addr
        self.allocated: bool = allocated


class MemoryAllocator:
    def __init__(self, memory_capacity: int):
        self.memory_capacity = memory_capacity
        self.memory_blocks: _LinkedList = _LinkedList()
        self.memory_blocks.append(_Node(MemoryBlock(0, self.memory_capacity, False)))
        self.allocated_size = 0

    def malloc(self, request_size: int) -> Optional[int]:
        # 输入 申请的内存大小, 返回内存地址
        def _malloc():
            for index, node in enumerate(self.memory_blocks):
                memory_block: MemoryBlock = node.payload
                start_addr, end_addr, allocated = memory_block.start_addr, memory_block.end_addr, memory_block.allocated
                if allocated:
                    continue
                block_size = end_addr - start_addr

                if block_size >= request_size:
                    allocated_addr = start_addr

                    if block_size > request_size:
                        node.payload = MemoryBlock(start_addr, start_addr + request_size, True)
                        new_node = _Node(MemoryBlock(start_addr + request_size, end_addr, False))
                        self.memory_blocks.insert_after(node, new_node)
                    else:
                        node.payload = MemoryBlock(start_addr, end_addr, True)

                    self.allocated_size += request_size
                    return allocated_addr
            return None

        addr = _malloc()
        if addr is None:
            self.merge_free_blocks()
            return _malloc()
        return addr

    def free(self, addr: int):
        # 释放相应的内存地址
        for index, node in enumerate(self.memory_blocks):
            memory_block = node.payload
            if memory_block.start_addr == addr:
                memory_block.allocated = False
                self.allocated_size -= (memory_block.end_addr - memory_block.start_addr)

    def merge_free_blocks(self):
        prev_node = None
        for node in self.memory_blocks:
            if prev_node is None:
                prev_node = node
                continue
            if (not prev_node.payload.allocated) and (not node.payload.allocated):
                assert prev_node.payload.end_addr == node.payload.start_addr
                # merge
                start_addr = prev_node.payload.start_addr
                end_addr = node.payload.end_addr
                prev_node.payload = MemoryBlock(start_addr, end_addr, False)

                node.remove()
                # prev node 不用向后移动
            else:
                prev_node = node
