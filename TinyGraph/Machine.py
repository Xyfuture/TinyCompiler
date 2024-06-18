from __future__ import annotations

from math import ceil
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel
import json


class XbarArrayConfig(BaseModel):
    xbar_cell_bit: int = 1
    xbar_size: Tuple[int, int] = (256, 256)


class CoreConfig(BaseModel):
    xbar_array: XbarArrayConfig = XbarArrayConfig()
    xbar_array_cnt: int = 64
    local_buffer_size: int = 1024 * 1024  # 1 MBytes

    weight_precision: int = 16  # bits


class ChipConfig(BaseModel):
    core_cnt: int = 64
    offchip_memory_size: int = 1024 * 1024 * 1024  # 1GB
    core_config: CoreConfig = CoreConfig()

    mapping_strategy: str = "performance"


class Core:
    id_counter: int = 0
    id_map: Dict[int, Core] = {}

    def __init__(self, core_config: CoreConfig = CoreConfig()):
        Core.id_counter += 1
        self.core_id = Core.id_counter
        Core.id_map[self.core_id] = self

        self.core_config = core_config
        self.memory_allocator = MemoryAllocator(self.core_config.local_buffer_size)  # 设置内存分配器

        from TinyGraph.MachineOps import MachineOp
        self.machine_op_list: List[MachineOp] = []
        self.inst_list: List[Dict] = []
        self.dummy_inst = []

        self.xbar_allocator = XbarAllocator(self.core_config.xbar_array_cnt)

    def assign_group(self, request_xbar_cnt: int):
        return self.xbar_allocator.assign_xbar_group(request_xbar_cnt)

    @classmethod
    def get_core_by_id(cls, core_id: int):
        if core_id >= 1:
            return cls.id_map[core_id]
        return None

    def inst_code_gen(self):
        for index, op in enumerate(self.machine_op_list):
            inst = op.lower_to_inst()
            if isinstance(inst, list):
                self.inst_list.extend(inst)
            else:
                self.inst_list.append(inst)


class Chip:
    current_chip: Optional[Chip] = None

    # 基于 chip 去做资源分配
    def __init__(self, chip_config: ChipConfig = ChipConfig()):
        self.chip_config = chip_config

        self.core_array: List[Core] = []
        for i in range(self.chip_config.core_cnt):
            self.core_array.append(Core(self.chip_config.core_config))

        self.dram_allocator = MemoryAllocator(self.chip_config.offchip_memory_size)

        # 为实现 mapping 需要记录的变量
        self.next_mapping_index = 0  # 每次都是从这个index开始，向后查找能mapping的 core

    def lower_to_inst(self):
        for core in self.core_array:
            core.inst_code_gen()

    def dump_inst_to_file(self, file_path: str):
        for core in self.core_array:
            with open(file_path + f'core_{core.core_id}.json', "w") as f:
                json_data = json.dumps(core.inst_list, separators=(',', ': '))
                f.write(json_data)

    def dump_output_to_file(self, file_path: str):
        # 生成mapping 信息,方便模拟器报告利用率
        mapping_dict: Dict[str, Dict[int, int]] = {}

        for core in self.core_array:
            core_xbar_mapping_dict = core.xbar_allocator.xbar_group_id_cnt_map
            mapping_dict[f'core_{core.core_id}'] = core_xbar_mapping_dict

        # 生成指令信息
        inst_dict: Dict[str, List[Dict]] = {}

        for core in self.core_array:
            inst_dict[f'core_{core.core_id}'] = core.inst_list

        output_dict = {"mapping": mapping_dict, "inst": inst_dict}
        # 写入到文件中
        with open(file_path, 'w') as f:
            json_data = json.dumps(output_dict, separators=(',', ':'))
            json_data = json_data.replace('},', '},\n')
            f.write(json_data)

    def mapping_matrix_to_core(self, matrix_shape: Tuple[int, int]):
        xbar_cell_bit = self.chip_config.core_config.xbar_array.xbar_cell_bit
        xbar_size = self.chip_config.core_config.xbar_array.xbar_size

        xbar_rows, xbar_cols = xbar_size

        matrix_rows, matrix_cols = matrix_shape

        xbar_cnt_per_group = ceil(
            (matrix_cols * self.chip_config.core_config.weight_precision / xbar_cell_bit) / xbar_cols)
        group_cnt = ceil(matrix_rows / xbar_rows)

        # assign_list: List[Tuple[int, int]] = []  # (core_id,group_id)

        if self.chip_config.mapping_strategy == 'utilization':
            return self.utilization_first_mapping(group_cnt, xbar_cnt_per_group)
        elif self.chip_config.mapping_strategy == 'performance':
            return self.performance_first_mapping(group_cnt, xbar_cnt_per_group)

        raise NotImplemented

    def get_xbar_usage(self):
        used_xbar_cnt = 0
        total_xbar_cnt = 0

        for core in self.core_array:
            total_xbar_cnt += core.xbar_allocator.xbar_cnt
            used_xbar_cnt += core.xbar_allocator.xbar_cnt - core.xbar_allocator.empty_xbar_cnt

        return used_xbar_cnt / total_xbar_cnt, used_xbar_cnt, total_xbar_cnt

    # 如何分配新的核
    def performance_first_mapping(self, group_cnt: int, xbar_cnt_per_group: int):
        # 返回core id 和 xbar group id
        # 每次都是从没有map过的core 开始 map

        assign_list = []
        assigned_cnt = 0

        while assigned_cnt < group_cnt:
            if self.next_mapping_index >= self.chip_config.core_cnt:
                raise "Resource Error"

            cur_core = self.core_array[self.next_mapping_index]
            while cur_core.xbar_allocator.empty_xbar_cnt >= xbar_cnt_per_group and \
                    assigned_cnt < group_cnt:
                xbar_group_id = cur_core.xbar_allocator.assign_xbar_group(xbar_cnt_per_group)
                assign_list.append((cur_core.core_id, xbar_group_id))
                assigned_cnt += 1
            self.next_mapping_index += 1


        return assign_list

    def utilization_first_mapping(self, group_cnt: int, xbar_cnt_per_group: int):
        # 返回 core id 和 xbar group id 的 list
        assign_list: List[Tuple[int, int]] = []
        assigned_cnt = 0
        for core in self.core_array:
            while core.xbar_allocator.empty_xbar_cnt >= xbar_cnt_per_group and \
                    assigned_cnt < group_cnt:
                xbar_group_id = core.assign_group(xbar_cnt_per_group)
                assign_list.append((core.core_id, xbar_group_id))
                assigned_cnt += 1

        if assigned_cnt != group_cnt:
            print("No Enough Xbars! Please use larger chip!")
        return assign_list


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

    @property
    def empty_size(self) -> int:
        return self.memory_capacity - self.allocated_size

    def get_block_info(self, block_start_addr):
        # 返回地址对应block的信息, 起始地址,大小和占用情况
        for index, node in enumerate(self.memory_blocks):
            memory_block: MemoryBlock = node.payload
            start_addr, end_addr, allocated = memory_block.start_addr, memory_block.end_addr, memory_block.allocated
            if start_addr == block_start_addr:
                return start_addr, end_addr - start_addr, allocated

        # 没有找到的情况
        return None

    def malloc(self, request_size: int) -> Optional[int]:
        # 输入 申请的内存大小, 返回内存地址 如果内存空间不够,就返回None 记得 check 一下
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
            # return _malloc()
            addr = _malloc()
            if addr is None:
                raise "Out of Memory, No Enough Memory Capacity"

        return addr

    def free(self, addr: int):
        # 释放相应的内存地址
        is_freed = False
        for index, node in enumerate(self.memory_blocks):
            memory_block = node.payload
            if memory_block.start_addr == addr:
                memory_block.allocated = False
                self.allocated_size -= (memory_block.end_addr - memory_block.start_addr)
                is_freed = True
        assert is_freed

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


class XbarAllocator:
    def __init__(self, xbar_cnt: int):
        self.xbar_cnt = xbar_cnt
        self.xbar_group_cnt = 0
        self.xbar_group_id_cnt_map: Dict[int, int] = {}  # 映射 xbar_group_id 到 该group xbar_cnt 的一个映射
        self.xbar_group_id_addr_map: Dict[int, int] = {}  # 映射 xbar_group_id 到 其在 memory_allocator中的地址

        self.allocator = MemoryAllocator(xbar_cnt)

    @property
    def empty_xbar_cnt(self):
        return self.allocator.empty_size

    def assign_xbar_group(self, request_xbar_cnt):
        addr = self.allocator.malloc(request_xbar_cnt)
        if addr is not None:
            self.xbar_group_id_addr_map[self.xbar_group_cnt] = addr
            self.xbar_group_id_cnt_map[self.xbar_group_cnt] = request_xbar_cnt
            xbar_group_id = self.xbar_group_cnt
            self.xbar_group_cnt += 1
            return xbar_group_id
        else:
            return None  # 分配失败
