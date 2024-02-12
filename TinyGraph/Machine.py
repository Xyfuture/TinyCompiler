from __future__ import annotations
from typing import List, Dict, Tuple

from pydantic import BaseModel


class MachineOp:
    def __init__(self, core_id: int, op_name: str, *args, **kwargs):
        self.core_id = core_id
        self.op_name = op_name
        self.args = args
        self.kwargs = kwargs


class CoreConfig(BaseModel):
    xbar_cell_bit: int = 2
    xbar_size: Tuple[int, int] = (128, 128)
    xbar_cnt: int = 8


class Core:
    id_counter: int = 0
    id_map: Dict[int, Core] = {}

    def __init__(self):
        Core.id_counter += 1
        self.core_id = Core.id_counter
        Core.id_map[self.core_id] = self

        self.inst: List[MachineOp] = []
        self.dummy_inst = []

    @classmethod
    def get_core_by_id(cls, core_id: int):
        if core_id >= 1:
            return cls.id_map[core_id]
        return None


class Chip:
    # 基于 chip 去做资源分配
    def __init__(self):
        pass

