from __future__ import  annotations
from typing import List, Dict


class MicroOp:
    def __init__(self, core_id: int, op_name: str, *args, **kwargs):
        self.core_id = core_id
        self.op_name = op_name
        self.args = args
        self.kwargs = kwargs


class Core:
    id_counter: int = 0
    id_map: Dict[int, Core] = {}

    def __init__(self, core_id: int):
        Core.id_counter += 1
        self.core_id = Core.id_counter
        Core.id_map[self.core_id] = self

        self.inst: List[MicroOp] = []

    @classmethod
    def get_core_by_id(cls, core_id: int):
        return cls.id_map[core_id]

