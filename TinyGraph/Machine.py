class MicroOp:
    def __init__(self, core_id: int, op_name: str, *args, **kwargs):
        self.core_id = core_id
        self.op_name = op_name
        self.args = args
        self.kwargs = kwargs


class Core:
    def __init__(self, core_id: int):
        self.core_id = core_id

        self.inst = []