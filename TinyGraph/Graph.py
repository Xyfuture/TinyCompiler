class MicroNode:
    """
    生成核上的指令
    """

    def __init__(self, *args, **kwargs):
        self.input_nodes: Dict[MicroNode, None] = {}
        self.output_nodes: Dict[MicroNode, None] = {}

    def replace_all_uses_with(self, replace_with: MicroNode):
        pass

    def replace_input_with(self, old_input: MicroNode, new_input: MicroNode):
        pass


class MicroGraph:
    def __init__(self):
        pass


class DepTensor:
    """
    tensor 中记录 该tensor 来自于哪一个micro node
    主要是方便数据传输
    """

    def __init__(self):
        pass

    @property
    def shape(self):
        return 0

