from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import numpy as np

class MicroNode:
    """
    生成核上的指令
    """

    def __init__(self, graph: MicroGraph, input_nodes: Tuple[MicroNode], *args, **kwargs):
        self.graph = graph

        self._input_nodes: Dict[MicroNode, None] = {}
        self._output_nodes: Dict[MicroNode, None] = {}

        self.output_tensor: Optional[DepTensor] = None

        for node in input_nodes:
            self._input_nodes.setdefault(node)
            node._output_nodes.setdefault(self)

    def replace_all_uses_with(self, replace_with: MicroNode):
        to_process = list(self._output_nodes)

        for node in to_process:
            node._input_nodes.pop(self)
            node._input_nodes.setdefault(replace_with)

        return to_process

    def replace_input_with(self, old_input: MicroNode, new_input: MicroNode):
        self._input_nodes.pop(old_input)
        self._input_nodes.setdefault(new_input)


class MicroGraph:
    def __init__(self):
        self.nodes: List[MicroNode] = []

    def add_node(self, node: MicroNode):
        self.nodes.append(node)


class DepTensor:
    """
    tensor 中记录 该tensor 来自于哪一个micro node
    主要是方便数据传输
    """

    def __init__(self, tensor_shape: Tuple[int, ...]):
        self.tensor_shape = tensor_shape



    @property
    def shape(self):
        return 0
