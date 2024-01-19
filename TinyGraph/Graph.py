from __future__ import annotations
from typing import Dict, List, Optional, Callable
import numpy as np

from TinyGraph.DSL import DepTensor


class MicroOp:
    def __init__(self):
        self.node: Optional[MicroNode] = None

        # self.output_dep_tensor:Optional[DepTensor] = None

    def register_node(self, node: MicroNode):
        self.node = node

    def code_gen(self):
        pass


class MicroNode:
    """
    生成核上的指令
    """

    def __init__(self, graph: MicroGraph, input_nodes: List[MicroNode], micro_op: MicroOp, *args, **kwargs):
        self.graph = graph
        self.micro_op = micro_op
        micro_op.register_node(self)

        self._input_nodes: Dict[MicroNode, None] = {}
        self._output_nodes: Dict[MicroNode, None] = {}

        self.__update_input_nodes(input_nodes)

    def replace_all_uses_with(self, replace_with: MicroNode,
                              delete_user_cb: Callable[[MicroNode], bool] = lambda user: True, ):
        to_process = list(self._output_nodes)
        skipped = []

        for node in to_process:
            if not delete_user_cb(node):
                skipped.append(node)
                continue

            new_input_nodes = node._input_nodes
            new_input_nodes.pop(self)
            new_input_nodes.setdefault(replace_with)

            node.__update_input_nodes(list(new_input_nodes))

        return [node for node in to_process if node not in skipped]

    def insert_node_after_with(self, append_with: MicroNode):
        self.replace_all_uses_with(append_with, delete_user_cb=lambda node: True if node is not append_with else False)
        append_with.__update_input_nodes([self])

    def insert_node_before_with(self, prepend_with: MicroNode):
        input_nodes = self._input_nodes
        self.__update_input_nodes([prepend_with])

        prepend_with.__update_input_nodes(list(input_nodes))

    def replace_input_with(self, old_input: MicroNode, new_input: MicroNode):
        new_input_nodes = self._input_nodes

        new_input_nodes.pop(old_input)
        new_input_nodes.setdefault(new_input)

        self.__update_input_nodes(list(new_input_nodes))

    def __update_input_nodes(self, new_input_nodes: List[MicroNode]):
        """
        only input nodes can be changed
        DO NOT CHANGE OUTPUT NODES DIRECTLY
        """
        for old_input_node in self._input_nodes:
            old_input_node._output_nodes.pop(self)

        self._input_nodes = {}

        for new_input_node in new_input_nodes:
            self._input_nodes.setdefault(new_input_node)
            new_input_node._output_nodes.setdefault(self)


class MicroGraph:
    """
    主要是维护一个偏序关系,不一定存在实际的输入输出关系,只是保证正常的顺序
    """

    current_graph: Optional[MicroGraph] = None

    def __init__(self):
        self.nodes: List[MicroNode] = []

        self._node_creator = _NodeCreate(self)

        self.create_node: Callable[[List[MicroNode], MicroOp], MicroNode] = self._node_creator.simple_create_node

    def add_node(self, node: MicroNode):
        self.nodes.append(node)

    def use_sequential_node_dep(self):
        self._node_creator.set_graph_creator_in_context(self._node_creator.sequential_create_node)
        return self._node_creator


class _NodeCreate:
    def __init__(self, graph: MicroGraph):
        self.graph = graph

        self._sequential_last_node: List[Optional[MicroNode]] = []

        self.pre_graph_creator = []
        self.next_graph_creator = None

    def simple_create_node(self, input_nodes: List[MicroNode], micro_op: MicroOp, *args, **kwargs) -> MicroNode:
        node = MicroNode(self.graph, input_nodes, micro_op, *args, **kwargs)
        self.graph.add_node(node)

        return node

    def sequential_create_node(self, input_nodes: List[MicroNode], micro_op: MicroOp, *args, **kwargs) -> MicroNode:
        if self._sequential_last_node and self._sequential_last_node[-1]:
            input_nodes.append(self._sequential_last_node[-1])

        node = MicroNode(self.graph, input_nodes, micro_op, *args, **kwargs)
        self.graph.add_node(node)
        self._sequential_last_node[-1] = node

        return node

    def set_graph_creator_in_context(self, creator):
        self.next_graph_creator = creator

    def __enter__(self):
        self.pre_graph_creator.append(self.graph.create_node)
        self.graph.create_node = self.next_graph_creator

        if self.graph.create_node is self.sequential_create_node:
            self._sequential_last_node.append(None)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.graph.create_node is self.sequential_create_node:
            self._sequential_last_node.pop()

        self.graph.create_node = self.pre_graph_creator.pop()
        self.next_graph_creator = None
