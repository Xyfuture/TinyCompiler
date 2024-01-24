from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Callable, Deque, Tuple
import numpy as np


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

            node.__remove_input_nodes(self)
            node.__add_input_nodes(replace_with)

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

    def __remove_input_nodes(self,to_remove:MicroNode):
        to_remove._output_nodes.pop(self)
        self._input_nodes.pop(to_remove)

    def __add_input_nodes(self,to_add:MicroNode):
        to_add._output_nodes.setdefault(self)
        self._input_nodes.setdefault(to_add)

    def check_connection(self):
        for node in self._input_nodes:
            assert self in node._output_nodes

        for node in self._output_nodes:
            assert self in node._input_nodes


class MicroGraph:
    """
    主要是维护一个偏序关系,不一定存在实际的输入输出关系,只是保证正常的顺序
    """

    current_graph: Optional[MicroGraph] = None

    def __init__(self):
        self.nodes: List[MicroNode] = []

        self._node_creator = _NodeCreate(self)

        self.create_node: Callable[[List[MicroNode], MicroOp], MicroNode] = self._node_creator.__call__

    def add_node(self, node: MicroNode):
        self.nodes.append(node)

    def use_sequential_node_dep(self):
        self._node_creator.set_graph_creator_in_context('sequential')
        return self._node_creator


def topo_sort(graph: MicroGraph) -> List[MicroNode]:
    sorted_nodes: List[MicroNode] = []
    pending_queue: Deque[MicroNode] = deque()
    dep_map: Dict[MicroNode, int] = {}

    for node in graph.nodes:
        dep_map[node] = len(node._input_nodes)
        if len(node._input_nodes) == 0:
            pending_queue.append(node)

    while len(pending_queue):
        current_node = pending_queue.popleft()
        sorted_nodes.append(current_node)
        for node in current_node._output_nodes.keys():
            dep_map[node] -= 1
            if dep_map[node] == 0:
                pending_queue.append(node)

    return sorted_nodes


class _NodeCreate:
    def __init__(self, graph: MicroGraph):
        self.graph = graph

        self._sequential_last_node: List[Optional[MicroNode]] = []

        self.create_func_stack: List[Tuple[Callable, str]] = [(self.simple_create_node, 'simple')]

        self._current_create_func: Callable[[List[MicroNode], MicroOp], MicroNode] = self.simple_create_node
        self._current_create_func_name: str = 'simple'

        self._next_create_func_name: str = ""

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

    def set_graph_creator_in_context(self, creator_name: str):
        self._next_create_func_name = creator_name

    def get_create_func_by_name(self, create_func_name: str) -> Callable:
        if create_func_name == "simple":
            return self.simple_create_node
        elif create_func_name == 'sequential':
            return self.sequential_create_node

    def __enter__(self):
        self._current_create_func_name = self._next_create_func_name
        self._current_create_func = self.get_create_func_by_name(self._next_create_func_name)
        self.create_func_stack.append((self._current_create_func, self._current_create_func_name))

        if self._current_create_func_name == 'sequential':
            self._sequential_last_node.append(None)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._current_create_func_name == 'sequential':
            self._sequential_last_node.pop()

        self.create_func_stack.pop()
        self._current_create_func, self._current_create_func_name = self.create_func_stack[-1]

        self._next_create_func_name = None

    def __call__(self, input_nodes: List[MicroNode], micro_op: MicroOp) -> MicroNode:
        return self._current_create_func(input_nodes, micro_op)
