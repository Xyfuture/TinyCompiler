from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Callable, Deque, Tuple, Union

from TinyGraph.Machine import Core
from TinyGraph.MachineOps import MachineOp, SharedAddressManager, AddressManager


class MicroOp:
    id_counter = {}

    def __init__(self, core_id: int = -1, vector_size: int = 0, src_ops: Union[List[MicroOp],MicroOp,None] = None,
                 shr_manager_id: int = 0):
        self.op_id = MicroOp.id_counter.get(self.__class__, 1)
        MicroOp.id_counter[self.__class__] = self.op_id + 1
        self.node: Optional[MicroNode] = None

        self.core_id = core_id  # 这个主要指写内存的core id (针对 transfer的这种情况 )

        self.vector_size = vector_size  # 这个是输出的大小
        self.shr_manager_id = shr_manager_id

        if src_ops:  # 这个机制可能还要修改,暂时暂时没想好应该怎么动 ...
            if isinstance(src_ops,List):
                self.src_ops = src_ops
            else:
                self.src_ops = [src_ops]
        else:
            self.src_ops = []

        self.output_machine_op: Optional[MachineOp] = None

    def get_core_address_manager(self):
        # 获得基于core memory allocator的 address manager
        core = Core.get_core_by_id(self.core_id)
        if self.shr_manager_id:
            return SharedAddressManager.get_shr_addr_manager(
                self.shr_manager_id, self.vector_size, core.memory_allocator
            )
        else:
            return AddressManager(self.vector_size, core.memory_allocator)

    def register_node(self, node: MicroNode):
        self.node = node

    def replace_with_new_src_op(self, old_src_op: MicroOp, new_src_op: MicroOp):
        for index, src_op in enumerate(self.src_ops):
            if src_op == old_src_op:
                self.src_ops[index] = new_src_op

    def machine_op_gen(self):
        pass

    def dummy_code_gen(self):
        pass

    def __repr__(self):
        return f"MicroOp"

    def full_info(self):
        return ""


class RootOp(MicroOp):
    def __init__(self):
        super().__init__()


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

        # 使用双向链表管理
        self._prev: MicroNode = self
        self._next: MicroNode = self

        self._erased = False

        self.__update_input_nodes(input_nodes)

    @property
    def next(self) -> MicroNode:
        return self._next

    @property
    def prev(self) -> MicroNode:
        return self._prev

    def prepend(self, x: MicroNode):
        if self == x:
            return

        x._remove_from_list()
        p = self._prev
        p._next, x._prev = x, p
        x._next, self._prev = self, x

    def append(self, x: MicroNode):
        self._next.append(x)

    def _remove_from_list(self):
        p, n = self._prev, self.next
        p._next, n._prev = n, p

    @property
    def all_input_nodes(self) -> List[MicroNode]:
        return list(self._input_nodes)

    @property
    def all_output_nodes(self) -> List[MicroNode]:
        return list(self._output_nodes)

    # TODO 要在更改这些node 的同时更改micro op相应的信息, 同时注意 micro op 的链接是单向的
    def replace_all_uses_with(self, replace_with: MicroNode,
                              delete_user_cb: Callable[[MicroNode], bool] = lambda user: True, ):
        to_process = list(self._output_nodes)
        skipped = []

        for node in to_process:
            if not delete_user_cb(node):
                skipped.append(node)
                continue

            # node.__remove_input_node(self)
            # node.__add_input_node(replace_with)

            node.__replace_input_with(self,replace_with)

        return [node for node in to_process if node not in skipped]

    def insert_node_after_with(self, append_with: MicroNode):
        self.replace_all_uses_with(append_with, delete_user_cb=lambda node: True if node is not append_with else False)
        append_with.__add_input_node(self)

    def replace_input_with(self, old_input: MicroNode, new_input: MicroNode):
        self.__replace_input_with(old_input,new_input)

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

    def add_input_nodes(self, nodes: List[MicroNode]):
        for node in nodes:
            self.__add_input_node(node)

    def __remove_input_node(self, to_remove: MicroNode):
        to_remove._output_nodes.pop(self)
        self._input_nodes.pop(to_remove)

    def __add_input_node(self, to_add: MicroNode):
        to_add._output_nodes.setdefault(self)
        self._input_nodes.setdefault(to_add)

    def __replace_input_with(self,old_input:MicroNode,new_input:MicroNode):
        # 替换 micro node的同时实现 micro op中 src op的替换工作
        # 替换 src op
        old_micro_op = old_input.micro_op
        for index,src_op in enumerate(self.micro_op.src_ops):
            if src_op == old_micro_op:
                self.micro_op.src_ops[index] = new_input.micro_op
        # 替换 micro node
        old_input._output_nodes.pop(self)
        self._input_nodes.pop(old_input)

        new_input._output_nodes.setdefault(self)
        self._input_nodes.setdefault(new_input)

    def check_connection(self):
        for node in self._input_nodes:
            assert self in node._output_nodes

        for node in self._output_nodes:
            assert self in node._input_nodes


class _NodeList:
    def __init__(self, graph: MicroGraph, direction: str = '_next'):
        assert direction in ['_next', '_previous']
        self.graph = graph
        self.direction = direction

    def __len__(self):
        return self.graph._len

    def __iter__(self):
        root, direction = self.graph._root, self.direction
        cur = getattr(root, direction)
        while cur is not root:
            yield cur
            cur = getattr(cur, direction)

    def __reversed__(self):
        return _NodeList(self.graph, '_next' if self.direction == '_prev' else '_prev')


class _InsertPoint:
    def __init__(self, graph: MicroGraph, new_insert):
        self.graph = graph
        self.original_insert, self.new_insert = graph._insert, new_insert

    def __enter__(self):
        self.graph._insert = self.new_insert

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.graph._insert = self.original_insert


class MicroGraph:
    """
    主要是维护一个偏序关系,不一定存在实际的输入输出关系,只是保证正常的顺序
    """

    current_graph: Optional[MicroGraph] = None

    def __init__(self):
        self._root: MicroNode = MicroNode(self, [], RootOp())
        self._insert = self._root.prepend
        self._len = 0

        self._node_creator = _NodeCreate(self)
        self.create_node: Callable[[List[MicroNode], MicroOp], MicroNode] = self._node_creator.__call__

    @property
    def nodes(self) -> _NodeList:
        return _NodeList(self)

    def insert_before(self, node: Optional[MicroNode] = None):
        if node is None:
            return self.insert_after(self._root)

        return _InsertPoint(self, node.prepend)

    def insert_after(self, node: Optional[MicroNode] = None):
        if node is None:
            return self.insert_before(self._root)
        return _InsertPoint(self, node.append)

    def add_node(self, node: MicroNode):
        self._insert(node)
        self._len += 1

    def use_sequential_node_dep(self):
        self._node_creator.set_graph_creator_in_context('sequential')
        return self._node_creator

    def erase_node(self, node: MicroNode):
        # 确保没有其他节点使用该node
        if len(node.all_output_nodes) > 0:
            assert False

        if node._erased:
            return

        node._remove_from_list()
        node._erased = True
        self._len -= 1
        node._input_nodes = {}


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

    for k,v in dep_map.items():
        if v != 0:
            input_nodes = k._input_nodes
            dep_input_nodes = []
            for node in input_nodes:
                if node not in sorted_nodes:
                    dep_input_nodes.append(node)
            print(f"{repr(k.micro_op)} -- {repr([mnode.micro_op for mnode in dep_input_nodes])}")
    assert len(sorted_nodes) == len(dep_map)


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
