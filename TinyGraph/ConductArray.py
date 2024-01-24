from __future__ import annotations

import copy
from typing import Tuple, Optional, Union

import numpy as np


class ObjectProxy:
    def __init__(self, ptr: Optional[object] = None):
        self.ptr = ptr

    def set(self, ptr):
        self.ptr = ptr

    def get(self):
        return self.ptr

    def __repr__(self):
        return f"<Proxy:{self.ptr}>"


def set_object_proxy_from_proxy(a: ObjectProxy, b: ObjectProxy) -> None:
    a.ptr = b.ptr


def set_object_proxy_from_value(a: ObjectProxy, b) -> None:
    a.ptr = b


def set_object_proxy(a: ObjectProxy, b: Union[ObjectProxy, object]) -> None:
    # 根据b数组的值修改a数组的值,把前面两个函数合起来,效率略低
    if isinstance(b, ObjectProxy):
        a.ptr = b.ptr
    else:
        a.ptr = b


def make_object_proxy(a) -> ObjectProxy:
    # 返回一个新的array
    if isinstance(a, ObjectProxy):
        return a
    else:
        return ObjectProxy(a)


class ConductArray:
    vec_set_from_proxy = np.vectorize(set_object_proxy_from_proxy)
    vec_set_from_value = np.vectorize(set_object_proxy_from_value)
    vec_set_proxy = np.vectorize(set_object_proxy)
    vec_make_proxy = np.vectorize(make_object_proxy)
    vec_proxy_ctor = np.vectorize(ObjectProxy)

    def __init__(self, shape: Optional[Tuple[int, ...]] = None, array_value: Union[np.ndarray, object] = None):
        # 不要再 ConductArray类之外调用init，使用full创建新的Conduct Array
        if isinstance(array_value, np.ndarray):
            # 这个ndarray 内部必须放的是 object proxy
            # 减少复制的情况
            self._ndarray = array_value[:]
        else:
            # 其他情况,实现full的功能
            # self._ndarray = np.zeros(shape, array_value,dtype=object)
            self._ndarray = self.vec_proxy_ctor(np.full(shape, array_value, dtype=object))

        self.shape = self._ndarray.shape

    @staticmethod
    def full(shape: Tuple[int, ...], value):
        return ConductArray(shape, value)

    def item_is_element(self, item: Tuple):
        if isinstance(self._ndarray[item], np.ndarray):
            return False
        return True

    def __getitem__(self, item):
        # 与 ndarray行为尽量相似，可以返回一个值 or 一个新的 conduct array
        if self.item_is_element(item):
            # 单个值 在DepTensor中应该不会遇到
            return self._ndarray[item].get()
        tmp_array = self._ndarray[item]
        return ConductArray(tmp_array.shape, tmp_array)

    def __setitem__(self, key, value):
        if self.item_is_element(key):
            # 直接对值及进行赋值
            # 在DepTensor中应该不会遇到
            self._ndarray[key].set(value)
        else:
            if isinstance(value, ConductArray):
                # 遍历改指针
                self.vec_set_from_proxy(self._ndarray[key], value._ndarray)
            elif isinstance(value, np.ndarray):
                # 遍历赋值 value是一个array
                self.vec_set_from_value(self._ndarray[key], value)
            else:
                # 遍历赋值 value是单个值 借助np.vectorize 实现
                self.vec_set_from_value(self._ndarray[key], value)

    def reshape(self, shape) -> ConductArray:
        new_ndarray = self._ndarray.reshape(shape)

        return ConductArray(new_ndarray.shape, new_ndarray)

    def flat(self):
        return [obj_proxy.get() for obj_proxy in self._ndarray.flat]

    def enum(self):
        return [(index, obj_proxy.get()) for index, obj_proxy in np.ndenumerate(self._ndarray)]

    @staticmethod
    def pad(array: ConductArray, pad_width: int, pad_value) -> ConductArray:
        pad_array = np.pad(array._ndarray, pad_width, constant_values=pad_value)
        pad_array = ConductArray.vec_make_proxy(pad_array)
        return ConductArray(pad_array.shape, pad_array)

    def __repr__(self):
        return f"ConductArray(\n{self._ndarray})"

    def __deepcopy__(self, memodict={}):
        new_ndarray = self.vec_proxy_ctor(np.full(self.shape, None, dtype=object))
        self.vec_set_from_proxy(new_ndarray, self._ndarray)

        return self.__class__(copy.deepcopy(self.shape), new_ndarray)
