from __future__ import annotations

from typing import Tuple, Optional, Union

import numpy as np


class ObjectProxy:
    def __init__(self, ptr: Optional[object] = None):
        self.ptr = ptr

    def set(self, ptr):
        self.ptr = ptr

    def get(self):
        return self.ptr


def set_object_proxy_from_proxy(a: ObjectProxy, b: ObjectProxy):
    a.ptr = b.ptr


def set_object_proxy_from_value(a: ObjectProxy, b):
    a.ptr = b


def make_object_proxy(a):
    if isinstance(a,ObjectProxy):
        return a
    else:
        return ObjectProxy(a)


class ConductArray:
    vec_set_from_proxy = np.vectorize(set_object_proxy_from_proxy)
    vec_set_from_value = np.vectorize(set_object_proxy_from_value)
    vec_make_proxy = np.vectorize(make_object_proxy)
    vec_proxy_ctor = np.vectorize(ObjectProxy)

    def __init__(self, shape: Optional[Tuple[int, ...]] = None, array_value: Union[np.ndarray, object] = None):
        self._ndarray = np.zeros(shape, dtype=object)
        if isinstance(array_value, np.ndarray):
            # 这个ndarray 内部必须放的是 object proxy
            self._ndarray = array_value[:]
        else:
            # 其他情况,实现full的功能
            self._ndarray[:] = self.vec_proxy_ctor(array_value)

        self.shape = self._ndarray.shape

    @staticmethod
    def full(shape: Tuple[int, ...], value):
        return ConductArray(shape, value)

    @staticmethod
    def item_is_element(item: Tuple):
        if isinstance(item, tuple):
            if not any(isinstance(x, slice) for x in item):
                return True
        elif isinstance(item, int):
            return True

        return False

    def __getitem__(self, item):
        if self.item_is_element(item):
            return self._ndarray[item].get()
        return self._ndarray[item]

    def __setitem__(self, key, value):
        if self.item_is_element(key):
            self._ndarray[key].set(value)
        else:
            if isinstance(value, ConductArray):
                # 遍历改指针
                self.vec_set_from_proxy(self._ndarray, value._ndarray)
            if isinstance(value, np.ndarray):
                # 遍历赋值 value是一个array
                self.vec_set_from_value(self._ndarray, value)
            else:
                self.vec_set_from_value(self._ndarray, value)
                # 遍历赋值 value是单个值

    def reshape(self, shape) -> ConductArray:
        new_ndarray = self._ndarray.reshape(shape)

        return ConductArray(new_ndarray.shape, new_ndarray)

    def flat(self):
        return [obj_proxy.get() for obj_proxy in self._ndarray.flat]

    def enum(self):
        return [(index, obj_proxy.get()) for index, obj_proxy in np.ndenumerate(self._ndarray)]

    @staticmethod
    def pad(array: ConductArray, pad_width: int, pad_value) -> ConductArray:
        pad_array = np.pad(array._ndarray, pad_width,constant_values=pad_value)
        pad_array = ConductArray.vec_make_proxy(pad_array)
        return ConductArray(pad_array.shape,pad_array)

