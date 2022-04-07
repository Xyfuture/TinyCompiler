from typing import List

from TinyDSL.DataType.tensor import TensorVar
from TinyDSL.DataType.vector import VectorVar

# 针对tensor的，vector基本不需要concat操作
def concat(ten_list:List[TensorVar],c_dim):
    # c_dim 为合并的维度 在这里不是连续的意思
    def check(a,b):
        assert a.dim == b.dim and a.dim == dims
        assert a.bitwidth == b.bitwidth

        for i in range(dims):
            if i != c_dim:
                assert a.ten_shape[i] == b.ten_shape[i]

    sample = ten_list[0]
    dims = sample.dim
    shape = sample.ten_shape

    outer_cnt = 1 # result_vec 需要进行多少次循环
    for d,i in enumerate(shape):
        if d<c_dim:
            outer_cnt *= i

    inner_offset_list = [] # 每一个待合并的tensor都有自己的情况
    for t in ten_list:
        tmp_len = 1
        for d,i in enumerate(t.ten_shape):
            if d>=c_dim:
                tmp_len *= i
        inner_offset_list.append(tmp_len)

    step_sum=0 # result_vec一次循环需要偏移这么多
    for i in inner_offset_list:
        step_sum += i

    for i in range(len(ten_list)-1):
        check(ten_list[i],ten_list[i+1])

    def gen(result_ten:TensorVar):
        # 通过[]把对应的元素取出来，然后通过copy依次进行赋值
        for i in range(outer_cnt):
            cur_sum = 0
            for t,l in zip(ten_list,inner_offset_list):
                result_ten.get_vec_offset(i*step_sum+cur_sum,l).copy(
                    t.get_vec_offset(i*l,l)
                )
                cur_sum += l

    return gen

