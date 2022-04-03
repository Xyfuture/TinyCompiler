import torch
import torch.nn
import math



def number_decompose(num):
    sq = math.ceil(math.sqrt(num))
    decompose = []
    for i in range(1,sq+1):
        if num % i == 0:
            decompose.append((i,num//i))
            decompose.append((num//i,i))
    return decompose
