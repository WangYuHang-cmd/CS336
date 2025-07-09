import torch
import torch.nn as nn
from einops import rearrange, reduce, einsum
from torch import Tensor
from jaxtyping import Float, Int
import math


def softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    if dim < 0:
        dim += x.ndim

    in_type = x.dtype
    x = x.to(torch.float64)
    
    perm = list(range(x.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_moved = x.permute(*perm)

    x_max = reduce(x_moved, "... n -> ... 1", "max")
    x_exp = (x_moved - x_max).exp()

    x_enom = reduce(x_exp, "... n -> ... 1", "sum")

    out_moved = x_exp / x_enom

    out = out_moved.permute(*perm)
    
    return out.to(in_type)
