import torch
import torch.nn as nn
from jaxtyping import Float, Int
from einops import rearrange, reduce, einsum
from torch import Tensor

from .linear import Linear

import math


def SiLU(x: Tensor):
    in_type = x.dtype
    x = x.to(torch.float32)
    return (x * torch.sigmoid(x)).to(in_type)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: str = None, 
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        std1 = (2.0 / (d_model + d_ff)) ** 0.5
        std2 = (2.0 / (d_ff + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.w1_weight, mean=0.0, std=std1, a=-3*std1, b=3*std1)
        torch.nn.init.trunc_normal_(self.w2_weight, mean=0.0, std=std2, a=-3*std2, b=3*std2)
        torch.nn.init.trunc_normal_(self.w3_weight, mean=0.0, std=std1, a=-3*std1, b=3*std1)


    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    )  -> Float[Tensor, "... d_model"]:
        in_type = x.dtype
        x = x.to(torch.float32)
        w1x = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        w3x = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        
        out = einsum(SiLU(w1x) * w3x, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")
        return out.to(in_type)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff , d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        gate = SiLU(self.w1(x)) * self.w3(x)
        return self.w2(gate)