import torch
import torch.nn as nn
from jaxtyping import Float, Int
from einops import rearrange, einsum
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: [Tensor, "... d_model"]) -> [Tensor, "... d_model"]:
        in_type = x.dtype
        x = x.to(torch.float32)
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        out = x / norm * self.weight
        return out.to(in_type)
