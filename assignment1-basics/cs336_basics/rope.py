import torch
import torch.nn as nn
from einops import rearrange, reduce, einsum
from torch import Tensor
from jaxtyping import Float, Int
import math


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None,dtype=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        self.half_dim = d_k // 2
        freq_seq = torch.arange(self.half_dim, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (theta ** (freq_seq / self.half_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        freqs = einsum(t, inv_freq, "i, j -> i j")
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)


    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "...  seq_len d_k"]:
        
        assert x.shape[-1] == self.d_k, f"x's last dim {x.shape[-1]} != d_k {self.d_k}"
        assert self.d_k % 2 == 0, "d_k must be even for RoPE"
        
        in_type = x.dtype
        x = x.to(torch.float32)
        
        # (... seq_len d_k) ->  (... seq_len d_pair 2) 2D-Tensor
        x_pair = rearrange(x, "... seq_len (d_pair two) -> ... seq_len d_pair two", two = 2)
        
        # cos/sin tensor build
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        cos = rearrange(cos, "... s d -> ... 1 s d")
        sin = rearrange(sin, "... s d -> ... 1 s d")
        
        x1, x2 = x_pair.unbind(dim=-1)
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        x_rot = torch.stack((rot1, rot2), dim = -1)
        
        out = rearrange(x_rot, "... s d two -> ... s (d two)", two=2)
        
        return out.to(in_type)