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
        rot_mat = torch.stack(
            (
                torch.stack((cos, -sin), dim = -1),
                torch.stack((sin, cos), dim = -1),
            ),
            dim = -2,
        )
        
        # rotate "i j, j -> i"
        x_rot = einsum(rot_mat, x_pair, "... d_pair i j, ... d_pair j -> ... d_pair i")
        out = rearrange(x_rot, "... seq_len d_pair two -> ... seq_len (d_pair two)", two = 2)
        
        return  out.to(in_type)
        


    # def forward(
    #     self,
    #     x: Float[Tensor, "... seq_len d_k"],
    #     token_positions: Int[Tensor, "... seq_len"],
    # ) -> Float[Tensor, "...  seq_len d_k"]:
        
    #     assert x.shape[-1] == self.d_k, f"x's last dim {x.shape[-1]} != d_k {self.d_k}"
    #     assert self.d_k % 2 == 0, "d_k must be even for RoPE"
        
    #     in_type = x.dtype
    #     x = x.to(torch.float32)

    #     x_even = x[..., 0::2]            # (..., seq_len, d_k/2)
    #     x_odd  = x[..., 1::2]

    #     cos = self.cos_cached[token_positions]  # [..., seq_len half_dim]
    #     sin = self.sin_cached[token_positions]

    #     rot_even = x_even * cos - x_odd * sin
    #     rot_odd = x_odd * cos + x_even * sin

    #     out = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
    #     return out.to(in_type)
