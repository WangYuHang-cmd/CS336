import torch, math, torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange, einsum, reduce

from .rmsnorm import RMSNorm
from .swiglu import SiLU, SwiGLU, SwiGLUFFN  # 只用激活函数，权重自己建
from .rope import RoPE
from .softmax import softmax
from .linear import Linear
from .attention import MultiheadSelfAttentionWithRoPE


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)

        self.attn = MultiheadSelfAttentionWithRoPE(
            d_model, num_heads, max_seq_len, theta, device, dtype
        )
        
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLUFFN(d_model, d_ff, device, dtype)

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        token_positions: Int[Tensor, "batch seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        if token_positions is None:
            token_positions = torch.arange(x.size(1), device=x.device).expand(
                x.size(0), -1
            )

        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
