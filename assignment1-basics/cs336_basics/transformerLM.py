import torch, math, torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange, einsum, reduce

from .rmsnorm import RMSNorm
from .swiglu import SiLU, SwiGLU, SwiGLUFFN  # 只用激活函数，权重自己建
from .rope import RoPE
from .utils import softmax, cross_entropy_loss
from .linear import Linear
from .attention import MultiheadSelfAttentionWithRoPE
from .embedding import Embedding
from .mylayerlist import MyLayerList


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


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        theta: float,
        num_layers: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.context_length = context_length
        self.theta = theta
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        param_dtype = (
            dtype
            if (
                dtype is not None
                and torch.is_floating_point(torch.tensor([], dtype=dtype))
            )
            else torch.float32
        )

        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=param_dtype
        )
        self.layers = MyLayerList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=param_dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=param_dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=param_dtype)

    @torch.no_grad()
    def forward(
        self,
        input_indices: Int[Tensor, "batch seq_len"],
        token_positions: Int[Tensor, "batch seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len vocab_size"]:
        x = self.token_embeddings(input_indices)

        if token_positions is None:
            token_positions = torch.arange(x.size(1), device=x.device).expand(
                x.size(0), -1
            )

        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
