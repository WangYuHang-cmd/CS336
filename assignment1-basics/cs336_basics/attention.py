import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce
from torch import Tensor
from jaxtyping import Float,Int
import math

from .linear   import Linear
from .rope     import RoPE
from .softmax  import softmax

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, "... queries d_v"]:
        d_k = Q.shape[-1]
        # (batch..., queries, keys)
        score = einsum(
            Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
        ) / math.sqrt(d_k)
        if mask is not None:
            score = score.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(score, dim=-1)  # softmax over "keys"
        # (batch..., queries, d_v)
        out = einsum(
            attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v"
        )
        return out


class MultiheadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_model d_in"],
        k_proj_weight: Float[Tensor, " d_model d_in"],
        v_proj_weight: Float[Tensor, " d_model d_in"],
        o_proj_weight: Float[Tensor, " d_model d_model"],
        in_features: Float[Tensor, " ... seq_len d_in"],
    ) -> Float[Tensor, " ... seq_len d_model"]:
        head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        seq_len = in_features.shape[-2]
        Q = einsum(
            in_features, q_proj_weight, "... s d_in, d_model d_in -> ... s d_model"
        )
        K = einsum(
            in_features, k_proj_weight, "... s d_in, d_model d_in -> ... s d_model"
        )
        V = einsum(
            in_features, v_proj_weight, "... s d_in, d_model d_in -> ... s d_model"
        )

        Q = rearrange(Q, "... s (h d) -> ... h s d", h=num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=num_heads)

        scale = 1.0 / math.sqrt(head_dim)
        attn_logits = einsum(Q, K, "... h q d, ... h k d -> ... h q k") * scale

        seq_len = Q.shape[-2]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=1
        )
        attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))
        attn_weights = attn_logits.softmax(dim=-1)
        attn_out = einsum(attn_weights, V, "... h q k, ... h k d -> ... h q d")
        attn_out = rearrange(attn_out, "... h s d -> ... s (h d)")
        out = einsum(attn_out, o_proj_weight, "... s d, d_out d -> ... s d_out")
        return out

class MultiheadSelfAttentionWithRoPE(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        # 投影（无 bias）
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

        self.rope   = RoPE(theta, self.head_dim, max_seq_len, device, dtype)

    def forward(
        self,
        x: Float[Tensor, "b s d_model"],
        token_positions: Int[Tensor, "b s"],
    ) -> Float[Tensor, "b s d_model"]:
        b, s, _ = x.shape
        h, d    = self.num_heads, self.head_dim

        Q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=h)
        K = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=h)
        V = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=h)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # Scaled dot-product
        logits = einsum(Q, K, "b h q d, b h k d -> b h q k") / math.sqrt(d)
        logits = logits.masked_fill(torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), 1),
                                    float("-inf"))
        attn   = softmax(logits, dim=-1)

        ctx = einsum(attn, V, "b h q k, b h k d -> b h q d")
        ctx = rearrange(ctx, "b h s d -> b s (h d)")
        return self.output_proj(ctx)
