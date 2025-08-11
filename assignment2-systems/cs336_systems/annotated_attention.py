# cs336_systems/annotated_attention.py
from __future__ import annotations
import torch, torch.nn as nn
import torch.cuda.nvtx as nvtx
from einops import rearrange, einsum
import math

# 引入原始实现类型，方便 isinstance 判别
from cs336_basics.attention import MultiheadSelfAttentionWithRoPE as _OrigMHA

class NVTXWrappedMHA(nn.Module):
    def __init__(self, orig: _OrigMHA):
        super().__init__()
        self.orig = orig
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        h, d = self.orig.num_heads, self.orig.head_dim

        with nvtx.range("MHA(total)"):
            with nvtx.range("QKV projections"):
                Q = rearrange(self.orig.q_proj(x), "b s (h d) -> b h s d", h=h)
                K = rearrange(self.orig.k_proj(x), "b s (h d) -> b h s d", h=h)
                V = rearrange(self.orig.v_proj(x), "b s (h d) -> b h s d", h=h)

            with nvtx.range("RoPE(Q)"):
                Q = self.orig.rope(Q, token_positions)
            with nvtx.range("RoPE(K)"):
                K = self.orig.rope(K, token_positions)

            with nvtx.range("attn_logits (QK^T) + mask"):
                logits = einsum(Q, K, "b h q d, b h k d -> b h q k") / math.sqrt(d)
                s = x.size(1)
                causal = torch.triu(
                    torch.ones(s, s, dtype=torch.bool, device=x.device), 1
                )
                logits = logits.masked_fill(causal, float("-inf"))

            with nvtx.range("softmax"):
                attn = torch.softmax(logits, dim=-1)

            with nvtx.range("attn*V (context)"):
                ctx = einsum(attn, V, "b h q k, b h k d -> b h q d")
                ctx = rearrange(ctx, "b h s d -> b s (h d)")

            with nvtx.range("output proj"):
                out = self.orig.output_proj(ctx)

        return out
    
    

def instrument_model_mha_with_nvtx(model: nn.Module) -> None:

    for name, child in list(model.named_children()):
        if isinstance(child, _OrigMHA):
            setattr(model, name, NVTXWrappedMHA(child))
        else:
            instrument_model_mha_with_nvtx(child)
    
