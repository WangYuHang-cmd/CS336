import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Type
import math
from torch.autograd import Function
from torch.nn import Parameter
from jaxtyping import Float, Int
from einops import rearrange, reduce, einsum

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd, # query tile
    stride_vb, stride_vq, stride_vd, # 
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.contextpr,
    Q_TILE_SIZE: tl.contextpr,
    K_TILE_SIZE: tl.contextpr
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape = (stride_qq, stride_qb),
        strides = (stride_qq, stride_qd),
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        orders = (1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        
    )

    V_block_ptr = tl.make_block_ptr(

    )



class FlashAttentionTriton():
    pass


class FlashAttentionPytorch(Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        dtype, device = Q.dtype, Q.device
        BLOCK_N, BLOCK_M = 64, 64

        B, H, L, D = Q.shape
        scale = 1.0 / math.sqrt(D)

        out = torch.zeros_like(Q, dtype=dtype, device=device)
        logsumexp = torch.full((B, H, L), float("-inf"), device=device, dtype=dtype)

        for i in range(0, L, BLOCK_N):
            row_start = i
            row_end = min(i + BLOCK_N, L)

            q_block = Q[:, :, row_start:row_end, :]  # (B, H, BLOCK_N, D)

            o_block = torch.zeros(
                (B, H, row_end - row_start, D), dtype=dtype, device=device
            )
            l_i = torch.zeros((B, H, row_end - row_start), device=device, dtype=dtype)
            m_i = torch.full(
                (B, H, row_end - row_start), float("-inf"), device=device, dtype=dtype
            )  # previous maximum one

            for j in range(0, L, BLOCK_M):
                col_start = j
                col_end = min(j + BLOCK_M, L)

                k_block = K[:, :, col_start:col_end, :]  # (B, H, BLOCK_M, D)
                v_block = V[:, :, col_start:col_end, :]  # (B, H, BLOCK_M, D)

                attn_score = (
                    einsum(q_block, k_block, "b h i d, b h j d -> b h i j") * scale
                )  # (B, H, BLOCK_N, BLOCK_M)

                if is_causal:
                    mask = torch.full(
                        (row_end - row_start, col_end - col_start),
                        float("-inf"),
                        device=device,
                        dtype=dtype,
                    )

                    for row_idx in range(row_end - row_start):
                        for col_idx in range(col_end - col_start):
                            actual_row = row_start + row_idx
                            actual_col = col_start + col_idx
                            if actual_col <= actual_row:
                                mask[row_idx, col_idx] = 0.0

                    attn_score = attn_score + mask.unsqueeze(0).unsqueeze(0)

                m_ij = torch.max(attn_score, dim=-1, keepdim=True)[
                    0
                ]  # the max in current each row, (B, H, BLOCK_N, 1).

                p_ij = torch.exp(attn_score - m_ij)  # (B, H, BLOCK_N, BLOCK_M)
                l_ij = torch.sum(
                    p_ij, dim=-1, keepdim=True
                )  # (B, H, BLOCK_N, 1), \sum_{i=1}^{BLOCK_N} e^{p_ij}

                m_i_new = torch.max(
                    m_i, m_ij.squeeze(-1)
                )  # (B, H, BLOCK_N)， m_i_new = max(m_ij, m_i), the largest value of all
                alpha = torch.exp(
                    m_i.unsqueeze(-1) - m_i_new.unsqueeze(-1)
                )  # (B, H, BLOCK_N, 1) scale up numbers
                beta = torch.exp(m_ij - m_i_new.unsqueeze(-1))  # (B, H, BLOCK_N, 1)

                l_i_new = alpha.squeeze(-1) * l_i + beta.squeeze(-1) * l_ij.squeeze(
                    -1
                )  # (B, H, BLOCK_N)

                if j == 0:
                    o_block = (
                        beta * einsum(p_ij, v_block, "b h i j, b h j d -> b h i d")
                    ) / l_i_new.unsqueeze(-1)
                else:
                    o_block = (
                        o_block * l_i.unsqueeze(-1) * alpha
                        + einsum(p_ij, v_block, "b h i j, b h j d -> b h i d") * beta
                    ) / l_i_new.unsqueeze(-1)
                    # new_contribution = beta.un

                m_i = m_i_new
                l_i = l_i_new

            out[:, :, row_start:row_end, :] = o_block  # / l_i.unsqueeze(-1)
            logsumexp[:, :, row_start:row_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(Q, K, V, out, logsumexp)
        ctx.is_causal = is_causal

        return out

    @staticmethod
    def backward(ctx, grad_output):
        return NotImplementedError


# 用于对比的标准attention实现（方程4-6和12）
def standard_attention(Q, K, V, is_causal=False):
    """
    方程4: S = QK^T
    方程5: P = softmax(S)
    方程6: O = PV
    方程12: L = logsumexp(S) (用于数值稳定性)
    """
    B, H, L, D = Q.shape
    d_k = 1.0 / math.sqrt(D)
    # 方程4: S = QK^T / sqrt(d_k)
    S = einsum(Q, K, "b h i d, b h j d -> b h i j") * d_k
    # 应用causal mask（如果需要）
    if is_causal:
        mask = torch.triu(torch.ones(L, L, device=Q.device, dtype=Q.dtype), diagonal=1)
        S = S.masked_fill(mask.bool(), float("-inf"))
    # 方程12: L = logsumexp(S) (数值稳定的softmax分母的对数)
    L = torch.logsumexp(S, dim=-1)
    # 方程5: P = softmax(S)
    P = torch.softmax(S, dim=-1)
    # 方程6: O = PV
    O = einsum(P, V, "b h i j, b h j d -> b h i d")
    return O, L


# 测试函数
def test_flashattention():
    B, H, L, D = 256, 16, 256, 768
    device = "cuda"

    Q = torch.randn(B, H, L, D, device=device, dtype=torch.float32, requires_grad=True)
    K = torch.randn(B, H, L, D, device=device, dtype=torch.float32, requires_grad=True)
    V = torch.randn(B, H, L, D, device=device, dtype=torch.float32, requires_grad=True)

    sum = 0.0


    for i in range(1000):
        # FlashAttention结果
        flash_out = FlashAttentionPytorch.apply(Q, K, V, False)
        # 标准attention结果
        std_out, std_logsumexp = standard_attention(Q, K, V, False)

        diff = torch.max(torch.abs(flash_out - std_out)).item()
        
        sum += diff

    print(
        f"FlashAttention和标准attention的输出差异: {sum}"
    )
    print(f"输出是否接近: {torch.allclose(flash_out, std_out, atol=1e-4)}")


if __name__ == "__main__":
    test_flashattention()
