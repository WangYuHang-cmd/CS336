import math
from typing import Type

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Parameter
from jaxtyping import Float, Int
from einops import rearrange, reduce, einsum

import triton
import triton.language as tl


# ----------------------- Triton kernels -----------------------

_BWD_SAFE_CONFIGS = [
    triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}, num_warps=4, num_stages=2),
    triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 64}, num_warps=4, num_stages=2),
    triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_warps=4, num_stages=2),
]


@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 64}, num_warps=8, num_stages=5),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}, num_warps=8, num_stages=5),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=4, num_stages=2),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_warps=8, num_stages=5),
    ],
    key=['N_QUERIES', 'N_KEYS', 'IS_CAUSAL'],
)

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vq, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    bh = tl.program_id(1)
    q_start = q_tile_idx * Q_TILE_SIZE

    Q = tl.make_block_ptr(
        base=Q_ptr + bh * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K = tl.make_block_ptr(
        base=K_ptr + bh * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V = tl.make_block_ptr(
        base=V_ptr + bh * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vq, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O = tl.make_block_ptr(
        base=O_ptr + bh * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L = tl.make_block_ptr(
        base=L_ptr + bh * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q = tl.load(Q, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    q_idx = q_start + tl.arange(0, Q_TILE_SIZE)

    for k_tile_idx in range(0, n_k_tiles):
        k = tl.load(K, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        v = tl.load(V, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        scores = tl.dot(q, tl.trans(k)) * scale

        k_start = k_tile_idx * K_TILE_SIZE
        k_idx = k_start + tl.arange(0, K_TILE_SIZE)[None, :]
        valid_k = k_idx < N_KEYS
        if IS_CAUSAL:
            causal = q_idx[:, None] >= k_idx
            valid_k = valid_k & causal

        scores = tl.where(valid_k, scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        row_has_any = tl.sum(valid_k, axis=1) > 0
        safe_m = tl.where(row_has_any, m_ij, 0.0)
        p = tl.exp(scores - safe_m[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_j = tl.dot(p, v)

        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)

        l_i = alpha * l_i + beta * l_ij
        acc = alpha[:, None] * acc + beta[:, None] * acc_j
        m_i = m_new

        K = tl.advance(K, (K_TILE_SIZE, 0))
        V = tl.advance(V, (K_TILE_SIZE, 0))

    o = acc / l_i[:, None]
    if OUT_DTYPE == 0:
        o_store = o
    elif OUT_DTYPE == 1:
        o_store = o.to(tl.float16)
    else:
        o_store = o.to(tl.bfloat16)
    tl.store(O, o_store, boundary_check=(0, 1))
    tl.store(L, m_i + tl.log(l_i), boundary_check=(0,))


@triton.autotune(configs=_BWD_SAFE_CONFIGS, key=['N_QUERIES', 'N_KEYS', 'D', 'IS_CAUSAL'])
@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vq, stride_vd,
    stride_gob, stride_goq, stride_god,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    bh = tl.program_id(1)
    q_start = q_tile_idx * Q_TILE_SIZE

    Q = tl.make_block_ptr(
        base=Q_ptr + bh * stride_qb,
        shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    K = tl.make_block_ptr(
        base=K_ptr + bh * stride_kb,
        shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    V = tl.make_block_ptr(
        base=V_ptr + bh * stride_vb,
        shape=(N_KEYS, D), strides=(stride_vq, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    dO = tl.make_block_ptr(
        base=dO_ptr + bh * stride_gob,
        shape=(N_QUERIES, D), strides=(stride_goq, stride_god),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    O = tl.make_block_ptr(
        base=O_ptr + bh * stride_ob,
        shape=(N_QUERIES, D), strides=(stride_oq, stride_od),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    L = tl.make_block_ptr(
        base=L_ptr + bh * stride_lb,
        shape=(N_QUERIES,), strides=(stride_lq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,), order=(0,),
    )
    dQ = tl.make_block_ptr(
        base=dQ_ptr + bh * stride_dqb,
        shape=(N_QUERIES, D), strides=(stride_dqq, stride_dqd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )

    q = tl.load(Q, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    go = tl.load(dO, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    o  = tl.load(O,  boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    L_row = tl.load(L, boundary_check=(0,), padding_option="zero").to(tl.float32)

    delta = tl.sum(go * o, axis=1)

    n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    q_idx = q_start + tl.arange(0, Q_TILE_SIZE)
    valid_q = q_idx < N_QUERIES
    dQ_acc = tl.zeros_like(q)

    K_it = K
    V_it = V
    for k_tile_idx in range(0, n_k_tiles):
        k = tl.load(K_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        v = tl.load(V_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale

        k_start = k_tile_idx * K_TILE_SIZE
        k_idx = k_start + tl.arange(0, K_TILE_SIZE)[None, :]
        valid_k = k_idx < N_KEYS
        valid = valid_k & valid_q[:, None]
        if IS_CAUSAL:
            causal = q_idx[:, None] >= k_idx
            valid = valid & causal
        scores = tl.where(valid, scores, -float("inf"))

        P = tl.exp(scores - L_row[:, None])

        dp = tl.dot(go, tl.trans(v))    # <gO_i, V_j>
        dS = P * (dp - delta[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)

        dQ_acc += tl.dot(dS, k) * scale

        K_it = tl.advance(K_it, (K_TILE_SIZE, 0))
        V_it = tl.advance(V_it, (K_TILE_SIZE, 0))

    if OUT_DTYPE == 0:
        dQ_store = dQ_acc
    elif OUT_DTYPE == 1:
        dQ_store = dQ_acc.to(tl.float16)
    else:
        dQ_store = dQ_acc.to(tl.bfloat16)
    tl.store(dQ, dQ_store, boundary_check=(0, 1))


@triton.autotune(configs=_BWD_SAFE_CONFIGS, key=['N_QUERIES', 'N_KEYS', 'D', 'IS_CAUSAL'])
@triton.jit
def flash_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vq, stride_vd,
    stride_gob, stride_goq, stride_god,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dkb, stride_dkq, stride_dkd,
    stride_dvb, stride_dvq, stride_dvd,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    k_tile_idx = tl.program_id(0)
    bh = tl.program_id(1)
    k_start = k_tile_idx * K_TILE_SIZE

    K_tile = tl.make_block_ptr(
        base=K_ptr + bh * stride_kb,
        shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    V_tile = tl.make_block_ptr(
        base=V_ptr + bh * stride_vb,
        shape=(N_KEYS, D), strides=(stride_vq, stride_vd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    dK_tile = tl.make_block_ptr(
        base=dK_ptr + bh * stride_dkb,
        shape=(N_KEYS, D), strides=(stride_dkq, stride_dkd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    dV_tile = tl.make_block_ptr(
        base=dV_ptr + bh * stride_dvb,
        shape=(N_KEYS, D), strides=(stride_dvq, stride_dvd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )

    Q_blk = tl.make_block_ptr(
        base=Q_ptr + bh * stride_qb,
        shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    dO_blk = tl.make_block_ptr(
        base=dO_ptr + bh * stride_gob,
        shape=(N_QUERIES, D), strides=(stride_goq, stride_god),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    O_blk = tl.make_block_ptr(
        base=O_ptr + bh * stride_ob,
        shape=(N_QUERIES, D), strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    L_blk = tl.make_block_ptr(
        base=L_ptr + bh * stride_lb,
        shape=(N_QUERIES,), strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,), order=(0,),
    )

    k = tl.load(K_tile, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    v = tl.load(V_tile, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    dK_acc = tl.zeros_like(k)
    dV_acc = tl.zeros_like(v)

    n_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    k_idx = k_start + tl.arange(0, K_TILE_SIZE)[None, :]
    valid_k = k_idx < N_KEYS

    Q_it, dO_it, O_it, L_it = Q_blk, dO_blk, O_blk, L_blk
    for q_tile_idx in range(0, n_q_tiles):
        q_start = q_tile_idx * Q_TILE_SIZE
        q_idx = q_start + tl.arange(0, Q_TILE_SIZE)
        valid_q = q_idx < N_QUERIES

        q  = tl.load(Q_it,  boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        go = tl.load(dO_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        o  = tl.load(O_it,  boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        L_row = tl.load(L_it, boundary_check=(0,), padding_option="zero").to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale

        valid = valid_k & valid_q[:, None]
        if IS_CAUSAL:
            causal = q_idx[:, None] >= k_idx
            valid = valid & causal
        scores = tl.where(valid, scores, -float("inf"))

        P = tl.exp(scores - L_row[:, None])

        dV_acc += tl.dot(tl.trans(P), go)

        delta = tl.sum(go * o, axis=1)
        dp = tl.dot(go, tl.trans(v))
        dS = P * (dp - delta[:, None])

        dK_acc += tl.dot(tl.trans(dS), q) * scale

        Q_it  = tl.advance(Q_it,  (Q_TILE_SIZE, 0))
        dO_it = tl.advance(dO_it, (Q_TILE_SIZE, 0))
        O_it  = tl.advance(O_it,  (Q_TILE_SIZE, 0))
        L_it  = tl.advance(L_it,  (Q_TILE_SIZE,))

    if OUT_DTYPE == 0:
        dK_store = dK_acc
        dV_store = dV_acc
    elif OUT_DTYPE == 1:
        dK_store = dK_acc.to(tl.float16)
        dV_store = dV_acc.to(tl.float16)
    else:
        dK_store = dK_acc.to(tl.bfloat16)
        dV_store = dV_acc.to(tl.bfloat16)

    tl.store(dK_tile, dK_store, boundary_check=(0, 1))
    tl.store(dV_tile, dV_store, boundary_check=(0, 1))



# ----------------------- Autograd wrapper (Triton) -----------------------

class FlashAttentionTriton(Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, H, L, D = Q.shape
        assert K.shape == V.shape == (B, H, L, D)

        Qr = rearrange(Q, 'b h l d -> (b h) l d').contiguous()
        Kr = rearrange(K, 'b h l d -> (b h) l d').contiguous()
        Vr = rearrange(V, 'b h l d -> (b h) l d').contiguous()

        Or = torch.empty_like(Qr)
        L_tensor = torch.empty(Qr.shape[0], L, device=Q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(D)

        if Q.dtype == torch.float32:
            out_flag = 0
        elif Q.dtype == torch.float16:
            out_flag = 1
        else:
            out_flag = 2  # bfloat16

        grid = lambda META: (triton.cdiv(L, META['Q_TILE_SIZE']), B * H)
        
        flash_fwd_kernel[grid](
            Qr, Kr, Vr, Or, L_tensor,
            Qr.stride(0), Qr.stride(1), Qr.stride(2),
            Kr.stride(0), Kr.stride(1), Kr.stride(2),
            Vr.stride(0), Vr.stride(1), Vr.stride(2),
            Or.stride(0), Or.stride(1), Or.stride(2),
            L_tensor.stride(0), L_tensor.stride(1),
            N_QUERIES=L, N_KEYS=L, scale=scale,
            D=D, IS_CAUSAL=is_causal,
            OUT_DTYPE=out_flag,   # <--- 新增
        )

        O = Or.view(B, H, L, D)
        L_final = L_tensor.view(B, H, L)

        ctx.save_for_backward(Q, K, V, O, L_final)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        B, H, Lq, D = Q.shape
        Kh = Lq

        Qr  = rearrange(Q,  'b h l d -> (b h) l d').contiguous()
        Kr  = rearrange(K,  'b h l d -> (b h) l d').contiguous()
        Vr  = rearrange(V,  'b h l d -> (b h) l d').contiguous()
        Or  = rearrange(O,  'b h l d -> (b h) l d').contiguous()
        dOr = rearrange(grad_output, 'b h l d -> (b h) l d').contiguous()
        Lr  = rearrange(L,  'b h l   -> (b h) l'   ).contiguous()

        dQr = torch.empty_like(Qr)
        dKr = torch.empty_like(Kr)
        dVr = torch.empty_like(Vr)

        scale = 1.0 / math.sqrt(D)

        if Q.dtype == torch.float32:
            out_flag = 0
        elif Q.dtype == torch.float16:
            out_flag = 1
        else:
            out_flag = 2  # bfloat16

        grid_k = lambda META: (triton.cdiv(Kh, META['K_TILE_SIZE']), B * H)
        grid_q = lambda META: (triton.cdiv(Lq, META['Q_TILE_SIZE']), B * H)
\
        flash_bwd_dkv_kernel[grid_k](
            Qr, Kr, Vr,                 # Q K V
            dOr, Or, Lr,                # dO, O, L
            dKr, dVr,                   # outputs
            # strides: Q K V
            Qr.stride(0), Qr.stride(1), Qr.stride(2),
            Kr.stride(0), Kr.stride(1), Kr.stride(2),
            Vr.stride(0), Vr.stride(1), Vr.stride(2),
            # strides: dO, O
            dOr.stride(0), dOr.stride(1), dOr.stride(2),
            Or.stride(0),  Or.stride(1),  Or.stride(2),
            # strides: L
            Lr.stride(0), Lr.stride(1),
            # strides: dK, dV
            dKr.stride(0), dKr.stride(1), dKr.stride(2),
            dVr.stride(0), dVr.stride(1), dVr.stride(2),
            N_QUERIES=Lq, N_KEYS=Kh, scale=scale,
            D=D, IS_CAUSAL=ctx.is_causal,
            OUT_DTYPE=out_flag,         
        )

        flash_bwd_dq_kernel[grid_q](
            Qr, Kr, Vr,
            dOr, Or, Lr,   
            dQr,
            # strides: Q K V
            Qr.stride(0), Qr.stride(1), Qr.stride(2),
            Kr.stride(0), Kr.stride(1), Kr.stride(2),
            Vr.stride(0), Vr.stride(1), Vr.stride(2),
            # strides: dO, O
            dOr.stride(0), dOr.stride(1), dOr.stride(2),
            Or.stride(0),  Or.stride(1),  Or.stride(2),
            # strides: L
            Lr.stride(0), Lr.stride(1),
            # strides: dQ
            dQr.stride(0), dQr.stride(1), dQr.stride(2),
            N_QUERIES=Lq, N_KEYS=Kh, scale=scale,
            D=D, IS_CAUSAL=ctx.is_causal,
            OUT_DTYPE=out_flag,     
        )

        dQ = dQr.view(B, H, Lq, D)
        dK = dKr.view(B, H, Kh, D)
        dV = dVr.view(B, H, Kh, D)
        return dQ, dK, dV, None


def _flash_attention_bwd(Q, K, V, O, L, gO, is_causal: bool, BLOCK_N: int = 64, BLOCK_M: int = 64):
    B, H, Lseq, D = Q.shape
    device = Q.device
    scale = 1.0 / math.sqrt(D)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    for i0 in range(0, Lseq, BLOCK_N):
        i1 = min(i0 + BLOCK_N, Lseq)

        q_blk = Q[:, :, i0:i1, :]
        o_blk = O[:, :, i0:i1, :]
        go_blk = gO[:, :, i0:i1, :]
        L_blk = L[:, :, i0:i1]

        delta = (go_blk * o_blk).sum(dim=-1)  # (B, H, I)

        for j0 in range(0, Lseq, BLOCK_M):
            j1 = min(j0 + BLOCK_M, Lseq)

            k_blk = K[:, :, j0:j1, :]
            v_blk = V[:, :, j0:j1, :]

            S = einsum(q_blk, k_blk, "b h i d, b h j d -> b h i j") * scale

            if is_causal:
                i_idx = torch.arange(i0, i1, device=device)
                j_idx = torch.arange(j0, j1, device=device)
                causal = (j_idx[None, :] <= i_idx[:, None])
                S = S.masked_fill(~causal[None, None, :, :], float("-inf"))

            P = torch.exp(S - L_blk.unsqueeze(-1))

            dV[:, :, j0:j1, :] += einsum(P, go_blk, "b h i j, b h i d -> b h j d")
            dp = einsum(go_blk, v_blk, "b h i d, b h j d -> b h i j")
            dS = P * (dp - delta.unsqueeze(-1))
            dQ[:, :, i0:i1, :] += einsum(dS, k_blk, "b h i j, b h j d -> b h i d") * scale
            dK[:, :, j0:j1, :] += einsum(dS, q_blk, "b h i j, b h i d -> b h j d") * scale

    return dQ, dK, dV


flash_backward_compiled = torch.compile(_flash_attention_bwd, mode="max-autotune", fullgraph=False)


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

            q_block = Q[:, :, row_start:row_end, :]

            o_block = torch.zeros((B, H, row_end - row_start, D), dtype=dtype, device=device)
            l_i = torch.zeros((B, H, row_end - row_start), device=device, dtype=dtype)
            m_i = torch.full((B, H, row_end - row_start), float("-inf"), device=device, dtype=dtype)

            for j in range(0, L, BLOCK_M):
                col_start = j
                col_end = min(j + BLOCK_M, L)

                k_block = K[:, :, col_start:col_end, :]
                v_block = V[:, :, col_start:col_end, :]

                attn_score = einsum(q_block, k_block, "b h i d, b h j d -> b h i j") * scale

                if is_causal:
                    mask = torch.full(
                        (row_end - row_start, col_end - col_start),
                        float("-inf"), device=device, dtype=dtype,
                    )
                    for row_idx in range(row_end - row_start):
                        for col_idx in range(col_end - col_start):
                            actual_row = row_start + row_idx
                            actual_col = col_start + col_idx
                            if actual_col <= actual_row:
                                mask[row_idx, col_idx] = 0.0
                    attn_score = attn_score + mask.unsqueeze(0).unsqueeze(0)

                m_ij = torch.max(attn_score, dim=-1, keepdim=True)[0]
                row_has_any = torch.isfinite(m_ij)
                safe_m_ij = torch.where(row_has_any, m_ij, torch.zeros_like(m_ij))
                p_ij = torch.exp(attn_score - safe_m_ij)
                l_ij = torch.sum(p_ij, dim=-1, keepdim=True)

                m_i_new = torch.max(m_i, m_ij.squeeze(-1))
                alpha = torch.exp(m_i.unsqueeze(-1) - m_i_new.unsqueeze(-1))
                beta = torch.exp(m_ij - m_i_new.unsqueeze(-1))

                l_i_new = alpha.squeeze(-1) * l_i + beta.squeeze(-1) * l_ij.squeeze(-1)

                if j == 0:
                    o_block = (beta * einsum(p_ij, v_block, "b h i j, b h j d -> b h i d")) / l_i_new.unsqueeze(-1)
                else:
                    o_block = (o_block * l_i.unsqueeze(-1) * alpha
                               + einsum(p_ij, v_block, "b h i j, b h j d -> b h i d") * beta) / l_i_new.unsqueeze(-1)

                m_i = m_i_new
                l_i = l_i_new

            out[:, :, row_start:row_end, :] = o_block
            logsumexp[:, :, row_start:row_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(Q, K, V, out, logsumexp)
        ctx.is_causal = is_causal

        return out

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = flash_backward_compiled(Q, K, V, O, L, grad_output, is_causal, 64, 64)
        return dQ, dK, dV, None


# ----------------------- Baseline attention -----------------------

def standard_attention(Q, K, V, is_causal=False):
    """
    方程4: S = QK^T / sqrt(d_k)
    方程5: P = softmax(S)
    方程6: O = P V
    方程12: L = logsumexp(S) （数值稳定）
    """
    B, H, L, D = Q.shape
    d_k = 1.0 / math.sqrt(D)
    S = einsum(Q, K, "b h i d, b h j d -> b h i j") * d_k
    if is_causal:
        mask = torch.triu(torch.ones(L, L, device=Q.device, dtype=Q.dtype), diagonal=1)
        S = S.masked_fill(mask.bool(), float("-inf"))
    Lout = torch.logsumexp(S, dim=-1)
    P = torch.softmax(S, dim=-1)
    O = einsum(P, V, "b h i j, b h j d -> b h i d")
    return O, Lout



def test_flashattention():
    import os
    torch.manual_seed(0)
    device = "cuda"

    B = int(os.getenv("B", 8))
    H = int(os.getenv("H", 8))
    L = int(os.getenv("L", 512))
    D = int(os.getenv("D", 64))
    iters = int(os.getenv("ITERS", 50))
    dtype = torch.float32

    ATOL_LOOP = float(os.getenv("ATOL_LOOP", "1e-4"))
    RTOL_LOOP = float(os.getenv("RTOL_LOOP", "1e-4"))
    ATOL_TRI  = float(os.getenv("ATOL_TRI",  "3e-3"))
    RTOL_TRI  = float(os.getenv("RTOL_TRI",  "1e-3"))

    def gen_inputs():
        Q = torch.randn(B, H, L, D, device=device, dtype=dtype)
        K = torch.randn(B, H, L, D, device=device, dtype=dtype)
        V = torch.randn(B, H, L, D, device=device, dtype=dtype)
        return Q, K, V

    def bench(fn, iters=iters, warmup=10):
        torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(warmup):
                _ = fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(iters):
                _ = fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def metrics(a, b, name, atol=1e-4, rtol=1e-4):
        diff = (a - b).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        mean_rel = (diff / b.abs().clamp_min(1e-8)).mean().item()
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        print(f"  [{name}] max|Δ|={max_abs:.3e}, mean|Δ|={mean_abs:.3e}, mean rel err={mean_rel:.3e}, allclose(atol={atol},rtol={rtol})={ok}")
        return ok

    for is_causal in (False, True):
        print(f"\n===== Test (B={B}, H={H}, L={L}, D={D}, is_causal={is_causal}) =====")
        Q, K, V = gen_inputs()

        with torch.no_grad():
            std_O, _ = standard_attention(Q, K, V, is_causal=is_causal)
            loop_O = FlashAttentionPytorch.apply(Q, K, V, is_causal)
            tri_O  = FlashAttentionTriton.apply(Q, K, V, is_causal)

        print("Numerical accuracy vs. standard attention:")
        metrics(loop_O, std_O, "Loop  vs Std", atol=ATOL_LOOP, rtol=RTOL_LOOP)
        metrics(tri_O,  std_O, "Triton vs Std", atol=ATOL_TRI,  rtol=RTOL_TRI)
        metrics(tri_O,  loop_O, "Triton vs Loop", atol=max(ATOL_LOOP, ATOL_TRI), rtol=max(RTOL_LOOP, RTOL_TRI))

        f_std  = lambda: standard_attention(Q, K, V, is_causal=is_causal)[0]
        f_loop = lambda: FlashAttentionPytorch.apply(Q, K, V, is_causal)
        f_tri  = lambda: FlashAttentionTriton.apply(Q, K, V, is_causal)

        t_std  = bench(f_std)
        t_loop = bench(f_loop)
        t_tri  = bench(f_tri)

        print("Throughput (ms per run):")
        print(f"  Std     : {t_std:.3f} ms")
        print(f"  Loop    : {t_loop:.3f} ms  (speedup vs Std: ×{t_std / max(t_loop,1e-9):.2f})")
        print(f"  Triton  : {t_tri:.3f} ms  (speedup vs Std: ×{t_std / max(t_tri,1e-9):.2f}, vs Loop: ×{t_loop / max(t_tri,1e-9):.2f})")

    print("\nAll tests done.")


def run_flash_benchmarks(
    B: int = 8, H: int = 8, L: int = 512, D: int = 64,
    iters_fwd: int = 50, iters_bwd: int = 20,
    dtype: torch.dtype = torch.float32, device: str = "cuda",
    seed: int = 0,
    atol_fwd_loop: float = 1e-4, rtol_fwd_loop: float = 1e-4,
    atol_fwd_tri: float = 3e-3, rtol_fwd_tri: float = 1e-3,
    atol_bwd: float = 3e-3, rtol_bwd: float = 1e-3,
):
    import time

    torch.manual_seed(seed)
    assert torch.cuda.is_available(), "CUDA device required"
    device = torch.device(device)

    def gen_inputs():
        Q = torch.randn(B, H, L, D, device=device, dtype=dtype)
        K = torch.randn(B, H, L, D, device=device, dtype=dtype)
        V = torch.randn(B, H, L, D, device=device, dtype=dtype)
        return Q, K, V

    def sync():
        torch.cuda.synchronize()

    def bench(fn, iters, warmup=10):
        sync()
        with torch.no_grad():
            for _ in range(warmup):
                _ = fn()
        sync()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(iters):
                _ = fn()
        end.record()
        sync()
        return start.elapsed_time(end) / max(iters, 1)

    def bench_backward(forward_apply, is_causal, iters, warmup=5):
        Q0, K0, V0 = gen_inputs()
        gO_const = torch.randn(B, H, L, D, device=device, dtype=dtype)

        def one_step():
            Q = Q0.detach().requires_grad_(True)
            K = K0.detach().requires_grad_(True)
            V = V0.detach().requires_grad_(True)
            O = forward_apply(Q, K, V, is_causal)
            loss = (O * gO_const).sum()
            loss.backward()
            return loss

        sync()
        for _ in range(warmup):
            _ = one_step()
        sync()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = one_step()
        end.record()
        sync()
        return start.elapsed_time(end) / max(iters, 1)

    def metrics(a, b, name, atol=1e-4, rtol=1e-4):
        diff = (a - b).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        rel = (diff / b.abs().clamp_min(1e-8)).mean().item()
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        print(f"  [{name}] max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}  mean rel={rel:.3e}  allclose={ok}")
        return ok, max_abs, mean_abs, rel

    for is_causal in (False, True):
        print(f"\n===== Bench (B={B}, H={H}, L={L}, D={D}, dtype={dtype}, is_causal={is_causal}) =====")

        with torch.no_grad():
            Q, K, V = gen_inputs()
            std_O, std_L = standard_attention(Q, K, V, is_causal=is_causal)
            loop_O = FlashAttentionPytorch.apply(Q, K, V, is_causal)
            tri_O  = FlashAttentionTriton.apply(Q, K, V, is_causal)

        print("Forward correctness vs. standard:")
        metrics(loop_O, std_O, "Loop  vs Std", atol_fwd_loop, rtol_fwd_loop)
        metrics(tri_O,  std_O, "Triton vs Std", atol_fwd_tri,  rtol_fwd_tri)

        f_std  = lambda: standard_attention(Q, K, V, is_causal=is_causal)[0]
        f_loop = lambda: FlashAttentionPytorch.apply(Q, K, V, is_causal)
        f_tri  = lambda: FlashAttentionTriton.apply(Q, K, V, is_causal)

        t_std  = bench(f_std,  iters_fwd)
        t_loop = bench(f_loop, iters_fwd)
        t_tri  = bench(f_tri,  iters_fwd)

        print("Forward throughput (ms/run):")
        print(f"  Std     : {t_std:.3f} ms")
        print(f"  Loop    : {t_loop:.3f} ms  (×{t_std/max(t_loop,1e-9):.2f} vs Std)")
        print(f"  Triton  : {t_tri:.3f} ms  (×{t_std/max(t_tri,1e-9):.2f} vs Std, ×{t_loop/max(t_tri,1e-9):.2f} vs Loop)")

        # 基准反向（standard）+ 固定 gO
        def grads_standard(Q, K, V, is_causal):
            Qr = Q.clone().detach().requires_grad_(True)
            Kr = K.clone().detach().requires_grad_(True)
            Vr = V.clone().detach().requires_grad_(True)
            O, _ = standard_attention(Qr, Kr, Vr, is_causal=is_causal)
            gO = torch.randn_like(O)
            (O * gO).sum().backward()
            return (Qr.grad, Kr.grad, Vr.grad), gO

        def grads_loop(Q, K, V, is_causal, gO):
            Qr = Q.clone().detach().requires_grad_(True)
            Kr = K.clone().detach().requires_grad_(True)
            Vr = V.clone().detach().requires_grad_(True)
            O = FlashAttentionPytorch.apply(Qr, Kr, Vr, is_causal)
            (O * gO).sum().backward()
            return Qr.grad, Kr.grad, Vr.grad

        def grads_triton(Q, K, V, is_causal, gO):
            Qr = Q.clone().detach().requires_grad_(True)
            Kr = K.clone().detach().requires_grad_(True)
            Vr = V.clone().detach().requires_grad_(True)
            O = FlashAttentionTriton.apply(Qr, Kr, Vr, is_causal)
            (O * gO).sum().backward()
            return Qr.grad, Kr.grad, Vr.grad

        try:
            (dQ_std, dK_std, dV_std), gO = grads_standard(Q, K, V, is_causal)
            try:
                dQ_loop, dK_loop, dV_loop = grads_loop(Q, K, V, is_causal, gO)
                print("Backward correctness (Loop vs Std):")
                metrics(dQ_loop, dQ_std, "dQ", atol_bwd, rtol_bwd)
                metrics(dK_loop, dK_std, "dK", atol_bwd, rtol_bwd)
                metrics(dV_loop, dV_std, "dV", atol_bwd, rtol_bwd)
            except Exception as e:
                print(f"Backward (Loop) failed: {e}")

            try:
                dQ_tri, dK_tri, dV_tri = grads_triton(Q, K, V, is_causal, gO)
                print("Backward correctness (Triton vs Std):")
                metrics(dQ_tri, dQ_std, "dQ", atol_bwd, rtol_bwd)
                metrics(dK_tri, dK_std, "dK", atol_bwd, rtol_bwd)
                metrics(dV_tri, dV_std, "dV", atol_bwd, rtol_bwd)
            except Exception as e:
                print(f"Backward (Triton) failed: {e}")
        except RuntimeError as e:
            print(f"Standard backward failed (skipping bwd checks): {e}")

        print("Backward throughput (ms/run):")
        try:
            t_bwd_std = bench_backward(lambda Q, K, V, c: standard_attention(Q, K, V, c)[0],
                                       is_causal, iters_bwd)
            print(f"  Std     : {t_bwd_std:.3f} ms")
        except Exception as e:
            print(f"  Std     : failed ({e})")

        try:
            t_bwd_loop = bench_backward(lambda Q, K, V, c: FlashAttentionPytorch.apply(Q, K, V, c),
                                        is_causal, iters_bwd)
            print(f"  Loop    : {t_bwd_loop:.3f} ms")
        except Exception as e:
            print(f"  Loop    : failed ({e})")

        try:
            t_bwd_tri = bench_backward(lambda Q, K, V, c: FlashAttentionTriton.apply(Q, K, V, c),
                                       is_causal, iters_bwd)
            print(f"  Triton  : {t_bwd_tri:.3f} ms")
        except Exception as e:
            print(f"  Triton  : failed ({e})")

    print("\nAll benchmarks done.")


if __name__ == "__main__":
    test_flashattention()
    run_flash_benchmarks()
