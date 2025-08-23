# GEMM_TEST.py
# Compare three implementations:
#   1) PyTorch/cuBLAS matmul (baseline)
#   2) Triton single-kernel GEMM (2D tiling, reduce along K)
#   3) Triton Split-K two-stage reduction (no atomics): stage-1 partials + stage-2 reduction
#
# Example:
#   python GEMM_TEST.py --M 4096 --K 8192 --N 4096 --dtype fp16 --iters 50 --warmup 10 --split-k 2,4,8
#
# Notes:
# - Accumulators use fp32 for numerical stability; inputs can be fp16/bf16/fp32.
# - Split-K uses two kernels (partials + reduction), no atomics.

import argparse
import time
import torch
import triton
import triton.language as tl


# =========================
# Triton kernels
# =========================

@triton.jit
def gemm_fwd(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # A strides
    stride_bk, stride_bn,       # B strides
    stride_cm, stride_cn,       # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    A_blk = tl.make_block_ptr(
        base=A_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(m0, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    B_blk = tl.make_block_ptr(
        base=B_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, n0), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0)
    )
    C_blk = tl.make_block_ptr(
        base=C_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(m0, n0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    for _ in range(k_tiles):
        A_tile = tl.load(A_blk, boundary_check=(0, 1), padding_option="zero")
        B_tile = tl.load(B_blk, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(A_tile, B_tile)
        A_blk = A_blk.advance((0, BLOCK_K))
        B_blk = B_blk.advance((BLOCK_K, 0))

    tl.store(C_blk, acc, boundary_check=(0, 1))


@triton.jit
def gemm_splitk_stage1(
    A_ptr, B_ptr, PART_ptr,     # PART: [S, M, N]
    M, N, K, S,                 # S = split_k
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ps, stride_pm, stride_pn,  # partials strides (S, M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)  # which K-slice

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    ks = (K * pid_s) // S
    ke = (K * (pid_s + 1)) // S
    k_len = ke - ks

    A_blk = tl.make_block_ptr(
        base=A_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(m0, ks), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    B_blk = tl.make_block_ptr(
        base=B_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(ks, n0), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0)
    )
    PART_blk = tl.make_block_ptr(
        base=PART_ptr, shape=(S, M, N), strides=(stride_ps, stride_pm, stride_pn),
        offsets=(pid_s, m0, n0), block_shape=(1, BLOCK_M, BLOCK_N), order=(2, 1, 0)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(k_len, BLOCK_K)
    for _ in range(k_tiles):
        A_tile = tl.load(A_blk, boundary_check=(0, 1), padding_option="zero")
        B_tile = tl.load(B_blk, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(A_tile, B_tile)
        A_blk = A_blk.advance((0, BLOCK_K))
        B_blk = B_blk.advance((BLOCK_K, 0))

    tl.store(PART_blk, acc[None, :, :], boundary_check=(0, 1, 2))


@triton.jit
def splitk_reduce_stage2(
    PART_ptr, C_ptr,
    M, N, S,
    stride_ps, stride_pm, stride_pn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    C_blk = tl.make_block_ptr(
        base=C_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(m0, n0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    PART_blk = tl.make_block_ptr(
        base=PART_ptr, shape=(S, M, N), strides=(stride_ps, stride_pm, stride_pn),
        offsets=(0, m0, n0), block_shape=(1, BLOCK_M, BLOCK_N), order=(2, 1, 0)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _s in range(S):
        tile = tl.load(PART_blk, boundary_check=(0, 1, 2), padding_option="zero")  # (1,BM,BN)
        acc += tl.sum(tile, axis=0)  # 等价于 tile[0, :, :]，避免局部张量切片限制
        PART_blk = PART_blk.advance((1, 0, 0))

    tl.store(C_blk, acc, boundary_check=(0, 1))


# =========================
# Python helpers
# =========================

def tflops(M, N, K, ms):
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


def run_triton_single(A, B, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=3):
    M, K = A.shape
    K2, N = B.shape
    assert K2 == K
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_fwd[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return C


def run_triton_splitk(A, B, split_k=4, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=3):
    M, K = A.shape
    K2, N = B.shape
    assert K2 == K
    device = A.device
    PART = torch.empty((split_k, M, N), device=device, dtype=torch.float32)
    C = torch.empty((M, N), device=device, dtype=torch.float32)

    grid1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), split_k)
    gemm_splitk_stage1[grid1](
        A, B, PART, M, N, K, split_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        PART.stride(0), PART.stride(1), PART.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    splitk_reduce_stage2[grid2](
        PART, C, M, N, split_k,
        PART.stride(0), PART.stride(1), PART.stride(2),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=2,
    )
    return C


def _bytes_to_mib(x): return x / (1024.0 ** 2)


def _peak_reserved_bytes():
    # 兼容不同版本 PyTorch：优先 torch.cuda.memory.max_memory_reserved()
    try:
        return torch.cuda.memory.max_memory_reserved()
    except AttributeError:
        return torch.cuda.max_memory_reserved()


def benchmark(fn, warmup, iters, sync=True, track_mem=True, print_mem_summary=False, label=""):
    # Warmup
    for _ in range(warmup):
        out = fn()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    peak_alloc_mib = peak_reserved_mib = None
    if track_mem and torch.cuda.is_available():
        # 重置峰值统计起点（allocated/reserved 都会被重置）
        torch.cuda.reset_peak_memory_stats()  # 官方建议的统一入口 :contentReference[oaicite:1]{index=1}

    # Measure
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    if track_mem and torch.cuda.is_available():
        # 读峰值：allocated（张量实际占用）与 reserved（缓存分配器管理的池）:contentReference[oaicite:2]{index=2}
        peak_alloc_mib = _bytes_to_mib(torch.cuda.max_memory_allocated())
        peak_reserved_mib = _bytes_to_mib(_peak_reserved_bytes())
        if print_mem_summary:
            try:
                summary = torch.cuda.memory_summary()
            except AttributeError:
                summary = torch.cuda.memory.memory_summary()
            print(f"\n[MEM SUMMARY] {label}\n{summary}\n")

    avg_ms = (t1 - t0) * 1000.0 / iters
    return out, avg_ms, peak_alloc_mib, peak_reserved_mib


def check_close(C_ref, C_test, name, atol=1e-2, rtol=1e-2):
    diff = (C_ref - C_test).abs().max().item()
    ok = torch.allclose(C_ref, C_test, atol=atol, rtol=rtol)
    print(f"[CHECK] {name:18s} | max_abs_diff={diff:.3e} | allclose={ok}")
    return ok


# =========================
# CLI
# =========================

# Modified GEMM_TEST.py to support implementation selection
# Add this to your existing GEMM_TEST.py or create a new version

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--split-k", type=str, default="4", help="comma-separated, e.g. 2,4,8")
    ap.add_argument("--allow-tf32", action="store_true", help="enable TF32 matmul for FP32")
    ap.add_argument("--seed", type=int, default=0)
    # NEW: Add implementation selection
    ap.add_argument("--impl", type=str, default="all", 
                    help="Implementation to test: 'cublas', 'triton-1x', 'splitK-N' (e.g., 'splitK-4'), or 'all'")
    # memory tracking flags
    ap.add_argument("--no-track-mem", action="store_true", help="disable peak memory tracking")
    ap.add_argument("--print-mem-summary", action="store_true",
                    help="print torch.cuda.memory_summary() after each benchmark")
    return ap.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.allow_tf32:
        torch.set_float32_matmul_precision("high")

    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    M, K, N = args.M, args.K, args.N
    print(f"# GEMM benchmark  M={M} K={K} N={N}  dtype={args.dtype}  device={device}")
    print(f"# iters={args.iters}, warmup={args.warmup}, implementation={args.impl}")
    print(f"# track_mem={not args.no_track_mem}, print_mem_summary={args.print_mem_summary}")

    A = torch.randn(M, K, device=device, dtype=dt)
    B = torch.randn(K, N, device=device, dtype=dt)

    results = []

    # Determine which implementations to run
    run_cublas = args.impl == "all" or args.impl == "cublas"
    run_triton_1x = args.impl == "all" or args.impl == "triton-1x"
    run_splitk = args.impl == "all" or args.impl.startswith("splitK-")

    # 1) cuBLAS baseline
    if run_cublas:
        def run_cublas():
            return (A @ B).to(torch.float32)

        C_ref, ms_cublas, alloc_cu, reserv_cu = benchmark(
            run_cublas, args.warmup, args.iters,
            track_mem=(not args.no_track_mem),
            print_mem_summary=args.print_mem_summary,
            label="cuBLAS"
        )
        print(f"[cublas]    {ms_cublas:8.3f} ms/it   {tflops(M,N,K,ms_cublas):6.2f} TFLOP/s"
              + ("" if alloc_cu is None else f"   peak_alloc={alloc_cu:.1f} MiB   peak_reserved={reserv_cu:.1f} MiB"))
        results.append(("cublas", ms_cublas, tflops(M,N,K,ms_cublas), alloc_cu, reserv_cu))

    # 2) Triton single-kernel GEMM
    if run_triton_1x:
        def run_single():
            return run_triton_single(A, B)

        C_single, ms_single, alloc_1x, reserv_1x = benchmark(
            run_single, args.warmup, args.iters,
            track_mem=(not args.no_track_mem),
            print_mem_summary=args.print_mem_summary,
            label="triton-1x"
        )
        print(f"[triton-1x] {ms_single:8.3f} ms/it   {tflops(M,N,K,ms_single):6.2f} TFLOP/s"
              + ("" if alloc_1x is None else f"   peak_alloc={alloc_1x:.1f} MiB   peak_reserved={reserv_1x:.1f} MiB"))
        
        if run_cublas:
            check_close(C_ref, C_single, "triton-1x")
        results.append(("triton-1x", ms_single, tflops(M,N,K,ms_single), alloc_1x, reserv_1x))

    # 3) Split-K (two-stage, no atomics)
    if run_splitk:
        if args.impl.startswith("splitK-"):
            # Extract specific split-k value
            splitk_list = [int(args.impl.split("-")[1])]
        else:
            # Use all split-k values
            splitk_list = [int(x) for x in args.split_k.split(",") if x.strip()]
        
        for S in splitk_list:
            def run_splitk():
                return run_triton_splitk(A, B, split_k=S)

            C_splitk, ms_splitk, alloc_sk, reserv_sk = benchmark(
                run_splitk, args.warmup, args.iters,
                track_mem=(not args.no_track_mem),
                print_mem_summary=args.print_mem_summary,
                label=f"splitK-{S}"
            )
            print(f"[splitK-{S:2d}] {ms_splitk:8.3f} ms/it   {tflops(M,N,K,ms_splitk):6.2f} TFLOP/s"
                  + ("" if alloc_sk is None else f"   peak_alloc={alloc_sk:.1f} MiB   peak_reserved={reserv_sk:.1f} MiB"))
            
            if run_cublas:
                check_close(C_ref, C_splitk, f"splitK-{S}")
            elif run_triton_1x:
                check_close(C_single, C_splitk, f"splitK-{S}")
            results.append((f"splitK-{S}", ms_splitk, tflops(M,N,K,ms_splitk), alloc_sk, reserv_sk))

    print("# Done.")
    return results

if __name__ == "__main__":
    main()


# python GEMM_TEST.py --M 4096 --K 8192 --N 4096 --dtype fp16 --iters 500 --warmup 100 --split-k 2,4,8 --print-mem-summary

# 