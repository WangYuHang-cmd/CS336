#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专为你当前 CSV 表头定制：
NVTX:  Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Range
KERNEL: Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
"""

import os, glob, csv, argparse
from collections import defaultdict

def find_csv(prefix: str, kind: str) -> str:
    # 兼容你这里的 “重复后缀” 情况：*_nvtx_pushpop_sum.csv_nvtx_pushpop_sum.csv
    pats = [
        f"{prefix}_{kind}.csv",
        f"{prefix}_{kind}.csv*",
        f"{prefix}*_{kind}.csv",
        f"{prefix}*{kind}.csv*",
    ]
    cands = []
    for p in pats:
        cands.extend(glob.glob(p))
    return sorted(cands, key=lambda s: (len(s), s))[-1] if cands else ""

def read_rows(path: str):
    if not path: return []
    with open(path, "r", newline="") as f:
        # 你这里看起来是逗号分隔
        rdr = csv.DictReader(f, delimiter=",")
        return list(rdr)

def ns_to_ms(x: float) -> float:
    return x / 1e6

def kernel_cat(name: str) -> str:
    n = name.lower()
    # 针对你当前的情况，把 Kernel2 当成 matmul
    if name.strip() == "Kernel2":
        return "matmul"

    if ("gemm" in n) or ("cublas" in n) or ("mma" in n):
        return "matmul"
    if ("softmax" in n) or ("sdpa" in n):
        return "softmax"
    if ("layernorm" in n) or ("rmsnorm" in n) or (" ln_" in n) or ("layer_norm" in n):
        return "layernorm"
    if "dropout" in n:
        return "dropout"
    if ("transpose" in n) or ("permute" in n) or ("reduce" in n) or ("reduction" in n) \
       or ("memcpy" in n) or ("memset" in n):
        return "memory/reduction"
    return "other"


# def kernel_cat(name: str) -> str:
#     n = name.lower()
#     if ("gemm" in n) or ("cublas" in n) or ("mma" in n):
#         return "matmul"
#     if ("softmax" in n) or ("sdpa" in n):
#         return "softmax"
#     if ("layernorm" in n) or ("rmsnorm" in n) or (" ln_" in n) or ("layer_norm" in n):
#         return "layernorm"
#     if "dropout" in n:
#         return "dropout"
#     if ("transpose" in n) or ("permute" in n) or ("reduce" in n) or ("reduction" in n) or ("memcpy" in n) or ("memset" in n):
#         return "memory/reduction"
#     return "other"

def analyze(prefix: str):
    nvtx = find_csv(prefix, "nvtx_pushpop_sum")
    kern = find_csv(prefix, "cuda_gpu_kern_sum_base")

    nvtx_rows = read_rows(nvtx)
    kern_rows = read_rows(kern)

    # ---- (a): NVTX 里 FORWARD 总时长（单位 ns -> ms），注意范围名前有冒号 ----
    fwd_ms = 0.0
    bwd_ms = 0.0
    opt_ms = 0.0
    mha_blocks = {}  # (e) 用
    for r in nvtx_rows:
        rname = str(r.get("Range",""))
        t_ns = float(r.get("Total Time (ns)", "0") or 0)
        if rname == ":FORWARD":
            fwd_ms += ns_to_ms(t_ns)
        elif rname == ":BACKWARD":
            bwd_ms += ns_to_ms(t_ns)
        elif rname in (":OPTIMIZER", ":OPTIMIZER.step", ":OPTIMIZER.zero_grad"):
            opt_ms += ns_to_ms(t_ns)
        # MHA 子块
        for key in [
            ":MHA(total)",
            ":attn_logits (QK^T) + mask",
            ":softmax",
            ":attn*V (context)",
            ":QKV projections", ":RoPE(Q)", ":RoPE(K)", ":output proj",
        ]:
            if rname == key:
                mha_blocks[key] = mha_blocks.get(key, 0.0) + ns_to_ms(t_ns)

    # ---- (b)(c)(d): Kernel summary（仅有全局总和，无法按 NVTX 过滤；用整体 & 通过 fwd-only vs train 对比）----
    tot_gpu_ms = 0.0
    by_cat = defaultdict(float)
    top_name, top_calls, top_ms = "", 0, 0.0

    for r in kern_rows:
        t_ns = float(r.get("Total Time (ns)", "0") or 0)
        t_ms = ns_to_ms(t_ns)
        tot_gpu_ms += t_ms
        name = str(r.get("Name",""))
        by_cat[kernel_cat(name)] += t_ms
        # top kernel
        if t_ms > top_ms:
            top_ms = t_ms
            top_name = name
            try:
                top_calls = int(r.get("Instances","0") or 0)
            except:
                top_calls = 0

    def pct(x): return (100.0 * x / tot_gpu_ms) if tot_gpu_ms > 0 else 0.0

    print(f"\n== Prefix: {prefix} ==")
    print(f"[a] NVTX time (ms): FORWARD={fwd_ms:.3f}, BACKWARD={bwd_ms:.3f}, OPTIMIZER={opt_ms:.3f}")
    print(f"[b] Top CUDA kernel: name='{top_name}', calls={top_calls}, time_ms={top_ms:.3f}")
    print("[c] Non-matmul categories (time_ms, share%):")
    for k in sorted(by_cat.keys(), key=lambda k: by_cat[k], reverse=True):
        if k == "matmul": continue
        print(f"     - {k:<18s} {by_cat[k]:10.3f} ms  ({pct(by_cat[k]):5.2f}%)")
    print(f"[d] Matmul share: {pct(by_cat['matmul']):.2f}%  (out of total GPU kernel time {tot_gpu_ms:.1f} ms)")

    # (e) attention 局部（来自 NVTX 子块）
    if mha_blocks:
        print("[e] Attention (from NVTX ranges, ms):")
        for k in [":MHA(total)", ":attn_logits (QK^T) + mask", ":softmax", ":attn*V (context)"]:
            if k in mha_blocks:
                print(f"     - {k[1:]:<28s} {mha_blocks[k]:10.3f} ms")
    else:
        print("[e] 未检测到 MHA 子块（确保运行时传了 --annotate-mha）")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="例如 nsys_train_medium_ctx512_bf16")
    args = ap.parse_args()
    analyze(args.prefix)
