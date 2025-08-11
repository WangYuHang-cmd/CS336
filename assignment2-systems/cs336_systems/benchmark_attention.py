#!/usr/bin/env python3
# benchmark_attention.py
import time, sys, os
import math
import itertools
import json
import argparse
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cs336-basics'))
from cs336_basics.attention import ScaledDotProductAttention
torch.set_float32_matmul_precision("high")



def naive_attention(Q, K, V):
    """
    Single-head attention (no masking), shapes:
      Q,K,V: [B, S, D]
    Returns:
      O: [B, S, D]
    """
    B, S, D = Q.shape
    # [B, S, S]
    logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(D)
    attn = torch.softmax(logits, dim=-1)
    # [B, S, D]
    out = torch.matmul(attn, V)
    return out

def format_mib(bytes_):
    return bytes_ / (1024.0 ** 2)

def run_one_config(B, S, D, device, warmup=10, iters=100):
    """
    Returns dict with timings and memory. Handles OOM robustly.
    """
    result = {
        "batch": B, "seq_len": S, "d_model": D,
        "fwd_ms": None, "bwd_ms": None,
        "mem_before_bwd_MiB": None, "status": "OK"
    }
    
    attn = ScaledDotProductAttention().to(device)
    attn.eval()  # No gradients needed for this benchmark
    attn = torch.compile(attn, dynamic=True)

    try:
        # ------------ Forward timing (with no grad graph) ------------
        Q = torch.randn(B, S, D, device=device, dtype=torch.float32)
        K = torch.randn(B, S, D, device=device, dtype=torch.float32)
        V = torch.randn(B, S, D, device=device, dtype=torch.float32)

        # Warmup (no grad)
        for _ in range(warmup):
            with torch.no_grad():
                _ = attn(Q, K, V)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

        # Measure 100 forward passes
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                _ = attn(Q, K, V)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        result["fwd_ms"] = (t1 - t0) * 1000.0

        # ------------ Memory before backward ------------
        # Build a single forward graph and measure memory before calling backward.
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        Qg = torch.randn(B, S, D, device=device, dtype=torch.float32, requires_grad=True)
        Kg = torch.randn(B, S, D, device=device, dtype=torch.float32, requires_grad=True)
        Vg = torch.randn(B, S, D, device=device, dtype=torch.float32, requires_grad=True)

        out = attn(Qg, Kg, Vg)
        # record allocated memory *after* forward, *before* backward
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            mem_before_bwd = torch.cuda.memory_allocated()
            result["mem_before_bwd_MiB"] = format_mib(mem_before_bwd)
        else:
            result["mem_before_bwd_MiB"] = 0.0

        # Clear graph (we'll time backward using fresh graphs per-iter)
        loss = out.sum()
        loss.backward()
        # Drop tensors so graph can be freed
        del Qg, Kg, Vg, out, loss
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # ------------ Backward timing ------------
        # Each iteration builds its own forward graph then runs backward
        # (no retain_graph) to reflect training cost realistically.
        # Warmup
        for _ in range(warmup):
            Qg = torch.randn(B, S, D, device=device, dtype=torch.float32, requires_grad=True)
            Kg = torch.randn(B, S, D, device=device, dtype=torch.float32, requires_grad=True)
            Vg = torch.randn(B, S, D, device=device, dtype=torch.float32, requires_grad=True)
            out = attn(Qg, Kg, Vg)
            loss = out.sum()
            loss.backward()
            del Qg, Kg, Vg, out, loss
            if device.startswith("cuda"):
                torch.cuda.synchronize()

        # Measure 100 backward passes
        t0 = time.perf_counter()
        for _ in range(iters):
            Qg = torch.randn(B, S, D, device=device, dtype=torch.float32, requires_grad=True)
            Kg = torch.randn(B, S, D, device=device, dtype=torch.float32, requires_grad=True)
            Vg = torch.randn(B, S, D, device=device, dtype=torch.float32, requires_grad=True)
            out = attn(Qg, Kg, Vg)
            loss = out.sum()
            loss.backward()
            del Qg, Kg, Vg, out, loss
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        result["bwd_ms"] = (t1 - t0) * 1000.0

    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda error" in msg or "cublas" in msg:
            result["status"] = "OOM"
        else:
            result["status"] = f"ERROR: {e}"
    finally:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="attention_bench.json",
                        help="Path to write JSON results (also printed as CSV to stdout)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = "cuda"
    B = 8
    d_models = [32, 64, 128, 256, 512]
    seq_lens = [64, 128, 256, 1024, 4096, 8192, 16384]

    print("# PyTorch single-head attention benchmark")
    print("# B=8, dtype=fp32, device=cuda")
    print("d_model,seq_len,fwd_ms,bwd_ms,mem_before_bwd_MiB,status")

    results = []
    for D, S in itertools.product(d_models, seq_lens):
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        res = run_one_config(B, S, D, device, warmup=args.warmup, iters=args.iters)
        results.append(res)
        print("{},{},{:.3f},{:.3f},{:.1f},{}".format(
            res["d_model"], res["seq_len"],
            -1.0 if res["fwd_ms"] is None else res["fwd_ms"],
            -1.0 if res["bwd_ms"] is None else res["bwd_ms"],
            -1.0 if res["mem_before_bwd_MiB"] is None else res["mem_before_bwd_MiB"],
            res["status"],
        ))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"# Saved JSON to {args.out}")

if __name__ == "__main__":
    main()
