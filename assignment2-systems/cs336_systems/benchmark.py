#!/usr/bin/env python3
import argparse, time, json, contextlib
import torch, torch.nn as nn
import torch.cuda.nvtx as nvtx
from cs336_basics.transformerLM import TransformerLM

# --------- 预设模型规模（按 A2 要求可扩） ---------
SIZE2CFG = {
    "small":  dict(d_model=768,  d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
}
VOCAB_SIZE = 10_000

def set_matmul_precision(fp32_mode: str):
    # 'highest' | 'high' | 'medium'，高/中等将使用 TF32 / BF16 内部实现（CUDA matmul）
    torch.set_float32_matmul_precision(fp32_mode)

def device_sync_if_needed(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()

def autocast_ctx(precision: str, device: str):
    if device.startswith("cuda") and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()

def build_model(size: str, ctx_len: int, device: str, param_dtype: torch.dtype):
    cfg = SIZE2CFG[size]
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        context_length=ctx_len,
        theta=10_000.0,
        num_layers=cfg["num_layers"],
        device=device,
        dtype=param_dtype,
    ).to(device)
    return model

def run_step(model: nn.Module, x: torch.Tensor, mode: str, optimizer=None):
    logits = model(x)
    if mode == "fwd":
        return logits
    # 构造一个轻量损失，保证后向能跑
    loss = logits.float().mean()
    loss.backward()
    if optimizer is not None:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZE2CFG.keys()), default="small")
    ap.add_argument("--ctx", type=int, default=256, choices=[128,256,512,1024])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--mode", choices=["fwd","fwd_bwd"], default="fwd")
    ap.add_argument("--precision", choices=["fp32","bf16"], default="fp32")
    ap.add_argument("--fp32-matmul", choices=["highest","high","medium"], default="high")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--nvtx", action="store_true")
    ap.add_argument("--mem-snapshot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    # 5090: 建议开启 matmul “high”（TF32 路径），对 FP32 测试更接近实际最佳性能
    set_matmul_precision(args.fp32_matmul)  # docs: torch.set_float32_matmul_precision

    # 参数维持 FP32；激活由 autocast 控制（BF16 时）
    model = build_model(args.size, args.ctx, device, torch.float32).train()
    if args.compile:
        model = torch.compile(model)

    B, T = args.batch, args.ctx
    x = torch.randint(VOCAB_SIZE, (B, T), device=device, dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) if args.mode=="fwd_bwd" else None

    # Warmup
    if args.nvtx: nvtx.range_push("warmup")
    for _ in range(args.warmup):
        with autocast_ctx(args.precision, device):
            run_step(model, x, args.mode, optimizer)
        device_sync_if_needed(device)
    if args.nvtx: nvtx.range_pop()

    # 可选：显存历史（供可视化）
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        if args.mem_snapshot:
            torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    # Measure
    iters = []
    if args.nvtx: nvtx.range_push("measure_loop")
    for _ in range(args.steps):
        t0 = time.perf_counter()
        tag = "fwd" if args.mode=="fwd" else "train_step"
        if args.nvtx: nvtx.range_push(tag)
        with autocast_ctx(args.precision, device):
            run_step(model, x, args.mode, optimizer)
        if args.nvtx: nvtx.range_pop()
        device_sync_if_needed(device)
        t1 = time.perf_counter()
        iters.append(t1 - t0)
    if args.nvtx: nvtx.range_pop()

    avg = sum(iters)/len(iters)
    std = (sum((t-avg)**2 for t in iters)/(len(iters)-1))**0.5 if len(iters)>1 else 0.0

    # 统计 throughput & 显存峰值
    toks = B * T
    toks_per_s = toks / avg
    peak_alloc = peak_reserved = None
    if device.startswith("cuda"):
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)

    result = {
        "size": args.size, "ctx": T, "batch": B, "mode": args.mode,
        "precision": args.precision, "fp32_matmul": args.fp32_matmul,
        "compile": args.compile,
        "avg_ms": avg*1000, "std_ms": std*1000,
        "tokens_per_s": toks_per_s,
        "peak_mem_alloc_MiB": peak_alloc, "peak_mem_reserved_MiB": peak_reserved,
    }
    print("[RESULT]", json.dumps(result, ensure_ascii=False, indent=2))

    # 导出内存快照
    if device.startswith("cuda") and args.mem_snapshot:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        print("[MEM] dumped memory_snapshot.pickle (open https://pytorch.org/memory_viz)")

if __name__ == "__main__":
    main()


# TEST
"""
PYTHONPATH="$PWD/cs336-basics:$PYTHONPATH" \
uv run --no-project --python /home/henry/miniconda3/envs/llm/bin/python --  \
  python cs336_systems/benchmark.py --size small --ctx 256 --batch 4 \
  --mode fwd --precision fp32 --warmup 5 --steps 20

PYTHONPATH="$PWD/cs336-basics:$PYTHONPATH" \
uv run --no-project --python /home/henry/miniconda3/envs/llm/bin/python --  \
  python cs336_systems/benchmark.py --size small --ctx 512 --batch 4 \
  --mode fwd --precision bf16 --warmup 5 --steps 20


PYTHONPATH="$PWD/cs336-basics:$PYTHONPATH" \
uv run --no-project --python /home/henry/miniconda3/envs/llm/bin/python --  \
  python cs336_systems/benchmark.py --size medium --ctx 512 --batch 4 \
  --mode fwd_bwd --precision fp32 --warmup 5 --steps 20


PYTHONPATH="$PWD/cs336-basics:$PYTHONPATH" \
uv run --no-project --python /home/henry/miniconda3/envs/llm/bin/python --  \
  python cs336_systems/benchmark.py --size medium --ctx 512 --batch 4 \
  --mode fwd_bwd --precision bf16 --warmup 5 --steps 20


PYTHONPATH="$PWD/cs336-basics:$PYTHONPATH" \
uv run --no-project --python /home/henry/miniconda3/envs/llm/bin/python --  \
  nsys profile --trace=cuda,nvtx --capture-range=nvtx --nvtx-capture=measure_loop \
  --capture-range-end=stop -o nsys_out \
  python cs336_systems/benchmark.py --size medium --ctx 512 --batch 4 \
    --mode fwd_bwd --precision bf16 --nvtx --warmup 5 --steps 50




PYTHONPATH="$PWD/cs336-basics:$PYTHONPATH" uv run --no-project --python /home/henry/miniconda3/envs/llm/bin/python --   python cs336_systems/benchmark.py --size small --ctx 256 --mode fwd
"""