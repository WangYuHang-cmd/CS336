#!/usr/bin/env python3
import argparse, time, json, contextlib, os
import torch, torch.nn as nn
import torch.cuda.nvtx as nvtx
from cs336_basics.transformerLM import TransformerLM

# ---------- Model size presets ----------
SIZE2CFG = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}
VOCAB_SIZE = 10_000


def set_matmul_precision(fp32_mode: str):
    # Controls TF32 usage on Ampere+/Ada+ for FP32 matmuls
    torch.set_float32_matmul_precision(fp32_mode)


def device_sync_if_needed(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def autocast_ctx(precision: str, device: str):
    if device.startswith("cuda") and precision in {"bf16", "fp16"}:
        dt = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dt)
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


def add_mha_nvtx_if_requested(model: nn.Module, enabled: bool):
    if not enabled:
        return
    from cs336_systems.annotated_attention import instrument_model_mha_with_nvtx
    instrument_model_mha_with_nvtx(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZE2CFG.keys()), default="small")
    ap.add_argument("--ctx", type=int, default=256, choices=[64, 128, 256, 512, 1024])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--mode", choices=["fwd", "fwd_bwd"], default="fwd")
    ap.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    ap.add_argument("--fp32-matmul", choices=["highest", "high", "medium"], default="high")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--nvtx", action="store_true")
    ap.add_argument("--annotate-mha", action="store_true", help="Wrap MHA internals with NVTX")

    # --- memory profiling options ---
    ap.add_argument("--mem-snapshot", action="store_true",
                    help="Record memory history during measure loop and dump a pickle for memory_viz")
    ap.add_argument("--mem-snapshot-out", type=str, default="memory_snapshot.pickle",
                    help="Path to write memory_viz pickle")
    ap.add_argument("--mem-summary", action="store_true",
                    help="Dump torch.cuda.memory_summary() to a text file after run")
    ap.add_argument("--mem-summary-out", type=str, default="memory_summary.txt",
                    help="Path to write memory_summary text")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    # Basics
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_matmul_precision(args.fp32_matmul)
    torch.backends.cudnn.benchmark = True  # harmless here

    # Model (keep params in FP32; autocast handles compute dtype)
    model = build_model(args.size, args.ctx, device, torch.float32).train()
    add_mha_nvtx_if_requested(model, args.annotate_mha)
    if args.compile:
        model = torch.compile(model)

    # Data
    B, T = args.batch, args.ctx
    x = torch.randint(VOCAB_SIZE, (B, T), device=device, dtype=torch.long)

    # Optimizer (only used in training mode)
    optimizer = None
    if args.mode == "fwd_bwd":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # GradScaler for fp16 training
    scaler = None
    if args.mode == "fwd_bwd" and device.startswith("cuda") and args.precision == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Warmup
    if args.nvtx:
        nvtx.range_push("warmup")
    for _ in range(args.warmup):
        with autocast_ctx(args.precision, device):
            if args.mode == "fwd":
                with nvtx.range("FORWARD") if args.nvtx else contextlib.nullcontext():
                    _ = model(x)
            else:
                with nvtx.range("FORWARD") if args.nvtx else contextlib.nullcontext():
                    logits = model(x)
                with nvtx.range("LOSS") if args.nvtx else contextlib.nullcontext():
                    loss = logits.float().mean()

                if scaler is not None:
                    with nvtx.range("BACKWARD") if args.nvtx else contextlib.nullcontext():
                        scaler.scale(loss).backward()
                    with nvtx.range("OPTIMIZER") if args.nvtx else contextlib.nullcontext():
                        with nvtx.range("OPTIMIZER.step") if args.nvtx else contextlib.nullcontext():
                            scaler.step(optimizer)
                            scaler.update()
                        with nvtx.range("OPTIMIZER.zero_grad") if args.nvtx else contextlib.nullcontext():
                            optimizer.zero_grad(set_to_none=True)
                else:
                    with nvtx.range("BACKWARD") if args.nvtx else contextlib.nullcontext():
                        loss.backward()
                    with nvtx.range("OPTIMIZER") if args.nvtx else contextlib.nullcontext():
                        with nvtx.range("OPTIMIZER.step") if args.nvtx else contextlib.nullcontext():
                            optimizer.step()
                        with nvtx.range("OPTIMIZER.zero_grad") if args.nvtx else contextlib.nullcontext():
                            optimizer.zero_grad(set_to_none=True)
        device_sync_if_needed(device)
    if args.nvtx:
        nvtx.range_pop()

    # ----- Memory profiling setup -----
    recording_mem = False
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        if args.mem_snapshot:
            torch.cuda.memory._record_memory_history(max_entries=1_000_000)
            recording_mem = True

    # Measurement
    iters = []
    toks_per_iter = B * T
    if args.nvtx:
        nvtx.range_push("measure_loop")  # pairs with nsys --capture-range=nvtx
    try:
        for _ in range(args.steps):
            t0 = time.perf_counter()
            with autocast_ctx(args.precision, device):
                if args.mode == "fwd":
                    with nvtx.range("FORWARD") if args.nvtx else contextlib.nullcontext():
                        _ = model(x)
                else:
                    with nvtx.range("FORWARD") if args.nvtx else contextlib.nullcontext():
                        logits = model(x)
                    with nvtx.range("LOSS") if args.nvtx else contextlib.nullcontext():
                        loss = logits.float().mean()

                    if scaler is not None:
                        with nvtx.range("BACKWARD") if args.nvtx else contextlib.nullcontext():
                            scaler.scale(loss).backward()
                        with nvtx.range("OPTIMIZER") if args.nvtx else contextlib.nullcontext():
                            with nvtx.range("OPTIMIZER.step") if args.nvtx else contextlib.nullcontext():
                                scaler.step(optimizer)
                                scaler.update()
                            with nvtx.range("OPTIMIZER.zero_grad") if args.nvtx else contextlib.nullcontext():
                                optimizer.zero_grad(set_to_none=True)
                    else:
                        with nvtx.range("BACKWARD") if args.nvtx else contextlib.nullcontext():
                            loss.backward()
                        with nvtx.range("OPTIMIZER") if args.nvtx else contextlib.nullcontext():
                            with nvtx.range("OPTIMIZER.step") if args.nvtx else contextlib.nullcontext():
                                optimizer.step()
                            with nvtx.range("OPTIMIZER.zero_grad") if args.nvtx else contextlib.nullcontext():
                                optimizer.zero_grad(set_to_none=True)
            device_sync_if_needed(device)
            iters.append(time.perf_counter() - t0)
    finally:
        if args.nvtx:
            nvtx.range_pop()
        if device.startswith("cuda") and recording_mem:
            out = args.mem_snapshot_out
            torch.cuda.memory._dump_snapshot(out)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"[MEM] dumped {out} (open https://pytorch.org/memory_viz)")

    # Stats
    avg = sum(iters) / max(1, len(iters))
    std = (sum((t - avg) ** 2 for t in iters) / max(1, (len(iters) - 1))) ** 0.5
    toks_per_s = toks_per_iter / avg if avg > 0 else float("nan")

    # Basic CUDA peaks
    peak_alloc = peak_reserved = None
    extra_mem_stats = {}
    if device.startswith("cuda"):
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)

        # More detailed CUDA memory stats
        stats = torch.cuda.memory_stats()
        def b2mib(v): return v / (1024**2)
        extra_mem_stats = {
            "active_bytes_peak_MiB": b2mib(stats.get("active_bytes.all.peak", 0)),
            "inactive_split_bytes_peak_MiB": b2mib(stats.get("inactive_split_bytes.all.peak", 0)),
            "allocated_bytes_peak_MiB": b2mib(stats.get("allocated_bytes.all.peak", 0)),
            "reserved_bytes_peak_MiB": b2mib(stats.get("reserved_bytes.all.peak", 0)),
            "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
            "num_ooms": int(stats.get("num_ooms", 0)),
        }

        # Optional memory summary text
        if args.mem_summary:
            try:
                summary_txt = torch.cuda.memory_summary()
                out_path = args.mem_summary_out
                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(out_path, "w") as f:
                    f.write(summary_txt)
                print(f"[MEM] wrote memory_summary to {out_path}")
            except Exception as e:
                print(f"[WARN] failed to write memory_summary: {e}")

    result = {
        "size": args.size,
        "ctx": T,
        "batch": B,
        "mode": args.mode,
        "precision": args.precision,
        "fp32_matmul": args.fp32_matmul,
        "compile": args.compile,
        "avg_ms": avg * 1000,
        "std_ms": std * 1000,
        "tokens_per_s": toks_per_s,
        "peak_mem_alloc_MiB": peak_alloc,
        "peak_mem_reserved_MiB": peak_reserved,
        "steps": args.steps,
        "warmup": args.warmup,
        "used_grad_scaler": bool(scaler is not None),
        # extra memory stats
        **extra_mem_stats,
    }
    print("[RESULT]", json.dumps(result, ensure_ascii=False, indent=2))

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
