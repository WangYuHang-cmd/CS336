# Nsight Systems Profiler

## Results

- Test on RTX 5090:

| size   | ctx | mode     | precision           |     avg\_ms |    std\_ms |    tokens/s | peak\_alloc (MiB) |
| ------ | --: | -------- | ------------------- | ----------: | ---------: | ----------: | ----------------: |
| small  | 256 | fwd      | fp32                |   **9.117** |      0.026 | **112,312** |             2,359 |
| medium | 512 | fwd\_bwd | fp32                | **215.206** |      0.076 |   **9,516** |            18,911 |
| medium | 512 | fwd\_bwd | **bf16**            | **182.841** |      0.031 |  **11,201** |        **17,534** |
| medium | 512 | fwd\_bwd | fp32（**warm-up=0**） | **233.300** | **57.388** |   **8,778** |            18,911 |


End-to-end analysis:

First, run the nvtx to get the running results:

```bash
# pure 

OUT=nsys_train_medium_ctx512_bf16
PYTHONPATH="$PWD/cs336-basics:$PYTHONPATH" \
uv run --no-project --python /home/henry/miniconda3/envs/llm/bin/python -- \
  nsys profile \
    --trace=cuda,nvtx,cublas,cudnn \
    --pytorch=autograd-shapes-nvtx \
    --capture-range=nvtx --nvtx-capture=measure_loop --capture-range-end=stop \
    -o "$OUT" --force-overwrite=true \
    python cs336_systems/benchmark.py \
      --size medium --ctx 512 --batch 4 \
      --mode fwd_bwd --precision bf16 \
      --warmup 5 --steps 30 \
      --nvtx --annotate-mha

nsys stats \
  --force-export=true \
  --report cuda_api_sum \
  --report cuda_gpu_kern_sum:base \
  --report cuda_gpu_mem_time_sum \
  --report nvtx_pushpop_sum \
  --format csv,csv,csv,csv \
  --output @"tee ${OUT}_cuda_api_sum.csv",@"tee ${OUT}_cuda_gpu_kern_sum_base.csv",@"tee ${OUT}_cuda_gpu_mem_time_sum.csv","${OUT}_nvtx_pushpop_sum.csv" \
  "$OUT.nsys-rep"
```


Then, run the scripts/analyze_nsys.py to analyze the results:

```python
python scripts/analyze_nsys_fit_yours.py --prefix nsys_train_medium_ctx512_bf16
```

```
== Prefix: nsys_train_medium_ctx512_bf16 ==
[a] NVTX time (ms): FORWARD=1139.483, BACKWARD=1782.316, OPTIMIZER=544.128
[b] Top CUDA kernel: name='Kernel2', calls=19530, time_ms=1002.105
[c] Non-matmul categories (time_ms, share%):
     - other                1567.143 ms  (57.85%)
     - softmax               104.440 ms  ( 3.86%)
     - memory/reduction       35.127 ms  ( 1.30%)
[d] Matmul share: 36.99%  (out of total GPU kernel time 2708.8 ms)
[e] Attention (from NVTX ranges, ms):
     - MHA(total)                      780.154 ms
     - attn_logits (QK^T) + mask       109.037 ms
     - softmax                          22.079 ms
     - attn*V (context)                 89.342 ms

```


Next we will compare the speed of the FP32, BF16 and FP16


- FP 32

```sh
uv run --no-project --python "$PYBIN" -- \
  python cs336_systems/benchmark.py \
    --size $SIZE --ctx $CTX --batch $BATCH \
    --mode fwd_bwd --precision fp32 \
    --warmup $WARMUP --steps $STEPS \
    --out-json fp_comparision/metrics_${SIZE}_ctx${CTX}_fp32.json

# nsys 

OUT=fp_comparision/nsys_${SIZE}_ctx${CTX}_fp32
uv run --no-project --python "$PYBIN" -- \
  nsys profile \
    --trace=cuda,nvtx,cublas,cudnn \
    --pytorch=autograd-shapes-nvtx \
    --capture-range=nvtx --nvtx-capture=measure_loop --capture-range-end=stop \
    -o "$OUT" --force-overwrite=true \
    python cs336_systems/benchmark.py \
      --size $SIZE --ctx $CTX --batch $BATCH \
      --mode fwd_bwd --precision fp32 \
      --warmup $WARMUP --steps $STEPS \
      --nvtx --annotate-mha \
      --out-json fp_comparision/metrics_${SIZE}_ctx${CTX}_fp32_profiled.json

nsys stats \
  --force-export=true \
  --report cuda_api_sum \
  --report cuda_gpu_kern_sum:base \
  --report cuda_gpu_mem_time_sum \
  --report nvtx_pushpop_sum \
  --format csv,csv,csv,csv \
  --output @"tee ${OUT}_cuda_api_sum.csv",@"tee ${OUT}_cuda_gpu_kern_sum_base.csv",@"tee ${OUT}_cuda_gpu_mem_time_sum.csv","${OUT}_nvtx_pushpop_sum.csv" \
  "$OUT.nsys-rep"

```

- BF 16

```sh
uv run --no-project --python "$PYBIN" -- \
  python cs336_systems/benchmark.py \
    --size $SIZE --ctx $CTX --batch $BATCH \
    --mode fwd_bwd --precision bf16 \
    --warmup $WARMUP --steps $STEPS \
    --out-json fp_comparision/metrics_${SIZE}_ctx${CTX}_bf16.json

OUT=fp_comparision/nsys_${SIZE}_ctx${CTX}_bf16
uv run --no-project --python "$PYBIN" -- \
  nsys profile \
    --trace=cuda,nvtx,cublas,cudnn \
    --pytorch=autograd-shapes-nvtx \
    --capture-range=nvtx --nvtx-capture=measure_loop --capture-range-end=stop \
    -o "$OUT" --force-overwrite=true \
    python cs336_systems/benchmark.py \
      --size $SIZE --ctx $CTX --batch $BATCH \
      --mode fwd_bwd --precision bf16 \
      --warmup $WARMUP --steps $STEPS \
      --nvtx --annotate-mha \
      --out-json fp_comparision/metrics_${SIZE}_ctx${CTX}_bf16_profiled.json

nsys stats \
  --force-export=true \
  --report cuda_api_sum \
  --report cuda_gpu_kern_sum:base \
  --report cuda_gpu_mem_time_sum \
  --report nvtx_pushpop_sum \
  --format csv,csv,csv,csv \
  --output @"tee ${OUT}_cuda_api_sum.csv",@"tee ${OUT}_cuda_gpu_kern_sum_base.csv",@"tee ${OUT}_cuda_gpu_mem_time_sum.csv","${OUT}_nvtx_pushpop_sum.csv" \
  "$OUT.nsys-rep"
```


- FP 16

```sh
uv run --no-project --python "$PYBIN" -- \
  python cs336_systems/benchmark.py \
    --size $SIZE --ctx $CTX --batch $BATCH \
    --mode fwd_bwd --precision fp16 \
    --warmup $WARMUP --steps $STEPS \
    --out-json fp_comparision/metrics_${SIZE}_ctx${CTX}_fp16.json

OUT=fp_comparision/nsys_${SIZE}_ctx${CTX}_fp16
uv run --no-project --python "$PYBIN" -- \
  nsys profile \
    --trace=cuda,nvtx,cublas,cudnn \
    --pytorch=autograd-shapes-nvtx \
    --capture-range=nvtx --nvtx-capture=measure_loop --capture-range-end=stop \
    -o "$OUT" --force-overwrite=true \
    python cs336_systems/benchmark.py \
      --size $SIZE --ctx $CTX --batch $BATCH \
      --mode fwd_bwd --precision fp16 \
      --warmup $WARMUP --steps $STEPS \
      --nvtx --annotate-mha \
      --out-json fp_comparision/metrics_${SIZE}_ctx${CTX}_fp16_profiled.json

nsys stats \
  --force-export=true \
  --report cuda_api_sum \
  --report cuda_gpu_kern_sum:base \
  --report cuda_gpu_mem_time_sum \
  --report nvtx_pushpop_sum \
  --format csv,csv,csv,csv \
  --output @"tee ${OUT}_cuda_api_sum.csv",@"tee ${OUT}_cuda_gpu_kern_sum_base.csv",@"tee ${OUT}_cuda_gpu_mem_time_sum.csv","${OUT}_nvtx_pushpop_sum.csv" \
  "$OUT.nsys-rep"

```


| Precision | Avg Time per Step (ms) | Tokens / s | Peak Mem Alloc (MiB) | Peak Mem Reserved (MiB) |
| --------- | ---------------------- | ---------- | -------------------- | ----------------------- |
| FP32      | 212.11                 | 9,655.32   | 18,910.47            | 19,400.00               |
| BF16      | 180.06                 | 11,374.00  | 17,534.59            | 17,820.00               |
| FP16      | 181.91                 | 11,258.06  | 17,534.59            | 17,820.00               |


## mixed_precision_accumulation

```python
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)
```




## benchmarking_mixed_precision

Autocast will keep the origin precision of the data. But for operation like layernorm or softmax, it will keep fp 32 and for matmul or conv it will transfer it to low precision.







# Profiling Memory



| ctx | avg\_ms | tokens/s | peak\_mem\_alloc\_MiB | peak\_mem\_reserved\_MiB |
| --: | ------: | -------: | --------------------: | -----------------------: |
|  64 |   26.94 |    9,504 |             15,910.84 |                15,976.00 |
| 128 |   46.99 |   10,895 |             19,310.64 |                19,562.00 |
| 256 |   98.05 |   10,444 |             27,962.59 |                28,288.00 |


