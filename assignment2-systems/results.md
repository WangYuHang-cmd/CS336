# Nsight Systems Profiler

## Results

- Test on RTX 5090:

| size   | ctx | mode     | precision           |     avg\_ms |    std\_ms |    tokens/s | peak\_alloc (MiB) |
| ------ | --: | -------- | ------------------- | ----------: | ---------: | ----------: | ----------------: |
| small  | 256 | fwd      | fp32                |   **9.117** |      0.026 | **112,312** |             2,359 |
| medium | 512 | fwd\_bwd | fp32                | **215.206** |      0.076 |   **9,516** |            18,911 |
| medium | 512 | fwd\_bwd | **bf16**            | **182.841** |      0.031 |  **11,201** |        **17,534** |
| medium | 512 | fwd\_bwd | fp32（**warm-up=0**） | **233.300** | **57.388** |   **8,778** |            18,911 |


End-to-end benchmark analysis:

The benchmark.py will stat the results of running an attention module, by running the nvtx to get the running results:

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
python scripts/analyze_nsys.py --prefix nsys_train_medium_ctx512_bf16
```


| Section                            | Metric                         |   Total (ms) | Per-step (ms) | Share                            |
| ---------------------------------- | ------------------------------ | -----------: | ------------: | -------------------------------- |
| **Run**                            | Steps / Batch / Ctx            | 30 / 4 / 512 |             – | –                                |
| **NVTX (inclusive)**               | FORWARD                        | **1139.483** |    **37.983** | –                                |
|                                    | BACKWARD                       | **1782.316** |    **59.411** | –                                |
|                                    | OPTIMIZER (inclusive)          |  **272.168** |     **9.072** | –                                |
|                                    | OPTIMIZER (exclusive)\*        |    **0.207** |     **0.007** | –                                |
| **Attention (forward-only, NVTX)** | MHA (total)                    |  **780.154** |    **26.005** | **68.5% of FORWARD**             |
|                                    | QKV projections                |      220.429 |         7.348 | 28.25% of MHA                    |
|                                    | output proj                    |       59.793 |         1.993 | 7.66% of MHA                     |
|                                    | RoPE(Q)                        |      142.246 |         4.742 | 18.23% of MHA                    |
|                                    | RoPE(K)                        |      126.224 |         4.207 | 16.18% of MHA                    |
|                                    | attn\_logits (QK^T) + mask     |      109.037 |         3.635 | 13.98% of MHA                    |
|                                    | attn\*V (context)              |   **89.342** |     **2.978** | 11.45% of MHA                    |
|                                    | softmax                        |       22.079 |         0.736 | 2.83% of MHA                     |
|                                    | **MHA linear (QKV+out\_proj)** |  **280.222** |             – | **35.9% of MHA**                 |
|                                    | **MHA residual (misc.)**       |   **11.004** |     **0.367** | **1.41% of MHA**                 |
|                                    | **Non-attention FORWARD**      |  **359.329** |    **11.978** | **31.5% of FORWARD**             |
| **GPU kernels (total)**            | Total GPU kernel time          |   **2712.7** |    **90.424** | –                                |
|                                    | matmul                         | **1004.806** |             – | **37.04% of GPU**                |
|                                    | elementwise                    |      963.182 |             – | 35.51% of GPU                    |
|                                    | optimizer/multi\_tensor        |      602.932 |             – | 22.23% of GPU                    |
|                                    | softmax (bwd)                  |       72.982 |             – | 2.69% of GPU                     |
|                                    | softmax (fwd)                  |       31.417 |             – | 1.16% of GPU                     |
|                                    | reduction                      |       35.158 |             – | 1.30% of GPU                     |
|                                    | other                          |        2.243 |             – | 0.08% of GPU                     |
| **Top kernel**                     | `Kernel2`                      | **1004.806** |             – | **37.04% of GPU** (19,530 calls) |





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

https://docs.pytorch.org/memory_viz

![alt text](image.png)



### Problem (pytorch_attention)

`uv run --no-project --python /home/henry/miniconda3/envs/llm/bin/python --   python benchmark_attention.py --warmup 50 --iters 200 --out ../mem_profiles/attention_bench.json`


| d\_model | seq\_len |   fwd\_ms |   bwd\_ms | mem\_before\_bwd\_MiB | status  |
| -------: | -------: | --------: | --------: | --------------------: | :------ |
|       32 |       64 |     7.730 |    40.655 |                   8.8 | OK      |
|       32 |      128 |     7.779 |    39.064 |                  17.8 | OK      |
|       32 |      256 |     8.545 |    43.571 |                  20.2 | OK      |
|       32 |     1024 |    21.786 |    67.080 |                  56.2 | OK      |
|       32 |     4096 |   499.066 |  1422.735 |                 560.2 | OK      |
|       32 |     8192 |  1986.323 |  5576.747 |                2128.2 | OK      |
|       32 |    16384 |  7868.635 |         – |                8336.2 | **OOM** |
|       64 |       64 |     7.829 |    38.369 |                  17.4 | OK      |
|       64 |      128 |     7.768 |    39.185 |                  18.8 | OK      |
|       64 |      256 |     8.536 |    40.298 |                  22.2 | OK      |
|       64 |     1024 |    24.012 |    73.223 |                  64.2 | OK      |
|       64 |     4096 |   555.080 |  1588.352 |                 592.2 | OK      |
|       64 |     8192 |  2219.932 |  6236.688 |                2192.2 | OK      |
|       64 |    16384 |  8705.040 |         – |                8464.2 | **OOM** |
|      128 |       64 |     7.774 |    39.148 |                  18.4 | OK      |
|      128 |      128 |     7.864 |    39.764 |                  20.8 | OK      |
|      128 |      256 |     8.544 |    40.366 |                  26.2 | OK      |
|      128 |     1024 |    38.299 |   123.351 |                  80.2 | OK      |
|      128 |     4096 |   835.214 |  2363.091 |                 656.2 | OK      |
|      128 |     8192 |  3339.936 |  9388.587 |                2320.2 | OK      |
|      128 |    16384 | 13403.818 |         – |                8720.2 | **OOM** |
|      256 |       64 |     7.870 |    39.730 |                  20.4 | OK      |
|      256 |      128 |     7.937 |    39.676 |                  24.8 | OK      |
|      256 |      256 |     8.905 |    40.249 |                  34.2 | OK      |
|      256 |     1024 |    67.905 |   221.925 |                 112.2 | OK      |
|      256 |     4096 |  1333.069 |  3751.283 |                 784.2 | OK      |
|      256 |     8192 |  5320.389 | 14889.777 |                2576.2 | OK      |
|      256 |    16384 | 21380.786 |         – |                9232.2 | **OOM** |
|      512 |       64 |     9.407 |    39.516 |                  24.4 | OK      |
|      512 |      128 |     9.978 |    40.317 |                  32.8 | OK      |
|      512 |      256 |    12.576 |    44.317 |                  50.2 | OK      |
|      512 |     1024 |   134.736 |   404.050 |                 176.2 | OK      |
|      512 |     4096 |  2312.945 |  6476.550 |                1040.2 | OK      |
|      512 |     8192 |  9246.937 | 25590.666 |                3088.2 | OK      |
|      512 |    16384 | 37243.737 |         – |               10256.2 | **OOM** |



with torch.compile and TF 32:

| d\_model | seq\_len | fwd\_ms (No TF32) | fwd\_ms (TF32) | Δ fwd      | bwd\_ms (No TF32) | bwd\_ms (TF32) | Δ bwd      |
| -------- | -------- | ----------------- | -------------- | ---------- | ----------------- | -------------- | ---------- |
| 32       | 64       | 11.154            | 11.482         | ↑0.3%      | 57.279            | 57.955         | ↑1.2%      |
| 32       | 128      | 11.404            | 11.356         | ↓0.4%      | 53.135            | 51.516         | ↓3.0%      |
| 32       | 256      | 11.449            | 11.271         | ↓1.6%      | 53.098            | 52.571         | ↓1.0%      |
| 32       | 1024     | 23.117            | 17.230         | **↓25.5%** | 76.556            | 58.628         | **↓23.4%** |
| 32       | 4096     | 411.509           | 324.396        | **↓21.2%** | 1114.305          | 908.258        | ↓18.5%     |
| 32       | 8192     | 1868.241          | 1559.212       | ↓16.5%     | 4718.115          | 3989.070       | ↓15.4%     |
| 32       | 16384    | 5816.936          | 4507.615       | ↓22.5%     | 15369.401         | 12071.501      | ↓21.5%     |
| 64       | 64       | 11.426            | 11.709         | ↑2.5%      | 51.993            | 58.528         | ↑12.6%     |
| 64       | 128      | 11.590            | 11.603         | ≈0         | 61.219            | 53.442         | ↓12.7%     |
| 64       | 256      | 13.645            | 12.664         | ↓7.2%      | 58.017            | 54.102         | ↓6.7%      |
| 64       | 1024     | 32.811            | 26.550         | ↓19.1%     | 83.606            | 66.943         | ↓19.9%     |
| 64       | 4096     | 423.220           | 288.840        | **↓31.7%** | 1139.737          | 790.209        | ↓30.7%     |
| 64       | 8192     | 1685.137          | 1134.789       | ↓32.7%     | 4508.481          | 3077.354       | ↓31.8%     |
| 64       | 16384    | 6773.545          | 4733.361       | ↓30.1%     | 17968.918         | 12362.552      | ↓31.2%     |
| 128      | 64       | 11.726            | 11.691         | ≈0         | 53.587            | 52.403         | ↓2.2%      |
| 128      | 128      | 11.841            | 11.442         | ↓3.4%      | 54.697            | 52.976         | ↓3.1%      |
| 128      | 256      | 13.774            | 12.512         | ↓9.2%      | 54.719            | 53.711         | ↓1.8%      |
| 128      | 1024     | 46.283            | 35.650         | ↓23.0%     | 130.749           | 98.651         | ↓24.5%     |
| 128      | 4096     | 706.171           | 360.357        | **↓49.0%** | 1925.811          | 1002.524       | ↓48.0%     |
| 128      | 8192     | 2815.868          | 1462.557       | ↓48.0%     | 7615.674          | 3951.136       | ↓48.1%     |
| 128      | 16384    | 11338.673         | 5981.148       | ↓47.2%     | 30600.758         | 16107.158      | ↓47.4%     |
| 256      | 64       | 12.394            | 11.623         | ↓6.2%      | 61.246            | 53.436         | ↓12.7%     |
| 256      | 128      | 12.566            | 11.591         | ↓7.8%      | 60.587            | 53.153         | ↓12.3%     |
| 256      | 256      | 14.981            | 12.548         | ↓16.2%     | 60.107            | 53.699         | ↓10.7%     |
| 256      | 1024     | 74.493            | 40.466         | **↓45.7%** | 228.783           | 111.487        | ↓51.3%     |
| 256      | 4096     | 1214.021          | 490.728        | ↓59.6%     | 3356.252          | 1398.923       | ↓58.3%     |
| 256      | 8192     | 4820.486          | 1917.672       | ↓60.2%     | 13304.750         | 5357.126       | ↓59.7%     |
| 256      | 16384    | 19420.524         | 7606.391       | ↓60.8%     | 52846.051         | 20982.350      | ↓60.3%     |
| 512      | 64       | 14.730            | 11.655         | ↓20.9%     | 54.081            | 54.813         | ≈0         |
| 512      | 128      | 14.801            | 11.523         | ↓22.1%     | 53.955            | 61.749         | ↑14.4%     |
| 512      | 256      | 18.696            | 15.245         | ↓18.5%     | 62.015            | 57.921         | ↓6.6%      |
| 512      | 1024     | 135.862           | 60.335         | ↓55.6%     | 407.043           | 179.551        | ↓55.9%     |
| 512      | 4096     | 2230.277          | 772.080        | ↓65.4%     | 6151.606          | 2255.915       | ↓63.3%     |
| 512      | 8192     | 8838.995          | 2970.841       | ↓66.4%     | 24000.653         | 8476.593       | ↓64.7%     |
| 512      | 16384    | 35320.222         | 11968.748      | ↓66.1%     | 96698.815         | 33706.693      | ↓65.1%     |


