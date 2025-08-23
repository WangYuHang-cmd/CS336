#!/bin/bash

# =======================================================================
# RTX 5090 优化版完整GEMM性能测试套件
# 32GB显存优化 + 3秒/2测试的性能预估
# =======================================================================

set -e  # 遇到错误立即退出

# RTX 5090 优化配置
IMPLEMENTATIONS="cublas,triton-1x,splitK-2,splitK-4,splitK-8,splitK-16,splitK-32,splitK-64"
DTYPE="fp16"
MEMORY_LIMIT="30.0"  # 保留2GB安全裕量
BASE_ITERS=200       # RTX 5090性能强劲，增加迭代数提高精度
BASE_WARMUP=50       # 增加warmup确保稳定性

# 创建主输出目录
MAIN_OUTPUT_DIR="rtx5090_complete_gemm_analysis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MAIN_OUTPUT_DIR"

echo "======================================================================="
echo "🚀 RTX 5090 完整GEMM性能测试套件"
echo "显卡: RTX 5090 (32GB VRAM)"
echo "性能: ~3秒/2测试"  
echo "时间: $(date)"
echo "输出目录: $MAIN_OUTPUT_DIR"
echo "======================================================================="
echo

# 记录开始时间
START_TIME=$(date +%s)

# 估算函数
estimate_test_time() {
    local num_configs=$1
    local num_impls=$2
    local total_tests=$((num_configs * num_impls))
    local estimated_minutes=$((total_tests * 3 / 2 / 60))  # 3秒/2测试
    echo "预估: ${num_configs}配置 × ${num_impls}实现 = ${total_tests}测试 ≈ ${estimated_minutes}分钟"
}

# =======================================================================
# 测试套件1: 高性能基准测试 (RTX 5090优化)
# =======================================================================
echo "🏆 [1/12] RTX 5090高性能基准测试..."
echo "目标: 充分发挥RTX 5090的计算能力，建立高性能基线"

# 配置数计算: 6M × 6K × 6N = 216 configs
estimate_test_time 216 8

python multi_impl_gemm_suite.py sweep \
    --implementations $IMPLEMENTATIONS \
    --dtype $DTYPE \
    --iters $BASE_ITERS \
    --warmup $BASE_WARMUP \
    --M0 8192 --K0 16384 --N0 8192 \
    --sweep-M 2048,4096,8192,16384,24576,32768 \
    --sweep-K 4096,8192,16384,32768,49152,65536 \
    --sweep-N 2048,4096,8192,16384,24576,32768 \
    --memory-limit-gb $MEMORY_LIMIT \
    --output "$MAIN_OUTPUT_DIR/01_rtx5090_baseline" \
    || echo "❌ RTX 5090基准测试失败，继续下一个测试..."

echo "✅ RTX 5090基准测试完成"
echo

# =======================================================================
# 测试套件2: 极小矩阵 + 巨大K维度 (Split-K黄金场景)
# =======================================================================
echo "🎯 [2/12] 极小矩阵+巨大K维度测试 (Split-K黄金场景)..."
echo "目标: 在RTX 5090上测试Split-K的理论最优场景"

# 配置数: 7M × 6K × 7N = 294 configs  
estimate_test_time 294 8

python multi_impl_gemm_suite.py sweep \
    --implementations $IMPLEMENTATIONS \
    --dtype $DTYPE \
    --iters 300 \
    --warmup 100 \
    --M0 64 --K0 262144 --N0 64 \
    --sweep-M 8,16,32,64,128,256,512 \
    --sweep-K 131072,262144,524288,1048576,1572864,2097152 \
    --sweep-N 8,16,32,64,128,256,512 \
    --memory-limit-gb $MEMORY_LIMIT \
    --output "$MAIN_OUTPUT_DIR/02_tiny_matrix_huge_k" \
    || echo "❌ 极小矩阵测试失败，继续下一个测试..."

echo "✅ 极小矩阵测试完成"
echo

# =======================================================================
# 测试套件3: 超瘦高矩阵测试 (极端GPU利用率不足)
# =======================================================================
echo "📏 [3/12] 超瘦高矩阵测试..."
echo "目标: 测试极端长宽比下Split-K的优势"

# 配置数: 6M × 5K × 6N = 180 configs
estimate_test_time 180 8

python multi_impl_gemm_suite.py sweep \
    --implementations $IMPLEMENTATIONS \
    --dtype $DTYPE \
    --iters $BASE_ITERS \
    --warmup $BASE_WARMUP \
    --M0 32768 --K0 131072 --N0 16 \
    --sweep-M 8192,16384,32768,65536,98304,131072 \
    --sweep-K 65536,131072,262144,524288,1048576 \
    --sweep-N 4,8,16,32,64,128 \
    --memory-limit-gb $MEMORY_LIMIT \
    --output "$MAIN_OUTPUT_DIR/03_ultra_tall_thin" \
    || echo "❌ 超瘦高矩阵测试失败，继续下一个测试..."

echo "✅ 超瘦高矩阵测试完成"
echo

# =======================================================================
# 测试套件4: 超扁宽矩阵测试
# =======================================================================
echo "📐 [4/12] 超扁宽矩阵测试..."
echo "目标: 测试另一种极端长宽比场景"

# 配置数: 6M × 5K × 6N = 180 configs
estimate_test_time 180 8

python multi_impl_gemm_suite.py sweep \
    --implementations $IMPLEMENTATIONS \
    --dtype $DTYPE \
    --iters $BASE_ITERS \
    --warmup $BASE_WARMUP \
    --M0 16 --K0 131072 --N0 32768 \
    --sweep-M 4,8,16,32,64,128 \
    --sweep-K 65536,131072,262144,524288,1048576 \
    --sweep-N 8192,16384,32768,65536,98304,131072 \
    --memory-limit-gb $MEMORY_LIMIT \
    --output "$MAIN_OUTPUT_DIR/04_ultra_short_wide" \
    || echo "❌ 超扁宽矩阵测试失败，继续下一个测试..."

echo "✅ 超扁宽矩阵测试完成"
echo

# =======================================================================
# 测试套件5: 大型矩阵 + 巨大K维度 (内存带宽极限测试)
# =======================================================================
echo "🚀 [5/12] 大型矩阵+巨大K维度测试..."
echo "目标: 测试RTX 5090内存带宽极限下的性能"

# 配置数: 5M × 4K × 5N = 100 configs
estimate_test_time 100 8

python multi_impl_gemm_suite.py sweep \
    --implementations $IMPLEMENTATIONS \
    --dtype $DTYPE \
    --iters 150 \
    --warmup 30 \
    --M0 4096 --K0 524288 --N0 4096 \
    --sweep-M 1024,2048,4096,8192,16384 \
    --sweep-K 262144,524288,1048576,2097152 \
    --sweep-N 1024,2048,4096,8192,16384 \
    --memory-limit-gb $MEMORY_LIMIT \
    --output "$MAIN_OUTPUT_DIR/05_large_matrix_huge_k" \
    || echo "❌ 大型矩阵测试失败，继续下一个测试..."

echo "✅ 大型矩阵测试完成"
echo

# =======================================================================
# 测试套件6: 高性能批处理模拟
# =======================================================================
echo "📦 [6/12] 高性能批处理模拟测试..."
echo "目标: 模拟RTX 5090处理多个并发任务的场景"

# 配置数: 6M × 5K × 6N = 180 configs
estimate_test_time 180 8

python multi_impl_gemm_suite.py sweep \
    --implementations $IMPLEMENTATIONS \
    --dtype $DTYPE \
    --iters $BASE_ITERS \
    --warmup $BASE_WARMUP \
    --M0 1024 --K0 65536 --N0 1024 \
    --sweep-M 256,512,1024,2048,4096,8192 \
    --sweep-K 32768,65536,131072,262144,524288 \
    --sweep-N 256,512,1024,2048,4096,8192 \
    --memory-limit-gb $MEMORY_LIMIT \
    --output "$MAIN_OUTPUT_DIR/06_high_perf_batch" \
    || echo "❌ 高性能批处理测试失败，继续下一个测试..."

echo "✅ 高性能批处理测试完成"
echo

# =======================================================================
# 测试套件7: 大规模Transformer模拟
# =======================================================================
echo "🔤 [7/12] 大规模Transformer模拟测试..."
echo "目标: 模拟GPT/LLaMA等大模型在RTX 5090上的计算"

# 配置数: 5M × 4K × 5N = 100 configs
estimate_test_time 100 8

python multi_impl_gemm_suite.py sweep \
    --implementations $IMPLEMENTATIONS \
    --dtype $DTYPE \
    --iters $BASE_ITERS \
    --warmup $BASE_WARMUP \
    --M0 16384 --K0 262144 --N0 1024 \
    --sweep-M 4096,8192,16384,32768,65536 \
    --sweep-K 131072,262144,524288,1048576 \
    --sweep-N 512,1024,2048,4096,8192 \
    --memory-limit-gb $MEMORY_LIMIT \
    --output "$MAIN_OUTPUT_DIR/07_large_transformer" \
    || echo "❌ 大规模Transformer测试失败，继续下一个测试..."

echo "✅ 大规模Transformer测试完成"
echo

# =======================================================================
# 测试套件8: 极端Split-K压力测试 (包含splitK-128)
# =======================================================================
echo "💥 [8/12] 极端Split-K压力测试..."
echo "目标: 测试RTX 5090上Split-K的极限性能"

# 配置数: 5M × 3K × 5N = 75 configs, 9 impls (包含splitK-128)
estimate_test_time 75 9

python multi_impl_gemm_suite.py sweep \
    --implementations "cublas,triton-1x,splitK-4,splitK-8,splitK-16,splitK-32,splitK-64,splitK-128,splitK-256" \
    --dtype $DTYPE \
    --iters 250 \
    --warmup 50 \
    --M0 256 --K0 1048576 --N0 256 \
    --sweep-M 64,128,256,512,1024 \
    --sweep-K 524288,1048576,2097152 \
    --sweep-N 64,128,256,512,1024 \
    --memory-limit-gb $MEMORY_LIMIT \
    --output "$MAIN_OUTPUT_DIR/08_extreme_splitk" \
    || echo "❌ 极端Split-K测试失败，继续下一个测试..."

echo "✅ 极端Split-K测试完成"
echo

# =======================================================================
# 测试套件9: 微型矩阵超大K (Split-K理论最优)
# =======================================================================
echo "🔬 [9/12] 微型矩阵超大K测试..."
echo "目标: 测试Split-K在绝对理论最优条件下的表现"

# 配置数: 6M × 4K × 6N = 144 configs
estimate_test_time 144 8

python multi_impl_gemm_suite.py sweep \
    --implementations $IMPLEMENTATIONS \
    --dtype $DTYPE \
    --iters 400 \
    --warmup 100 \
    --M0 32 --K0 2097152 --N0 32 \
    --sweep-M 4,8,16,32,64,128 \
    --sweep-K 1048576,2097152,4194304,8388608 \
    --sweep-N 4,8,16,32,64,128 \
    --memory-limit-gb $MEMORY_LIMIT \
    --output "$MAIN_OUTPUT_DIR/09_micro_matrix_mega_k" \
    || echo "❌ 微型矩阵测试失败，继续下一个测试..."

echo "✅ 微型矩阵测试完成"
echo

# =======================================================================
# 测试套件10: 方形矩阵全尺寸测试
# =======================================================================
echo "⬜ [10/12] 方形矩阵全尺寸测试..."
echo "目标: 测试方形矩阵在不同尺寸下的性能特征"

# 配置数: 8M × 6K × 1N = 48 configs (N=M，所以实际是方形)
estimate_test_time 48 8

python multi_impl_gemm_suite.py sweep \
    --implementations $IMPLEMENTATIONS \
    --dtype $DTYPE \
    --iters $BASE_ITERS \
    --warmup $BASE_WARMUP \
    --M0 4096 --K0 32768 --N0 4096 \
    --sweep-M 256,512,1024,2048,4096,8192,16384,32768 \
    --sweep-K 4096,8192,16384,32768,65536,131072 \
    --sweep-N 256,512,1024,2048,4096,8192,16384,32768 \
    --memory-limit-gb $MEMORY_LIMIT \
    --output "$MAIN_OUTPUT_DIR/10_square_matrix_full_range" \
    || echo "❌ 方形矩阵测试失败，继续下一个测试..."

echo "✅ 方形矩阵测试完成"
echo

# =======================================================================
# 测试套件11: 内存效率优化测试
# =======================================================================
echo "🧠 [11/12] 内存效率优化测试..."
echo "目标: 测试不同实现的内存效率和带宽利用率"

# 配置数: 5M × 5K × 5N = 125 configs
estimate_test_time 125 8

python multi_impl_gemm_suite.py sweep \
    --implementations $IMPLEMENTATIONS \
    --dtype $DTYPE \
    --iters $BASE_ITERS \
    --warmup $BASE_WARMUP \
    --M0 2048 --K0 131072 --N0 2048 \
    --sweep-M 512,1024,2048,4096,8192 \
    --sweep-K 65536,131072,262144,524288,1048576 \
    --sweep-N 512,1024,2048,4096,8192 \
    --memory-limit-gb $MEMORY_LIMIT \
    --output "$MAIN_OUTPUT_DIR/11_memory_efficiency" \
    || echo "❌ 内存效率测试失败，继续下一个测试..."

echo "✅ 内存效率测试完成"
echo

# =======================================================================
# 测试套件12: RTX 5090极限性能探索
# =======================================================================
echo "🏁 [12/12] RTX 5090极限性能探索..."
echo "目标: 探索RTX 5090的绝对性能极限"

# 配置数: 4M × 3K × 4N = 48 configs
estimate_test_time 48 8

python multi_impl_gemm_suite.py sweep \
    --implementations $IMPLEMENTATIONS \
    --dtype $DTYPE \
    --iters 100 \
    --warmup 20 \
    --M0 16384 --K0 65536 --N0 16384 \
    --sweep-M 8192,16384,32768,49152 \
    --sweep-K 32768,65536,131072 \
    --sweep-N 8192,16384,32768,49152 \
    --memory-limit-gb $MEMORY_LIMIT \
    --output "$MAIN_OUTPUT_DIR/12_rtx5090_peak_performance" \
    || echo "❌ RTX 5090极限测试失败，继续下一个测试..."

echo "✅ RTX 5090极限测试完成"
echo

# =======================================================================
# 合并所有结果
# =======================================================================
echo "📊 合并所有测试结果..."

# 创建合并结果目录
MERGED_DIR="$MAIN_OUTPUT_DIR/merged_results"
mkdir -p "$MERGED_DIR"

# 合并所有CSV文件
echo "scenario,var,M,K,N,dtype,impl,ms,tflops,peak_alloc_MiB,peak_reserved_MiB,matrix_size,flops,memory_efficiency,shape_category" > "$MERGED_DIR/all_results.csv"

# 找到所有结果文件并合并
total_records=0
for result_dir in "$MAIN_OUTPUT_DIR"/*/; do
    if [ -f "$result_dir/results.csv" ]; then
        echo "合并 $result_dir/results.csv"
        records=$(tail -n +2 "$result_dir/results.csv" | wc -l)
        total_records=$((total_records + records))
        tail -n +2 "$result_dir/results.csv" >> "$MERGED_DIR/all_results.csv" || echo "跳过损坏的文件: $result_dir/results.csv"
    fi
done

echo "✅ 结果合并完成，总计 $total_records 条记录"
echo

# =======================================================================
# 生成分析报告和可视化
# =======================================================================
echo "📈 生成分析报告和可视化..."

# 测试套件信息
test_suites=(
    "01_rtx5090_baseline:RTX 5090基准测试"
    "02_tiny_matrix_huge_k:极小矩阵巨大K"
    "03_ultra_tall_thin:超瘦高矩阵"
    "04_ultra_short_wide:超扁宽矩阵"
    "05_large_matrix_huge_k:大型矩阵巨大K"
    "06_high_perf_batch:高性能批处理"
    "07_large_transformer:大规模Transformer"
    "08_extreme_splitk:极端Split-K"
    "09_micro_matrix_mega_k:微型矩阵超大K"
    "10_square_matrix_full_range:方形矩阵全尺寸"
    "11_memory_efficiency:内存效率优化"
    "12_rtx5090_peak_performance:RTX 5090极限性能"
)

# 为每个测试套件生成单独的分析
for suite_info in "${test_suites[@]}"; do
    IFS=':' read -r suite_dir suite_name <<< "$suite_info"
    
    if [ -f "$MAIN_OUTPUT_DIR/$suite_dir/results.csv" ]; then
        echo "分析 $suite_name..."
        python multi_impl_gemm_suite.py analyze \
            --csv "$MAIN_OUTPUT_DIR/$suite_dir/results.csv" \
            --output "$MAIN_OUTPUT_DIR/analysis_$suite_dir" \
            || echo "❌ $suite_name 分析失败"
    else
        echo "⚠️  未找到 $suite_name 的结果文件"
    fi
done

# 生成总体分析
echo "生成RTX 5090综合分析..."
python multi_impl_gemm_suite.py analyze \
    --csv "$MERGED_DIR/all_results.csv" \
    --output "$MAIN_OUTPUT_DIR/rtx5090_comprehensive_analysis" \
    || echo "❌ 综合分析失败"

echo "✅ 分析报告生成完成"
echo

# =======================================================================
# 生成RTX 5090专用测试总结报告
# =======================================================================
echo "📋 生成RTX 5090专用测试总结报告..."

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# 计算实际测试速度
if [ "$total_records" -gt 0 ]; then
    tests_per_second=$(echo "scale=2; $total_records / $DURATION" | bc -l 2>/dev/null || echo "N/A")
    seconds_per_test=$(echo "scale=2; $DURATION / $total_records" | bc -l 2>/dev/null || echo "N/A")
else
    tests_per_second="N/A"
    seconds_per_test="N/A"
fi

SUMMARY_FILE="$MAIN_OUTPUT_DIR/RTX5090_TEST_SUMMARY.md"

cat > "$SUMMARY_FILE" << EOF
# RTX 5090 完整GEMM性能测试报告 🚀

## 硬件配置
- **显卡**: NVIDIA RTX 5090
- **显存**: 32GB GDDR7
- **测试日期**: $(date -d @$START_TIME)
- **完成时间**: $(date -d @$END_TIME)
- **总耗时**: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒

## 性能统计
- **总测试数**: $total_records 个
- **实际测试速度**: ${tests_per_second} 测试/秒 (${seconds_per_test} 秒/测试)
- **预期速度**: 0.67 测试/秒 (1.5 秒/测试)
- **性能比预期**: $(echo "scale=1; $tests_per_second / 0.67" | bc -l 2>/dev/null || echo "N/A")x

## 测试实现
- **cuBLAS**: NVIDIA优化的高性能GEMM库
- **Triton-1x**: 单kernel Triton实现
- **Split-K**: 2,4,8,16,32,64,128,256分片并行

## 测试套件详解

### 🏆 1. RTX 5090基准测试 (01_rtx5090_baseline)
**目标**: 充分发挥RTX 5090计算能力
**配置**: M,N ∈ [2048, 32768], K ∈ [4096, 65536]
**特点**: 大尺寸矩阵，测试峰值性能

### 🎯 2. 极小矩阵+巨大K (02_tiny_matrix_huge_k) ⭐ 核心
**目标**: Split-K理论最优场景
**配置**: M,N ∈ [8, 512], K ∈ [131072, 2097152]
**特点**: Split-K应该大显身手的场景

### 📏 3. 超瘦高矩阵 (03_ultra_tall_thin)
**目标**: 极端长宽比下的GPU利用率
**配置**: M ∈ [8192, 131072], N ∈ [4, 128], K ∈ [65536, 1048576]

### 📐 4. 超扁宽矩阵 (04_ultra_short_wide)
**目标**: 另一种极端长宽比场景
**配置**: M ∈ [4, 128], N ∈ [8192, 131072], K ∈ [65536, 1048576]

### 🚀 5. 大型矩阵+巨大K (05_large_matrix_huge_k)
**目标**: 内存带宽极限测试
**配置**: M,N ∈ [1024, 16384], K ∈ [262144, 2097152]
**特点**: 测试RTX 5090内存带宽上限

### 📦 6. 高性能批处理 (06_high_perf_batch)
**目标**: 多任务并发场景
**配置**: 中等尺寸矩阵的多种组合

### 🔤 7. 大规模Transformer (07_large_transformer)
**目标**: 模拟GPT/LLaMA计算模式
**配置**: 模拟真实大模型的矩阵形状

### 💥 8. 极端Split-K (08_extreme_splitk) ⭐ 核心
**目标**: Split-K极限性能
**配置**: 包含splitK-256的极端测试
**特点**: 测试Split-K的性能边界

### 🔬 9. 微型矩阵超大K (09_micro_matrix_mega_k) ⭐ 核心
**目标**: Split-K绝对理论最优
**配置**: M,N ∈ [4, 128], K ∈ [1048576, 8388608]
**特点**: K维度达到800万的极端测试

### ⬜ 10. 方形矩阵全尺寸 (10_square_matrix_full_range)
**目标**: 方形矩阵性能特征
**配置**: 从256×256到32768×32768

### 🧠 11. 内存效率优化 (11_memory_efficiency)
**目标**: 内存带宽利用率分析
**配置**: 专门测试内存效率的矩阵组合

### 🏁 12. RTX 5090极限性能 (12_rtx5090_peak_performance)
**目标**: 探索绝对性能上限
**配置**: 大尺寸高性能组合

## 关键分析文件

### 📊 综合分析
- \`rtx5090_comprehensive_analysis/implementation_comparison.png\`: **总体对比图**
- \`rtx5090_comprehensive_analysis/splitk_analysis.png\`: **Split-K专项分析**
- \`rtx5090_comprehensive_analysis/analysis_report.txt\`: **详细报告**

### 🎯 Split-K重点分析
- \`analysis_02_tiny_matrix_huge_k/\`: **Split-K黄金场景分析**
- \`analysis_08_extreme_splitk/\`: **极端Split-K性能**
- \`analysis_09_micro_matrix_mega_k/\`: **微型矩阵超大K分析**

### 🚀 RTX 5090性能分析
- \`analysis_01_rtx5090_baseline/\`: **RTX 5090基础性能**
- \`analysis_12_rtx5090_peak_performance/\`: **RTX 5090极限性能**
- \`analysis_11_memory_efficiency/\`: **内存效率分析**

## 数据文件
- \`merged_results/all_results.csv\`: **完整原始数据** ($total_records 条记录)
- 各测试目录: 分场景原始数据

## 快速查看方式
\`\`\`bash
cd $MAIN_OUTPUT_DIR
./rtx5090_quick_view.sh
\`\`\`

---
**RTX 5090测试完成**: $(date)  
**数据总量**: $total_records 测试记录  
**实际性能**: ${tests_per_second} 测试/秒
EOF

echo "✅ RTX 5090测试总结报告已生成: $SUMMARY_FILE"
echo

# =======================================================================
# 最终输出和RTX 5090专用快速查看脚本
# =======================================================================
echo "======================================================================="
echo "🎉 RTX 5090完整测试套件执行完成！"
echo "======================================================================="
echo
echo "📊 **性能统计**:"
echo "   🚀 显卡: RTX 5090 (32GB)"
echo "   📈 测试数: $total_records 个"
echo "   ⏱️  耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "   🏃 速度: ${tests_per_second} 测试/秒"
echo
echo "📁 **主要结果**:"
echo "   📋 测试总结: $MAIN_OUTPUT_DIR/RTX5090_TEST_SUMMARY.md"
echo "   📊 综合分析: $MAIN_OUTPUT_DIR/rtx5090_comprehensive_analysis/"
echo "   🎯 Split-K优势: $MAIN_OUTPUT_DIR/analysis_02_tiny_matrix_huge_k/"
echo "   💥 极端Split-K: $MAIN_OUTPUT_DIR/analysis_08_extreme_splitk/"  
echo "   🔬 微型矩阵: $MAIN_OUTPUT_DIR/analysis_09_micro_matrix_mega_k/"
echo "   🏁 RTX5090极限: $MAIN_OUTPUT_DIR/analysis_12_rtx5090_peak_performance/"
echo
echo "🔍 **重点查看Split-K优势的场景**:"
echo "   1️⃣  极小矩阵+巨大K: analysis_02_tiny_matrix_huge_k/"
echo "   2️⃣  微型矩阵+超大K: analysis_09_micro_matrix_mega_k/"  
echo "   3️⃣  极端Split-K测试: analysis_08_extreme_splitk/"
echo

# 创建RTX 5090专用快速查看脚本
cat > "$MAIN_OUTPUT_DIR/rtx5090_quick_view.sh" << 'EOF'
#!/bin/bash
echo "🚀 RTX 5090 GEMM测试结果快速查看"
echo "=================================="
echo

# 显示硬件信息
echo "💻 硬件配置:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | while IFS=',' read -r name mem_total mem_used gpu_util; do
        echo "   GPU: $name"
        echo "   显存: ${mem_used}MB / ${mem_total}MB 已用"
        echo "   GPU利用率: ${gpu_util}%"
    done
else
    echo "   GPU: RTX 5090 (nvidia-smi不可用)"
fi
echo

# 显示测试总结
echo "📊 测试总结:"
if [ -f "RTX5090_TEST_SUMMARY.md" ]; then
    grep -A 20 "## 性能统计" RTX5090_TEST_SUMMARY.md
else
    echo "   未找到总结文件"
fi
echo

# 自动打开关键图片
echo "📈 打开关键分析图表..."
image_dirs=(
    "rtx5090_comprehensive_analysis"
    "analysis_02_tiny_matrix_huge_k" 
    "analysis_08_extreme_splitk"
    "analysis_09_micro_matrix_mega_k"
)

for dir in "${image_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "📊 查看 $dir 的分析结果..."
        
        # 尝试不同的图片查看器
        if command -v feh >/dev/null 2>&1; then
            feh "$dir"/*.png 2>/dev/null &
        elif command -v eog >/dev/null 2>&1; then
            eog "$dir"/*.png 2>/dev/null &
        elif command -v xdg-open >/dev/null 2>&1; then
            for img in "$dir"/*.png; do
                [ -f "$img" ] && xdg-open "$img" &
            done
        else
            echo "   请手动查看: $dir/*.png"
        fi
    fi
done

# 显示Split-K分析摘要
echo
echo "🎯 Split-K性能分析摘要:"
echo "------------------------"

# 从综合分析中提取Split-K相关信息
if [ -f "rtx5090_comprehensive_analysis/analysis_report.txt" ]; then
    echo "从综合报告中提取Split-K信息:"
    grep -A 10 -B 2 "Split-K" "rtx5090_comprehensive_analysis/analysis_report.txt" | head -20
else
    echo "未找到综合分析报告"
fi

echo
echo "🔍 详细分析文件位置:"
echo "   📊 综合对比图: rtx5090_comprehensive_analysis/implementation_comparison.png"
echo "   🎯 Split-K分析: rtx5090_comprehensive_analysis/splitk_analysis.png"
echo "   📋 详细报告: rtx5090_comprehensive_analysis/analysis_report.txt"
echo "   🏆 最佳场景: analysis_02_tiny_matrix_huge_k/"
echo "   💥 极限测试: analysis_08_extreme_splitk/"
echo

# 显示性能排行榜
echo "🏆 性能排行榜 (从综合数据):"
if [ -f "merged_results/all_results.csv" ]; then
    echo "正在分析综合数据..."
    python3 -c "
import csv
import sys
from collections import defaultdict

try:
    # 读取数据
    data = []
    with open('merged_results/all_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    if not data:
        print('   无数据可分析')
        sys.exit(0)
        
    # 按实现分组计算平均性能
    impl_perf = defaultdict(list)
    for row in data:
        try:
            tflops = float(row['tflops'])
            impl_perf[row['impl']].append(tflops)
        except:
            continue
    
    # 计算平均值并排序
    avg_perf = {}
    for impl, perfs in impl_perf.items():
        avg_perf[impl] = sum(perfs) / len(perfs) if perfs else 0
    
    # 按性能排序
    sorted_impls = sorted(avg_perf.items(), key=lambda x: x[1], reverse=True)
    
    print('   实现方式         平均性能    测试数')
    print('   ' + '-' * 40)
    for impl, avg in sorted_impls:
        count = len(impl_perf[impl])
        print(f'   {impl:<15} {avg:>8.1f} TFLOPS  {count:>6}')
        
except Exception as e:
    print(f'   数据分析出错: {e}')
" 2>/dev/null || echo "   Python分析失败，请手动查看CSV文件"
else
    echo "   未找到合并数据文件"
fi

echo
echo "🎮 下一步操作建议:"
echo "   1. 查看综合对比图了解整体情况"
echo "   2. 重点分析Split-K优势场景的结果"  
echo "   3. 对比不同Split-K值在各场景下的表现"
echo "   4. 查看RTX 5090的极限性能测试结果"
echo
EOF

chmod +x "$MAIN_OUTPUT_DIR/rtx5090_quick_view.sh"

# 创建性能总结脚本
cat > "$MAIN_OUTPUT_DIR/performance_summary.py" << 'EOF'
#!/usr/bin/env python3
"""
RTX 5090 性能测试结果快速总结脚本
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

def analyze_performance_data():
    """分析性能数据并生成总结"""
    
    csv_file = Path("merged_results/all_results.csv")
    if not csv_file.exists():
        print("❌ 未找到合并的结果文件")
        return
    
    print("🚀 RTX 5090 GEMM性能测试结果分析")
    print("=" * 50)
    
    # 读取数据
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['tflops'] = float(row['tflops'])
                row['ms'] = float(row['ms'])
                row['peak_alloc_MiB'] = float(row['peak_alloc_MiB'])
                row['M'] = int(row['M'])
                row['K'] = int(row['K'])
                row['N'] = int(row['N'])
                data.append(row)
            except ValueError:
                continue
    
    print(f"📊 总数据量: {len(data)} 条测试记录")
    print()
    
    # 按实现统计
    impl_stats = defaultdict(lambda: {'tflops': [], 'memory': [], 'count': 0})
    scenario_stats = defaultdict(lambda: defaultdict(list))
    
    for row in data:
        impl = row['impl']
        scenario = row['scenario']
        
        impl_stats[impl]['tflops'].append(row['tflops'])
        impl_stats[impl]['memory'].append(row['peak_alloc_MiB'])
        impl_stats[impl]['count'] += 1
        
        scenario_stats[scenario][impl].append(row['tflops'])
    
    # 总体性能排行
    print("🏆 总体性能排行榜:")
    print("-" * 60)
    print(f"{'实现':<15} {'平均TFLOPS':<12} {'最大TFLOPS':<12} {'平均内存(MiB)':<15} {'测试数':<8}")
    print("-" * 60)
    
    impl_avg = {}
    for impl, stats in impl_stats.items():
        avg_tflops = sum(stats['tflops']) / len(stats['tflops'])
        max_tflops = max(stats['tflops'])
        avg_memory = sum(stats['memory']) / len(stats['memory'])
        count = stats['count']
        
        impl_avg[impl] = avg_tflops
        print(f"{impl:<15} {avg_tflops:<12.2f} {max_tflops:<12.2f} {avg_memory:<15.1f} {count:<8}")
    
    # 按平均性能排序
    sorted_impls = sorted(impl_avg.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🥇 性能冠军: {sorted_impls[0][0]} ({sorted_impls[0][1]:.1f} TFLOPS)")
    
    # Split-K vs 基准对比
    print(f"\n🎯 Split-K vs 基准对比:")
    print("-" * 40)
    
    if 'triton-1x' in impl_avg:
        baseline = impl_avg['triton-1x']
        print(f"Triton-1x基准: {baseline:.2f} TFLOPS")
        
        splitk_impls = [impl for impl in sorted_impls if impl[0].startswith('splitK-')]
        if splitk_impls:
            print("Split-K加速比:")
            for impl, avg_perf in splitk_impls:
                speedup = avg_perf / baseline
                status = "🚀" if speedup > 1.1 else "⚡" if speedup > 1.0 else "🐌"
                print(f"  {impl:<12}: {speedup:>5.2f}x {status}")
    
    # 场景分析
    print(f"\n📋 各场景最佳实现:")
    print("-" * 50)
    
    for scenario, impls in scenario_stats.items():
        if not impls:
            continue
            
        # 计算各实现在该场景下的平均性能
        scenario_avg = {}
        for impl, perfs in impls.items():
            if perfs:
                scenario_avg[impl] = sum(perfs) / len(perfs)
        
        if scenario_avg:
            best_impl = max(scenario_avg.items(), key=lambda x: x[1])
            print(f"{scenario:<25}: {best_impl[0]:<12} ({best_impl[1]:.1f} TFLOPS)")
    
    # Split-K优势场景分析
    print(f"\n🎯 Split-K理论优势场景表现:")
    print("-" * 60)
    
    splitk_advantage_scenarios = ['tiny_matrix_huge_k', 'micro_matrix_mega_k', 'extreme_splitk']
    
    for scenario_key in splitk_advantage_scenarios:
        matching_scenarios = [s for s in scenario_stats.keys() if scenario_key.replace('_', '') in s.replace('_', '')]
        
        for scenario in matching_scenarios:
            print(f"\n📊 {scenario}:")
            impls = scenario_stats[scenario]
            
            if not impls:
                continue
                
            scenario_results = []
            for impl, perfs in impls.items():
                if perfs:
                    avg_perf = sum(perfs) / len(perfs)
                    scenario_results.append((impl, avg_perf))
            
            scenario_results.sort(key=lambda x: x[1], reverse=True)
            
            for i, (impl, perf) in enumerate(scenario_results[:5]):  # 显示前5名
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                print(f"  {rank_emoji} {impl:<12}: {perf:>7.2f} TFLOPS")
    
    # 寻找Split-K真正占优的场景
    print(f"\n🔍 Split-K真正占优的场景:")
    print("-" * 40)
    
    splitk_wins = []
    for scenario, impls in scenario_stats.items():
        if not impls or 'cublas' not in impls:
            continue
            
        # 找到该场景下的最佳Split-K
        splitk_perfs = {}
        for impl, perfs in impls.items():
            if impl.startswith('splitK-') and perfs:
                splitk_perfs[impl] = sum(perfs) / len(perfs)
        
        if not splitk_perfs:
            continue
            
        best_splitk = max(splitk_perfs.items(), key=lambda x: x[1])
        cublas_avg = sum(impls['cublas']) / len(impls['cublas']) if impls['cublas'] else 0
        
        if best_splitk[1] > cublas_avg:
            speedup = best_splitk[1] / cublas_avg
            splitk_wins.append((scenario, best_splitk[0], speedup, best_splitk[1]))
    
    if splitk_wins:
        splitk_wins.sort(key=lambda x: x[2], reverse=True)  # 按加速比排序
        for scenario, impl, speedup, perf in splitk_wins:
            print(f"  {scenario:<25}: {impl} ({speedup:.2f}x, {perf:.1f} TFLOPS)")
    else:
        print("  ❌ 未发现Split-K明显优于cuBLAS的场景")
    
    # 生成JSON格式的总结
    summary = {
        'total_tests': len(data),
        'implementations': list(impl_avg.keys()),
        'performance_ranking': sorted_impls,
        'splitk_wins': len(splitk_wins),
        'best_overall': sorted_impls[0][0] if sorted_impls else None
    }
    
    with open('performance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ 详细分析完成，结果已保存到 performance_summary.json")

if __name__ == "__main__":
    analyze_performance_data()
EOF

chmod +x "$MAIN_OUTPUT_DIR/performance_summary.py"

echo "✅ RTX 5090专用脚本创建完成"
echo
echo "🎮 **快速使用指南**:"
echo "   📱 快速查看: cd $MAIN_OUTPUT_DIR && ./rtx5090_quick_view.sh"
echo "   📊 性能分析: cd $MAIN_OUTPUT_DIR && python3 performance_summary.py"
echo "   📋 测试总结: cat $MAIN_OUTPUT_DIR/RTX5090_TEST_SUMMARY.md"
echo
echo "💡 **优化建议**:"
if [ "$tests_per_second" != "N/A" ] && [ "$(echo "$tests_per_second > 1.0" | bc -l 2>/dev/null)" == "1" ]; then
    echo "   🚀 RTX 5090性能超出预期！实际速度 ${tests_per_second} 测试/秒"
    echo "   💪 可以考虑增加更多测试场景或更大的矩阵尺寸"
else
    echo "   ⚡ RTX 5090性能符合预期"
fi
echo "   🎯 重点关注Split-K在极小矩阵+超大K场景下的表现"
echo "   📈 对比不同Split-K值找到最优参数"
echo
echo "======================================================================="
echo "🎉 RTX 5090完整测试套件准备就绪！开始执行测试..."
echo "======================================================================="