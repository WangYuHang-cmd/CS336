#!/usr/bin/env python3
"""
Complete Multi-Implementation GEMM Analysis System
基于现有的 GEMM_TEST.py，提供完整的多实现性能测试和可视化分析

使用方法:
1. python multi_impl_gemm_suite.py sweep --help  # 查看测试选项
2. python multi_impl_gemm_suite.py analyze --help  # 查看分析选项
"""

import argparse
import subprocess
import sys
import csv
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple

class GEMMTestSuite:
    """GEMM测试套件，基于现有的GEMM_TEST.py"""
    
    def __init__(self, gemm_script_path: str = "GEMM_TEST.py"):
        self.gemm_script = Path(gemm_script_path)
        if not self.gemm_script.exists():
            raise FileNotFoundError(f"GEMM_TEST.py not found at {gemm_script_path}")
    
    def run_single_test(self, M: int, K: int, N: int, dtype: str, 
                       impl: str, iters: int = 50, warmup: int = 10,
                       allow_tf32: bool = False, print_mem_summary: bool = False,
                       debug: bool = False) -> Dict[str, Any]:
        """运行单个GEMM测试"""
        cmd = [
            sys.executable, str(self.gemm_script),
            "--M", str(M), "--K", str(K), "--N", str(N),
            "--dtype", dtype, "--impl", impl,
            "--iters", str(iters), "--warmup", str(warmup)
        ]
        
        if allow_tf32:
            cmd.append("--allow-tf32")
        if print_mem_summary:
            cmd.append("--print-mem-summary")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"Error running test: {result.stderr}")
                if debug:
                    print(f"STDOUT: {result.stdout}")
                return None
            
            if debug:
                print(f"Raw output for {impl}:")
                print(result.stdout)
                print("-" * 50)
            
            return self._parse_gemm_output(result.stdout, impl, M, K, N, dtype)
        except subprocess.TimeoutExpired:
            print(f"Test timed out for {impl} M={M} K={K} N={N}")
            return None
        except Exception as e:
            print(f"Error running test: {e}")
            return None
    
    def _parse_gemm_output(self, output: str, impl: str, M: int, K: int, N: int, dtype: str) -> Dict[str, Any]:
        """解析GEMM_TEST.py的输出"""
        lines = output.strip().split('\n')
        
        for line in lines:
            # 处理Split-K的特殊格式：[splitK- 2] 而不是 [splitK-2]
            if impl.startswith('splitK-'):
                splitk_val = impl.split('-')[1]
                # 匹配 [splitK- 2] 或 [splitK-2] 格式
                if line.startswith(f"[splitK- {splitk_val}]") or line.startswith(f"[splitK-{splitk_val}]"):
                    pass  # 找到匹配行
                else:
                    continue
            elif line.startswith(f"[{impl}]"):
                pass  # 找到匹配行
            else:
                continue
            
            # 解析这一行
            try:
                # 使用正则表达式提取数值
                import re
                
                # 提取ms/it的数值
                ms_match = re.search(r'(\d+\.\d+)\s+ms/it', line)
                if not ms_match:
                    print(f"Could not find ms/it in line: {line}")
                    continue
                ms = float(ms_match.group(1))
                
                # 提取TFLOP/s的数值
                tflops_match = re.search(r'(\d+\.\d+)\s+TFLOP/s', line)
                if not tflops_match:
                    print(f"Could not find TFLOP/s in line: {line}")
                    continue
                tflops = float(tflops_match.group(1))
                
                # 提取内存信息
                alloc_match = re.search(r'peak_alloc=(\d+\.\d+)\s+MiB', line)
                reserved_match = re.search(r'peak_reserved=(\d+\.\d+)\s+MiB', line)
                
                alloc_mib = float(alloc_match.group(1)) if alloc_match else None
                reserved_mib = float(reserved_match.group(1)) if reserved_match else None
                
                return {
                    'M': M, 'K': K, 'N': N, 'dtype': dtype,
                    'impl': impl, 'ms': ms, 'tflops': tflops,
                    'peak_alloc_MiB': alloc_mib, 'peak_reserved_MiB': reserved_mib
                }
            except Exception as e:
                print(f"Failed to parse line: {line}, error: {e}")
                continue
        
        print(f"No valid results found for {impl}")
        return None

class GEMMTestRunner:
    """GEMM测试运行器"""
    
    def __init__(self, gemm_suite: GEMMTestSuite):
        self.suite = gemm_suite
        self.results = []
    
    def estimate_memory_gb(self, M: int, K: int, N: int, dtype: str = "fp16") -> float:
        """估算内存使用量（GB）- 为32GB显存优化"""
        element_size = 2 if dtype in ("fp16", "bf16") else 4
        
        # 基础矩阵内存：A: M*K, B: K*N, C: M*N (fp32)
        base_memory = (M * K + K * N) * element_size + M * N * 4
        
        # Split-K会需要额外的partial results存储
        # 估算最大可能的Split-K开销 (最大split=64时)
        max_split_overhead = M * N * 4 * 64  # 最坏情况的partial results
        
        # 总内存 = 基础内存 + Split-K开销 + 50%的安全裕量
        total_bytes = (base_memory + max_split_overhead) * 1.5
        
        return total_bytes / (1024**3)
    
    def generate_test_configurations(self, base_config: Dict, enable_curated: bool = True) -> List[Tuple[str, int, int, int, int]]:
        """生成测试配置"""
        configs = []
        
        # 基础扫描
        M0, K0, N0 = base_config['M0'], base_config['K0'], base_config['N0']
        
        # M维度扫描
        for M in base_config.get('sweep_M', [1024, 2048, 4096, 6144, 8192]):
            configs.append(("sweep_M", M, M, K0, N0))
        
        # K维度扫描  
        for K in base_config.get('sweep_K', [2048, 4096, 8192, 12288, 16384]):
            configs.append(("sweep_K", K, M0, K, N0))
        
        # N维度扫描
        for N in base_config.get('sweep_N', [1024, 2048, 4096, 6144, 8192]):
            configs.append(("sweep_N", N, M0, K0, N))
        
        # 精选场景
        if enable_curated:
            configs.extend(self._get_curated_scenarios(M0, K0, N0))
        
        return configs
    
    def _get_curated_scenarios(self, base_M: int, base_K: int, base_N: int) -> List[Tuple[str, int, int, int, int]]:
        """获取精选测试场景"""
        scenarios = []
        
        # 小矩阵大K场景 (Split-K友好)
        for M in [32, 64, 128]:
            for N in [32, 64, 128]:
                for K in [8192, 16384, 32768, 65536]:
                    scenarios.append(("tinyMN_wideK", K, M, K, N))
        
        # 高瘦矩阵
        for M in [2048, 4096, 8192]:
            for N in [32, 64, 128]:
                for K in [8192, 16384, 32768]:
                    scenarios.append(("tall_skinny", K, M, K, N))
        
        # 短胖矩阵
        for M in [32, 64, 128]:
            for N in [2048, 4096, 8192]:
                for K in [8192, 16384, 32768]:
                    scenarios.append(("short_fat", K, M, K, N))
        
        # 方形矩阵
        for size in [512, 1024, 2048, 4096]:
            for K in [4096, 8192, 16384]:
                scenarios.append(("square", K, size, K, size))
        
        # 原始基准
        scenarios.append(("baseline", base_K, base_M, base_K, base_N))
        
        return scenarios
    
    def run_comprehensive_sweep(self, 
                              implementations: List[str],
                              base_config: Dict,
                              test_params: Dict,
                              memory_limit_gb: float = 24.0,
                              enable_curated: bool = True) -> List[Dict[str, Any]]:
        """运行综合性能扫描"""
        
        configs = self.generate_test_configurations(base_config, enable_curated)
        total_tests = len(configs) * len(implementations)
        
        print(f"总配置数: {len(configs)}")
        print(f"实现数: {len(implementations)}")
        print(f"总测试数: {total_tests}")
        print(f"预估时间: {total_tests * test_params['iters'] * 0.001:.1f} 分钟")
        print()
        
        results = []
        test_count = 0
        
        for scenario, var, M, K, N in configs:
            # 内存安全检查
            if self.estimate_memory_gb(M, K, N, base_config['dtype']) > memory_limit_gb:
                print(f"[SKIP] M={M} K={K} N={N} 超出内存限制 ({memory_limit_gb:.1f}GB)")
                continue
            
            for impl in implementations:
                test_count += 1
                print(f"[{test_count}/{total_tests}] 测试 {impl}: M={M}, K={K}, N={N} ({scenario})")
                
                result = self.suite.run_single_test(
                    M, K, N, base_config['dtype'], impl,
                    test_params['iters'], test_params['warmup'],
                    test_params.get('allow_tf32', False),
                    test_params.get('print_mem_summary', False),
                    debug=(test_count <= 3)  # 前3个测试启用debug
                )
                
                if result:
                    result.update({
                        'scenario': scenario,
                        'var': var
                    })
                    results.append(result)
                    print(f"  ✓ {result['tflops']:.1f} TFLOPS, {result['ms']:.2f} ms, {result['peak_alloc_MiB']:.1f} MiB")
                else:
                    print(f"  ✗ 测试失败")
                
                time.sleep(0.1)  # 短暂暂停避免GPU过热
        
        print(f"\n完成! 成功测试: {len(results)}/{total_tests}")
        return results

class GEMMAnalyzer:
    """GEMM结果分析器"""
    
    def __init__(self, results: List[Dict[str, Any]] = None, csv_path: str = None):
        if results:
            self.data = results
        elif csv_path:
            self.data = self.load_from_csv(csv_path)
        else:
            raise ValueError("必须提供results或csv_path")
        
        self._prepare_data()
    
    def load_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """从CSV文件加载数据"""
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换数值类型
                row.update({
                    'M': int(row['M']), 'K': int(row['K']), 'N': int(row['N']),
                    'var': int(row['var']), 'ms': float(row['ms']), 'tflops': float(row['tflops']),
                    'peak_alloc_MiB': float(row['peak_alloc_MiB']), 
                    'peak_reserved_MiB': float(row['peak_reserved_MiB'])
                })
                data.append(row)
        return data
    
    def _prepare_data(self):
        """准备分析数据"""
        for row in self.data:
            # 计算派生指标
            row['matrix_size'] = row['M'] * row['N'] * row['K']
            row['flops'] = 2.0 * row['matrix_size']
            row['memory_efficiency'] = row['tflops'] / (row['peak_alloc_MiB'] / 1024)  # TFLOPS per GiB
            
            # 矩阵形状分类
            M, N, K = row['M'], row['N'], row['K']
            if M <= 128 and N <= 128:
                shape_cat = 'Tiny'
            elif abs(M - N) / max(M, N) < 0.2:
                shape_cat = 'Square'
            elif M > 2 * N:
                shape_cat = 'Tall'
            elif N > 2 * M:
                shape_cat = 'Wide'
            else:
                shape_cat = 'Moderate'
            row['shape_category'] = shape_cat
    
    def save_to_csv(self, output_path: str):
        """保存结果到CSV"""
        if not self.data:
            return
        
        fieldnames = list(self.data[0].keys())
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data)
    
    def create_implementation_comparison(self, output_dir: Path):
        """创建实现方式对比图表"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        implementations = list(set(d['impl'] for d in self.data))
        impl_colors = {
            'cublas': '#1f77b4', 'triton-1x': '#ff7f0e',
            'splitK-2': '#2ca02c', 'splitK-4': '#d62728', 
            'splitK-8': '#9467bd', 'splitK-16': '#8c564b'
        }
        
        # 1. 总体性能对比
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GEMM Implementation Performance Comparison', fontsize=16, fontweight='bold')
        
        # 性能 vs 矩阵大小
        ax = axes[0, 0]
        for impl in implementations:
            impl_data = [d for d in self.data if d['impl'] == impl]
            if impl_data:
                x = [d['matrix_size'] / 1e9 for d in impl_data]
                y = [d['tflops'] for d in impl_data]
                color = impl_colors.get(impl, f'C{hash(impl) % 10}')
                ax.scatter(x, y, label=impl, alpha=0.7, s=50, color=color)
        
        ax.set_xlabel('Matrix Size (Billion Elements)')
        ax.set_ylabel('Performance (TFLOPS)')
        ax.set_title('Performance vs Matrix Size')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 内存 vs 性能
        ax = axes[0, 1]
        for impl in implementations:
            impl_data = [d for d in self.data if d['impl'] == impl]
            if impl_data:
                x = [d['peak_alloc_MiB'] for d in impl_data]
                y = [d['tflops'] for d in impl_data]
                color = impl_colors.get(impl, f'C{hash(impl) % 10}')
                ax.scatter(x, y, label=impl, alpha=0.7, s=50, color=color)
        
        ax.set_xlabel('Peak Memory (MiB)')
        ax.set_ylabel('Performance (TFLOPS)')
        ax.set_title('Memory vs Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # K维度影响
        ax = axes[0, 2]
        for impl in implementations:
            impl_data = [d for d in self.data if d['impl'] == impl]
            if impl_data:
                x = [d['K'] for d in impl_data]
                y = [d['tflops'] for d in impl_data]
                color = impl_colors.get(impl, f'C{hash(impl) % 10}')
                ax.scatter(x, y, label=impl, alpha=0.7, s=50, color=color)
        
        ax.set_xlabel('K Dimension')
        ax.set_ylabel('Performance (TFLOPS)')
        ax.set_title('K Dimension Impact')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 延迟对比
        ax = axes[1, 0]
        for impl in implementations:
            impl_data = [d for d in self.data if d['impl'] == impl]
            if impl_data:
                x = [d['ms'] for d in impl_data]
                y = [d['tflops'] for d in impl_data]
                color = impl_colors.get(impl, f'C{hash(impl) % 10}')
                ax.scatter(x, y, label=impl, alpha=0.7, s=50, color=color)
        
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Performance (TFLOPS)')
        ax.set_title('Latency vs Performance')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 内存效率
        ax = axes[1, 1]
        for impl in implementations:
            impl_data = [d for d in self.data if d['impl'] == impl]
            if impl_data:
                x = [d['matrix_size'] / 1e9 for d in impl_data]
                y = [d['memory_efficiency'] for d in impl_data]
                color = impl_colors.get(impl, f'C{hash(impl) % 10}')
                ax.scatter(x, y, label=impl, alpha=0.7, s=50, color=color)
        
        ax.set_xlabel('Matrix Size (Billion Elements)')
        ax.set_ylabel('Memory Efficiency (TFLOPS/GiB)')
        ax.set_title('Memory Efficiency')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 性能分布
        ax = axes[1, 2]
        impl_data = {}
        for impl in implementations:
            impl_data[impl] = [d['tflops'] for d in self.data if d['impl'] == impl]
        
        if impl_data:
            bp = ax.boxplot([impl_data[impl] for impl in implementations], 
                           labels=implementations, patch_artist=True)
            ax.set_ylabel('Performance (TFLOPS)')
            ax.set_title('Performance Distribution')
            ax.tick_params(axis='x', rotation=45)
            
            # 着色
            colors = [impl_colors.get(impl, f'C{i}') for i, impl in enumerate(implementations)]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'implementation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_scenario_analysis(self, output_dir: Path):
        """按场景分析各实现表现"""
        scenarios = list(set(d['scenario'] for d in self.data))
        implementations = list(set(d['impl'] for d in self.data))
        
        for scenario in scenarios:
            scenario_data = [d for d in self.data if d['scenario'] == scenario]
            if len(scenario_data) < 2:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f'Scenario Analysis: {scenario}', fontsize=14, fontweight='bold')
            
            impl_colors = {
                'cublas': '#1f77b4', 'triton-1x': '#ff7f0e',
                'splitK-2': '#2ca02c', 'splitK-4': '#d62728', 
                'splitK-8': '#9467bd', 'splitK-16': '#8c564b'
            }
            
            # 性能 vs K
            ax = axes[0, 0]
            for impl in implementations:
                impl_data = [d for d in scenario_data if d['impl'] == impl]
                if impl_data:
                    # 按K排序
                    impl_data.sort(key=lambda x: x['K'])
                    x = [d['K'] for d in impl_data]
                    y = [d['tflops'] for d in impl_data]
                    color = impl_colors.get(impl, f'C{hash(impl) % 10}')
                    ax.plot(x, y, 'o-', label=impl, color=color, markersize=6)
            
            ax.set_xlabel('K Dimension')
            ax.set_ylabel('Performance (TFLOPS)')
            ax.set_title('Performance vs K')
            if len(set(d['K'] for d in scenario_data)) > 1:
                ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 内存使用 vs K
            ax = axes[0, 1]
            for impl in implementations:
                impl_data = [d for d in scenario_data if d['impl'] == impl]
                if impl_data:
                    impl_data.sort(key=lambda x: x['K'])
                    x = [d['K'] for d in impl_data]
                    y = [d['peak_alloc_MiB'] for d in impl_data]
                    color = impl_colors.get(impl, f'C{hash(impl) % 10}')
                    ax.plot(x, y, 'o-', label=impl, color=color, markersize=6)
            
            ax.set_xlabel('K Dimension')
            ax.set_ylabel('Peak Memory (MiB)')
            ax.set_title('Memory Usage vs K')
            if len(set(d['K'] for d in scenario_data)) > 1:
                ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 延迟 vs K
            ax = axes[1, 0]
            for impl in implementations:
                impl_data = [d for d in scenario_data if d['impl'] == impl]
                if impl_data:
                    impl_data.sort(key=lambda x: x['K'])
                    x = [d['K'] for d in impl_data]
                    y = [d['ms'] for d in impl_data]
                    color = impl_colors.get(impl, f'C{hash(impl) % 10}')
                    ax.plot(x, y, 'o-', label=impl, color=color, markersize=6)
            
            ax.set_xlabel('K Dimension')
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Latency vs K')
            if len(set(d['K'] for d in scenario_data)) > 1:
                ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 效率对比
            ax = axes[1, 1]
            for impl in implementations:
                impl_data = [d for d in scenario_data if d['impl'] == impl]
                if impl_data:
                    impl_data.sort(key=lambda x: x['K'])
                    x = [d['K'] for d in impl_data]
                    y = [d['memory_efficiency'] for d in impl_data]
                    color = impl_colors.get(impl, f'C{hash(impl) % 10}')
                    ax.plot(x, y, 'o-', label=impl, color=color, markersize=6)
            
            ax.set_xlabel('K Dimension')
            ax.set_ylabel('Memory Efficiency (TFLOPS/GiB)')
            ax.set_title('Memory Efficiency vs K')
            if len(set(d['K'] for d in scenario_data)) > 1:
                ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            safe_name = scenario.replace('/', '_').replace(' ', '_')
            plt.savefig(output_dir / f'scenario_{safe_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_splitk_analysis(self, output_dir: Path):
        """Split-K专项分析"""
        splitk_data = [d for d in self.data if 'splitK' in d['impl']]
        if not splitk_data:
            print("No Split-K data found")
            return
        
        triton_1x_data = [d for d in self.data if d['impl'] == 'triton-1x']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Split-K Analysis', fontsize=16, fontweight='bold')
        
        # Split-K值提取
        splitk_values = list(set(int(d['impl'].split('-')[1]) for d in splitk_data))
        splitk_values.sort()
        
        # 1. 不同Split-K值的性能对比
        ax = axes[0, 0]
        scenarios = list(set(d['scenario'] for d in splitk_data))[:5]  # 限制场景数
        
        for scenario in scenarios:
            scenario_splitk_data = [d for d in splitk_data if d['scenario'] == scenario]
            if scenario_splitk_data:
                # 按Split-K值分组
                splitk_perf = {}
                for d in scenario_splitk_data:
                    splitk_val = int(d['impl'].split('-')[1])
                    if splitk_val not in splitk_perf:
                        splitk_perf[splitk_val] = []
                    splitk_perf[splitk_val].append(d['tflops'])
                
                x_vals = sorted(splitk_perf.keys())
                y_vals = [np.mean(splitk_perf[k]) for k in x_vals]
                ax.plot(x_vals, y_vals, 'o-', label=scenario, markersize=6)
        
        ax.set_xlabel('Split-K Value')
        ax.set_ylabel('Average Performance (TFLOPS)')
        ax.set_title('Performance vs Split-K Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Split-K vs 单核加速比
        ax = axes[0, 1]
        if triton_1x_data:
            # 建立配置映射
            single_kernel_map = {}
            for d in triton_1x_data:
                key = (d['M'], d['N'], d['K'], d['scenario'])
                single_kernel_map[key] = d['tflops']
            
            # 计算加速比
            speedup_data = {k: [] for k in splitk_values}
            for d in splitk_data:
                key = (d['M'], d['N'], d['K'], d['scenario'])
                if key in single_kernel_map:
                    speedup = d['tflops'] / single_kernel_map[key]
                    splitk_val = int(d['impl'].split('-')[1])
                    speedup_data[splitk_val].append(speedup)
            
            # 绘制箱线图
            valid_vals = [k for k in splitk_values if speedup_data[k]]
            if valid_vals:
                bp = ax.boxplot([speedup_data[k] for k in valid_vals], 
                               labels=valid_vals, patch_artist=True)
                ax.set_xlabel('Split-K Value')
                ax.set_ylabel('Speedup vs Triton-1x')
                ax.set_title('Split-K Speedup Distribution')
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No speedup')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 着色
                colors = plt.cm.viridis(np.linspace(0, 1, len(valid_vals)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
        
        # 3. 内存开销 vs Split-K
        ax = axes[1, 0]
        for scenario in scenarios:
            scenario_splitk_data = [d for d in splitk_data if d['scenario'] == scenario]
            if scenario_splitk_data:
                splitk_mem = {}
                for d in scenario_splitk_data:
                    splitk_val = int(d['impl'].split('-')[1])
                    if splitk_val not in splitk_mem:
                        splitk_mem[splitk_val] = []
                    splitk_mem[splitk_val].append(d['peak_alloc_MiB'])
                
                x_vals = sorted(splitk_mem.keys())
                y_vals = [np.mean(splitk_mem[k]) for k in x_vals]
                ax.plot(x_vals, y_vals, 'o-', label=scenario, markersize=6)
        
        ax.set_xlabel('Split-K Value')
        ax.set_ylabel('Average Memory Usage (MiB)')
        ax.set_title('Memory Usage vs Split-K Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 最优Split-K值分布
        ax = axes[1, 1]
        if triton_1x_data:
            # 按矩阵形状分类找最优Split-K
            shape_optimal = {}
            shapes = list(set(d['shape_category'] for d in splitk_data))
            
            for shape in shapes:
                shape_data = [d for d in splitk_data if d['shape_category'] == shape]
                if shape_data:
                    # 按配置分组，找到每个配置的最佳Split-K
                    config_best = {}
                    for d in shape_data:
                        config_key = (d['M'], d['N'], d['K'])
                        if config_key not in config_best:
                            config_best[config_key] = {'best_perf': 0, 'best_splitk': 0}
                        
                        if d['tflops'] > config_best[config_key]['best_perf']:
                            config_best[config_key]['best_perf'] = d['tflops']
                            config_best[config_key]['best_splitk'] = int(d['impl'].split('-')[1])
                    
                    # 统计最佳Split-K值的分布
                    best_counts = {}
                    for config in config_best.values():
                        splitk_val = config['best_splitk']
                        best_counts[splitk_val] = best_counts.get(splitk_val, 0) + 1
                    
                    shape_optimal[shape] = best_counts
            
            # 创建堆叠柱状图
            if shape_optimal:
                width = 0.8
                x_pos = np.arange(len(shapes))
                bottom = np.zeros(len(shapes))
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(splitk_values)))
                for i, splitk_val in enumerate(splitk_values):
                    heights = []
                    for shape in shapes:
                        count = shape_optimal.get(shape, {}).get(splitk_val, 0)
                        heights.append(count)
                    
                    ax.bar(x_pos, heights, width, label=f'Split-K {splitk_val}', 
                          bottom=bottom, color=colors[i], alpha=0.8)
                    bottom += heights
                
                ax.set_xlabel('Matrix Shape Category')
                ax.set_ylabel('Number of Optimal Configurations')
                ax.set_title('Optimal Split-K Value by Shape')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(shapes, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'splitk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, output_dir: Path):
        """生成详细分析报告"""
        implementations = list(set(d['impl'] for d in self.data))
        scenarios = list(set(d['scenario'] for d in self.data))
        
        with open(output_dir / 'analysis_report.txt', 'w') as f:
            f.write("GEMM Multi-Implementation Performance Analysis Report\n")
            f.write("=" * 70 + "\n\n")
            
            # 测试概况
            f.write(f"测试概况:\n")
            f.write(f"  总配置数: {len(self.data)}\n")
            f.write(f"  实现方式: {', '.join(implementations)}\n")
            f.write(f"  测试场景: {', '.join(scenarios)}\n\n")
            
            # 整体性能统计
            f.write("整体性能统计:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'实现方式':<15} {'测试数':<8} {'平均TFLOPS':<12} {'最大TFLOPS':<12} {'平均内存(MiB)':<15}\n")
            f.write("-" * 70 + "\n")
            
            for impl in implementations:
                impl_data = [d for d in self.data if d['impl'] == impl]
                if impl_data:
                    avg_tflops = np.mean([d['tflops'] for d in impl_data])
                    max_tflops = np.max([d['tflops'] for d in impl_data])
                    avg_mem = np.mean([d['peak_alloc_MiB'] for d in impl_data])
                    count = len(impl_data)
                    
                    f.write(f"{impl:<15} {count:<8} {avg_tflops:<12.2f} {max_tflops:<12.2f} {avg_mem:<15.1f}\n")
            
            # 按场景分析
            f.write(f"\n\n按场景性能分析:\n")
            f.write("-" * 50 + "\n")
            
            for scenario in scenarios:
                f.write(f"\n场景: {scenario}\n")
                f.write("-" * 30 + "\n")
                
                scenario_data = [d for d in self.data if d['scenario'] == scenario]
                if scenario_data:
                    f.write(f"{'实现方式':<15} {'平均TFLOPS':<12} {'最佳配置':<40}\n")
                    f.write("-" * 70 + "\n")
                    
                    for impl in implementations:
                        impl_scenario_data = [d for d in scenario_data if d['impl'] == impl]
                        if impl_scenario_data:
                            avg_tflops = np.mean([d['tflops'] for d in impl_scenario_data])
                            best_config = max(impl_scenario_data, key=lambda x: x['tflops'])
                            best_desc = f"M={best_config['M']}, N={best_config['N']}, K={best_config['K']}"
                            
                            f.write(f"{impl:<15} {avg_tflops:<12.2f} {best_desc:<40}\n")
            
            # Split-K特殊分析
            splitk_data = [d for d in self.data if 'splitK' in d['impl']]
            triton_1x_data = [d for d in self.data if d['impl'] == 'triton-1x']
            
            if splitk_data and triton_1x_data:
                f.write(f"\n\nSplit-K分析:\n")
                f.write("-" * 30 + "\n")
                
                # 计算平均加速比
                single_kernel_map = {}
                for d in triton_1x_data:
                    key = (d['M'], d['N'], d['K'], d['scenario'])
                    single_kernel_map[key] = d['tflops']
                
                splitk_values = list(set(int(d['impl'].split('-')[1]) for d in splitk_data))
                splitk_values.sort()
                
                f.write("相对于Triton-1x的平均加速比:\n")
                f.write("-" * 40 + "\n")
                
                for splitk_val in splitk_values:
                    speedups = []
                    splitk_subset = [d for d in splitk_data if d['impl'] == f'splitK-{splitk_val}']
                    
                    for d in splitk_subset:
                        key = (d['M'], d['N'], d['K'], d['scenario'])
                        if key in single_kernel_map:
                            speedup = d['tflops'] / single_kernel_map[key]
                            speedups.append(speedup)
                    
                    if speedups:
                        avg_speedup = np.mean(speedups)
                        win_rate = sum(1 for s in speedups if s > 1.0) / len(speedups) * 100
                        f.write(f"Split-K {splitk_val}: {avg_speedup:.2f}x (胜率: {win_rate:.1f}%, 样本: {len(speedups)})\n")
            
            # 推荐总结
            f.write(f"\n\n性能推荐:\n")
            f.write("-" * 20 + "\n")
            
            # 找出每种场景的最佳实现
            for scenario in scenarios:
                scenario_data = [d for d in self.data if d['scenario'] == scenario]
                if scenario_data:
                    # 按实现分组，计算平均性能
                    impl_avg_perf = {}
                    for impl in implementations:
                        impl_data = [d for d in scenario_data if d['impl'] == impl]
                        if impl_data:
                            impl_avg_perf[impl] = np.mean([d['tflops'] for d in impl_data])
                    
                    if impl_avg_perf:
                        best_impl = max(impl_avg_perf, key=impl_avg_perf.get)
                        f.write(f"{scenario}: 推荐 {best_impl} (平均 {impl_avg_perf[best_impl]:.1f} TFLOPS)\n")


def main():
    parser = argparse.ArgumentParser(description='Complete GEMM Multi-Implementation Analysis Suite')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Run performance sweep')
    sweep_parser.add_argument('--gemm-script', type=str, default='GEMM_TEST.py',
                             help='Path to GEMM_TEST.py script')
    sweep_parser.add_argument('--implementations', type=str, 
                             default='cublas,triton-1x,splitK-2,splitK-4,splitK-8',
                             help='Comma-separated list of implementations')
    sweep_parser.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    sweep_parser.add_argument('--iters', type=int, default=100, help='Number of iterations')
    sweep_parser.add_argument('--warmup', type=int, default=20, help='Warmup iterations')
    sweep_parser.add_argument('--allow-tf32', action='store_true')
    sweep_parser.add_argument('--print-mem-summary', action='store_true')
    sweep_parser.add_argument('--memory-limit-gb', type=float, default=24.0)
    
    # Base configuration
    sweep_parser.add_argument('--M0', type=int, default=4096, help='Base M dimension')
    sweep_parser.add_argument('--K0', type=int, default=8192, help='Base K dimension') 
    sweep_parser.add_argument('--N0', type=int, default=4096, help='Base N dimension')
    
    # Sweep ranges
    sweep_parser.add_argument('--sweep-M', type=str, default='1024,2048,4096,6144,8192')
    sweep_parser.add_argument('--sweep-K', type=str, default='2048,4096,8192,12288,16384')
    sweep_parser.add_argument('--sweep-N', type=str, default='1024,2048,4096,6144,8192')
    sweep_parser.add_argument('--enable-curated', action='store_true', default=True,
                             help='Enable curated test scenarios')
    
    sweep_parser.add_argument('--output', type=str, default='gemm_results',
                             help='Output directory')
    
    # Analyze command  
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument('--csv', type=str, required=True,
                               help='Path to results CSV file')
    analyze_parser.add_argument('--output', type=str, default='gemm_analysis',
                               help='Output directory for analysis')
    
    args = parser.parse_args()
    
    if args.command == 'sweep':
        # Run performance sweep
        print("Starting GEMM Multi-Implementation Performance Sweep...")
        print(f"Target implementations: {args.implementations}")
        
        # Setup
        suite = GEMMTestSuite(args.gemm_script)
        runner = GEMMTestRunner(suite)
        
        # Configuration
        implementations = [impl.strip() for impl in args.implementations.split(',')]
        
        base_config = {
            'M0': args.M0, 'K0': args.K0, 'N0': args.N0,
            'dtype': args.dtype,
            'sweep_M': [int(x) for x in args.sweep_M.split(',') if x.strip()],
            'sweep_K': [int(x) for x in args.sweep_K.split(',') if x.strip()],
            'sweep_N': [int(x) for x in args.sweep_N.split(',') if x.strip()],
        }
        
        test_params = {
            'iters': args.iters,
            'warmup': args.warmup, 
            'allow_tf32': args.allow_tf32,
            'print_mem_summary': args.print_mem_summary
        }
        
        # Run sweep
        results = runner.run_comprehensive_sweep(
            implementations, base_config, test_params,
            args.memory_limit_gb, args.enable_curated
        )
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analyzer = GEMMAnalyzer(results=results)
        analyzer.save_to_csv(output_dir / 'results.csv')
        
        # Save configuration
        with open(output_dir / 'config.json', 'w') as f:
            config = {
                'implementations': implementations,
                'base_config': base_config,
                'test_params': test_params,
                'memory_limit_gb': args.memory_limit_gb,
                'enable_curated': args.enable_curated
            }
            json.dump(config, f, indent=2)
        
        print(f"\n✓ Sweep completed! Results saved to {output_dir}")
        print(f"  - results.csv: Raw performance data")
        print(f"  - config.json: Test configuration")
        print(f"\nTo analyze results, run:")
        print(f"  python {sys.argv[0]} analyze --csv {output_dir}/results.csv")
        
    elif args.command == 'analyze':
        # Analyze results
        print(f"Analyzing results from {args.csv}...")
        
        analyzer = GEMMAnalyzer(csv_path=args.csv)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Creating implementation comparison plots...")
        analyzer.create_implementation_comparison(output_dir)
        
        print("Creating scenario analysis plots...")  
        analyzer.create_scenario_analysis(output_dir)
        
        print("Creating Split-K analysis...")
        analyzer.create_splitk_analysis(output_dir)
        
        print("Generating summary report...")
        analyzer.generate_summary_report(output_dir)
        
        print(f"\n✓ Analysis completed! Results saved to {output_dir}")
        print(f"Generated files:")
        print(f"  - implementation_comparison.png: 总体实现对比")
        print(f"  - scenario_*.png: 各场景详细分析")  
        print(f"  - splitk_analysis.png: Split-K专项分析")
        print(f"  - analysis_report.txt: 详细分析报告")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    
'''
python multi_impl_gemm_suite.py sweep \
    --implementations cublas,triton-1x,splitK-2,splitK-4,splitK-8 \
    --dtype fp16 \
    --iters 100 \
    --warmup 20 \
    --M0 4096 --K0 8192 --N0 4096 \
    --sweep-M 1024,2048,4096,8192 \
    --sweep-K 4096,8192,16384,32768 \
    --sweep-N 1024,2048,4096,8192 \
    --memory-limit-gb 24.0 \
    --output my_gemm_results
    
python multi_impl_gemm_suite.py analyze --csv my_gemm_results/results.csv
'''
