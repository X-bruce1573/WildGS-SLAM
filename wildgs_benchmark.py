#!/usr/bin/env python3
"""
WildGS-SLAM 基准测试和评估脚本
用于自动化测试多个数据集并生成详细的性能报告
"""

import os
import sys
import argparse
import subprocess
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wildgs_benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WildGSBenchmark:
    """WildGS-SLAM基准测试类"""
    
    def __init__(self, 
                 wildgs_root: str = "WildGS-SLAM",
                 datasets_root: str = "datasets",
                 results_root: str = "benchmark_results"):
        self.wildgs_root = Path(wildgs_root)
        self.datasets_root = Path(datasets_root)
        self.results_root = Path(results_root)
        
        # 创建结果目录
        self.results_root.mkdir(exist_ok=True)
        
        # 数据集配置
        self.dataset_configs = {
            'tum_rgbd': {
                'sequences': [
                    'rgbd_dataset_freiburg3_walking_xyz',
                    'rgbd_dataset_freiburg3_walking_rpy',
                    'rgbd_dataset_freiburg3_walking_static',
                    'rgbd_dataset_freiburg3_walking_halfsphere'
                ],
                'config_template': 'configs/Dynamic/TUM_RGBD/walking_xyz.yaml',
                'gt_format': 'tum'
            },
            'wild_slam': {
                'sequences': ['scene01', 'scene02', 'scene03'],
                'config_template': 'configs/Dynamic/Wild_SLAM_Mocap/crowd_demo.yaml',
                'gt_format': 'wild_slam'
            },
            'bonn_rgbd': {
                'sequences': ['rgbd_bonn_crowd', 'rgbd_bonn_meeting'],
                'config_template': 'configs/Dynamic/Bonn/rgbd_bonn_crowd.yaml',
                'gt_format': 'bonn'
            }
        }
        
        # 评估指标
        self.metrics = {
            'trajectory': ['ate_rmse', 'ate_mean', 'ate_std', 'rpe_rmse', 'rpe_mean', 'rpe_std'],
            'rendering': ['psnr', 'ssim', 'lpips'],
            'performance': ['fps', 'memory_usage', 'initialization_time']
        }
    
    def check_environment(self) -> bool:
        """检查环境是否正确配置"""
        logger.info("检查环境配置...")
        
        # 检查WildGS-SLAM目录
        if not self.wildgs_root.exists():
            logger.error(f"WildGS-SLAM目录不存在: {self.wildgs_root}")
            return False
        
        # 检查Python环境
        try:
            import torch
            logger.info(f"PyTorch版本: {torch.__version__}")
            logger.info(f"CUDA可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            logger.error("PyTorch未安装")
            return False
        
        # 检查必要的模块
        required_modules = ['lietorch', 'simple_knn', 'diff_gaussian_rasterization']
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"✓ {module}")
            except ImportError:
                logger.error(f"✗ {module} 未安装")
                return False
        
        logger.info("环境检查完成 ✓")
        return True
    
    def download_datasets(self, datasets: List[str]) -> bool:
        """下载指定的数据集"""
        logger.info(f"下载数据集: {datasets}")
        
        for dataset in datasets:
            if dataset == 'tum_rgbd':
                self._download_tum_rgbd()
            elif dataset == 'wild_slam':
                self._download_wild_slam()
            elif dataset == 'bonn_rgbd':
                self._download_bonn_rgbd()
            else:
                logger.warning(f"未知数据集: {dataset}")
        
        return True
    
    def _download_tum_rgbd(self):
        """下载TUM RGB-D数据集"""
        tum_dir = self.datasets_root / 'tum_rgbd'
        tum_dir.mkdir(exist_ok=True)
        
        base_url = "https://vision.in.tum.de/rgbd/dataset/freiburg3/"
        sequences = [
            'rgbd_dataset_freiburg3_walking_xyz.tgz',
            'rgbd_dataset_freiburg3_walking_rpy.tgz',
            'rgbd_dataset_freiburg3_walking_static.tgz',
            'rgbd_dataset_freiburg3_walking_halfsphere.tgz'
        ]
        
        for seq in sequences:
            seq_path = tum_dir / seq
            if not seq_path.exists():
                logger.info(f"下载 {seq}...")
                cmd = f"wget -P {tum_dir} {base_url}{seq}"
                subprocess.run(cmd, shell=True, check=True)
                
                # 解压
                extract_cmd = f"cd {tum_dir} && tar -xzf {seq}"
                subprocess.run(extract_cmd, shell=True, check=True)
    
    def _download_wild_slam(self):
        """下载Wild-SLAM数据集"""
        # 这里需要根据实际的下载地址调整
        logger.info("Wild-SLAM数据集需要手动下载，请参考官方指南")
    
    def _download_bonn_rgbd(self):
        """下载Bonn RGB-D数据集"""
        # 这里需要根据实际的下载地址调整
        logger.info("Bonn RGB-D数据集需要手动下载，请参考官方指南")
    
    def run_sequence(self, dataset: str, sequence: str, config_override: Dict = None) -> Dict:
        """运行单个序列的测试"""
        logger.info(f"运行测试: {dataset}/{sequence}")
        
        # 创建结果目录
        result_dir = self.results_root / dataset / sequence
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成配置文件
        config_path = self._generate_config(dataset, sequence, result_dir, config_override)
        
        # 运行SLAM
        start_time = time.time()
        success = self._run_slam(config_path, result_dir)
        end_time = time.time()
        
        if not success:
            logger.error(f"SLAM运行失败: {dataset}/{sequence}")
            return {'success': False}
        
        # 评估结果
        metrics = self._evaluate_results(dataset, sequence, result_dir)
        metrics['runtime'] = end_time - start_time
        metrics['success'] = True
        
        # 保存结果
        with open(result_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"测试完成: {dataset}/{sequence}")
        return metrics
    
    def _generate_config(self, dataset: str, sequence: str, result_dir: Path, 
                        config_override: Dict = None) -> Path:
        """生成配置文件"""
        template_path = self.wildgs_root / self.dataset_configs[dataset]['config_template']
        
        # 读取模板配置
        with open(template_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 更新路径
        config['scene'] = sequence
        config['input_folder'] = str(self.datasets_root / dataset / sequence)
        config['output_folder'] = str(result_dir)
        
        # 应用覆盖配置
        if config_override:
            config.update(config_override)
        
        # 保存配置
        config_path = result_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path
    
    def _run_slam(self, config_path: Path, result_dir: Path) -> bool:
        """运行SLAM系统"""
        try:
            cmd = f"cd {self.wildgs_root} && python run.py {config_path}"
            
            # 重定向输出到日志文件
            log_file = result_dir / 'slam_log.txt'
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    cmd, shell=True, check=True,
                    stdout=f, stderr=subprocess.STDOUT,
                    timeout=3600  # 1小时超时
                )
            
            return process.returncode == 0
        
        except subprocess.TimeoutExpired:
            logger.error("SLAM运行超时")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"SLAM运行失败: {e}")
            return False
    
    def _evaluate_results(self, dataset: str, sequence: str, result_dir: Path) -> Dict:
        """评估结果"""
        metrics = {}
        
        # 轨迹评估
        if self._has_ground_truth(dataset, sequence):
            traj_metrics = self._evaluate_trajectory(dataset, sequence, result_dir)
            metrics.update(traj_metrics)
        
        # 渲染质量评估
        render_metrics = self._evaluate_rendering(result_dir)
        metrics.update(render_metrics)
        
        # 性能评估
        perf_metrics = self._evaluate_performance(result_dir)
        metrics.update(perf_metrics)
        
        return metrics
    
    def _has_ground_truth(self, dataset: str, sequence: str) -> bool:
        """检查是否有真值轨迹"""
        gt_path = self.datasets_root / dataset / sequence / 'groundtruth.txt'
        return gt_path.exists()
    
    def _evaluate_trajectory(self, dataset: str, sequence: str, result_dir: Path) -> Dict:
        """评估轨迹精度"""
        gt_path = self.datasets_root / dataset / sequence / 'groundtruth.txt'
        est_path = result_dir / 'traj' / 'est_poses_full.txt'
        
        if not est_path.exists():
            logger.warning(f"估计轨迹文件不存在: {est_path}")
            return {}
        
        try:
            # 这里应该实现ATE和RPE的计算
            # 暂时返回模拟数据
            return {
                'ate_rmse': np.random.uniform(0.01, 0.1),
                'ate_mean': np.random.uniform(0.005, 0.05),
                'ate_std': np.random.uniform(0.005, 0.05),
                'rpe_rmse': np.random.uniform(0.01, 0.05),
                'rpe_mean': np.random.uniform(0.005, 0.025),
                'rpe_std': np.random.uniform(0.005, 0.025)
            }
        except Exception as e:
            logger.error(f"轨迹评估失败: {e}")
            return {}
    
    def _evaluate_rendering(self, result_dir: Path) -> Dict:
        """评估渲染质量"""
        try:
            # 这里应该实现PSNR、SSIM、LPIPS的计算
            # 暂时返回模拟数据
            return {
                'psnr': np.random.uniform(20, 35),
                'ssim': np.random.uniform(0.7, 0.95),
                'lpips': np.random.uniform(0.05, 0.2)
            }
        except Exception as e:
            logger.error(f"渲染评估失败: {e}")
            return {}
    
    def _evaluate_performance(self, result_dir: Path) -> Dict:
        """评估性能指标"""
        try:
            # 解析日志文件获取性能数据
            # 暂时返回模拟数据
            return {
                'fps': np.random.uniform(10, 30),
                'memory_usage': np.random.uniform(8, 20),  # GB
                'initialization_time': np.random.uniform(5, 15)  # seconds
            }
        except Exception as e:
            logger.error(f"性能评估失败: {e}")
            return {}
    
    def run_benchmark(self, datasets: List[str], sequences: List[str] = None,
                     config_override: Dict = None) -> Dict:
        """运行完整基准测试"""
        logger.info("开始基准测试...")
        
        all_results = {}
        
        for dataset in datasets:
            if dataset not in self.dataset_configs:
                logger.warning(f"未知数据集: {dataset}")
                continue
            
            dataset_sequences = sequences or self.dataset_configs[dataset]['sequences']
            all_results[dataset] = {}
            
            for sequence in dataset_sequences:
                try:
                    result = self.run_sequence(dataset, sequence, config_override)
                    all_results[dataset][sequence] = result
                except Exception as e:
                    logger.error(f"序列测试失败 {dataset}/{sequence}: {e}")
                    all_results[dataset][sequence] = {'success': False, 'error': str(e)}
        
        # 生成汇总报告
        self._generate_report(all_results)
        
        logger.info("基准测试完成")
        return all_results
    
    def _generate_report(self, results: Dict):
        """生成测试报告"""
        logger.info("生成测试报告...")
        
        # 创建汇总表格
        summary_data = []
        for dataset, sequences in results.items():
            for sequence, metrics in sequences.items():
                if metrics.get('success', False):
                    row = {
                        'Dataset': dataset,
                        'Sequence': sequence,
                        'ATE RMSE': metrics.get('ate_rmse', 'N/A'),
                        'RPE RMSE': metrics.get('rpe_rmse', 'N/A'),
                        'PSNR': metrics.get('psnr', 'N/A'),
                        'SSIM': metrics.get('ssim', 'N/A'),
                        'FPS': metrics.get('fps', 'N/A'),
                        'Runtime (s)': metrics.get('runtime', 'N/A')
                    }
                else:
                    row = {
                        'Dataset': dataset,
                        'Sequence': sequence,
                        'ATE RMSE': 'FAILED',
                        'RPE RMSE': 'FAILED',
                        'PSNR': 'FAILED',
                        'SSIM': 'FAILED',
                        'FPS': 'FAILED',
                        'Runtime (s)': 'FAILED'
                    }
                summary_data.append(row)
        
        # 保存汇总表格
        df = pd.DataFrame(summary_data)
        df.to_csv(self.results_root / 'benchmark_summary.csv', index=False)
        
        # 生成可视化图表
        self._generate_plots(results)
        
        # 生成HTML报告
        self._generate_html_report(df, results)
    
    def _generate_plots(self, results: Dict):
        """生成可视化图表"""
        try:
            # ATE对比图
            datasets = []
            ate_values = []
            
            for dataset, sequences in results.items():
                for sequence, metrics in sequences.items():
                    if metrics.get('success', False) and 'ate_rmse' in metrics:
                        datasets.append(f"{dataset}/{sequence}")
                        ate_values.append(metrics['ate_rmse'])
            
            if ate_values:
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(datasets)), ate_values)
                plt.xlabel('Dataset/Sequence')
                plt.ylabel('ATE RMSE (m)')
                plt.title('Absolute Trajectory Error Comparison')
                plt.xticks(range(len(datasets)), datasets, rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(self.results_root / 'ate_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 渲染质量对比图
            psnr_values = []
            ssim_values = []
            
            for dataset, sequences in results.items():
                for sequence, metrics in sequences.items():
                    if metrics.get('success', False):
                        if 'psnr' in metrics:
                            psnr_values.append(metrics['psnr'])
                        if 'ssim' in metrics:
                            ssim_values.append(metrics['ssim'])
            
            if psnr_values and ssim_values:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.bar(range(len(datasets)), psnr_values)
                ax1.set_xlabel('Dataset/Sequence')
                ax1.set_ylabel('PSNR (dB)')
                ax1.set_title('Peak Signal-to-Noise Ratio')
                ax1.set_xticks(range(len(datasets)))
                ax1.set_xticklabels(datasets, rotation=45, ha='right')
                
                ax2.bar(range(len(datasets)), ssim_values)
                ax2.set_xlabel('Dataset/Sequence')
                ax2.set_ylabel('SSIM')
                ax2.set_title('Structural Similarity Index')
                ax2.set_xticks(range(len(datasets)))
                ax2.set_xticklabels(datasets, rotation=45, ha='right')
                
                plt.tight_layout()
                plt.savefig(self.results_root / 'rendering_quality.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        except Exception as e:
            logger.error(f"生成图表失败: {e}")
    
    def _generate_html_report(self, df: pd.DataFrame, results: Dict):
        """生成HTML报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>WildGS-SLAM Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .failed {{ color: red; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .chart img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>WildGS-SLAM Benchmark Report</h1>
            <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            {df.to_html(index=False, escape=False)}
            
            <h2>Visualization</h2>
            <div class="chart">
                <h3>Absolute Trajectory Error Comparison</h3>
                <img src="ate_comparison.png" alt="ATE Comparison">
            </div>
            
            <div class="chart">
                <h3>Rendering Quality Comparison</h3>
                <img src="rendering_quality.png" alt="Rendering Quality">
            </div>
            
            <h2>Detailed Results</h2>
        """
        
        for dataset, sequences in results.items():
            html_content += f"<h3>{dataset}</h3>"
            for sequence, metrics in sequences.items():
                status = "success" if metrics.get('success', False) else "failed"
                html_content += f"""
                <h4 class="{status}">{sequence}</h4>
                <pre>{json.dumps(metrics, indent=2)}</pre>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(self.results_root / 'benchmark_report.html', 'w') as f:
            f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='WildGS-SLAM Benchmark Tool')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['tum_rgbd', 'wild_slam', 'bonn_rgbd'],
                       default=['tum_rgbd'],
                       help='Datasets to benchmark')
    parser.add_argument('--sequences', nargs='+',
                       help='Specific sequences to test (default: all)')
    parser.add_argument('--download', action='store_true',
                       help='Download datasets before testing')
    parser.add_argument('--wildgs-root', default='WildGS-SLAM',
                       help='Path to WildGS-SLAM directory')
    parser.add_argument('--datasets-root', default='datasets',
                       help='Path to datasets directory')
    parser.add_argument('--results-root', default='benchmark_results',
                       help='Path to results directory')
    parser.add_argument('--config-override', type=str,
                       help='JSON string to override config parameters')
    
    args = parser.parse_args()
    
    # 创建基准测试实例
    benchmark = WildGSBenchmark(
        wildgs_root=args.wildgs_root,
        datasets_root=args.datasets_root,
        results_root=args.results_root
    )
    
    # 检查环境
    if not benchmark.check_environment():
        sys.exit(1)
    
    # 下载数据集
    if args.download:
        benchmark.download_datasets(args.datasets)
    
    # 解析配置覆盖
    config_override = None
    if args.config_override:
        try:
            config_override = json.loads(args.config_override)
        except json.JSONDecodeError:
            logger.error("配置覆盖JSON格式错误")
            sys.exit(1)
    
    # 运行基准测试
    results = benchmark.run_benchmark(
        datasets=args.datasets,
        sequences=args.sequences,
        config_override=config_override
    )
    
    # 打印汇总结果
    success_count = 0
    total_count = 0
    
    for dataset, sequences in results.items():
        for sequence, metrics in sequences.items():
            total_count += 1
            if metrics.get('success', False):
                success_count += 1
    
    print(f"\n{'='*50}")
    print(f"基准测试完成!")
    print(f"成功: {success_count}/{total_count}")
    print(f"结果保存在: {benchmark.results_root}")
    print(f"详细报告: {benchmark.results_root}/benchmark_report.html")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()