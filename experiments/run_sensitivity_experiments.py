#!/usr/bin/env python3
"""
敏感度实验自动化脚本
测试不同超参组合对模型性能的影响
"""
import os
import sys
import subprocess
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / 'sensitivity_results'
RESULTS_DIR.mkdir(exist_ok=True)
class SensitivityExperiment:
    def __init__(self):
        self.results = []
        self.summary_csv = RESULTS_DIR / 'sensitivity_experiment_summary.csv'
    def run_experiments(self, learning_rates, batch_sizes):
        """运行所有超参组合的实验"""
        total_combos = len(learning_rates) * len(batch_sizes)
        current = 0
        print(f"\n{'='*80}")
        print(f"🚀 敏感度实验开始")
        print(f"学习率: {learning_rates}")
        print(f"批大小: {batch_sizes}")
        print(f"总实验数: {total_combos}")
        print(f"{'='*80}\n")
        for lr, batch_size in product(learning_rates, batch_sizes):
            current += 1
            print(f"\n[{current}/{total_combos}] 运行: LR={lr}, Batch={batch_size}")
            print(f"{'-'*60}")
            result = self.run_single_experiment(lr, batch_size)
            if result:
                self.results.append(result)
            # 防止过载
            time.sleep(2)
        self._save_summary()
        self._generate_report()
    def run_single_experiment(self, lr, batch_size):
        """运行单个实验"""
        try:
            # 使用Python动态执行训练脚本，修改超参数
            train_code = self._prepare_train_script(lr, batch_size)
            # 写入临时脚本
            temp_script = RESULTS_DIR / f'train_lr{lr}_bs{batch_size}.py'
            with open(temp_script, 'w') as f:
                f.write(train_code)
            # 执行
            result = subprocess.run(
                [sys.executable, str(temp_script)],
                cwd=PROJECT_DIR,
                capture_output=True,
                timeout=3600,
                text=True
            )
            if result.returncode == 0:
                print(f"✅ 完成: LR={lr}, Batch={batch_size}")
                # 解析结果 (从最新的CSV文件)
                return self._parse_results(lr, batch_size)
            else:
                print(f"❌ 失败: {result.stderr[:200]}")
                return None
        except Exception as e:
            print(f"❌ 异常: {e}")
            return None
    def _prepare_train_script(self, lr, batch_size):
        """准备修改了超参数的训练脚本"""
        with open(PROJECT_DIR / 'run_ml1m_benchmark_final.py', 'r') as f:
            content = f.read()
        # 替换超参
        content = content.replace('BATCH_SIZE = 2048', f'BATCH_SIZE = {batch_size}')
        content = content.replace('BATCH_SIZE=2048', f'BATCH_SIZE = {batch_size}')
        # 添加学习率参数到optimizer
        if 'learning_rate' not in content:
            # 在model.compile后添加
            content = content.replace(
                'model.compile("adam", "binary_crossentropy", metrics=["auc"])',
                f'model.compile("adam", "binary_crossentropy", metrics=["auc"], lr={lr})'
            )
        return content
    def _parse_results(self, lr, batch_size):
        """解析实验结果"""
        import glob
        # 查找最新生成的结果CSV
        result_files = sorted(
            glob.glob(str(PROJECT_DIR / 'thesis_final_benchmark_*.csv')),
            key=os.path.getctime
        )
        if not result_files:
            return None
        latest_file = result_files[-1]
        df = pd.read_csv(latest_file)
        # 整理结果
        results = {
            'lr': lr,
            'batch_size': batch_size,
            'timestamp': datetime.now().isoformat(),
            'result_file': os.path.basename(latest_file)
        }
        # 添加模型性能指标
        for idx, row in df.iterrows():
            model_result = dict(results)
            model_result.update({
                'model': row['Model'],
                'auc': row['AUC'],
                'ndcg_10': row['NDCG@10'],
                'mrr_10': row['MRR@10'],
                'recall_10': row['Recall@10'],
                'precision_10': row['Precision@10'],
                'f1_10': row['F1@10'],
                'hitrate_10': row['HitRate@10']
            })
            self.results.append(model_result)
        return results
    def _save_summary(self):
        """保存结果汇总"""
        if not self.results:
            print("⚠️ 没有实验结果")
            return
        df = pd.DataFrame(self.results)
        df.to_csv(self.summary_csv, index=False)
        print(f"\n✅ 结果已保存: {self.summary_csv}")
    def _generate_report(self):
        """生成分析报告"""
        if not self.summary_csv.exists():
            return
        df = pd.read_csv(self.summary_csv)
        print(f"\n{'='*80}")
        print(f"📊 敏感度实验分析报告")
        print(f"{'='*80}")
        # 按LR和Batch分组
        grouped = df.groupby(['lr', 'batch_size'])
        for (lr, bs), group in grouped:
            print(f"\n🔍 LR={lr}, Batch={bs}")
            print(f"   样本数: {len(group)}")
            print(f"   AUC    - 均值: {group['auc'].mean():.6f}, 最大: {group['auc'].max():.6f}")
            print(f"   NDCG@10- 均值: {group['ndcg_10'].mean():.6f}, 最大: {group['ndcg_10'].max():.6f}")
            # 找出最佳模型
            best_model = group.loc[group['auc'].idxmax()]
            print(f"   🏆 最佳模型: {best_model['model']} (AUC={best_model['auc']:.6f})")
        # 全局最佳配置
        best_result = df.loc[df['auc'].idxmax()]
        print(f"\n{'='*80}")
        print(f"🏆 全局最佳配置")
        print(f"   LR={best_result['lr']}, Batch={best_result['batch_size']}")
        print(f"   模型: {best_result['model']}")
        print(f"   AUC={best_result['auc']:.6f}, NDCG@10={best_result['ndcg_10']:.6f}")
        print(f"{'='*80}\n")
        # 保存分析结果
        analysis_file = RESULTS_DIR / 'analysis_summary.txt'
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("敏感度实验分析报告\n")
            f.write("="*80 + "\n\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n\n")
            f.write("全局最佳配置:\n")
            f.write(f"  LR={best_result['lr']}, Batch={best_result['batch_size']}\n")
            f.write(f"  最佳模型: {best_result['model']}\n")
            f.write(f"  AUC={best_result['auc']:.6f}\n\n")
            f.write("详见CSV文件: " + str(self.summary_csv) + "\n")
if __name__ == "__main__":
    # 定义超参范围
    LEARNING_RATES = [1e-4, 1e-3, 5e-3]
    BATCH_SIZES = [512, 1024, 2048]
    experiment = SensitivityExperiment()
    experiment.run_experiments(LEARNING_RATES, BATCH_SIZES)
    print("\n✅ 所有实验已完成！")
    print(f"📁 结果目录: {RESULTS_DIR}")
