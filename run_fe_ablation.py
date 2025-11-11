#!/usr/bin/env python3
"""
运行FE-IDDQN消融实验
对比不同特征配置下FE-IDDQN vs HEFT的性能
"""

import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
VAL = ROOT / 'val_data_20250930_122033_labeled.csv'
OUT_DIR = ROOT / 'ablation_fe'
OUT_DIR.mkdir(exist_ok=True)

# 消融实验变体
VARIANTS = {
    'BASE': {
        'STRICT_HIST_DUR': '0',
        'FE_ENHANCED': '1',
        'EF_COEF': '3e-4',
        'HEFT_COEF': '0.15',
        'DISP_COEF': '1e-5',
        'FE_NO_TTYPE': '0',
        'FE_IGNORE_Q': '0'  # 确保运行FE-IDDQN
    },
    '-A_no_speed': {
        'STRICT_HIST_DUR': '1',  # 关闭speed_factor
        'FE_ENHANCED': '1',
        'EF_COEF': '3e-4',
        'HEFT_COEF': '0.15',
        'DISP_COEF': '1e-5',
        'FE_NO_TTYPE': '0',
        'FE_IGNORE_Q': '0'
    },
    '-B_no_upward': {
        'STRICT_HIST_DUR': '0',
        'FE_ENHANCED': '1',
        'EF_COEF': '3e-4',
        'HEFT_COEF': '0.0',      # 关闭upward_rank
        'CP_COEF': '0.0',
        'DISP_COEF': '1e-5',
        'FE_NO_TTYPE': '0',
        'FE_WARM_HEFT': '0',
        'FE_IGNORE_Q': '0'
    },
    '-C_no_ttype': {
        'STRICT_HIST_DUR': '0',
        'FE_ENHANCED': '1',
        'EF_COEF': '3e-4',
        'HEFT_COEF': '0.15',
        'DISP_COEF': '1e-5',
        'FE_NO_TTYPE': '1',      # 关闭任务类型特征
        'FE_IGNORE_Q': '0'
    }
}

def run_one(pid: int, env: dict, variant_label: str):
    """运行单个实验"""
    env_str = ' '.join([f"{k}={v}" for k, v in env.items()])
    metrics_file = OUT_DIR / f"metrics_{pid}_{variant_label}.csv"
    
    cmd = f"source .venv/bin/activate && {env_str} python create_gantt_chart_generic.py {pid} > /dev/null 2>&1"
    
    print(f"    运行 {pid}...", end='', flush=True)
    result = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    
    # 检查结果
    orig_metrics = ROOT / f"metrics_{pid}.csv"
    if orig_metrics.exists():
        import shutil
        shutil.move(str(orig_metrics), str(metrics_file))
        
        # 检查是否有FE_IDDQN数据
        try:
            df = pd.read_csv(metrics_file)
            if 'FE_IDDQN' in df['algorithm'].values:
                print(f" ✅")
                return True
            else:
                print(f" ⚠️ (没有FE_IDDQN)")
                return False
        except:
            print(f" ❌ (读取失败)")
            return False
    else:
        print(f" ❌ (无文件)")
        return False

def main():
    """主函数"""
    print("🔬 开始FE-IDDQN消融实验")
    print("="*80)
    
    # 读取验证数据
    df = pd.read_csv(VAL)
    
    # 改用XLarge规模（任务数更多，不会触发小规模处理）
    xlarge_pids = df[df['size'] == 'XLarge']['process_id'].unique()[:12]  # XLarge有12个
    print(f"\n📋 XLarge规模工作流: {len(xlarge_pids)} 个")
    print(f"Process IDs: {list(xlarge_pids)}\n")
    
    all_results = []
    
    for variant_name, env in VARIANTS.items():
        variant_label = f"XLarge{variant_name}"
        print(f"\n🧪 运行变体: {variant_label}")
        print(f"配置: {env}")
        print("-"*80)
        
        success_count = 0
        for pid in xlarge_pids:
            if run_one(int(pid), env, variant_label):
                success_count += 1
        
        print(f"\n✅ 成功: {success_count}/{len(xlarge_pids)}")
        
        # 收集结果
        variant_results = []
        for pid in xlarge_pids:
            metrics_file = OUT_DIR / f"metrics_{int(pid)}_{variant_label}.csv"
            if metrics_file.exists():
                try:
                    df_m = pd.read_csv(metrics_file)
                    df_m['variant'] = variant_label
                    df_m['variant_name'] = variant_name
                    variant_results.append(df_m)
                except:
                    pass
        
        if variant_results:
            all_results.append(pd.concat(variant_results, ignore_index=True))
    
    if not all_results:
        print("\n❌ 没有收集到任何结果")
        return
    
    # 合并所有结果
    results_df = pd.concat(all_results, ignore_index=True)
    
    # 保存原始数据
    out_csv = OUT_DIR / 'fe_ablation_results.csv'
    results_df.to_csv(out_csv, index=False)
    print(f"\n💾 原始数据保存到: {out_csv}")
    
    # 分析结果
    print("\n" + "="*80)
    print("📊 分析FE-IDDQN消融实验结果")
    print("="*80)
    
    # 只保留HEFT和FE_IDDQN
    df_compare = results_df[results_df['algorithm'].isin(['HEFT', 'FE_IDDQN'])].copy()
    
    if len(df_compare) == 0:
        print("❌ 没有可对比的数据")
        return
    
    # 按工作流和变体分组，计算改进
    pivot = df_compare.pivot_table(
        index=['process_id', 'variant_name'],
        columns='algorithm',
        values='makespan',
        aggfunc='min'
    ).reset_index()
    
    # 计算改进百分比
    pivot['improve_vs_heft'] = (pivot['HEFT'] - pivot['FE_IDDQN']) / pivot['HEFT']
    
    # 按变体分组统计
    summary = pivot.groupby('variant_name')['improve_vs_heft'].agg([
        'count', 'mean', 'std', 'median'
    ]).reset_index()
    
    summary['mean_pct'] = summary['mean'] * 100
    summary['std_pct'] = summary['std'] * 100
    summary['median_pct'] = summary['median'] * 100
    
    # 保存汇总
    summary_file = OUT_DIR / 'fe_ablation_summary.csv'
    summary.to_csv(summary_file, index=False)
    print(f"💾 汇总数据保存到: {summary_file}")
    
    # 打印表5.8
    print("\n" + "="*80)
    print("📝 表5.8 特征工程消融实验结果（XLarge规模）")
    print("="*80)
    print("\n| 特征集 | 平均改进(%) | 标准差(%) | 工作流数量 | 中位数(%) |")
    print("|--------|------------|----------|-----------|-----------|")
    
    variant_labels = {
        'BASE': 'XLargeBASE (完整特征)',
        '-A_no_speed': 'XLarge-A (去除speed_factor)',
        '-B_no_upward': 'XLarge-B (去除upward_rank)',
        '-C_no_ttype': 'XLarge-C (去除任务类型)'
    }
    
    for _, row in summary.iterrows():
        label = variant_labels.get(row['variant_name'], row['variant_name'])
        print(f"| {label:30} | {row['mean_pct']:>10.1f} | {row['std_pct']:>8.1f} | "
              f"{int(row['count']):>9d} | {row['median_pct']:>9.1f} |")
    
    # 详细分析
    print("\n" + "="*80)
    print("📈 详细分析")
    print("="*80)
    
    for variant in ['BASE', '-A_no_speed', '-B_no_upward', '-C_no_ttype']:
        variant_data = pivot[pivot['variant_name'] == variant]
        if len(variant_data) > 0:
            improvements = variant_data['improve_vs_heft'] * 100
            print(f"\n{variant}:")
            print(f"  样本数: {len(improvements)}")
            print(f"  平均改进: {improvements.mean():.1f}%")
            print(f"  标准差: {improvements.std():.1f}%")
            print(f"  中位数: {improvements.median():.1f}%")
            print(f"  最小值: {improvements.min():.1f}%")
            print(f"  最大值: {improvements.max():.1f}%")
            print(f"  25分位: {improvements.quantile(0.25):.1f}%")
            print(f"  75分位: {improvements.quantile(0.75):.1f}%")
    
    print("\n✅ 消融实验完成!")

if __name__ == '__main__':
    main()

