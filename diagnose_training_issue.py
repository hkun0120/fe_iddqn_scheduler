#!/usr/bin/env python3
"""
诊断训练过程中的模型保存问题
检查训练日志和验证阶段的具体情况
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

def diagnose_training_issue():
    """诊断训练问题"""
    print("🔍 开始诊断训练过程中的模型保存问题...")
    
    # 检查训练日志
    logs_dir = Path("fe_iddqn_training_system/logs")
    if logs_dir.exists():
        print(f"\n📋 检查训练日志目录: {logs_dir}")
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            # 找到最新的日志文件
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"📄 最新日志文件: {latest_log}")
            
            # 读取日志内容
            try:
                with open(latest_log, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                
                print(f"📊 日志文件大小: {len(log_content)} 字符")
                
                # 检查关键信息
                key_phrases = [
                    "保存初始模型",
                    "新的最佳模型",
                    "模型已保存到",
                    "avg_val_makespan",
                    "best_val_makespan",
                    "Epoch 1",
                    "训练完成",
                    "早停",
                    "验证阶段"
                ]
                
                print("\n🔍 关键信息检查:")
                for phrase in key_phrases:
                    if phrase in log_content:
                        print(f"   ✅ 找到: {phrase}")
                    else:
                        print(f"   ❌ 未找到: {phrase}")
                
                # 检查验证阶段的makespan值
                print("\n📈 验证Makespan值分析:")
                lines = log_content.split('\n')
                val_makespan_lines = [line for line in lines if 'Val' in line and 'Makespan' in line]
                if val_makespan_lines:
                    for line in val_makespan_lines[-5:]:  # 显示最后5行
                        print(f"   {line.strip()}")
                else:
                    print("   ❌ 没有找到验证Makespan信息")
                
            except Exception as e:
                print(f"   ❌ 读取日志文件失败: {e}")
        else:
            print("   ❌ 没有找到日志文件")
    else:
        print(f"   ❌ 日志目录不存在: {logs_dir}")
    
    # 检查结果文件
    results_dir = Path("fe_iddqn_training_system/results")
    if results_dir.exists():
        print(f"\n📊 检查结果目录: {results_dir}")
        result_files = list(results_dir.glob("*.json"))
        if result_files:
            for result_file in result_files:
                print(f"   📄 结果文件: {result_file}")
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    print(f"     📊 文件大小: {result_file.stat().st_size} bytes")
                    if isinstance(result_data, dict):
                        print(f"     🔑 包含键: {list(result_data.keys())}")
                except Exception as e:
                    print(f"     ❌ 读取失败: {e}")
        else:
            print("   ❌ 没有找到结果文件")
    else:
        print(f"   ❌ 结果目录不存在: {results_dir}")
    
    # 检查模型文件
    models_dir = Path("fe_iddqn_training_system/models")
    if models_dir.exists():
        print(f"\n🤖 检查模型目录: {models_dir}")
        model_files = list(models_dir.glob("*.pkl"))
        if model_files:
            for model_file in model_files:
                print(f"   📄 模型文件: {model_file}")
                print(f"     📊 文件大小: {model_file.stat().st_size} bytes")
                print(f"     📅 修改时间: {datetime.fromtimestamp(model_file.stat().st_mtime)}")
        else:
            print("   ❌ 没有找到模型文件")
    else:
        print(f"   ❌ 模型目录不存在: {models_dir}")
    
    # 模拟训练过程的关键检查点
    print(f"\n🧪 模拟训练过程检查:")
    
    # 检查1: 验证阶段是否执行
    print("   1. 检查验证阶段是否执行...")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            with open(latest_log, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            if "Val   - Reward:" in log_content:
                print("     ✅ 验证阶段已执行")
            else:
                print("     ❌ 验证阶段未执行")
        else:
            print("     ❌ 无法检查日志")
    
    # 检查2: avg_val_makespan值是否有效
    print("   2. 检查avg_val_makespan值...")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            with open(latest_log, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # 查找makespan值
            import re
            makespan_matches = re.findall(r'Makespan: ([\d.]+)s', log_content)
            if makespan_matches:
                print(f"     📊 找到makespan值: {makespan_matches}")
                # 检查是否有inf或nan
                inf_values = [v for v in makespan_matches if 'inf' in v.lower() or v == 'nan']
                if inf_values:
                    print(f"     ⚠️  发现无效值: {inf_values}")
                else:
                    print("     ✅ makespan值都是有效的")
            else:
                print("     ❌ 没有找到makespan值")
    
    # 检查3: 训练是否正常完成
    print("   3. 检查训练完成状态...")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            latest_log = max(logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime)
            with open(latest_log, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            if "训练完成" in log_content:
                print("     ✅ 训练正常完成")
            elif "早停" in log_content:
                print("     ⚠️  训练早停")
            else:
                print("     ❌ 训练可能未正常完成")

def main():
    print("=" * 80)
    print("🔍 训练过程诊断工具")
    print("=" * 80)
    
    diagnose_training_issue()
    
    print("\n" + "=" * 80)
    print("🎯 诊断完成！")
    print("💡 建议:")
    print("   1. 检查验证阶段是否正常执行")
    print("   2. 检查avg_val_makespan是否为有效数值")
    print("   3. 检查训练是否正常完成")
    print("   4. 如果问题持续，考虑添加更多调试信息")
    print("=" * 80)

if __name__ == "__main__":
    main()

