#!/usr/bin/env python3
"""
调整训练参数以适应现有数据规模
"""

import os
import sys
from pathlib import Path

def adjust_training_parameters():
    """调整训练参数以适应现有数据规模"""
    
    print("🔧 调整训练参数以适应现有数据规模...")
    
    # 1. 调整仿真器参数
    simulator_file = "environment/historical_replay_simulator.py"
    
    with open(simulator_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 调整参数以适应现有数据
    adjustments = [
        # 降低每轮工作流数量，适应现有3,606个工作流
        ('self.MAX_PROCESSES_PER_EPISODE = 100', 'self.MAX_PROCESSES_PER_EPISODE = 50'),
        
        # 降低每轮任务数量，适应现有83,172个任务
        ('self.MAX_TASKS_PER_EPISODE = 500', 'self.MAX_TASKS_PER_EPISODE = 200'),
    ]
    
    for old, new in adjustments:
        if old in content:
            content = content.replace(old, new)
            print(f"   ✅ 调整: {old} -> {new}")
        else:
            print(f"   ⚠️  未找到: {old}")
    
    with open(simulator_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 2. 调整训练脚本参数
    train_script = "fe_iddqn_training_system/train_with_preprocessed_data.py"
    
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 调整训练参数
    adjustments = [
        # 增加训练轮次，补偿数据量减少
        ('self.n_epochs = 200', 'self.n_epochs = 300'),
        
        # 增加早停耐心值
        ('patience = 30', 'patience = 40'),
    ]
    
    for old, new in adjustments:
        if old in content:
            content = content.replace(old, new)
            print(f"   ✅ 调整: {old} -> {new}")
        else:
            print(f"   ⚠️  未找到: {old}")
    
    with open(train_script, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 训练参数调整完成！")

def create_optimized_training_script():
    """创建优化的训练脚本"""
    
    print("📝 创建优化的训练脚本...")
    
    optimized_script = """#!/usr/bin/env python3
'''
优化训练脚本 - 适应现有数据规模
'''

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from fe_iddqn_training_system.train_with_preprocessed_data import FEIDDQNTrainerWithPreprocessedData

def run_optimized_training():
    '''运行优化训练'''
    
    print("🚀 开始优化训练...")
    print("📊 优化后的训练参数:")
    print("   - 训练轮次: 300 (增加50%)")
    print("   - 每轮工作流: 50 (适应现有数据)")
    print("   - 每轮任务: 200 (适应现有数据)")
    print("   - 早停耐心: 40 (增加33%)")
    print("   - 总训练轮次: 15,000 (vs 之前的20,000)")
    
    # 创建训练器
    trainer = FEIDDQNTrainerWithPreprocessedData(
        data_dir="fe_iddqn_training_system/data",
        models_dir="fe_iddqn_training_system/models",
        logs_dir="fe_iddqn_training_system/logs",
        results_dir="fe_iddqn_training_system/results"
    )
    
    # 运行训练
    agent, history = trainer.run_training_pipeline()
    
    print("✅ 优化训练完成！")
    return agent, history

if __name__ == "__main__":
    run_optimized_training()
"""
    
    with open("run_optimized_training.py", 'w', encoding='utf-8') as f:
        f.write(optimized_script)
    
    print("✅ 优化训练脚本创建完成: run_optimized_training.py")

def main():
    """主函数"""
    
    print("=" * 80)
    print("🔧 FE-IDDQN 训练参数优化")
    print("=" * 80)
    
    print("📊 当前数据规模:")
    print("   - 工作流实例: 3,606个")
    print("   - 任务实例: 83,172个")
    print("   - 平均每工作流任务数: 23.1个")
    
    print("\n🎯 优化策略:")
    print("   - 降低每轮数据需求，适应现有数据规模")
    print("   - 增加训练轮次，补偿数据量减少")
    print("   - 增加早停耐心值，给模型更多时间学习")
    
    # 1. 调整训练参数
    adjust_training_parameters()
    
    # 2. 创建优化训练脚本
    create_optimized_training_script()
    
    print("\n" + "=" * 80)
    print("📋 优化后的训练配置:")
    print("   - 训练轮次: 300 (vs 之前的200)")
    print("   - 每轮工作流: 50 (vs 之前的100)")
    print("   - 每轮任务: 200 (vs 之前的500)")
    print("   - 早停耐心: 40 (vs 之前的30)")
    print("   - 总训练轮次: 15,000 (vs 之前的20,000)")
    print("\n🚀 下一步:")
    print("1. 上传优化后的文件到远程服务器")
    print("2. 运行: python3 run_optimized_training.py")
    print("3. 预计训练时间: 8-12小时")
    print("=" * 80)

if __name__ == "__main__":
    main()

