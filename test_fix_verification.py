#!/usr/bin/env python3
"""
测试修复后的FE-IDDQN算法性能
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.data_loader import DataLoader
from experiments.experiment_runner import ExperimentRunner
from config.config import Config

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_fix_verification.log')
        ]
    )

def test_historical_replay_simulator():
    """测试历史重放仿真器的修复"""
    print("=" * 60)
    print("测试历史重放仿真器修复")
    print("=" * 60)
    
    # 加载数据
    data_loader = DataLoader(Config.RAW_DATA_DIR)
    data = data_loader.load_all_data()
    
    # 创建实验运行器
    experiment_runner = ExperimentRunner(
        data=data,
        features=pd.DataFrame(),  # 空特征DataFrame
        output_dir=Config.RESULTS_DIR,
        n_experiments=1  # 只运行1次测试
    )
    
    # 测试FE-IDDQN算法
    print("测试FE-IDDQN算法...")
    try:
        results = experiment_runner.run_algorithm("FE_IDDQN", use_historical_replay=True)
        print(f"FE-IDDQN结果: {results}")
        
        # 检查makespan是否合理
        fe_iddqn_result = results.get('FE_IDDQN', {})
        makespan = fe_iddqn_result.get('avg_makespan', 0)
        resource_util = fe_iddqn_result.get('avg_resource_utilization', 0)
        
        print(f"\nFE-IDDQN性能指标:")
        print(f"  Makespan: {makespan:.2f}")
        print(f"  资源利用率: {resource_util:.2f}")
        
        # 检查是否修复了问题
        if makespan > 0 and makespan < 50000:  # 合理的makespan范围
            print("✅ Makespan计算修复成功！")
        else:
            print("❌ Makespan仍然异常")
            
        if resource_util > 0:
            print("✅ 资源利用率计算修复成功！")
        else:
            print("❌ 资源利用率仍然为0")
            
    except Exception as e:
        print(f"❌ FE-IDDQN测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试传统算法作为对比
    print("\n测试传统算法作为对比...")
    try:
        fifo_results = experiment_runner.run_algorithm("FIFO", use_historical_replay=True)
        print(f"FIFO结果: {fifo_results}")
        
        fifo_result = fifo_results.get('FIFO', {})
        fifo_makespan = fifo_result.get('avg_makespan', 0)
        fifo_resource_util = fifo_result.get('avg_resource_utilization', 0)
        
        print(f"\nFIFO性能指标:")
        print(f"  Makespan: {fifo_makespan:.2f}")
        print(f"  资源利用率: {fifo_resource_util:.2f}")
        
    except Exception as e:
        print(f"❌ FIFO测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_dqn_algorithms():
    """测试DQN算法的修复"""
    print("\n" + "=" * 60)
    print("测试DQN算法修复")
    print("=" * 60)
    
    # 加载数据
    data_loader = DataLoader(Config.RAW_DATA_DIR)
    data = data_loader.load_all_data()
    
    # 创建实验运行器
    experiment_runner = ExperimentRunner(
        data=data,
        features=pd.DataFrame(),
        output_dir=Config.RESULTS_DIR,
        n_experiments=1
    )
    
    # 测试DQN算法
    algorithms_to_test = ["DQN", "DDQN"]
    
    for algo in algorithms_to_test:
        print(f"\n测试{algo}算法...")
        try:
            results = experiment_runner.run_algorithm(algo, use_historical_replay=True)
            print(f"{algo}结果: {results}")
            
            algo_result = results.get(algo, {})
            makespan = algo_result.get('avg_makespan', 0)
            resource_util = algo_result.get('avg_resource_utilization', 0)
            
            print(f"\n{algo}性能指标:")
            print(f"  Makespan: {makespan:.2f}")
            print(f"  资源利用率: {resource_util:.2f}")
            
            if makespan > 0:
                print(f"✅ {algo} Makespan修复成功！")
            else:
                print(f"❌ {algo} Makespan仍然为0")
                
        except Exception as e:
            print(f"❌ {algo}测试失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    setup_logging()
    
    print("开始测试修复效果...")
    
    # 测试历史重放仿真器修复
    test_historical_replay_simulator()
    
    # 测试DQN算法修复
    test_dqn_algorithms()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)