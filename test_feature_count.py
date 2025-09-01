#!/usr/bin/env python3
"""
测试任务特征数量
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from environment.historical_replay_simulator import HistoricalReplaySimulator

def test_feature_count():
    """测试任务特征数量"""
    print("=" * 60)
    print("测试任务特征数量")
    print("=" * 60)
    
    # 创建测试数据
    process_instances = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['process_1', 'process_2', 'process_3'],
        'process_definition_code': [101, 102, 103],
        'state': [7, 7, 7],
        'start_time': ['2024-01-01 10:00:00'] * 3
    })
    
    task_instances = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'process_instance_id': [1, 1, 2, 2, 3],
        'name': ['task_1', 'task_2', 'task_3', 'task_4', 'task_5'],
        'host': ['host_A', 'host_B', 'host_A', 'host_B', 'host_A'],
        'task_type': ['SQL', 'PYTHON', 'JAVA', 'SPARK', 'FLINK'],
        'start_time': ['2024-01-01 10:00:00'] * 5,
        'end_time': ['2024-01-01 10:01:00'] * 5,
        'cpu_req': [2, 4, 6, 8, 10],
        'memory_req': [4, 8, 12, 16, 20],
        'task_instance_priority': [1, 2, 3, 4, 5],
        'retry_times': [0, 1, 0, 2, 0],
        'process_definition_id': [101, 101, 102, 102, 103]
    })
    
    task_definitions = pd.DataFrame({
        'id': [101, 102, 103, 104, 105],
        'name': ['task_def_1', 'task_def_2', 'task_def_3', 'task_def_4', 'task_def_5'],
        'task_type': ['SQL', 'PYTHON', 'JAVA', 'SPARK', 'FLINK']
    })
    
    process_task_relations = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'process_definition_code': [101, 101, 102, 102, 103],
        'task_definition_code': [101, 102, 103, 104, 105],
        'pre_task_code': [None, 101, None, 103, None],
        'post_task_code': [102, None, 104, None, 106]
    })
    
    # 创建模拟器
    simulator = HistoricalReplaySimulator(process_instances, task_instances, task_definitions, process_task_relations)
    
    print("模拟器创建成功")
    print(f"进程数: {len(simulator.successful_processes)}")
    print(f"任务数: {len(simulator.current_process_tasks)}")
    
    print("\n测试任务特征提取:")
    
    # 测试每个任务的特征数量
    for i, task in simulator.current_process_tasks.iterrows():
        try:
            features = simulator._extract_task_features(task)
            print(f"任务 {i+1} ({task['name']}): {len(features)} 个特征")
            
            # 打印特征详情
            feature_names = [
                "SQL", "SHELL", "PYTHON", "JAVA", "SPARK", "FLINK", "HTTP",  # 7个任务类型
                "CPU需求", "内存需求", "执行时间", "优先级", "重试次数",  # 5个基础特征
                "复杂度评分", "依赖数量", "队列位置", "剩余任务数"  # 4个计算特征
            ]
            
            for j, (name, value) in enumerate(zip(feature_names, features)):
                print(f"  {j+1:2d}. {name:12s}: {value:8.3f}")
                
        except Exception as e:
            print(f"任务 {i+1} 特征提取失败: {e}")
    
    print("\n测试get_state方法:")
    try:
        state = simulator.get_state()
        task_features, resource_features = state
        
        print(f"任务特征形状: {task_features.shape}")
        print(f"资源特征形状: {resource_features.shape}")
        
        # 检查每个任务的特征数量
        for i in range(task_features.shape[1]):
            task_feat = task_features[0, i, :]
            print(f"任务 {i+1}: {len(task_feat)} 个特征")
            
    except Exception as e:
        print(f"get_state失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_feature_count()
