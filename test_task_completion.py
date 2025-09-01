#!/usr/bin/env python3
"""
测试任务完成统计逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from environment.historical_replay_simulator import HistoricalReplaySimulator

def test_task_completion():
    """测试任务完成统计逻辑"""
    print("=" * 60)
    print("测试任务完成统计逻辑")
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
    print(f"当前进程任务数: {len(simulator.current_process_tasks)}")
    
    print("\n初始状态:")
    process_info = simulator.get_current_process_info()
    print(f"进程ID: {process_info['process_id']}")
    print(f"总任务数: {process_info['total_tasks']}")
    print(f"已完成任务: {process_info['completed_tasks']}")
    print(f"剩余任务: {process_info['remaining_tasks']}")
    print(f"当前任务索引: {simulator.current_task_idx}")
    print(f"已完成任务集合: {simulator.completed_tasks}")
    
    print("\n模拟任务调度过程:")
    
    # 模拟调度第一个任务
    print("\n1. 调度第一个任务...")
    state, reward, done, info = simulator.step(0)  # 选择host_A
    print(f"   奖励: {reward}")
    print(f"   完成: {done}")
    print(f"   信息: {info}")
    
    process_info = simulator.get_current_process_info()
    print(f"   进程状态: 已完成 {process_info['completed_tasks']}/{process_info['total_tasks']}")
    print(f"   已完成任务集合: {simulator.completed_tasks}")
    print(f"   当前任务索引: {simulator.current_task_idx}")
    
    # 模拟调度第二个任务
    print("\n2. 调度第二个任务...")
    state, reward, done, info = simulator.step(1)  # 选择host_B
    print(f"   奖励: {reward}")
    print(f"   完成: {done}")
    print(f"   信息: {info}")
    
    process_info = simulator.get_current_process_info()
    print(f"   进程状态: 已完成 {process_info['completed_tasks']}/{process_info['total_tasks']}")
    print(f"   剩余任务: {process_info['remaining_tasks']}")
    print(f"   已完成任务集合: {simulator.completed_tasks}")
    print(f"   当前任务索引: {simulator.current_task_idx}")
    
    # 检查是否完成当前进程
    print(f"\n当前进程是否完成: {simulator.current_task_idx >= len(simulator.current_process_tasks)}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_task_completion()
