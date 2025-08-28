#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试历史重放模拟器的问题
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.historical_replay_simulator import HistoricalReplaySimulator
import pandas as pd
import numpy as np

def create_debug_data():
    """创建调试用的数据"""
    
    # 创建进程实例数据
    process_instances = pd.DataFrame({
        'id': [9977035, 9977036],
        'process_definition_code': ['WORKFLOW_001', 'WORKFLOW_002'],
        'process_definition_version': [1, 1],
        'name': ['测试工作流1', '测试工作流2'],
        'state': [7, 7],  # 7表示成功
        'start_time': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
        'end_time': ['2024-01-01 10:30:00', '2024-01-01 11:30:00']
    })
    
    # 创建任务实例数据
    task_instances = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'process_instance_id': [9977035, 9977035, 9977036, 9977036],
        'task_code': ['TASK_001', 'TASK_002', 'TASK_003', 'TASK_004'],
        'task_definition_version': [1, 1, 1, 1],
        'name': ['任务1', '任务2', '任务3', '任务4'],
        'task_type': ['SHELL', 'PYTHON', 'SQL', 'SHELL'],
        'start_time': ['2024-01-01 10:00:00', '2024-01-01 10:05:00', '2024-01-01 11:00:00', '2024-01-01 11:05:00'],
        'end_time': ['2024-01-01 10:05:00', '2024-01-01 10:10:00', '2024-01-01 11:05:00', '2024-01-01 11:10:00'],
        'state': [7, 7, 7, 7],
        'host': ['host1', 'host2', 'host1', 'host2'],
        'worker_group': ['default', 'default', 'default', 'default'],
        'task_instance_priority': [1, 2, 1, 2],
        'retry_times': [0, 0, 0, 0]
    })
    
    # 创建任务定义数据
    task_definitions = pd.DataFrame({
        'code': ['TASK_001', 'TASK_002', 'TASK_003', 'TASK_004'],
        'version': [1, 1, 1, 1],
        'name': ['任务1', '任务2', '任务3', '任务4'],
        'task_type': ['SHELL', 'PYTHON', 'SQL', 'SHELL']
    })
    
    # 创建进程任务关系数据
    process_task_relations = pd.DataFrame({
        'process_definition_code': ['WORKFLOW_001', 'WORKFLOW_001', 'WORKFLOW_002', 'WORKFLOW_002'],
        'process_definition_version': [1, 1, 1, 1],
        'pre_task_code': [None, 'TASK_001', None, 'TASK_003'],
        'post_task_code': ['TASK_001', 'TASK_002', 'TASK_003', 'TASK_004']
    })
    
    return {
        'process_instance': process_instances,
        'task_instance': task_instances,
        'task_definition': task_definitions,
        'process_task_relation': process_task_relations
    }

def debug_simulator():
    """调试模拟器"""
    print("=" * 60)
    print("调试历史重放模拟器")
    print("=" * 60)
    
    # 创建调试数据
    debug_data = create_debug_data()
    
    print("📊 调试数据概览:")
    for table_name, table_data in debug_data.items():
        print(f"  {table_name}: {len(table_data)} 行")
    print()
    
    # 创建模拟器
    simulator = HistoricalReplaySimulator(
        process_instances=debug_data['process_instance'],
        task_instances=debug_data['task_instance'],
        task_definitions=debug_data['task_definition'],
        process_task_relations=debug_data['process_task_relation']
    )
    
    print("🚀 模拟器初始化完成:")
    print(f"  成功进程数: {len(simulator.successful_processes)}")
    print(f"  当前进程索引: {simulator.current_process_idx}")
    print(f"  当前进程任务数: {len(simulator.current_process_tasks)}")
    print(f"  当前任务索引: {simulator.current_task_idx}")
    print(f"  可用资源: {list(simulator.available_resources.keys())}")
    print()
    
    # 测试多个step
    for step in range(10):
        print(f"--- Step {step + 1} ---")
        
        # 获取当前进程信息
        process_info = simulator.get_current_process_info()
        if process_info:
            print(f"  当前进程: {process_info['process_id']}")
            print(f"  已完成任务: {process_info['completed_tasks']}/{process_info['total_tasks']}")
        
        # 检查是否完成
        if simulator.is_done():
            print("  ✅ 所有任务已完成！")
            break
        
        # 检查当前任务
        if simulator.current_task_idx < len(simulator.current_process_tasks):
            current_task = simulator.current_process_tasks.iloc[simulator.current_task_idx]
            print(f"  当前任务: {current_task['name']} (ID: {current_task['id']})")
            print(f"  任务类型: {current_task['task_type']}")
            print(f"  主机: {current_task['host']}")
        else:
            print("  ⚠️  当前进程任务索引超出范围")
        
        # 检查资源状态
        print(f"  资源状态:")
        for host, resource in simulator.available_resources.items():
            print(f"    {host}: CPU {resource['cpu_used']:.1f}/{resource['cpu_capacity']:.1f}, "
                  f"Memory {resource['memory_used']:.1f}/{resource['memory_capacity']:.1f}")
        
        # 执行step
        action = 0  # 选择第一个资源
        print(f"  执行动作: 选择资源 {action}")
        
        try:
            next_state, reward, done, info = simulator.step(action)
            print(f"  执行结果:")
            print(f"    奖励: {reward}")
            print(f"    完成: {done}")
            print(f"    信息: {info}")
            
            # 检查状态变化
            print(f"    任务索引变化: {simulator.current_task_idx}")
            print(f"    进程索引变化: {simulator.current_process_idx}")
            print(f"    已完成任务: {len(simulator.completed_tasks)}")
            
        except Exception as e:
            print(f"  ❌ 执行step时出错: {e}")
            import traceback
            traceback.print_exc()
            break
        
        print()
    
    # 最终状态
    print("=" * 60)
    print("🏁 最终状态:")
    print(f"  总步数: {step + 1}")
    print(f"  已完成任务: {len(simulator.completed_tasks)}")
    print(f"  当前进程索引: {simulator.current_process_idx}")
    print(f"  当前任务索引: {simulator.current_task_idx}")
    print(f"  是否完成: {simulator.is_done()}")
    print("=" * 60)

if __name__ == "__main__":
    debug_simulator()
