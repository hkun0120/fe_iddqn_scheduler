#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用模拟数据测试workflow_simulator的step方法
模拟真实的数据结构和场景
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.workflow_simulator import WorkflowSimulator
import numpy as np
import pandas as pd

def create_mock_data():
    """创建模拟的真实数据"""
    
    # 模拟任务定义数据
    task_definitions = pd.DataFrame({
        'code': ['TASK_001', 'TASK_002', 'TASK_003', 'TASK_004', 'TASK_005'],
        'version': [1, 1, 1, 1, 1],
        'name': ['数据预处理', '特征提取', '模型训练', '结果评估', '报告生成'],
        'task_type': ['python', 'python', 'python', 'python', 'python'],
        'cpu_req': [2.0, 4.0, 8.0, 2.0, 1.0],
        'memory_req': [4.0, 8.0, 16.0, 4.0, 2.0],
        'estimated_duration': [300, 600, 1200, 300, 150]
    })
    
    # 模拟任务实例数据
    task_instances = pd.DataFrame({
        'id_instance': [1, 2, 3, 4, 5],
        'task_code': ['TASK_001', 'TASK_002', 'TASK_003', 'TASK_004', 'TASK_005'],
        'task_definition_version': [1, 1, 1, 1, 1],
        'process_instance_id': [1, 1, 1, 1, 1],
        'submit_time': ['2024-01-01 10:00:00'] * 5,
        'start_time': ['2024-01-01 10:00:00', '2024-01-01 10:05:00', '2024-01-01 10:15:00', '2024-01-01 10:35:00', '2024-01-01 10:40:00'],
        'end_time': ['2024-01-01 10:05:00', '2024-01-01 10:15:00', '2024-01-01 10:35:00', '2024-01-01 10:40:00', '2024-01-01 10:42:30'],
        'state': [7, 7, 7, 7, 7],  # 7表示成功
        'host': ['worker-01', 'worker-02', 'worker-03', 'worker-01', 'worker-02'],
        'worker_group': ['default', 'default', 'gpu', 'default', 'default'],
        'task_instance_priority': [1, 2, 1, 3, 2],
        'retry_times': [0, 0, 0, 0, 0]
    })
    
    # 模拟进程任务关系数据
    process_task_relations = pd.DataFrame({
        'process_definition_code': ['WORKFLOW_001'] * 4,
        'process_definition_version': [1] * 4,
        'pre_task_code': [None, 'TASK_001', 'TASK_002', 'TASK_003'],
        'post_task_code': ['TASK_001', 'TASK_002', 'TASK_003', 'TASK_004']
    })
    
    return {
        'task_definition': task_definitions,
        'task_instance': task_instances,
        'process_task_relation': process_task_relations
    }

def test_step_with_mock_data():
    """使用模拟数据测试step方法"""
    print("=" * 60)
    print("使用模拟数据测试 WorkflowSimulator 的 step 方法")
    print("=" * 60)
    
    # 创建模拟数据
    mock_data = create_mock_data()
    
    print("📊 模拟数据概览:")
    for table_name, table_data in mock_data.items():
        print(f"  {table_name}: {len(table_data)} 行")
    print()
    
    # 从模拟数据构建任务和依赖关系
    tasks = []
    dependencies = []
    
    # 构建任务列表
    for _, task_def in mock_data['task_definition'].iterrows():
        task_code = task_def['code']
        task_instance = mock_data['task_instance'][mock_data['task_instance']['task_code'] == task_code].iloc[0]
        
        # 计算实际持续时间（秒）
        start_time = pd.to_datetime(task_instance['start_time'])
        end_time = pd.to_datetime(task_instance['end_time'])
        duration = (end_time - start_time).total_seconds()
        
        tasks.append({
            'id': task_def['code'],
            'name': task_def['name'],
            'task_type': task_def['task_type'],
            'duration': duration,
            'cpu_req': task_def['cpu_req'],
            'memory_req': task_def['memory_req'],
            'submit_time': 0.0
        })
    
    # 构建依赖关系
    for _, relation in mock_data['process_task_relation'].iterrows():
        if relation['pre_task_code'] is not None:
            dependencies.append({
                'pre_task': relation['pre_task_code'],
                'post_task': relation['post_task_code']
            })
    
    print("🔧 构建的任务:")
    for task in tasks:
        print(f"  {task['id']}: {task['name']} (CPU: {task['cpu_req']}, Memory: {task['memory_req']}, Duration: {task['duration']:.1f}s)")
    
    print(f"\n🔗 依赖关系: {dependencies}")
    
    # 创建资源（基于模拟数据中的主机信息）
    resources = [
        {
            'id': 1,
            'cpu_capacity': 8.0,
            'memory_capacity': 16.0
        },
        {
            'id': 2,
            'cpu_capacity': 4.0,
            'memory_capacity': 8.0
        },
        {
            'id': 3,
            'cpu_capacity': 16.0,
            'memory_capacity': 32.0  # GPU节点
        }
    ]
    
    print(f"\n💻 资源配置:")
    for resource in resources:
        print(f"  资源{resource['id']}: CPU {resource['cpu_capacity']}, Memory {resource['memory_capacity']}")
    
    # 创建模拟器
    simulator = WorkflowSimulator(tasks, resources, dependencies)
    
    print(f"\n🚀 模拟器初始化完成:")
    print(f"  任务总数: {len(tasks)}")
    print(f"  资源总数: {len(resources)}")
    print(f"  依赖关系数: {len(dependencies)}")
    print(f"  初始可调度任务: {simulator.ready_tasks}")
    print(f"  当前时间: {simulator.current_time}")
    print()
    
    # 测试调度过程
    step_count = 0
    max_steps = 10
    
    while not simulator.is_done() and step_count < max_steps:
        step_count += 1
        print(f"--- Step {step_count} ---")
        
        # 获取当前状态
        current_state = simulator.get_state()
        task_features, resource_features = current_state
        
        print(f"  当前状态:")
        print(f"    任务特征形状: {task_features.shape}")
        print(f"    资源特征形状: {resource_features.shape}")
        print(f"    可调度任务: {simulator.ready_tasks}")
        print(f"    已完成任务: {simulator.completed_tasks}")
        print(f"    当前时间: {simulator.current_time}")
        
        # 检查是否完成
        if simulator.is_done():
            print("  ✅ 所有任务已完成！")
            break
            
        # 如果没有可调度的任务，等待
        if not simulator.ready_tasks:
            print("  ⏳ 没有可调度的任务，等待...")
            if simulator.task_end_times:
                next_completion = min(simulator.task_end_times.values())
                simulator.current_time = next_completion
                simulator.ready_tasks = simulator._get_ready_tasks()
                print(f"    更新当前时间到: {simulator.current_time}")
                print(f"    新的可调度任务: {simulator.ready_tasks}")
        
        # 执行step
        if simulator.ready_tasks:
            # 选择资源（这里简化处理，实际应该由算法决定）
            action = 0  # 选择第一个资源
            print(f"  🎯 执行动作: 选择资源 {action}")
            
            next_state, reward, done, info = simulator.step(action)
            next_task_features, next_resource_features = next_state
            
            print(f"  📊 执行结果:")
            print(f"    奖励: {reward:.2f}")
            print(f"    完成: {done}")
            print(f"    信息: {info}")
            
            # 检查任务分配
            print(f"    任务分配: {simulator.task_assignments}")
            print(f"    任务开始时间: {simulator.task_start_times}")
            print(f"    任务结束时间: {simulator.task_end_times}")
            print(f"    资源可用时间: {simulator.resource_available_time}")
        else:
            print("  ❌ 仍然没有可调度的任务")
        
        print()
    
    # 最终状态
    print("=" * 60)
    print("🏁 最终状态:")
    print(f"  总步数: {step_count}")
    print(f"  已完成任务: {simulator.completed_tasks}")
    print(f"  任务分配: {simulator.task_assignments}")
    print(f"  Makespan: {simulator.get_makespan():.2f}")
    print(f"  资源利用率: {simulator.get_resource_utilization():.2f}")
    
    if simulator.is_done():
        print("  ✅ 所有任务成功完成！")
    else:
        print(f"  ⚠️  任务未完全完成，剩余任务: {set(task['id'] for task in tasks) - simulator.completed_tasks}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_step_with_mock_data()
