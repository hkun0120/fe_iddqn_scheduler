#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试workflow_simulator的step方法
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.workflow_simulator import WorkflowSimulator
import numpy as np

def test_step_method():
    """测试step方法"""
    print("=" * 60)
    print("测试 WorkflowSimulator 的 step 方法")
    print("=" * 60)
    
    # 创建简单的测试任务
    tasks = [
        {
            'id': 1,
            'name': 'Task1',
            'duration': 10.0,
            'cpu_req': 2.0,
            'memory_req': 4.0,
            'submit_time': 0.0
        },
        {
            'id': 2,
            'name': 'Task2',
            'duration': 15.0,
            'cpu_req': 1.0,
            'memory_req': 2.0,
            'submit_time': 0.0
        },
        {
            'id': 3,
            'name': 'Task3',
            'duration': 8.0,
            'cpu_req': 3.0,
            'memory_req': 6.0,
            'submit_time': 0.0
        }
    ]
    
    # 创建资源
    resources = [
        {
            'id': 1,
            'cpu_capacity': 4.0,
            'memory_capacity': 8.0
        },
        {
            'id': 2,
            'cpu_capacity': 2.0,
            'memory_capacity': 4.0
        }
    ]
    
    # 创建依赖关系（Task1 -> Task2, Task1 -> Task3）
    dependencies = [
        {'pre_task': 1, 'post_task': 2},
        {'pre_task': 1, 'post_task': 3}
    ]
    
    # 创建模拟器
    simulator = WorkflowSimulator(tasks, resources, dependencies)
    
    print(f"初始状态:")
    print(f"  任务总数: {len(tasks)}")
    print(f"  资源总数: {len(resources)}")
    print(f"  依赖关系: {dependencies}")
    print(f"  可调度任务: {simulator.ready_tasks}")
    print(f"  已完成任务: {simulator.completed_tasks}")
    print(f"  当前时间: {simulator.current_time}")
    print()
    
    # 测试多个step
    for step in range(5):
        print(f"--- Step {step + 1} ---")
        
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
            print("  所有任务已完成！")
            break
            
        # 如果没有可调度的任务，等待
        if not simulator.ready_tasks:
            print("  没有可调度的任务，等待...")
            # 更新当前时间到下一个任务完成时间
            if simulator.task_end_times:
                next_completion = min(simulator.task_end_times.values())
                simulator.current_time = next_completion
                simulator.ready_tasks = simulator._get_ready_tasks()
                print(f"  更新当前时间到: {simulator.current_time}")
                print(f"  新的可调度任务: {simulator.ready_tasks}")
        
        # 执行step
        if simulator.ready_tasks:
            action = 0  # 选择第一个资源
            print(f"  执行动作: 选择资源 {action}")
            
            next_state, reward, done, info = simulator.step(action)
            next_task_features, next_resource_features = next_state
            
            print(f"  执行结果:")
            print(f"    奖励: {reward}")
            print(f"    完成: {done}")
            print(f"    信息: {info}")
            print(f"    下一步状态:")
            print(f"      任务特征形状: {next_task_features.shape}")
            print(f"      资源特征形状: {next_resource_features.shape}")
            
            # 检查任务分配
            print(f"    任务分配: {simulator.task_assignments}")
            print(f"    任务开始时间: {simulator.task_start_times}")
            print(f"    任务结束时间: {simulator.task_end_times}")
            print(f"    资源可用时间: {simulator.resource_available_time}")
        else:
            print("  仍然没有可调度的任务")
        
        print()
    
    # 最终状态
    print("=" * 60)
    print("最终状态:")
    print(f"  已完成任务: {simulator.completed_tasks}")
    print(f"  任务分配: {simulator.task_assignments}")
    print(f"  Makespan: {simulator.get_makespan()}")
    print(f"  资源利用率: {simulator.get_resource_utilization()}")
    print("=" * 60)

if __name__ == "__main__":
    test_step_method()
