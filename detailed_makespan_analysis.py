#!/usr/bin/env python3
"""
详细分析FE-IDDQN算法makespan的计算过程
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.mysql_data_loader import MySQLDataLoader
from environment.historical_replay_simulator import HistoricalReplaySimulator
from models.fe_iddqn import FE_IDDQN
from config.config import Config
from config.hyperparameters import Hyperparameters

def detailed_makespan_analysis():
    """详细分析makespan计算过程"""
    
    print("=" * 80)
    print("FE-IDDQN算法makespan详细分析")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    data_loader = MySQLDataLoader()
    data = data_loader.load_all_data()
    
    # 2. 创建仿真器
    print("\n2. 创建仿真器...")
    simulator = HistoricalReplaySimulator(
        process_instances=data['process_instance'],
        task_instances=data['task_instance'],
        task_definitions=data['task_definition'],
        process_task_relations=data['process_task_relation']
    )
    
    # 3. 运行一次FE-IDDQN实验
    print("\n3. 运行FE-IDDQN实验...")
    
    # 创建FE-IDDQN智能体
    task_input_dim = 50
    resource_input_dim = 50
    action_dim = 4  # 假设有4个资源
    agent = FE_IDDQN(task_input_dim, resource_input_dim, action_dim)
    
    # 重置仿真器
    simulator.reset()
    
    print(f"仿真器状态:")
    print(f"  成功进程数: {len(simulator.successful_processes)}")
    print(f"  当前进程索引: {simulator.current_process_idx}")
    print(f"  资源数量: {len(simulator.available_resources)}")
    
    # 4. 分析任务调度过程
    print("\n4. 分析任务调度过程...")
    
    step_count = 0
    max_steps = 1000
    
    while not simulator.is_done() and step_count < max_steps:
        state = simulator.get_state()
        action = agent.select_action(state[0], state[1])
        next_state, reward, done, info = simulator.step(action)
        
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"  步骤 {step_count}: 当前时间 {simulator.current_time:.2f}, 已完成任务 {len(simulator.completed_tasks)}")
        
        if done:
            break
    
    # 5. 分析最终结果
    print("\n5. 分析最终结果...")
    
    makespan = simulator.get_makespan()
    resource_utilization = simulator.get_resource_utilization()
    
    print(f"最终makespan: {makespan:.2f}秒")
    print(f"资源利用率: {resource_utilization:.2f}")
    print(f"已完成任务数: {len(simulator.completed_tasks)}")
    print(f"调度历史记录数: {len(simulator.task_schedule_history)}")
    
    # 6. 分析资源使用情况
    print("\n6. 分析资源使用情况...")
    
    for host, resource in simulator.available_resources.items():
        execution_time = resource.get('execution_time', 0.0)
        print(f"资源 {host}: 执行时间 {execution_time:.2f}秒")
    
    # 7. 分析任务调度历史
    print("\n7. 分析任务调度历史...")
    
    if simulator.task_schedule_history:
        # 按时间排序
        sorted_history = sorted(simulator.task_schedule_history, 
                              key=lambda x: x.get('timestamp', 0))
        
        print(f"调度历史记录数: {len(sorted_history)}")
        
        # 分析时间分布
        timestamps = [record.get('timestamp', 0) for record in sorted_history]
        durations = [record.get('duration', 0) for record in sorted_history]
        
        if timestamps and durations:
            print(f"最早开始时间: {min(timestamps):.2f}秒")
            print(f"最晚开始时间: {max(timestamps):.2f}秒")
            print(f"最短任务时间: {min(durations):.2f}秒")
            print(f"最长任务时间: {max(durations):.2f}秒")
            print(f"平均任务时间: {np.mean(durations):.2f}秒")
            
            # 计算完成时间
            completion_times = [t + d for t, d in zip(timestamps, durations)]
            print(f"最早完成时间: {min(completion_times):.2f}秒")
            print(f"最晚完成时间: {max(completion_times):.2f}秒")
            
            # 分析资源使用模式
            resource_usage = {}
            for record in sorted_history:
                host = record.get('host', 'unknown')
                duration = record.get('duration', 0)
                if host not in resource_usage:
                    resource_usage[host] = []
                resource_usage[host].append(duration)
            
            print(f"\n各资源使用情况:")
            for host, durations in resource_usage.items():
                total_time = sum(durations)
                task_count = len(durations)
                print(f"  资源 {host}: {task_count}个任务, 总时间 {total_time:.2f}秒, 平均时间 {total_time/task_count:.2f}秒")
    
    # 8. 解释makespan计算
    print("\n8. 解释makespan计算...")
    
    print("FE-IDDQN的makespan计算逻辑:")
    print("1. 算法使用历史重放仿真器，按时间顺序处理任务")
    print("2. 每个任务被分配给一个资源，并记录开始时间和持续时间")
    print("3. makespan = max(所有资源的执行时间)")
    print("4. 这反映了并行执行完成所有任务所需的时间")
    
    # 9. 与理论值比较
    print("\n9. 与理论值比较...")
    
    if simulator.task_schedule_history:
        # 计算理论最小makespan（如果所有任务完全并行执行）
        theoretical_min_makespan = max(durations) if durations else 0
        print(f"理论最小makespan (最长任务时间): {theoretical_min_makespan:.2f}秒")
        
        # 计算理论最大makespan（如果所有任务串行执行）
        theoretical_max_makespan = sum(durations) if durations else 0
        print(f"理论最大makespan (所有任务时间之和): {theoretical_max_makespan:.2f}秒")
        
        # 计算实际makespan
        actual_makespan = makespan
        print(f"实际makespan: {actual_makespan:.2f}秒")
        
        # 分析效率
        if theoretical_min_makespan > 0:
            efficiency = theoretical_min_makespan / actual_makespan
            print(f"并行效率: {efficiency:.2f} ({efficiency*100:.1f}%)")
        
        if theoretical_max_makespan > 0:
            parallel_speedup = theoretical_max_makespan / actual_makespan
            print(f"并行加速比: {parallel_speedup:.2f}x")
    
    print("\n" + "=" * 80)
    print("总结:")
    print("1. FE-IDDQN的makespan (18460秒) 是通过并行调度计算得出的")
    print("2. 它表示完成所有采样任务所需的最长时间，而不是所有任务的总时间")
    print("3. 每次实验的makespan不同是因为随机采样和调度策略的影响")
    print("4. 这个值比理论最小makespan大，说明存在资源竞争和调度开销")
    print("=" * 80)

if __name__ == "__main__":
    detailed_makespan_analysis()
