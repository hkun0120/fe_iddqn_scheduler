#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析工作流实例 292677，测试算法
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from data.data_loader import DataLoader
try:
    from data.data_preprocessor import DataPreprocessor
except:
    pass
try:
    from environment.workflow_simulator import WorkflowSimulator
except:
    pass
try:
    from models.fe_iddqn import FE_IDDQN
except:
    pass
from baselines.traditional_schedulers import FIFOScheduler, SJFScheduler, HEFTScheduler

def analyze_workflow_instance(instance_id=292677):
    """分析特定工作流实例"""
    
    print(f"\n{'='*60}")
    print(f"分析工作流实例: {instance_id}")
    print(f"{'='*60}\n")
    
    # 加载数据
    data_loader = DataLoader('./data/raw_data')
    all_data = data_loader.load_all_data()
    
    # 找到对应的实例
    process_instance_df = pd.read_csv('./data/Commercial_B_t_ds_process_instance.csv')
    
    # 查找实例信息
    instance = process_instance_df[process_instance_df['id'] == instance_id]
    if instance.empty:
        print(f"工作流实例 {instance_id} 不存在！")
        return None
    
    instance_info = instance.iloc[0]
    print(f"工作流名称: {instance_info['name']}")
    print(f"工作流状态: {instance_info['state']}")
    print(f"开始时间: {instance_info['start_time']}")
    print(f"结束时间: {instance_info['end_time']}")
    
    # 计算任务数和实际执行时间
    task_instance_df = pd.read_csv('./data/Commercial_B_t_ds_task_instance.csv', 
                                   usecols=['id', 'name', 'process_instance_id', 'start_time', 'end_time', 'state', 'task_code'])
    
    workflow_tasks = task_instance_df[task_instance_df['process_instance_id'] == instance_id].copy()
    print(f"\n工作流任务数: {len(workflow_tasks)}")
    
    if len(workflow_tasks) == 0:
        print("该实例没有任务记录！")
        return None
    
    # 显示任务信息
    print("\n任务列表:")
    print("-" * 80)
    for idx, task in workflow_tasks.iterrows():
        start = pd.to_datetime(task['start_time'])
        end = pd.to_datetime(task['end_time'])
        duration = (end - start).total_seconds()
        print(f"  {task['name']}: {duration:.2f}s")
    
    # 计算关键指标
    from datetime import datetime
    start_time = pd.to_datetime(instance_info['start_time'])
    end_time = pd.to_datetime(instance_info['end_time'])
    total_time = (end_time - start_time).total_seconds()
    
    task_times = []
    for idx, task in workflow_tasks.iterrows():
        t_start = pd.to_datetime(task['start_time'])
        t_end = pd.to_datetime(task['end_time'])
        duration = (t_end - t_start).total_seconds()
        task_times.append(duration)
    
    total_work = sum(task_times)
    num_tasks = len(workflow_tasks)
    
    print(f"\n关键指标:")
    print(f"  总执行时间: {total_time:.2f}s")
    print(f"  总工作量: {total_work:.2f}s")
    print(f"  任务数: {num_tasks}")
    print(f"  平均任务时长: {total_work/num_tasks:.2f}s")
    print(f"  串行度: {total_time/total_work:.2f} (理论最小值=1)")
    
    # 加载任务依赖关系
    process_task_relation_df = all_data['process_task_relation']
    process_def_code = instance_info['process_definition_code']
    process_def_version = instance_info['process_definition_version']
    
    relations = process_task_relation_df[
        (process_task_relation_df['process_definition_code'] == process_def_code) &
        (process_task_relation_df['process_definition_version'] == process_def_version)
    ]
    
    print(f"\n任务依赖关系:")
    if len(relations) == 0:
        print("  没有依赖关系（所有任务可并行执行）")
    else:
        print(f"  依赖数: {len(relations)}")
        for idx, rel in relations.head(10).iterrows():
            print(f"  {rel['pre_task_code']} -> {rel['post_task_code']}")
        if len(relations) > 10:
            print(f"  ... 共 {len(relations)} 个依赖关系")
    
    return {
        'instance_id': instance_id,
        'name': instance_info['name'],
        'total_time': total_time,
        'total_work': total_work,
        'num_tasks': num_tasks,
        'task_times': task_times,
        'tasks': workflow_tasks,
        'relations': relations,
        'process_def_code': process_def_code,
        'process_def_version': process_def_version
    }

def test_algorithm_on_workflow(workflow_info):
    """用算法测试工作流"""
    
    if workflow_info is None:
        return
    
    print(f"\n{'='*60}")
    print(f"算法测试")
    print(f"{'='*60}\n")
    
    instance_id = workflow_info['instance_id']
    total_time = workflow_info['total_time']
    
    # 模拟环境配置
    num_workers = 2  # 2个Worker
    task_times = workflow_info['task_times']
    
    # 创建模拟环境
    from environment.workflow_simulator import WorkflowSimulator
    
    print(f"模拟环境配置:")
    print(f"  Worker数: {num_workers}")
    print(f"  任务数: {len(task_times)}")
    print(f"  实际执行时间: {total_time:.2f}s\n")
    
    # 测试不同的调度策略
    try:
        schedulers = {
            'FIFO': FIFOScheduler(),
            'SJF': SJFScheduler(),
            'HEFT': HEFTScheduler(),
        }
    except:
        schedulers = {}
    
    if not schedulers:
        print("调度器加载失败，跳过调度器测试\n")
        return
    
    print(f"{'策略':<15} {'执行时间':<12} {'改进比例':<12}")
    print("-" * 40)
    
    for strategy_name, scheduler in schedulers.items():
        try:
            # 创建虚拟任务列表
            tasks = [{'id': i, 'duration': task_times[i]} for i in range(len(task_times))]
            
            # 使用简单的模拟调度
            # 没有依赖的情况下，按LPT（最长任务优先）分配
            sorted_tasks = sorted(enumerate(task_times), key=lambda x: x[1], reverse=True)
            schedule = {}
            worker_times = [0] * num_workers
            
            for idx, (task_id, duration) in enumerate(sorted_tasks):
                # 分配给负载最轻的worker
                min_worker = np.argmin(worker_times)
                schedule[task_id] = min_worker
                worker_times[min_worker] += duration
            
            completion_time = max(worker_times)
            improvement = (total_time - completion_time) / total_time * 100
            
            print(f"{strategy_name:<15} {completion_time:<12.2f} {improvement:<12.2f}%")
        except Exception as e:
            print(f"{strategy_name:<15} 计算失败: {e}")
    
    
    print(f"\n{'实际执行':<15} {total_time:<12.2f} {'0.00':<12}%")
    
    # 尝试加载和测试RL模型
    try:
        print(f"\n{'='*60}")
        print(f"尝试测试RL模型")
        print(f"{'='*60}\n")
        
        # 检查是否有保存的模型
        checkpoint_dir = './checkpoints'
        if os.path.exists(checkpoint_dir):
            checkpoints = os.listdir(checkpoint_dir)
            if checkpoints:
                latest_checkpoint = sorted(checkpoints)[-1]
                print(f"发现检查点: {latest_checkpoint}")
                print("（需要完整的模型配置和数据来测试）")
    except Exception as e:
        print(f"RL模型测试暂跳过: {e}")

if __name__ == '__main__':
    # 分析工作流292677
    workflow_info = analyze_workflow_instance(292677)
    
    # 测试算法
    if workflow_info:
        test_algorithm_on_workflow(workflow_info)
