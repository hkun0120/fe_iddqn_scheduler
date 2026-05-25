#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入分析真实数据的DAG结构
"""

import pandas as pd
import numpy as np
from collections import defaultdict

print('=== 加载真实数据 ===')

# 加载数据
process_df = pd.read_csv('data/__B_t_ds_process_instance.csv')
task_df = pd.read_csv('data/__B_t_ds_task_instance.csv')
relation_df = pd.read_csv('data/__B_t_ds_process_task_relation.csv')

# 加载任务定义，获取任务code
task_definition_df = pd.read_csv('data/__B_t_ds_task_definition.csv')
print(f'任务定义: {len(task_definition_df)} 条')
print(f'任务定义列: {list(task_definition_df.columns)}')

print()

# 分析依赖关系中的pre_task_code
print('=== 依赖关系分析 ===')
print(f'pre_task_code为0的关系数: {(relation_df["pre_task_code"] == 0).sum()}')
print(f'pre_task_code不为0的关系数: {(relation_df["pre_task_code"] != 0).sum()}')

print()

# 找一个具体工作流来分析完整的DAG结构
print('=== 分析具体工作流的DAG结构 ===')

# 选择一个成功的工作流
successful_processes = process_df[process_df['state'] == 7]

def analyze_workflow(process_id, process_code, verbose=True):
    """分析一个工作流的DAG结构"""
    
    # 获取该工作流的所有任务实例
    workflow_tasks = task_df[task_df['process_instance_id'] == process_id].copy()
    
    # 获取该工作流的任务定义
    task_defs = task_definition_df[task_definition_df['process_definition_code'] == process_code]
    
    # 获取该工作流的依赖关系
    relations = relation_df[relation_df['process_definition_code'] == process_code]
    
    if verbose:
        print(f'  任务实例数: {len(workflow_tasks)}')
        print(f'  任务定义数: {len(task_defs)}')
        print(f'  依赖关系数: {len(relations)}')
    
    # 建立task_code到task_name的映射
    code_to_name = dict(zip(task_defs['code'], task_defs['name']))
    
    # 分析DAG结构
    if len(relations) == 0:
        if verbose:
            print('  ⚠️ 无依赖关系，所有任务独立')
        return None
    
    # 构建邻接表
    graph = defaultdict(list)  # pre -> [post]
    reverse_graph = defaultdict(list)  # post -> [pre]
    all_nodes = set()
    
    for _, r in relations.iterrows():
        pre = r['pre_task_code']
        post = r['post_task_code']
        all_nodes.add(pre)
        all_nodes.add(post)
        if pre != 0:  # 0表示没有前置任务
            graph[pre].append(post)
            reverse_graph[post].append(pre)
    
    # 找入口任务（没有前置的任务）
    # pre_task_code=0的记录表示该任务是入口任务
    entry_tasks = set()
    for _, r in relations.iterrows():
        if r['pre_task_code'] == 0:
            entry_tasks.add(r['post_task_code'])
    
    # 找出口任务（没有后续的任务）
    all_posts = set(relations['post_task_code'])
    all_pres = set(relations['pre_task_code']) - {0}
    exit_tasks = all_posts - all_pres
    
    if verbose:
        print(f'  入口任务数: {len(entry_tasks)}')
        print(f'  出口任务数: {len(exit_tasks)}')
        print(f'  总节点数: {len(all_nodes) - (1 if 0 in all_nodes else 0)}')
        
        if len(entry_tasks) > 0:
            print(f'  入口任务: {[code_to_name.get(t, t) for t in list(entry_tasks)[:5]]}...')
        if len(exit_tasks) > 0:
            print(f'  出口任务: {[code_to_name.get(t, t) for t in list(exit_tasks)[:5]]}...')
    
    return {
        'task_count': len(workflow_tasks),
        'relation_count': len(relations),
        'entry_tasks': entry_tasks,
        'exit_tasks': exit_tasks,
        'graph': graph,
        'reverse_graph': reverse_graph,
        'code_to_name': code_to_name
    }

# 分析几个工作流
sample_count = 0
for idx, row in successful_processes.iterrows():
    process_id = row['id']
    process_code = row['process_definition_code']
    
    # 该工作流的任务
    tasks = task_df[task_df['process_instance_id'] == process_id]
    
    if len(tasks) >= 10:  # 只看有一定复杂度的工作流
        print(f'\n--- 工作流 {process_id} ({row["name"][:40]}...) ---')
        result = analyze_workflow(process_id, process_code)
        sample_count += 1
        if sample_count >= 3:
            break

print()
print('=== 任务实例的执行时间分析 ===')

# 检查任务有没有执行时间信息
print(f'任务实例列: {list(task_df.columns)}')

# 计算执行时间（如果有start_time和end_time）
if 'start_time' in task_df.columns and 'end_time' in task_df.columns:
    task_df['start_time'] = pd.to_datetime(task_df['start_time'])
    task_df['end_time'] = pd.to_datetime(task_df['end_time'])
    
    # 只看成功的任务
    successful_tasks = task_df[task_df['state'] == 7].copy()
    successful_tasks['duration'] = (successful_tasks['end_time'] - successful_tasks['start_time']).dt.total_seconds()
    
    # 过滤有效时长
    valid_tasks = successful_tasks[successful_tasks['duration'] > 0]
    
    print(f'\n成功任务数: {len(successful_tasks)}')
    print(f'有效执行时间的任务数: {len(valid_tasks)}')
    print(f'\n执行时间统计 (秒):')
    print(f'  最小: {valid_tasks["duration"].min():.2f}')
    print(f'  最大: {valid_tasks["duration"].max():.2f}')
    print(f'  平均: {valid_tasks["duration"].mean():.2f}')
    print(f'  中位数: {valid_tasks["duration"].median():.2f}')
    
    print(f'\n按任务类型的平均执行时间:')
    type_duration = valid_tasks.groupby('task_type')['duration'].mean().sort_values(ascending=False)
    for task_type, duration in type_duration.items():
        print(f'  {task_type}: {duration:.2f}秒')

print()
print('=== 资源信息分析 ===')

# 检查是否有worker/资源信息
if 'worker_group' in task_df.columns:
    print(f'Worker组分布:')
    print(task_df['worker_group'].value_counts().head(10))

if 'executor_id' in task_df.columns:
    print(f'\n执行器分布:')
    print(task_df['executor_id'].value_counts().head(10))

if 'host' in task_df.columns:
    print(f'\n执行主机分布:')
    print(task_df['host'].value_counts().head(10))
