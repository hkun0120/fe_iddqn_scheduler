#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入分析真实数据的DAG结构 - 修正版
"""

import pandas as pd
import numpy as np
from collections import defaultdict

print('=== 加载真实数据 ===')

# 加载数据
process_df = pd.read_csv('data/__B_t_ds_process_instance.csv')
task_df = pd.read_csv('data/__B_t_ds_task_instance.csv')
relation_df = pd.read_csv('data/__B_t_ds_process_task_relation.csv')

print(f'工作流实例: {len(process_df)} 条')
print(f'任务实例: {len(task_df)} 条')
print(f'依赖关系: {len(relation_df)} 条')

# 查看任务实例表的列
print(f'\n任务实例列: {list(task_df.columns)}')

print()

# 分析依赖关系中的pre_task_code
print('=== 依赖关系分析 ===')
print(f'pre_task_code为0的关系数: {(relation_df["pre_task_code"] == 0).sum()} (表示入口任务)')
print(f'pre_task_code不为0的关系数: {(relation_df["pre_task_code"] != 0).sum()} (表示有前置任务)')

print()
print('=== 分析具体工作流的DAG结构 ===')

# 选择成功的工作流
successful_processes = process_df[process_df['state'] == 7]

def analyze_workflow(process_id, process_code, verbose=True):
    """分析一个工作流的DAG结构"""
    
    # 获取该工作流的所有任务实例
    workflow_tasks = task_df[task_df['process_instance_id'] == process_id].copy()
    
    # 获取该工作流的依赖关系 - 通过process_definition_code匹配
    relations = relation_df[relation_df['process_definition_code'] == process_code]
    
    if verbose:
        print(f'  任务实例数: {len(workflow_tasks)}')
        print(f'  依赖关系数: {len(relations)}')
    
    if len(relations) == 0:
        if verbose:
            print('  ⚠️ 无依赖关系定义')
        return None
    
    # 建立task_code到task_name的映射 (从关系表中提取name列，如果有的话)
    code_to_name = {}
    if 'name' in relations.columns:
        # relation表中的name字段标识的是这个关系的来源任务
        pass  # 这个name可能不是任务名，而是关系名
    
    # 从任务实例表获取任务名
    task_code_to_info = {}
    if 'task_code' in workflow_tasks.columns:
        for _, t in workflow_tasks.iterrows():
            task_code_to_info[t['task_code']] = {
                'name': t['name'],
                'type': t.get('task_type', 'UNKNOWN'),
                'state': t['state']
            }
    
    # 构建邻接表
    graph = defaultdict(list)  # pre -> [post]
    reverse_graph = defaultdict(list)  # post -> [pre]
    all_nodes = set()
    
    for _, r in relations.iterrows():
        pre = r['pre_task_code']
        post = r['post_task_code']
        if post != 0:  # 确保post有效
            all_nodes.add(post)
            if pre != 0:  # 0表示没有前置任务（入口任务）
                all_nodes.add(pre)
                graph[pre].append(post)
                reverse_graph[post].append(pre)
    
    # 找入口任务（没有前置的任务，即pre_task_code=0的记录中的post_task_code）
    entry_tasks = set()
    for _, r in relations.iterrows():
        if r['pre_task_code'] == 0:
            entry_tasks.add(r['post_task_code'])
    
    # 找出口任务（没有后续的任务）
    all_posts = set(relations['post_task_code'])
    all_pres = set(relations['pre_task_code']) - {0}
    exit_tasks = all_posts - all_pres
    
    # 如果入口任务和所有任务相等，可能是并行无依赖的工作流
    actual_nodes = all_nodes
    
    if verbose:
        print(f'  总节点数: {len(actual_nodes)}')
        print(f'  入口任务数: {len(entry_tasks)} (无前置依赖)')
        print(f'  出口任务数: {len(exit_tasks)} (无后续任务)')
        print(f'  实际有依赖边的节点: {len(all_pres | all_posts)}')
        
        # 显示一些入口任务名
        if len(entry_tasks) > 0:
            entry_names = []
            for tc in list(entry_tasks)[:5]:
                if tc in task_code_to_info:
                    entry_names.append(f"{task_code_to_info[tc]['name']}")
                else:
                    entry_names.append(f"code={tc}")
            print(f'  入口任务示例: {entry_names}')
        
        # 打印几条边
        print(f'  依赖边示例:')
        edge_count = 0
        for pre, posts in list(graph.items())[:3]:
            for post in posts[:2]:
                pre_name = task_code_to_info.get(pre, {}).get('name', f'code={pre}')
                post_name = task_code_to_info.get(post, {}).get('name', f'code={post}')
                print(f'    {pre_name[:20]} -> {post_name[:20]}')
                edge_count += 1
                if edge_count >= 3:
                    break
            if edge_count >= 3:
                break
    
    return {
        'task_count': len(workflow_tasks),
        'relation_count': len(relations),
        'entry_tasks': entry_tasks,
        'exit_tasks': exit_tasks,
        'graph': graph,
        'reverse_graph': reverse_graph,
        'node_count': len(actual_nodes)
    }

# 分析几个工作流
sample_count = 0
workflow_stats = []

for idx, row in successful_processes.iterrows():
    process_id = row['id']
    process_code = row['process_definition_code']
    
    # 该工作流的任务
    tasks = task_df[task_df['process_instance_id'] == process_id]
    
    if len(tasks) >= 10:  # 只看有一定复杂度的工作流
        print(f'\n--- 工作流 {process_id} ({row["name"][:40]}...) ---')
        result = analyze_workflow(process_id, process_code)
        if result:
            workflow_stats.append(result)
        sample_count += 1
        if sample_count >= 3:
            break

print()
print('=== 任务实例的执行时间分析 ===')

# 计算执行时间
if 'start_time' in task_df.columns and 'end_time' in task_df.columns:
    task_df['start_time'] = pd.to_datetime(task_df['start_time'], errors='coerce')
    task_df['end_time'] = pd.to_datetime(task_df['end_time'], errors='coerce')
    
    # 只看成功的任务
    successful_tasks = task_df[task_df['state'] == 7].copy()
    successful_tasks['duration'] = (successful_tasks['end_time'] - successful_tasks['start_time']).dt.total_seconds()
    
    # 过滤有效时长
    valid_tasks = successful_tasks[(successful_tasks['duration'] > 0) & (successful_tasks['duration'] < 100000)]
    
    print(f'\n成功任务数: {len(successful_tasks)}')
    print(f'有效执行时间的任务数: {len(valid_tasks)}')
    print(f'\n执行时间统计 (秒):')
    print(f'  最小: {valid_tasks["duration"].min():.2f}')
    print(f'  最大: {valid_tasks["duration"].max():.2f}')
    print(f'  平均: {valid_tasks["duration"].mean():.2f}')
    print(f'  中位数: {valid_tasks["duration"].median():.2f}')
    
    print(f'\n按任务类型的平均执行时间:')
    type_duration = valid_tasks.groupby('task_type')['duration'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    for task_type, row in type_duration.iterrows():
        print(f'  {task_type}: 平均{row["mean"]:.2f}秒 (样本数: {int(row["count"])})')

print()
print('=== 资源/执行器信息分析 ===')

# 检查是否有worker/资源信息
if 'worker_group' in task_df.columns:
    print(f'Worker组分布:')
    print(task_df['worker_group'].value_counts().head(5))

if 'executor_id' in task_df.columns:
    print(f'\n执行器分布:')
    print(task_df['executor_id'].value_counts().head(5))

if 'host' in task_df.columns:
    print(f'\n执行主机分布:')
    hosts = task_df['host'].value_counts().head(10)
    for host, count in hosts.items():
        print(f'  {host}: {count}')
