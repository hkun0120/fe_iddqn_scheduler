#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析真实数据结构
"""

import pandas as pd
import numpy as np

print('=== 加载真实数据 ===')

# 工作流实例
process_df = pd.read_csv('data/__B_t_ds_process_instance.csv')
print(f'工作流实例: {len(process_df)} 条')
print(f'列: {list(process_df.columns)[:10]}...')
print(f'状态分布: ')
print(process_df['state'].value_counts())

print()

# 任务实例
task_df = pd.read_csv('data/__B_t_ds_task_instance.csv')
print(f'任务实例: {len(task_df)} 条')
print(f'列: {list(task_df.columns)[:10]}...')
print(f'任务类型分布:')
print(task_df['task_type'].value_counts().head(10))

print()

# 任务依赖关系
relation_df = pd.read_csv('data/__B_t_ds_process_task_relation.csv')
print(f'任务依赖关系: {len(relation_df)} 条')
print(f'列: {list(relation_df.columns)}')

print()

# 找一个有任务和依赖的工作流
print('=== 寻找有完整依赖关系的工作流 ===')

# 成功的工作流 (state=7)
successful_processes = process_df[process_df['state'] == 7]
print(f'成功的工作流数: {len(successful_processes)}')

# 找一个有多个任务的工作流
for idx, row in successful_processes.head(50).iterrows():
    process_id = row['id']
    process_code = row['process_definition_code']
    
    # 该工作流的任务
    tasks = task_df[task_df['process_instance_id'] == process_id]
    
    # 该工作流的依赖关系
    relations = relation_df[relation_df['process_definition_code'] == process_code]
    
    if len(tasks) >= 5 and len(relations) >= 3:
        print(f'\n找到示例工作流:')
        print(f'  Process ID: {process_id}')
        print(f'  Process Code: {process_code}')
        process_name = row['name']
        print(f'  名称: {process_name[:50]}...' if len(str(process_name)) > 50 else f'  名称: {process_name}')
        print(f'  任务数: {len(tasks)}')
        print(f'  依赖关系数: {len(relations)}')
        
        print(f'\n  任务列表:')
        for _, t in tasks.iterrows():
            print(f'    - {t["name"][:30]}: {t["task_type"]}, 状态={t["state"]}')
        
        print(f'\n  依赖关系:')
        for _, r in relations.head(10).iterrows():
            pre = r['pre_task_code']
            post = r['post_task_code']
            print(f'    {pre} -> {post}')
        
        break

# 统计工作流任务数分布
print('\n=== 工作流任务数分布 ===')
task_counts = task_df.groupby('process_instance_id').size()
print(f'最小任务数: {task_counts.min()}')
print(f'最大任务数: {task_counts.max()}')
print(f'平均任务数: {task_counts.mean():.2f}')
print(f'中位数: {task_counts.median():.2f}')

print('\n任务数分布:')
bins = [1, 2, 5, 10, 20, 50, 100, 1000]
for i in range(len(bins)-1):
    count = ((task_counts >= bins[i]) & (task_counts < bins[i+1])).sum()
    print(f'  {bins[i]}-{bins[i+1]-1}个任务: {count} 个工作流')
