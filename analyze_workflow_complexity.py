#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从MySQL数据库分析工作流复杂度，按大/中/小分类
考虑因素：任务数量、嵌套复杂度(SUB_PROCESS)、执行时间
"""

import pymysql
import pandas as pd
import numpy as np
from collections import defaultdict

def get_db_connection():
    return pymysql.connect(
        host='localhost', 
        user='root', 
        password='root', 
        database='whalesb',
        cursorclass=pymysql.cursors.DictCursor
    )

print('=== 从MySQL加载工作流数据 ===')

conn = get_db_connection()

# 1. 获取成功完成的工作流实例
query_process = """
SELECT 
    id as process_instance_id,
    name,
    process_definition_code,
    state,
    start_time,
    end_time,
    TIMESTAMPDIFF(SECOND, start_time, end_time) as duration_seconds
FROM t_ds_process_instance 
WHERE state = 7 
    AND start_time IS NOT NULL 
    AND end_time IS NOT NULL
    AND end_time > start_time
ORDER BY duration_seconds DESC
"""
process_df = pd.read_sql(query_process, conn)
print(f'成功完成的工作流实例: {len(process_df)}')

# 2. 获取所有任务实例
query_tasks = """
SELECT 
    id,
    name,
    task_type,
    task_code,
    process_instance_id,
    state,
    start_time,
    end_time,
    TIMESTAMPDIFF(SECOND, start_time, end_time) as duration_seconds
FROM t_ds_task_instance
WHERE state = 7
"""
task_df = pd.read_sql(query_tasks, conn)
print(f'成功的任务实例: {len(task_df)}')

# 3. 获取任务依赖关系
query_relations = """
SELECT 
    process_definition_code,
    pre_task_code,
    post_task_code
FROM t_ds_process_task_relation
"""
relation_df = pd.read_sql(query_relations, conn)
print(f'任务依赖关系: {len(relation_df)}')

conn.close()

print()
print('=== 分析工作流复杂度 ===')

# 计算每个工作流的复杂度指标
workflow_stats = []

for idx, proc in process_df.iterrows():
    pid = proc['process_instance_id']
    pcode = proc['process_definition_code']
    
    # 该工作流的任务
    tasks = task_df[task_df['process_instance_id'] == pid]
    
    if len(tasks) == 0:
        continue
    
    # 任务数量
    task_count = len(tasks)
    
    # SUB_PROCESS数量（嵌套复杂度）
    sub_process_count = len(tasks[tasks['task_type'] == 'SUB_PROCESS'])
    
    # 依赖关系数量
    relations = relation_df[relation_df['process_definition_code'] == pcode]
    relation_count = len(relations[relations['pre_task_code'] != 0])  # 排除入口任务
    
    # 计算DAG深度（通过拓扑排序）
    # 构建邻接表
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    nodes = set()
    
    for _, r in relations.iterrows():
        pre = r['pre_task_code']
        post = r['post_task_code']
        if pre != 0 and post != 0:
            graph[pre].append(post)
            in_degree[post] += 1
            nodes.add(pre)
            nodes.add(post)
        elif pre == 0:
            nodes.add(post)
    
    # 计算最长路径（DAG深度）
    dag_depth = 1
    if len(graph) > 0:
        # 拓扑排序计算深度
        depth = {n: 1 for n in nodes}
        queue = [n for n in nodes if in_degree[n] == 0]
        while queue:
            node = queue.pop(0)
            for neighbor in graph[node]:
                depth[neighbor] = max(depth[neighbor], depth[node] + 1)
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        dag_depth = max(depth.values()) if depth else 1
    
    # 执行时间
    duration = proc['duration_seconds'] or 0
    
    # 任务类型多样性
    task_types = tasks['task_type'].nunique()
    
    workflow_stats.append({
        'process_instance_id': pid,
        'name': proc['name'],
        'process_definition_code': pcode,
        'task_count': task_count,
        'sub_process_count': sub_process_count,
        'relation_count': relation_count,
        'dag_depth': dag_depth,
        'task_types': task_types,
        'duration_seconds': duration
    })

stats_df = pd.DataFrame(workflow_stats)
print(f'分析完成的工作流: {len(stats_df)}')

print()
print('=== 工作流统计 ===')
print(f'任务数量: 最小={stats_df["task_count"].min()}, 最大={stats_df["task_count"].max()}, '
      f'平均={stats_df["task_count"].mean():.1f}, 中位数={stats_df["task_count"].median():.0f}')
print(f'DAG深度: 最小={stats_df["dag_depth"].min()}, 最大={stats_df["dag_depth"].max()}, '
      f'平均={stats_df["dag_depth"].mean():.1f}')
print(f'执行时间: 最小={stats_df["duration_seconds"].min()}秒, 最大={stats_df["duration_seconds"].max()}秒, '
      f'平均={stats_df["duration_seconds"].mean():.0f}秒')

# 计算串行度（串行度 = DAG深度 / 任务数量，越低越并行）
stats_df['serialization_ratio'] = stats_df['dag_depth'] / stats_df['task_count']
print(f'串行度: 最小={stats_df["serialization_ratio"].min():.3f}, 最大={stats_df["serialization_ratio"].max():.3f}, '
      f'平均={stats_df["serialization_ratio"].mean():.3f}')

print()
print('=== 按复杂度分类工作流 ===')

# 计算复杂度得分 (综合考虑任务数、嵌套、DAG深度、执行时间)
stats_df['complexity_score'] = (
    stats_df['task_count'] / stats_df['task_count'].max() * 0.3 +
    stats_df['sub_process_count'] / max(stats_df['sub_process_count'].max(), 1) * 0.2 +
    stats_df['dag_depth'] / stats_df['dag_depth'].max() * 0.2 +
    stats_df['duration_seconds'] / stats_df['duration_seconds'].max() * 0.3
)

# 按复杂度分类
stats_df = stats_df.sort_values('complexity_score', ascending=False)

# 大型工作流（复杂度前20%）
large_threshold = stats_df['complexity_score'].quantile(0.8)
# 小型工作流（复杂度后20%）
small_threshold = stats_df['complexity_score'].quantile(0.2)

stats_df['category'] = 'medium'
stats_df.loc[stats_df['complexity_score'] >= large_threshold, 'category'] = 'large'
stats_df.loc[stats_df['complexity_score'] <= small_threshold, 'category'] = 'small'

print(f'大型工作流: {len(stats_df[stats_df["category"] == "large"])}')
print(f'中型工作流: {len(stats_df[stats_df["category"] == "medium"])}')
print(f'小型工作流: {len(stats_df[stats_df["category"] == "small"])}')

print()
print('=== 选择代表性工作流 ===')

# 首先，找出串行度最低的工作流（高并行度）
print('\n【串行度最低的工作流 TOP 20】（并行度最高）')
low_serialization = stats_df[stats_df['task_count'] >= 5].nsmallest(20, 'serialization_ratio')
for idx, w in low_serialization.iterrows():
    parallel_level = w['task_count'] / w['dag_depth']
    print(f'  ID={w["process_instance_id"]}: 任务={w["task_count"]}, 深度={w["dag_depth"]}, '
          f'串行度={w["serialization_ratio"]:.3f} (平均并行度={parallel_level:.1f})')
    print(f'    名称: {w["name"][:80]}...')

def select_representative(df, category, n=10):
    """选择具有代表性的工作流（有实际依赖关系的）"""
    cat_df = df[df['category'] == category].copy()
    # 优先选择有依赖关系的工作流
    cat_df = cat_df[cat_df['relation_count'] > 0]
    # 按复杂度排序
    if category == 'large':
        return cat_df.head(n)
    elif category == 'small':
        return cat_df.tail(n)
    else:
        # 中型取中间的
        mid_start = len(cat_df) // 2 - n // 2
        return cat_df.iloc[mid_start:mid_start + n]

print('\n【大型工作流 TOP 10】')
large_workflows = select_representative(stats_df, 'large', 10)
for _, w in large_workflows.iterrows():
    print(f'  ID={w["process_instance_id"]}: 任务={w["task_count"]}, 嵌套={w["sub_process_count"]}, '
          f'深度={w["dag_depth"]}, 时长={w["duration_seconds"]}秒')
    print(f'    名称: {w["name"][:60]}...')

print('\n【中型工作流 10个】')
medium_workflows = select_representative(stats_df, 'medium', 10)
for _, w in medium_workflows.iterrows():
    print(f'  ID={w["process_instance_id"]}: 任务={w["task_count"]}, 嵌套={w["sub_process_count"]}, '
          f'深度={w["dag_depth"]}, 时长={w["duration_seconds"]}秒')

print('\n【小型工作流 10个】')
small_workflows = select_representative(stats_df, 'small', 10)
for _, w in small_workflows.iterrows():
    print(f'  ID={w["process_instance_id"]}: 任务={w["task_count"]}, 嵌套={w["sub_process_count"]}, '
          f'深度={w["dag_depth"]}, 时长={w["duration_seconds"]}秒')

# 保存选中的工作流ID
selected_workflows = {
    'large': large_workflows['process_instance_id'].tolist(),
    'medium': medium_workflows['process_instance_id'].tolist(),
    'small': small_workflows['process_instance_id'].tolist(),
    'low_serialization': low_serialization['process_instance_id'].tolist()
}

print()
print('=== 保存分类结果 ===')
import json
with open('selected_workflows.json', 'w') as f:
    json.dump(selected_workflows, f, indent=2)
print('已保存到 selected_workflows.json')

# 也保存完整统计信息（包括串行度）
stats_df.to_csv('workflow_complexity_stats.csv', index=False)
print('完整统计已保存到 workflow_complexity_stats.csv')
