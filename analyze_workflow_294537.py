#!/usr/bin/env python3
"""
详细分析工作流 294537
为什么它的调度效率这么低？
"""

import pandas as pd
from sqlalchemy import create_engine
import networkx as nx
import matplotlib.pyplot as plt

engine = create_engine('mysql+pymysql://root:@localhost:3306/whalesb')

process_id = 294537

print("=" * 100)
print(f"工作流 {process_id} 详细分析")
print("=" * 100)

# 加载数据
process = pd.read_sql(f'SELECT * FROM t_ds_process_instance WHERE id = {process_id}', engine)
pdc = int(process.iloc[0]['process_definition_code'])

tasks = pd.read_sql(f'SELECT * FROM t_ds_task_instance WHERE process_instance_id = {process_id} AND state = 7', engine)
deps = pd.read_sql(f'SELECT * FROM t_ds_process_task_relation WHERE process_definition_code = {pdc}', engine)

# 加载任务定义
dep_codes = set(deps['pre_task_code'].unique()) | set(deps['post_task_code'].unique())
codes_str = ','.join(str(c) for c in list(dep_codes)[:100])
task_defs = pd.read_sql(f'SELECT code, name, task_type FROM t_ds_task_definition WHERE code IN ({codes_str})', engine)

def_code_to_name = dict(zip(task_defs['code'], task_defs['name']))

# 构建 DAG
G = nx.DiGraph()
for _, t in tasks.iterrows():
    G.add_node(t['name'])

for _, d in deps.iterrows():
    if d['pre_task_code'] != 0:
        pre = def_code_to_name.get(d['pre_task_code'])
        post = def_code_to_name.get(d['post_task_code'])
        if pre in G.nodes and post in G.nodes:
            G.add_edge(pre, post)

# 计算任务持续时间
def get_duration(name):
    t = tasks[tasks['name'] == name]
    if len(t) > 0:
        start = pd.to_datetime(t.iloc[0]['start_time'])
        end = pd.to_datetime(t.iloc[0]['end_time'])
        return max(1, (end - start).total_seconds())
    return 1

task_durations = {name: get_duration(name) for name in G.nodes}

# 计算关键路径
critical_path_length = {}
critical_predecessors = {}
for node in nx.topological_sort(G):
    preds = list(G.predecessors(node))
    if not preds:
        critical_path_length[node] = task_durations[node]
        critical_predecessors[node] = None
    else:
        best_pred = max(preds, key=lambda p: critical_path_length[p])
        critical_path_length[node] = critical_path_length[best_pred] + task_durations[node]
        critical_predecessors[node] = best_pred

cp_length = max(critical_path_length.values())
total_work = sum(task_durations.values())

# 实际执行时间
original_start = pd.to_datetime(process.iloc[0]['start_time'])
original_end = pd.to_datetime(process.iloc[0]['end_time'])
actual_makespan = (original_end - original_start).total_seconds()

print(f"\n基本指标:")
print(f"  任务数: {len(G.nodes)}")
print(f"  依赖边数: {len(G.edges)}")
print(f"")
print(f"时间指标:")
print(f"  关键路径长度: {cp_length:.0f}秒 ({cp_length/60:.1f}分钟)")
print(f"  总工作量: {total_work:.0f}秒 ({total_work/60:.1f}分钟)")
print(f"  实际执行时间: {actual_makespan:.0f}秒 ({actual_makespan/60:.1f}分钟)")
print(f"")
print(f"效率指标:")
print(f"  调度效率: {cp_length/actual_makespan*100:.1f}%")
print(f"  理论改进潜力: {(actual_makespan-cp_length)/actual_makespan*100:.1f}%")
print(f"  理论并行度: {total_work/cp_length:.1f}")
print(f"  实际利用率: {total_work/actual_makespan:.1f}")

# 找出关键路径
print(f"\n关键路径 ({cp_length:.0f}秒):")
current = max(critical_path_length, key=lambda x: critical_path_length[x])
path = []
while current is not None:
    path.append(current)
    current = critical_predecessors[current]
path.reverse()

for i, node in enumerate(path, 1):
    print(f"  {i}. {node[:50]:50} ({task_durations[node]:6.0f}s)")

# 分析实际执行时间线
print(f"\n任务执行时间线分析:")
print("-" * 100)

tasks_sorted = tasks.sort_values('start_time')
ref_start = pd.to_datetime(tasks_sorted.iloc[0]['start_time'])

print(f"{'任务名':50} | {'开始':>10} | {'结束':>10} | {'持续':>6} | {'实际开始':>10} | {'理想开始':>10}")
print("-" * 100)

# 计算每个任务的理想开始时间（基于关键路径）
ideal_start = {}
for node in nx.topological_sort(G):
    preds = list(G.predecessors(node))
    if not preds:
        ideal_start[node] = 0
    else:
        ideal_start[node] = max(ideal_start[p] + task_durations[p] for p in preds)

for _, task in tasks_sorted.iterrows():
    name = task['name']
    start = pd.to_datetime(task['start_time'])
    end = pd.to_datetime(task['end_time'])
    duration = (end - start).total_seconds()
    actual_offset = (start - ref_start).total_seconds()
    ideal_offset = ideal_start.get(name, 0)
    
    idle_time = actual_offset - ideal_offset
    
    print(f"{name[:50]:50} | {actual_offset:10.0f} | {actual_offset + duration:10.0f} | {duration:6.0f} | {ideal_offset:10.0f} | {idle_time:>10.0f}")

# 统计空闲时间
total_idle = sum((pd.to_datetime(t['start_time']) - ref_start).total_seconds() - ideal_start.get(t['name'], 0) for _, t in tasks.iterrows())

print(f"\n总空闲时间: {total_idle:.0f}秒 ({total_idle/actual_makespan*100:.1f}%)")

# 分析为什么有空闲时间
print(f"\n为什么调度效率低？")
print("-" * 100)

# 找出有空闲时间的任务
idle_tasks = []
for _, task in tasks.iterrows():
    name = task['name']
    start = pd.to_datetime(task['start_time'])
    actual_offset = (start - ref_start).total_seconds()
    ideal_offset = ideal_start.get(name, 0)
    idle = actual_offset - ideal_offset
    
    if idle > 1:  # 至少 1 秒的空闲
        preds = list(G.predecessors(name))
        pred_names = [p for p in preds]
        idle_tasks.append({
            'name': name,
            'idle_time': idle,
            'predecessors': pred_names
        })

idle_tasks.sort(key=lambda x: -x['idle_time'])

for i, task in enumerate(idle_tasks[:10], 1):
    print(f"{i}. {task['name'][:50]}")
    print(f"   空闲时间: {task['idle_time']:.0f}秒")
    if task['predecessors']:
        for pred in task['predecessors']:
            print(f"   依赖于: {pred[:50]}")
    print()

print("\n" + "=" * 100)
print("诊断结论:")
print("=" * 100)
if total_idle / actual_makespan > 0.3:
    print("⚠️ 此工作流有大量的空闲时间（> 30%）")
    print("原因可能是:")
    print("  1. 任务调度顺序不优化 - 优先调度关键路径任务")
    print("  2. 资源分配不均 - 某些资源被高效利用，其他闲置")
    print("  3. 任务依赖管理不当 - 某些任务被延迟执行")
    print("")
    print("深度强化学习可以学习到最优的任务调度和资源分配策略！")
