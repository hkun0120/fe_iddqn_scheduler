#!/usr/bin/env python3
"""
验证修复后的 DAG 构建逻辑
只处理工作流 293718
"""

import pandas as pd
from sqlalchemy import create_engine
import networkx as nx

print("连接数据库...")
engine = create_engine('mysql+pymysql://root:@localhost:3306/whalesb')

process_id = 293718

# 加载 293718 相关数据
print("加载数据...")
process_instance = pd.read_sql(f"SELECT * FROM t_ds_process_instance WHERE id = {process_id}", engine)
task_instance = pd.read_sql(f"SELECT * FROM t_ds_task_instance WHERE process_instance_id = {process_id}", engine)

process_def_code = int(process_instance.iloc[0]['process_definition_code'])
print(f"process_definition_code: {process_def_code}")

process_task_relation = pd.read_sql(
    f"SELECT * FROM t_ds_process_task_relation WHERE process_definition_code = {process_def_code}", 
    engine
)

# 只加载该工作流定义的任务定义
task_definition = pd.read_sql(
    f"""SELECT code, name, task_type FROM t_ds_task_definition 
        WHERE code IN (
            SELECT DISTINCT pre_task_code FROM t_ds_process_task_relation WHERE process_definition_code = {process_def_code}
            UNION
            SELECT DISTINCT post_task_code FROM t_ds_process_task_relation WHERE process_definition_code = {process_def_code}
        )""",
    engine
)

print(f"任务实例数: {len(task_instance)}")
print(f"任务定义数: {len(task_definition)}")
print(f"依赖边数: {len(process_task_relation)}")

# 成功任务
successful = task_instance[task_instance['state'] == 7].copy()
print(f"成功任务数: {len(successful)}")

# 建立 task_definition.code -> 任务名称 的映射
def_code_to_name = dict(zip(task_definition['code'], task_definition['name']))

# 构建 DAG（使用任务名称作为节点）
print("\n" + "=" * 80)
print("构建 DAG（使用任务名称作为节点）")
print("=" * 80)

G = nx.DiGraph()

# 添加节点
for _, task in successful.iterrows():
    G.add_node(task['name'], task_data=task.to_dict())

print(f"节点数: {len(G.nodes)}")

# 添加边
edges_added = 0
edges_failed = 0
for _, dep in process_task_relation.iterrows():
    pre_def = dep['pre_task_code']
    post_def = dep['post_task_code']
    
    if pd.notna(pre_def) and pd.notna(post_def) and pre_def != 0:
        pre_name = def_code_to_name.get(pre_def)
        post_name = def_code_to_name.get(post_def)
        
        if pre_name is not None and post_name is not None:
            if pre_name in G.nodes and post_name in G.nodes:
                G.add_edge(pre_name, post_name)
                edges_added += 1
            else:
                edges_failed += 1
                if pre_name not in G.nodes:
                    print(f"  节点不存在: {pre_name}")
                if post_name not in G.nodes:
                    print(f"  节点不存在: {post_name}")

print(f"成功添加的边数: {edges_added}")
print(f"未能添加的边数: {edges_failed}")

# 验证依赖关系
print("\n" + "=" * 80)
print("验证依赖关系:")
print("=" * 80)

# 按类型统计边
type_stats = {}
for u, v in G.edges():
    u_data = G.nodes[u].get('task_data', {})
    v_data = G.nodes[v].get('task_data', {})
    u_type = u_data.get('task_type', 'N/A')
    v_type = v_data.get('task_type', 'N/A')
    pair = f"{u_type} -> {v_type}"
    type_stats[pair] = type_stats.get(pair, 0) + 1

for pair, count in sorted(type_stats.items(), key=lambda x: -x[1]):
    print(f"  {pair}: {count}条")

# 拓扑排序
print("\n" + "=" * 80)
print("拓扑排序:")
print("=" * 80)

try:
    sorted_names = list(nx.topological_sort(G))
    print(f"拓扑排序成功! 共 {len(sorted_names)} 个任务")
    
    print("\n前10个任务:")
    for i, name in enumerate(sorted_names[:10], 1):
        task_type = G.nodes[name].get('task_data', {}).get('task_type', 'N/A')
        print(f"  {i}. [{task_type}] {name}")
except nx.NetworkXUnfeasible as e:
    print(f"拓扑排序失败（存在环）: {e}")

# 模拟调度
print("\n" + "=" * 80)
print("模拟调度（使用修复后的逻辑）:")
print("=" * 80)

def get_duration(task):
    try:
        start = pd.to_datetime(task.get('start_time'))
        end = pd.to_datetime(task.get('end_time'))
        return max(1, (end - start).total_seconds())
    except:
        return 10

num_resources = 5
resource_avail = {i: 0 for i in range(num_resources)}
task_finish = {}

sorted_tasks = [G.nodes[name]['task_data'] for name in sorted_names]

for idx, task in enumerate(sorted_tasks):
    task_name = task.get('name')
    duration = get_duration(task)
    
    # 计算最早开始时间
    earliest = 0
    if task_name in G:
        for pred in G.predecessors(task_name):
            if pred in task_finish:
                earliest = max(earliest, task_finish[pred])
    
    # 选择最早可用的资源
    selected = min(resource_avail.items(), key=lambda x: max(x[1], earliest))[0]
    
    start = max(resource_avail[selected], earliest)
    finish = start + duration
    
    resource_avail[selected] = finish
    task_finish[task_name] = finish

makespan = max(task_finish.values())
print(f"模拟调度 Makespan: {makespan:.0f} 秒 ({makespan/3600:.2f} 小时)")

# 对比原始执行
original_start = pd.to_datetime(process_instance.iloc[0]['start_time'])
original_end = pd.to_datetime(process_instance.iloc[0]['end_time'])
original_makespan = (original_end - original_start).total_seconds()
print(f"原始执行 Makespan: {original_makespan:.0f} 秒 ({original_makespan/3600:.2f} 小时)")
print(f"改进: {(original_makespan - makespan) / original_makespan * 100:.1f}%")
