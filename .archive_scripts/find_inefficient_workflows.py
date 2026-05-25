#!/usr/bin/env python3
"""
寻找调度效率较低的工作流执行
即实际执行时间 远大于 关键路径长度的情况
"""

import pandas as pd
from sqlalchemy import create_engine
import networkx as nx

engine = create_engine('mysql+pymysql://root:@localhost:3306/whalesb')

print('=' * 80)
print('寻找调度效率较低的工作流执行')
print('=' * 80)

# 获取执行成功且有足够任务的工作流
processes = pd.read_sql('''
    SELECT p.id, p.process_definition_code, p.start_time, p.end_time, p.name,
           COUNT(*) as task_count
    FROM t_ds_process_instance p
    JOIN t_ds_task_instance t ON t.process_instance_id = p.id
    WHERE p.state = 7 AND t.state = 7
    GROUP BY p.id
    HAVING task_count >= 30
    ORDER BY task_count DESC
    LIMIT 100
''', engine)

print(f'检查 {len(processes)} 个工作流...\n')

results = []

for _, proc in processes.iterrows():
    pid = proc['id']
    pdc = proc['process_definition_code']
    
    # 计算实际执行时间
    try:
        start = pd.to_datetime(proc['start_time'])
        end = pd.to_datetime(proc['end_time'])
        actual_makespan = (end - start).total_seconds()
        if actual_makespan <= 0:
            continue
    except:
        continue
    
    # 获取任务
    tasks = pd.read_sql(f'''
        SELECT name, start_time, end_time
        FROM t_ds_task_instance 
        WHERE process_instance_id = {pid} AND state = 7
    ''', engine)
    
    if len(tasks) < 30:
        continue
    
    # 计算总工作量
    total_work = 0
    task_durations = {}
    for _, t in tasks.iterrows():
        try:
            dur = (pd.to_datetime(t['end_time']) - pd.to_datetime(t['start_time'])).total_seconds()
            dur = max(1, dur)
        except:
            dur = 1
        task_durations[t['name']] = dur
        total_work += dur
    
    # 获取依赖关系
    deps = pd.read_sql(f'''
        SELECT pre_task_code, post_task_code 
        FROM t_ds_process_task_relation 
        WHERE process_definition_code = {pdc} AND pre_task_code != 0
    ''', engine)
    
    if len(deps) < 5:
        continue
    
    task_defs = pd.read_sql(f'''
        SELECT code, name FROM t_ds_task_definition 
        WHERE code IN (
            SELECT pre_task_code FROM t_ds_process_task_relation WHERE process_definition_code = {pdc}
            UNION
            SELECT post_task_code FROM t_ds_process_task_relation WHERE process_definition_code = {pdc}
        )
    ''', engine)
    
    def_code_to_name = dict(zip(task_defs['code'], task_defs['name']))
    
    # 构建 DAG
    G = nx.DiGraph()
    for name, dur in task_durations.items():
        G.add_node(name, duration=dur)
    
    for _, d in deps.iterrows():
        pre = def_code_to_name.get(d['pre_task_code'])
        post = def_code_to_name.get(d['post_task_code'])
        if pre in G.nodes and post in G.nodes:
            G.add_edge(pre, post)
    
    if len(G.edges) < 5:
        continue
    
    # 计算关键路径
    try:
        cp = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            if not preds:
                cp[node] = task_durations.get(node, 1)
            else:
                cp[node] = max(cp[p] for p in preds) + task_durations.get(node, 1)
        critical_path = max(cp.values())
    except:
        continue
    
    # 计算效率
    efficiency = critical_path / actual_makespan
    parallelism = total_work / critical_path
    slack = actual_makespan - critical_path
    slack_ratio = slack / actual_makespan
    
    results.append({
        'id': pid,
        'name': proc['name'][:50],
        'tasks': proc['task_count'],
        'edges': len(G.edges),
        'actual': actual_makespan,
        'critical_path': critical_path,
        'efficiency': efficiency,
        'parallelism': parallelism,
        'slack': slack,
        'slack_ratio': slack_ratio
    })

# 按效率排序（效率越低，优化空间越大）
df = pd.DataFrame(results)
df = df.sort_values('efficiency')

print(f'{"ID":>8} | {"任务":>5} | {"边":>5} | {"实际(s)":>10} | {"关键路径(s)":>12} | {"效率":>6} | {"并行度":>7} | {"优化空间":>8} | 名称')
print('-' * 140)

for _, r in df.head(20).iterrows():
    opt = r['slack_ratio'] * 100
    print(f'{r["id"]:>8} | {r["tasks"]:>5} | {r["edges"]:>5} | {r["actual"]:>10.0f} | {r["critical_path"]:>12.0f} | {r["efficiency"]*100:>5.1f}% | {r["parallelism"]:>7.1f} | {opt:>7.1f}% | {r["name"]}')

print('\n' + '=' * 80)
print('分析：效率 < 100% 意味着存在优化空间')
print('优化空间 = (实际时间 - 关键路径) / 实际时间')
print('=' * 80)

# 选择最适合的候选
if len(df) > 0:
    best = df[(df['efficiency'] < 0.9) & (df['parallelism'] > 5) & (df['edges'] > 20)]
    if len(best) > 0:
        print('\n最佳候选工作流（效率<90%, 并行度>5, 边>20）:')
        for _, r in best.head(5).iterrows():
            print(f'  工作流 {r["id"]}: {r["tasks"]}任务, {r["edges"]}边, 效率{r["efficiency"]*100:.0f}%, 可优化{r["slack_ratio"]*100:.0f}%')
