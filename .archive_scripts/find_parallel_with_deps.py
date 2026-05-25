#!/usr/bin/env python3
"""
查找有依赖关系且有并行潜力的工作流
重点：边数 > 0 的工作流
"""

import pandas as pd
from sqlalchemy import create_engine
import networkx as nx

engine = create_engine('mysql+pymysql://root:@localhost:3306/whalesb')

print('=' * 80)
print('查找有依赖关系且有并行潜力的工作流')
print('=' * 80)

# 先找有较多依赖边的工作流定义
print('\n1. 查找有较多依赖边的工作流定义...')
workflow_defs = pd.read_sql('''
    SELECT process_definition_code, COUNT(*) as edge_count
    FROM t_ds_process_task_relation
    WHERE pre_task_code != 0
    GROUP BY process_definition_code
    HAVING edge_count >= 15
    ORDER BY edge_count DESC
    LIMIT 100
''', engine)

print(f'   找到 {len(workflow_defs)} 个有较多依赖边的工作流定义')

# 找这些定义对应的实例
results = []
print('\n2. 分析这些工作流的实例...')

for idx, wfd in workflow_defs.iterrows():
    pdc = int(wfd['process_definition_code'])
    edge_count = int(wfd['edge_count'])
    
    try:
        # 找一个成功的实例
        instance = pd.read_sql(f'''
            SELECT pi.id, pi.name, pi.start_time, pi.end_time
            FROM t_ds_process_instance pi
            WHERE pi.process_definition_code = {pdc} AND pi.state = 7
            LIMIT 1
        ''', engine)
        
        if len(instance) == 0:
            continue
        
        pid = int(instance.iloc[0]['id'])
        
        # 获取任务
        tasks = pd.read_sql(f'''
            SELECT name, start_time, end_time, task_type 
            FROM t_ds_task_instance 
            WHERE process_instance_id = {pid} AND state = 7
        ''', engine)
        
        if len(tasks) < 10:
            continue
        
        # 计算总工作量
        total_work = 0
        for _, t in tasks.iterrows():
            try:
                dur = (pd.to_datetime(t['end_time']) - pd.to_datetime(t['start_time'])).total_seconds()
                total_work += max(1, dur)
            except:
                total_work += 1
        
        # 原始 makespan
        original_start = pd.to_datetime(instance.iloc[0]['start_time'])
        original_end = pd.to_datetime(instance.iloc[0]['end_time'])
        original_makespan = (original_end - original_start).total_seconds()
        
        if original_makespan <= 0:
            continue
        
        parallelism = total_work / original_makespan
        
        results.append({
            'id': pid,
            'name': instance.iloc[0]['name'][:60],
            'tasks': len(tasks),
            'edges': edge_count,
            'parallelism': parallelism,
            'original_makespan': original_makespan,
            'total_work': total_work
        })
        
    except Exception as e:
        pass

print(f'   分析完成，找到 {len(results)} 个有效工作流')

# 按并行度排序
results = sorted(results, key=lambda x: -x['parallelism'])

print('\n' + '=' * 80)
print('有依赖关系的高并行度工作流 Top 15:')
print('=' * 80)
print(f'{"ID":>8} | {"任务数":>6} | {"边数":>6} | {"并行度":>8} | {"原始执行":>12} | 名称')
print('-' * 100)

for r in results[:15]:
    makespan_str = f"{r['original_makespan']:.0f}s"
    if r['original_makespan'] > 3600:
        makespan_str = f"{r['original_makespan']/3600:.1f}h"
    print(f"{r['id']:>8} | {r['tasks']:>6} | {r['edges']:>6} | {r['parallelism']:>8.2f} | {makespan_str:>12} | {r['name'][:40]}")

# 选择一个进行详细分析
if results:
    print('\n' + '=' * 80)
    print('详细分析并行度最高的工作流:')
    print('=' * 80)
    
    best = results[0]
    pid = best['id']
    
    print(f"\n工作流 {pid}: {best['name']}")
    print(f"任务数: {best['tasks']}, 边数: {best['edges']}")
    print(f"并行度: {best['parallelism']:.2f}")
    
    # 构建 DAG
    pdc_query = pd.read_sql(f"SELECT process_definition_code FROM t_ds_process_instance WHERE id = {pid}", engine)
    pdc = int(pdc_query.iloc[0]['process_definition_code'])
    
    tasks = pd.read_sql(f'''
        SELECT name, start_time, end_time, task_type 
        FROM t_ds_task_instance 
        WHERE process_instance_id = {pid} AND state = 7
    ''', engine)
    
    deps = pd.read_sql(f'''
        SELECT pre_task_code, post_task_code 
        FROM t_ds_process_task_relation 
        WHERE process_definition_code = {pdc} AND pre_task_code != 0
    ''', engine)
    
    task_defs = pd.read_sql(f'''
        SELECT code, name FROM t_ds_task_definition 
        WHERE code IN (
            SELECT pre_task_code FROM t_ds_process_task_relation WHERE process_definition_code = {pdc}
            UNION
            SELECT post_task_code FROM t_ds_process_task_relation WHERE process_definition_code = {pdc}
        )
    ''', engine)
    
    def_code_to_name = dict(zip(task_defs['code'], task_defs['name']))
    
    G = nx.DiGraph()
    for _, t in tasks.iterrows():
        G.add_node(t['name'])
    
    for _, d in deps.iterrows():
        pre = def_code_to_name.get(d['pre_task_code'])
        post = def_code_to_name.get(d['post_task_code'])
        if pre in G.nodes and post in G.nodes:
            G.add_edge(pre, post)
    
    print(f"\nDAG: {len(G.nodes)} 节点, {len(G.edges)} 边")
    
    # 分析层级
    if len(G.edges) > 0:
        try:
            levels = {}
            for node in nx.topological_sort(G):
                preds = list(G.predecessors(node))
                if not preds:
                    levels[node] = 0
                else:
                    levels[node] = max(levels[p] for p in preds) + 1
            
            level_groups = {}
            for node, level in levels.items():
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append(node)
            
            print(f"\n层级分布 (并行机会):")
            for level in sorted(level_groups.keys()):
                nodes = level_groups[level]
                print(f"  Level {level}: {len(nodes)} 个任务")
                if len(nodes) > 1:
                    print(f"    -> {len(nodes)} 个任务可并行执行!")
        except Exception as e:
            print(f"层级分析失败: {e}")
