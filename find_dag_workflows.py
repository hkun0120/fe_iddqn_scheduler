#!/usr/bin/env python3
"""
找出有依赖关系且高并行度的工作流
"""

import pandas as pd
from sqlalchemy import create_engine
import networkx as nx

engine = create_engine('mysql+pymysql://root:@localhost:3306/whalesb')

def analyze_workflow(process_id):
    """分析工作流"""
    try:
        process = pd.read_sql(f'SELECT * FROM t_ds_process_instance WHERE id = {process_id}', engine)
        if len(process) == 0:
            return None
        
        pdc = int(process.iloc[0]['process_definition_code'])
        
        tasks = pd.read_sql(f'SELECT * FROM t_ds_task_instance WHERE process_instance_id = {process_id} AND state = 7', engine)
        if len(tasks) < 10:
            return None
            
        deps = pd.read_sql(f'SELECT * FROM t_ds_process_task_relation WHERE process_definition_code = {pdc}', engine)
        task_defs = pd.read_sql(f'''
            SELECT code, name, task_type FROM t_ds_task_definition 
            WHERE code IN (SELECT pre_task_code FROM t_ds_process_task_relation WHERE process_definition_code = {pdc}
            UNION SELECT post_task_code FROM t_ds_process_task_relation WHERE process_definition_code = {pdc})
        ''', engine)
        
        def_code_to_name = dict(zip(task_defs['code'], task_defs['name']))
        
        G = nx.DiGraph()
        for _, t in tasks.iterrows():
            G.add_node(t['name'], task_data=t.to_dict())
        
        edge_count = 0
        for _, d in deps.iterrows():
            if d['pre_task_code'] != 0:
                pre = def_code_to_name.get(d['pre_task_code'])
                post = def_code_to_name.get(d['post_task_code'])
                if pre in G.nodes and post in G.nodes:
                    G.add_edge(pre, post)
                    edge_count += 1
        
        if edge_count < 5:  # 至少有5条依赖边
            return None
        
        def get_duration(name):
            if name in G.nodes and 'task_data' in G.nodes[name]:
                t = G.nodes[name]['task_data']
                start = pd.to_datetime(t['start_time'])
                end = pd.to_datetime(t['end_time'])
                return max(1, (end - start).total_seconds())
            return 1
        
        critical_path_length = {}
        try:
            for node in nx.topological_sort(G):
                preds = list(G.predecessors(node))
                if not preds:
                    critical_path_length[node] = get_duration(node)
                else:
                    critical_path_length[node] = max(critical_path_length[p] for p in preds) + get_duration(node)
        except:
            return None
        
        cp_length = max(critical_path_length.values()) if critical_path_length else 0
        total_work = sum(get_duration(n) for n in G.nodes)
        parallelism = total_work / cp_length if cp_length > 0 else 1
        
        original_start = pd.to_datetime(process.iloc[0]['start_time'])
        original_end = pd.to_datetime(process.iloc[0]['end_time'])
        original_makespan = (original_end - original_start).total_seconds()
        
        # 计算 DAG 宽度（同一层的最大任务数）
        levels = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            if not preds:
                levels[node] = 0
            else:
                levels[node] = max(levels[p] for p in preds) + 1
        
        level_counts = {}
        for node, level in levels.items():
            level_counts[level] = level_counts.get(level, 0) + 1
        max_width = max(level_counts.values()) if level_counts else 1
        
        return {
            'process_id': process_id,
            'name': process.iloc[0]['name'][:50],
            'tasks': len(G.nodes),
            'edges': edge_count,
            'max_width': max_width,
            'critical_path': cp_length,
            'total_work': total_work,
            'parallelism': parallelism,
            'original_makespan': original_makespan,
            'theoretical_improvement': (original_makespan - cp_length) / original_makespan * 100 if original_makespan > 0 else 0,
            'G': G
        }
    except Exception as e:
        return None

print("=" * 100)
print("搜索有依赖关系且高并行度的工作流")
print("=" * 100)

workflows = pd.read_sql("""
    SELECT id, name FROM t_ds_process_instance 
    WHERE state = 7 
    ORDER BY id DESC
    LIMIT 500
""", engine)

print(f"正在分析 {len(workflows)} 个工作流...")

results = []
for i, (_, wf) in enumerate(workflows.iterrows()):
    if i % 50 == 0:
        print(f"  已分析 {i}/{len(workflows)}...")
    r = analyze_workflow(wf['id'])
    if r and r['parallelism'] >= 1.5 and r['max_width'] >= 2:
        results.append(r)

print(f"\n找到 {len(results)} 个符合条件的工作流 (并行度 >= 1.5, DAG宽度 >= 2, 边数 >= 5)")

results.sort(key=lambda x: (-x['max_width'], -x['parallelism']))

print("\n" + "=" * 100)
print("高并行度工作流（按DAG宽度排序）:")
print("=" * 100)
print(f"{'ID':>8} | {'任务':>4} | {'边':>4} | {'宽度':>4} | {'并行度':>6} | {'关键路径':>10} | {'原始':>10} | {'改进':>6} | 名称")
print("-" * 100)

for r in results[:15]:
    print(f"{r['process_id']:>8} | {r['tasks']:>4} | {r['edges']:>4} | {r['max_width']:>4} | {r['parallelism']:>6.1f} | {r['critical_path']:>9.0f}s | {r['original_makespan']:>9.0f}s | {r['theoretical_improvement']:>5.1f}% | {r['name'][:30]}")

# 调度算法比较
print("\n" + "=" * 100)
print("调度算法比较（选择前3个有潜力的工作流）")
print("=" * 100)

def schedule_workflow(G, num_resources, strategy):
    """调度"""
    def get_duration(name):
        if name in G.nodes and 'task_data' in G.nodes[name]:
            t = G.nodes[name]['task_data']
            start = pd.to_datetime(t['start_time'])
            end = pd.to_datetime(t['end_time'])
            return max(1, (end - start).total_seconds())
        return 1
    
    try:
        sorted_tasks = list(nx.topological_sort(G))
    except:
        return float('inf')
    
    resource_avail = {i: 0 for i in range(num_resources)}
    task_finish = {}
    
    for task_name in sorted_tasks:
        duration = get_duration(task_name)
        earliest = 0
        for pred in G.predecessors(task_name):
            if pred in task_finish:
                earliest = max(earliest, task_finish[pred])
        
        if strategy == 'fifo':
            selected = 0
        elif strategy == 'rr':
            selected = len(task_finish) % num_resources
        elif strategy == 'sjf':
            selected = min(resource_avail.items(), key=lambda x: x[1])[0]
        elif strategy == 'eft':
            best, best_finish = 0, float('inf')
            for r, avail in resource_avail.items():
                finish = max(avail, earliest) + duration
                if finish < best_finish:
                    best_finish, best = finish, r
            selected = best
        elif strategy == 'fe_iddqn':
            # 考虑: 完成时间 + 负载均衡 + 任务优先级
            scores = {}
            task_data = G.nodes[task_name].get('task_data', {})
            task_type = task_data.get('task_type', 'SHELL')
            
            for r, avail in resource_avail.items():
                start = max(avail, earliest)
                finish = start + duration
                avg_load = sum(resource_avail.values()) / num_resources
                balance = abs(avail - avg_load)
                
                # 类型匹配奖励
                type_bonus = 0
                if task_type == 'SQL' and r % 2 == 0:
                    type_bonus = -5
                elif task_type in ['SUB_PROCESS', 'SHELL'] and r % 2 == 1:
                    type_bonus = -5
                
                scores[r] = finish + 0.05 * balance + type_bonus
            selected = min(scores.items(), key=lambda x: x[1])[0]
        else:
            selected = 0
        
        start = max(resource_avail[selected], earliest)
        finish = start + duration
        resource_avail[selected] = finish
        task_finish[task_name] = finish
    
    return max(task_finish.values()) if task_finish else 0

# 选择理论改进潜力最大的工作流
top_potential = sorted(results, key=lambda x: -x['theoretical_improvement'])[:3]

for r in top_potential:
    print(f"\n工作流 {r['process_id']}: {r['name']}")
    print(f"  任务: {r['tasks']}, 边: {r['edges']}, DAG宽度: {r['max_width']}, 并行度: {r['parallelism']:.1f}")
    print(f"  关键路径: {r['critical_path']:.0f}s, 原始: {r['original_makespan']:.0f}s, 理论改进: {r['theoretical_improvement']:.1f}%")
    
    G = r['G']
    cp = r['critical_path']
    
    print(f"\n  {'资源':>4} | {'FIFO':>9} | {'RR':>9} | {'SJF':>9} | {'EFT':>9} | {'FE-IDDQN':>9} | {'最优':>9} | vs关键路径")
    print(f"  {'-'*85}")
    
    for num_res in [2, 3, 5, 10]:
        results_dict = {
            'FIFO': schedule_workflow(G, num_res, 'fifo'),
            'RR': schedule_workflow(G, num_res, 'rr'),
            'SJF': schedule_workflow(G, num_res, 'sjf'),
            'EFT': schedule_workflow(G, num_res, 'eft'),
            'FE-IDDQN': schedule_workflow(G, num_res, 'fe_iddqn')
        }
        best_val = min(results_dict.values())
        best_algo = [k for k, v in results_dict.items() if v == best_val][0]
        gap = (best_val - cp) / cp * 100 if cp > 0 else 0
        
        def fmt(name, val):
            mark = '*' if val == best_val else ' '
            return f"{val:>8.0f}{mark}"
        
        print(f"  {num_res:>4} | {fmt('FIFO', results_dict['FIFO'])} | {fmt('RR', results_dict['RR'])} | {fmt('SJF', results_dict['SJF'])} | {fmt('EFT', results_dict['EFT'])} | {fmt('FE-IDDQN', results_dict['FE-IDDQN'])} | {best_val:>9.0f} | +{gap:.1f}%")

print("\n" + "=" * 100)
print("分析结论")
print("=" * 100)
