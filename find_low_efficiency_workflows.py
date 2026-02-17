#!/usr/bin/env python3
"""
找出调度效率低的工作流
这些工作流是演示深度强化学习价值的最好候选
"""

import pandas as pd
from sqlalchemy import create_engine
import networkx as nx

engine = create_engine('mysql+pymysql://root:@localhost:3306/whalesb')

def analyze_workflow_efficiency(process_id):
    """分析单个工作流的调度效率"""
    try:
        process = pd.read_sql(f'SELECT * FROM t_ds_process_instance WHERE id = {process_id}', engine)
        if len(process) == 0:
            return None
        
        pdc = int(process.iloc[0]['process_definition_code'])
        
        tasks = pd.read_sql(f'SELECT * FROM t_ds_task_instance WHERE process_instance_id = {process_id} AND state = 7', engine)
        if len(tasks) == 0:
            return None
        
        deps = pd.read_sql(f'SELECT * FROM t_ds_process_task_relation WHERE process_definition_code = {pdc}', engine)
        
        # 只加载需要的任务定义
        dep_codes = set(deps['pre_task_code'].unique()) | set(deps['post_task_code'].unique())
        if not dep_codes:
            return None
        
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
        
        # 计算指标
        def get_duration(name):
            t = tasks[tasks['name'] == name]
            if len(t) > 0:
                start = pd.to_datetime(t.iloc[0]['start_time'])
                end = pd.to_datetime(t.iloc[0]['end_time'])
                return max(1, (end - start).total_seconds())
            return 1
        
        # 关键路径
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
        
        # 实际执行时间
        original_start = pd.to_datetime(process.iloc[0]['start_time'])
        original_end = pd.to_datetime(process.iloc[0]['end_time'])
        actual_makespan = (original_end - original_start).total_seconds()
        
        # 计算调度效率
        if actual_makespan > 0:
            scheduling_efficiency = cp_length / actual_makespan * 100
        else:
            scheduling_efficiency = 0
        
        # 理论并行度
        parallelism = total_work / cp_length if cp_length > 0 else 1
        
        # 理论改进潜力
        theoretical_improvement = (actual_makespan - cp_length) / actual_makespan * 100 if actual_makespan > 0 else 0
        
        return {
            'process_id': process_id,
            'num_tasks': len(G.nodes),
            'num_edges': len(G.edges),
            'critical_path': cp_length,
            'total_work': total_work,
            'actual_makespan': actual_makespan,
            'scheduling_efficiency': scheduling_efficiency,
            'parallelism': parallelism,
            'theoretical_improvement': theoretical_improvement
        }
    except Exception as e:
        return None

print("=" * 100)
print("寻找调度效率低且有并行潜力的工作流")
print("=" * 100)

# 获取 50 个成功的工作流
workflows = pd.read_sql("""
    SELECT id FROM t_ds_process_instance 
    WHERE state = 7 
    ORDER BY id DESC
    LIMIT 100
""", engine)

results = []
for _, wf in workflows.iterrows():
    r = analyze_workflow_efficiency(wf['id'])
    if r and r['num_tasks'] > 10 and r['num_edges'] > 5:  # 过滤太小的工作流
        results.append(r)
    if len(results) >= 50:
        break

print(f"分析了 {len(results)} 个有意义的工作流\n")

# 排序：按调度效率低排序
results_sorted = sorted(results, key=lambda x: x['scheduling_efficiency'])

print("调度效率最低的 10 个工作流:")
print("-" * 100)
print(f"{'PID':>8} | {'效率':>6} | {'任务数':>5} | {'边数':>4} | {'并行度':>6} | {'改进潜力':>8}")
print("-" * 100)

for r in results_sorted[:10]:
    print(f"{r['process_id']:8d} | {r['scheduling_efficiency']:6.1f}% | {r['num_tasks']:5d} | {r['num_edges']:4d} | {r['parallelism']:6.1f} | {r['theoretical_improvement']:7.1f}%")

print("\n" + "=" * 100)
print("调度效率 vs 理论改进潜力")
print("=" * 100)

# 找出最有价值的工作流：效率低 + 改进潜力大 + 有依赖关系
candidates = []
for r in results:
    if r['scheduling_efficiency'] < 80 and r['theoretical_improvement'] > 5 and r['num_edges'] > 10:
        candidates.append(r)

candidates = sorted(candidates, key=lambda x: x['theoretical_improvement'], reverse=True)

print(f"\n最有价值的候选工作流（效率 < 80% + 改进潜力 > 5% + 有依赖关系）:")
print("-" * 100)
print(f"{'PID':>8} | {'效率':>6} | {'改进潜力':>8} | {'任务数':>5} | {'边数':>4} | {'并行度':>6}")
print("-" * 100)

for r in candidates[:15]:
    print(f"{r['process_id']:8d} | {r['scheduling_efficiency']:6.1f}% | {r['theoretical_improvement']:7.1f}% | {r['num_tasks']:5d} | {r['num_edges']:4d} | {r['parallelism']:6.1f}")

if candidates:
    print(f"\n推荐用于演示深度强化学习的工作流: {candidates[0]['process_id']}")
    best = candidates[0]
    print(f"  - 调度效率: {best['scheduling_efficiency']:.1f}% (有 {100-best['scheduling_efficiency']:.1f}% 的优化空间)")
    print(f"  - 理论改进潜力: {best['theoretical_improvement']:.1f}%")
    print(f"  - 任务数: {best['num_tasks']}, 依赖边数: {best['num_edges']}")
    print(f"  - 并行度: {best['parallelism']:.1f}")
