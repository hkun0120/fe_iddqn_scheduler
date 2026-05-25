#!/usr/bin/env python3
"""
分析深度强化学习在调度中的作用
"""

import pandas as pd
from sqlalchemy import create_engine
import networkx as nx

engine = create_engine('mysql+pymysql://root:@localhost:3306/whalesb')

def analyze_workflow_parallelism(process_id):
    """分析单个工作流的并行度"""
    process = pd.read_sql(f'SELECT * FROM t_ds_process_instance WHERE id = {process_id}', engine)
    pdc = int(process.iloc[0]['process_definition_code'])
    
    tasks = pd.read_sql(f'SELECT * FROM t_ds_task_instance WHERE process_instance_id = {process_id} AND state = 7', engine)
    deps = pd.read_sql(f'SELECT * FROM t_ds_process_task_relation WHERE process_definition_code = {pdc}', engine)
    task_defs = pd.read_sql(f'''
        SELECT code, name, task_type FROM t_ds_task_definition 
        WHERE code IN (SELECT pre_task_code FROM t_ds_process_task_relation WHERE process_definition_code = {pdc}
        UNION SELECT post_task_code FROM t_ds_process_task_relation WHERE process_definition_code = {pdc})
    ''', engine)
    
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
    
    def get_duration(name):
        t = tasks[tasks['name'] == name]
        if len(t) > 0:
            start = pd.to_datetime(t.iloc[0]['start_time'])
            end = pd.to_datetime(t.iloc[0]['end_time'])
            return max(1, (end - start).total_seconds())
        return 1
    
    # 计算关键路径
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
    
    # 原始 makespan
    original_start = pd.to_datetime(process.iloc[0]['start_time'])
    original_end = pd.to_datetime(process.iloc[0]['end_time'])
    original_makespan = (original_end - original_start).total_seconds()
    
    return {
        'process_id': process_id,
        'tasks': len(G.nodes),
        'edges': len(G.edges),
        'critical_path': cp_length,
        'total_work': total_work,
        'parallelism': parallelism,
        'original_makespan': original_makespan,
        'theoretical_improvement': (original_makespan - cp_length) / original_makespan * 100 if original_makespan > 0 else 0
    }

print("=" * 80)
print("深度强化学习在工作流调度中的作用分析")
print("=" * 80)

# 分析 293718
print("\n工作流 293718 分析:")
print("-" * 60)
result = analyze_workflow_parallelism(293718)
if result:
    print(f"任务数: {result['tasks']}")
    print(f"边数: {result['edges']}")
    print(f"")
    print(f"关键路径长度: {result['critical_path']:.0f}秒 ({result['critical_path']/3600:.2f}小时)")
    print(f"总工作量: {result['total_work']:.0f}秒")
    print(f"原始执行时间: {result['original_makespan']:.0f}秒 ({result['original_makespan']/3600:.2f}小时)")
    print(f"")
    print(f"理论并行度: {result['parallelism']:.2f}")
    print(f"理论最大改进: {result['theoretical_improvement']:.1f}%")
    print("")
    if result['parallelism'] < 1.2:
        print("⚠️  结论: 此工作流高度串行！")
        print("   关键路径 ≈ 总工作量，无论用什么调度算法都无法显著改进。")

# 分析多个工作流
print("\n" + "=" * 80)
print("分析多个工作流的并行度:")
print("=" * 80)

# 获取一些工作流
workflows = pd.read_sql("""
    SELECT id, name FROM t_ds_process_instance 
    WHERE state = 7 
    ORDER BY RAND() 
    LIMIT 20
""", engine)

results = []
for _, wf in workflows.iterrows():
    try:
        r = analyze_workflow_parallelism(wf['id'])
        if r:
            results.append(r)
    except Exception as e:
        pass

if results:
    print(f"\n分析了 {len(results)} 个工作流:")
    print("")
    
    # 分类
    serial = [r for r in results if r['parallelism'] < 1.2]
    low_parallel = [r for r in results if 1.2 <= r['parallelism'] < 2.0]
    high_parallel = [r for r in results if r['parallelism'] >= 2.0]
    
    print(f"高度串行 (并行度 < 1.2): {len(serial)} 个 ({len(serial)/len(results)*100:.0f}%)")
    print(f"低并行度 (1.2 <= 并行度 < 2.0): {len(low_parallel)} 个 ({len(low_parallel)/len(results)*100:.0f}%)")
    print(f"高并行度 (并行度 >= 2.0): {len(high_parallel)} 个 ({len(high_parallel)/len(results)*100:.0f}%)")
    
    if high_parallel:
        print(f"\n高并行度工作流的理论改进潜力:")
        for r in sorted(high_parallel, key=lambda x: -x['theoretical_improvement'])[:5]:
            print(f"  工作流 {r['process_id']}: 并行度={r['parallelism']:.1f}, 理论改进={r['theoretical_improvement']:.1f}%")

print("\n" + "=" * 80)
print("深度强化学习的真正价值:")
print("=" * 80)
print("""
1. 对于高度串行的工作流（如 293718）:
   - 任何调度算法都无法显著改进
   - 优化空间受限于关键路径

2. 深度强化学习的优势场景:
   - 高并行度工作流（多个独立任务可并行）
   - 资源有限时的智能分配
   - 动态环境（任务执行时间不确定）
   - 多目标优化（时间+成本+可靠性）

3. 建议:
   - 先筛选出有优化空间的工作流（并行度 > 1.5）
   - 对这些工作流应用深度强化学习
   - 对串行工作流，考虑优化单个任务执行效率
""")
