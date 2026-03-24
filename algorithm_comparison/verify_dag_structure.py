#!/usr/bin/env python3
"""
深入验证：打印工作流 DAG 的真实层级结构（文字版），
直接对照数据库 relation 表数据，证明可视化图与源数据一致。
"""
import pymysql
import pandas as pd
import networkx as nx
from collections import defaultdict

conn = pymysql.connect(host='localhost', user='root', password='',
                       database='whalesb', port=3306, charset='utf8mb4')

# 与 compare_by_parallelism.py 一致的查询
q = """
SELECT pi.id, pi.name, pi.process_definition_code, COUNT(ti.id) AS task_count
FROM t_ds_process_instance pi
JOIN t_ds_task_instance ti ON ti.process_instance_id = pi.id
WHERE pi.state = 7 AND ti.state = 7
GROUP BY pi.id
HAVING task_count BETWEEN 10 AND 60
ORDER BY RAND(42)
LIMIT 10
"""
procs = pd.read_sql(q, conn)

for row_i in range(min(3, len(procs))):
    pid = int(procs.iloc[row_i]['id'])
    def_code = int(procs.iloc[row_i]['process_definition_code'])
    wf_name = procs.iloc[row_i]['name']

    # 加载任务
    tasks = pd.read_sql(f"""
        SELECT id, name, task_code, start_time, end_time
        FROM t_ds_task_instance
        WHERE process_instance_id = {pid} AND state = 7
        ORDER BY start_time
    """, conn)

    # 建立 task_code -> 简称 和 task_code -> idx 的映射
    code_to_name = {}
    code_to_idx = {}
    idx_to_code = {}
    idx_to_name = {}
    for idx, (_, t) in enumerate(tasks.iterrows()):
        code = int(t['task_code'])
        name = str(t['name'])
        code_to_name[code] = name
        code_to_idx[code] = idx
        idx_to_code[idx] = code
        idx_to_name[idx] = name

    # 加载依赖
    deps_all = pd.read_sql(f"""
        SELECT pre_task_code, post_task_code
        FROM t_ds_process_task_relation
        WHERE process_definition_code = {def_code}
          AND pre_task_code != 0
    """, conn)

    # 构建 DAG
    G = nx.DiGraph()
    G.add_nodes_from(range(len(tasks)))
    edges_detail = []
    for _, d in deps_all.iterrows():
        pre = int(d['pre_task_code'])
        post = int(d['post_task_code'])
        if pre in code_to_idx and post in code_to_idx:
            pi_ = code_to_idx[pre]
            po_ = code_to_idx[post]
            G.add_edge(pi_, po_)
            edges_detail.append((pi_, po_, code_to_name[pre][:20], code_to_name[post][:20]))

    # 检查是否DAG
    if not nx.is_directed_acyclic_graph(G):
        print(f"⚠ 工作流 {wf_name} 有环！")
        continue

    # 按拓扑层级分组
    depth = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            depth[node] = 0
        else:
            depth[node] = max(depth[p] for p in preds) + 1

    layers = defaultdict(list)
    for node, d in depth.items():
        layers[d].append(node)

    # 计算每个节点的入度和出度
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    print("\n" + "=" * 100)
    print(f"工作流: {wf_name}")
    print(f"PID={pid}, def_code={def_code}")
    print(f"任务数={len(tasks)}, 依赖数={len(edges_detail)}, DAG深度={max(layers.keys())+1}")
    print("=" * 100)

    print("\n--- DAG 层级结构 (每层内的任务可以并行执行) ---\n")
    for layer_idx in sorted(layers.keys()):
        nodes = sorted(layers[layer_idx])
        n_nodes = len(nodes)
        print(f"  Layer {layer_idx} ({n_nodes} 个任务{'  ← 可并行' if n_nodes > 1 else ''}):")
        for node in nodes:
            name = idx_to_name[node][:40]
            preds = list(G.predecessors(node))
            succs = list(G.successors(node))
            pred_str = ""
            if preds:
                pred_names = [idx_to_name[p][:15] for p in preds]
                pred_str = f"  ← 依赖: [{', '.join(pred_names)}]"
            succ_str = ""
            if succs:
                succ_str = f"  → 后续 {len(succs)} 个任务"
            print(f"    [{node:>2}] {name:42s} (in={in_deg[node]}, out={out_deg[node]}){pred_str}{succ_str}")
        if layer_idx < max(layers.keys()):
            # 画出本层到下层的连线
            edges_this = [(u, v) for u, v in G.edges() if depth[u] == layer_idx]
            if edges_this:
                down_nodes = set(v for _, v in edges_this)
                up_nodes = set(u for u, _ in edges_this)
                if len(up_nodes) <= 3 and len(down_nodes) > 3:
                    for u in up_nodes:
                        targets = [v for uu, v in edges_this if uu == u]
                        print(f"        [{node:>2}] ──┬──> [{', '.join(str(t) for t in targets)}]  (扇出 {len(targets)})")
                print(f"        {'│' * min(n_nodes, 10)}")

    # 关键统计
    fan_out_nodes = [(n, out_deg[n]) for n in G.nodes() if out_deg[n] > 3]
    fan_in_nodes = [(n, in_deg[n]) for n in G.nodes() if in_deg[n] > 3]
    isolated = [n for n in G.nodes() if in_deg[n] == 0 and out_deg[n] == 0]

    print(f"\n--- 关键拓扑特征 ---")
    if fan_out_nodes:
        print(f"  扇出节点 (out>3):")
        for n, deg in sorted(fan_out_nodes, key=lambda x: -x[1]):
            print(f"    [{n}] {idx_to_name[n][:40]}  出度={deg}")
    if fan_in_nodes:
        print(f"  扇入节点 (in>3):")
        for n, deg in sorted(fan_in_nodes, key=lambda x: -x[1]):
            print(f"    [{n}] {idx_to_name[n][:40]}  入度={deg}")
    if isolated:
        print(f"  孤立节点 (无依赖): {len(isolated)}")
        for n in isolated:
            print(f"    [{n}] {idx_to_name[n][:40]}")

    # 最大并行度
    max_par = max(len(nodes) for nodes in layers.values())
    avg_par = len(tasks) / (max(layers.keys()) + 1)
    print(f"\n  最大层宽度(最大并行度): {max_par}")
    print(f"  平均层宽度: {avg_par:.1f}")
    print(f"  DAG 深度: {max(layers.keys()) + 1}")

conn.close()
