#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化高并行度工作流的 DAG 结构图

从数据库加载与 compare_by_parallelism.py 相同的高并行度工作流，
绘制每个工作流的 DAG 图（任务节点 + 依赖边），按层级自上而下排列。
"""

import os
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pymysql
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'STHeiti', 'Heiti TC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
WORKFLOW_TASK_MIN = 10
WORKFLOW_TASK_MAX = 60
DB_CANDIDATE_LIMIT = 200
NUM_RESOURCES = 5

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "parallelism_comparison"


def load_workflows_by_parallelism():
    """与 compare_by_parallelism.py 完全一致的加载逻辑，返回高/低并行度两组"""
    random.seed(SEED)
    np.random.seed(SEED)

    conn = pymysql.connect(
        host="localhost", user="root", password="",
        database="whalesb", port=3306, charset="utf8mb4",
    )

    query = f"""
        SELECT pi.id, pi.name, pi.process_definition_code,
               pi.start_time, pi.end_time,
               COUNT(ti.id) AS task_count
        FROM t_ds_process_instance pi
        JOIN t_ds_task_instance ti ON ti.process_instance_id = pi.id
        WHERE pi.state = 7 AND ti.state = 7
        GROUP BY pi.id
        HAVING task_count BETWEEN {WORKFLOW_TASK_MIN} AND {WORKFLOW_TASK_MAX}
        ORDER BY RAND({SEED})
        LIMIT {DB_CANDIDATE_LIMIT}
    """
    processes_df = pd.read_sql(query, conn)
    print(f"候选工作流: {len(processes_df)}")

    all_workflows = []
    used_def_codes = set()

    for _, proc in processes_df.iterrows():
        pid = int(proc["id"])
        def_code = int(proc["process_definition_code"])
        if def_code in used_def_codes:
            continue
        used_def_codes.add(def_code)

        tasks_df = pd.read_sql(f"""
            SELECT id, name, task_type, task_code,
                   task_instance_priority, start_time, end_time
            FROM t_ds_task_instance
            WHERE process_instance_id = {pid} AND state = 7
            ORDER BY start_time
        """, conn)

        if len(tasks_df) < WORKFLOW_TASK_MIN:
            continue

        deps_df = pd.read_sql(f"""
            SELECT pre_task_code, post_task_code
            FROM t_ds_process_task_relation
            WHERE process_definition_code = {def_code}
              AND pre_task_code != 0
        """, conn)

        task_code_to_idx = {}
        task_names = []
        task_durations = []
        for idx, (_, t) in enumerate(tasks_df.iterrows()):
            code = int(t["task_code"]) if pd.notna(t.get("task_code")) else t["id"]
            task_code_to_idx[code] = idx
            task_names.append(str(t["name"]))
            duration = 30.0
            if pd.notna(t["start_time"]) and pd.notna(t["end_time"]):
                try:
                    dur = (pd.to_datetime(t["end_time"]) - pd.to_datetime(t["start_time"])).total_seconds()
                    if dur > 0:
                        duration = dur
                except Exception:
                    pass
            task_durations.append(duration)

        dep_edges = []
        for _, d in deps_df.iterrows():
            pre_code = int(d["pre_task_code"])
            post_code = int(d["post_task_code"])
            if pre_code in task_code_to_idx and post_code in task_code_to_idx:
                dep_edges.append((task_code_to_idx[pre_code], task_code_to_idx[post_code]))

        G = nx.DiGraph()
        G.add_nodes_from(range(len(task_names)))
        G.add_edges_from(dep_edges)

        if not nx.is_directed_acyclic_graph(G):
            try:
                for cycle in nx.simple_cycles(G):
                    if len(cycle) > 1:
                        G.remove_edge(cycle[-1], cycle[0])
                dep_edges = list(G.edges())
            except Exception:
                dep_edges = []
                G = nx.DiGraph()
                G.add_nodes_from(range(len(task_names)))

        # 并行度指标
        total_work = sum(task_durations)
        cp_length = total_work
        dag_depth = 1
        if G.number_of_edges() > 0 and nx.is_directed_acyclic_graph(G):
            cp_dict, depth_dict = {}, {}
            try:
                for node in nx.topological_sort(G):
                    preds = list(G.predecessors(node))
                    dur = task_durations[node]
                    if not preds:
                        cp_dict[node] = dur
                        depth_dict[node] = 1
                    else:
                        cp_dict[node] = max(cp_dict[p] for p in preds) + dur
                        depth_dict[node] = max(depth_dict[p] for p in preds) + 1
                cp_length = max(cp_dict.values()) if cp_dict else total_work
                dag_depth = max(depth_dict.values()) if depth_dict else 1
            except Exception:
                cp_length = total_work
                dag_depth = 1
        else:
            cp_length = max(task_durations) if task_durations else total_work
            dag_depth = 1

        par_ratio = total_work / cp_length if cp_length > 0 else 1.0
        ser_ratio = dag_depth / len(task_names) if len(task_names) > 0 else 1.0

        all_workflows.append({
            "name": str(proc["name"]),
            "task_names": task_names,
            "task_durations": task_durations,
            "dependencies": dep_edges,
            "num_tasks": len(task_names),
            "num_deps": len(dep_edges),
            "parallelism_ratio": par_ratio,
            "serialization_ratio": ser_ratio,
            "dag_depth": dag_depth,
            "graph": G,
        })

    conn.close()

    # 与 compare_by_parallelism.py 一致的筛选
    high_par = [w for w in all_workflows
                if w["parallelism_ratio"] >= 2.0 and w["serialization_ratio"] <= 0.5]
    high_par.sort(key=lambda w: -w["parallelism_ratio"])
    high_par = high_par[:25]

    low_par = [w for w in all_workflows
               if w["serialization_ratio"] >= 0.7 and w["parallelism_ratio"] < 2.0]
    low_par.sort(key=lambda w: -w["serialization_ratio"])
    low_par = low_par[:25]

    # 与实验一致的 shuffle + 取 test 部分
    random.seed(SEED)
    random.shuffle(high_par)
    high_test = high_par[15:25]

    random.seed(SEED)
    random.shuffle(low_par)
    low_test = low_par[15:25]

    print(f"高并行度测试集: {len(high_test)} 个工作流")
    print(f"低并行度测试集: {len(low_test)} 个工作流")
    return high_test, low_test


def compute_layer_layout(G, task_count):
    """
    用拓扑排序计算每个节点的层级，同层节点横向排列。
    返回 {node: (x, y)} 的位置字典。
    """
    if G.number_of_edges() == 0:
        # 无依赖：全部放在同一层
        cols = int(np.ceil(np.sqrt(task_count)))
        pos = {}
        for i in range(task_count):
            row = i // cols
            col = i % cols
            pos[i] = (col, -row)
        return pos

    # 计算每个节点的深度（层级）
    depth = {}
    try:
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            if not preds:
                depth[node] = 0
            else:
                depth[node] = max(depth[p] for p in preds) + 1
    except Exception:
        for i in range(task_count):
            depth[i] = 0

    # 确保所有节点都有深度
    for i in range(task_count):
        if i not in depth:
            depth[i] = 0

    # 按层级分组
    layers = defaultdict(list)
    for node, d in depth.items():
        layers[d].append(node)

    # 排列：每层内按节点 id 排序，居中对齐
    max_width = max(len(nodes) for nodes in layers.values()) if layers else 1
    pos = {}
    for layer_idx, nodes in sorted(layers.items()):
        nodes.sort()
        n = len(nodes)
        # 居中排列
        start_x = (max_width - n) / 2.0
        for i, node in enumerate(nodes):
            pos[node] = (start_x + i, -layer_idx)

    return pos


def shorten_task_name(name, max_len=12):
    """缩短任务名"""
    # 常见前缀
    for prefix in ["ods_", "dwd_", "dws_", "ads_", "dim_", "stg_"]:
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
            break
    # 去掉公共长前缀
    parts = name.split("_")
    if len(parts) > 3:
        name = "_".join(parts[-3:])
    if len(name) > max_len:
        name = name[:max_len-2] + ".."
    return name


def draw_workflow_dag(workflow, ax, idx):
    """在给定的 ax 上绘制一个工作流的 DAG"""
    G = workflow["graph"]
    task_names = workflow["task_names"]
    task_durations = workflow["task_durations"]
    n = workflow["num_tasks"]

    pos = compute_layer_layout(G, n)

    # 计算关键路径节点
    cp_nodes = set()
    if G.number_of_edges() > 0:
        try:
            cp_dict = {}
            for node in nx.topological_sort(G):
                preds = list(G.predecessors(node))
                dur = task_durations[node]
                if not preds:
                    cp_dict[node] = (dur, [node])
                else:
                    best_pred = max(preds, key=lambda p: cp_dict[p][0])
                    cp_dict[node] = (cp_dict[best_pred][0] + dur, cp_dict[best_pred][1] + [node])
            # 关键路径 = 最长路径上的节点
            if cp_dict:
                end_node = max(cp_dict, key=lambda k: cp_dict[k][0])
                cp_nodes = set(cp_dict[end_node][1])
        except Exception:
            pass

    # 节点颜色：关键路径红色，普通蓝色，独立节点灰色
    node_colors = []
    for i in range(n):
        if i in cp_nodes:
            node_colors.append("#FF6B6B")  # 关键路径：红
        elif G.degree(i) == 0:
            node_colors.append("#CCCCCC")  # 独立节点：灰
        else:
            node_colors.append("#4ECDC4")  # 普通：青

    # 节点大小：按 duration 归一化
    dur_arr = np.array(task_durations)
    min_d, max_d = dur_arr.min(), dur_arr.max()
    if max_d > min_d:
        node_sizes = 200 + 600 * (dur_arr - min_d) / (max_d - min_d)
    else:
        node_sizes = np.full(n, 400)

    # 绘制边
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        # 关键路径上的边用粗红线
        if u in cp_nodes and v in cp_nodes:
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#FF4444",
                                        lw=2.0, connectionstyle="arc3,rad=0.05"))
        else:
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#888888",
                                        lw=0.8, connectionstyle="arc3,rad=0.05"))

    # 绘制节点
    for i in range(n):
        x, y = pos[i]
        ax.scatter(x, y, s=node_sizes[i], c=node_colors[i],
                   edgecolors="black", linewidth=0.8, zorder=5)
        # 节点标签：编号
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=6, fontweight="bold", zorder=6)

    # 标题
    wf_name = workflow["name"]
    if len(wf_name) > 55:
        wf_name = wf_name[:52] + "..."
    ax.set_title(f"#{idx+1} {wf_name}\n"
                 f"Tasks={n}  Deps={workflow['num_deps']}  "
                 f"ParRatio={workflow['parallelism_ratio']:.1f}  "
                 f"Depth={workflow['dag_depth']}",
                 fontsize=8, pad=4)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("加载工作流...")
    high_test, low_test = load_workflows_by_parallelism()

    if not high_test:
        print("未找到高并行度工作流")
        return

    n_wf = len(high_test)

    # ── 画一张大图，每个工作流一个子图 ──
    cols = 2
    rows = (n_wf + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    fig.suptitle("高并行度工作流 DAG 结构\n"
                 "(红色节点/粗边 = 关键路径, 青色 = 普通任务, 灰色 = 独立任务, 节点大小 ∝ 执行时长)",
                 fontsize=13, fontweight="bold", y=0.995)

    for i, wf in enumerate(high_test):
        print(f"  绘制 {i+1}/{n_wf}: {wf['name'][:60]}  "
              f"(tasks={wf['num_tasks']}, deps={wf['num_deps']}, par={wf['parallelism_ratio']:.1f})")
        draw_workflow_dag(wf, axes[i], i)

    # 隐藏多余的子图
    for j in range(n_wf, len(axes)):
        axes[j].axis("off")

    # 图例
    legend_elements = [
        mpatches.Patch(color="#FF6B6B", label="关键路径节点 (Critical Path)"),
        mpatches.Patch(color="#4ECDC4", label="普通任务节点 (Normal Task)"),
        mpatches.Patch(color="#CCCCCC", label="独立任务 (No Dependencies)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
               frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.001))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path = OUTPUT_DIR / "high_parallelism_dag_structures.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nDAG 结构图已保存: {out_path}")

    if low_test:
        n_low = len(low_test)
        cols2 = 2
        rows2 = (n_low + cols2 - 1) // cols2
        fig2, axes2 = plt.subplots(rows2, cols2, figsize=(18, rows2 * 6))
        if rows2 == 1:
            axes2 = list(axes2)
        else:
            axes2 = axes2.flatten()
        fig2.suptitle("低并行度工作流 DAG 结构 (对比参考)\n"
                      "(红色节点/粗边 = 关键路径, 青色 = 普通任务, 节点大小 ∝ 执行时长)",
                      fontsize=13, fontweight="bold", y=0.995)
        for i, wf in enumerate(low_test):
            print(f"  绘制低并行度 {i+1}/{n_low}: {wf['name'][:60]}")
            draw_workflow_dag(wf, axes2[i], i)
        for j in range(n_low, len(axes2)):
            axes2[j].axis("off")
        fig2.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
                    frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 0.001))
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        out_path2 = OUTPUT_DIR / "low_parallelism_dag_structures.png"
        fig2.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"低并行度 DAG 结构图已保存: {out_path2}")


if __name__ == "__main__":
    main()
