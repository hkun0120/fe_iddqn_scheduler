#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按工作流并行度分组的 GA-HPO FE-IDDQN vs 原始 FE-IDDQN 对比实验

核心思路：
  用 DAG 结构分析工作流的真实并行度：
    并行度 = 总工作量 / 关键路径长度
    串行度 = DAG深度 / 任务数

  将工作流分为两组：
  1. 高并行度组：任务间依赖少、可大量并行执行
  2. 低并行度组：任务间依赖多、几乎串行执行

  分别在两组上训练 & 评估，验证 GA-HPO FE-IDDQN 在不同并行度下的表现差异。

使用方法:
    .venv/bin/python3 algorithm_comparison/compare_by_parallelism.py
"""

import json
import logging
import os
import random
import sys
import time
import warnings
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pymysql
import pandas as pd
import networkx as nx

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environment.enhanced_workflow_simulator import EnhancedWorkflowSimulator

warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 全局配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 42
DEVICE = "cpu"

TRAIN_EPISODES = 1000
EVAL_EPISODES = 5
NUM_RESOURCES = 5
COMPACT_DIM = 128

# 每组工作流数量
WORKFLOWS_PER_GROUP_TRAIN = 15
WORKFLOWS_PER_GROUP_TEST = 10

# 工作流任务数范围
WORKFLOW_TASK_MIN = 10
WORKFLOW_TASK_MAX = 60

# 从数据库加载的候选工作流数量
DB_CANDIDATE_LIMIT = 200

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "parallelism_comparison"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 从 MySQL 加载并分析工作流并行度
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_and_classify_workflows(logger) -> Tuple[List[Dict], List[Dict]]:
    """
    从 MySQL 加载工作流，计算 DAG 并行度指标，分为高/低并行度两组。

    并行度指标：
      - parallelism_ratio = total_work / critical_path_length
        高并行度：ratio >= 2.0（任务可以 2 路以上并行）
      - serialization_ratio = dag_depth / num_tasks
        低并行度：ratio >= 0.8（几乎串行）

    Returns:
        (high_parallelism_workflows, low_parallelism_workflows)
    """
    logger.info("连接 MySQL 数据库 (localhost/whalesb)...")
    conn = pymysql.connect(
        host="localhost", user="root", password="",
        database="whalesb", port=3306, charset="utf8mb4",
    )

    # 获取足够多的候选工作流
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
    logger.info(f"找到 {len(processes_df)} 个候选工作流")

    all_workflows = []
    used_def_codes = set()

    for _, proc in processes_df.iterrows():
        pid = int(proc["id"])
        def_code = int(proc["process_definition_code"])
        if def_code in used_def_codes:
            continue
        used_def_codes.add(def_code)

        # 加载任务
        tasks_df = pd.read_sql(f"""
            SELECT id, name, task_type, task_code,
                   task_instance_priority, start_time, end_time
            FROM t_ds_task_instance
            WHERE process_instance_id = {pid} AND state = 7
            ORDER BY start_time
        """, conn)

        if len(tasks_df) < WORKFLOW_TASK_MIN:
            continue

        # 加载依赖
        deps_df = pd.read_sql(f"""
            SELECT pre_task_code, post_task_code
            FROM t_ds_process_task_relation
            WHERE process_definition_code = {def_code}
              AND pre_task_code != 0
        """, conn)

        # 构建任务列表
        task_code_to_idx = {}
        task_list = []
        task_durations = []
        for idx, (_, t) in enumerate(tasks_df.iterrows()):
            code = int(t["task_code"]) if pd.notna(t.get("task_code")) else t["id"]
            task_code_to_idx[code] = idx
            duration = 30.0
            if pd.notna(t["start_time"]) and pd.notna(t["end_time"]):
                try:
                    dur = (pd.to_datetime(t["end_time"]) - pd.to_datetime(t["start_time"])).total_seconds()
                    if dur > 0:
                        duration = dur
                except Exception:
                    pass
            tt = str(t.get("task_type", "SHELL"))
            cpu_map = {"SQL": 1, "SHELL": 1, "PYTHON": 2, "JAVA": 2, "SPARK": 4, "FLINK": 4, "HTTP": 1}
            mem_map = {"SQL": 2, "SHELL": 1, "PYTHON": 4, "JAVA": 4, "SPARK": 8, "FLINK": 8, "HTTP": 1}
            priority = int(t.get("task_instance_priority", 0)) if pd.notna(t.get("task_instance_priority")) else 0

            task_list.append({
                "id": idx, "duration": duration,
                "cpu_req": cpu_map.get(tt, 1), "memory_req": mem_map.get(tt, 2),
                "priority": priority, "task_type": tt,
            })
            task_durations.append(duration)

        # 构建 DAG，计算并行度指标
        dep_edges = []
        for _, d in deps_df.iterrows():
            pre_code = int(d["pre_task_code"])
            post_code = int(d["post_task_code"])
            if pre_code in task_code_to_idx and post_code in task_code_to_idx:
                dep_edges.append((task_code_to_idx[pre_code], task_code_to_idx[post_code]))

        G = nx.DiGraph()
        G.add_nodes_from(range(len(task_list)))
        G.add_edges_from(dep_edges)

        # 修复环
        if not nx.is_directed_acyclic_graph(G):
            try:
                for cycle in nx.simple_cycles(G):
                    if len(cycle) > 1:
                        G.remove_edge(cycle[-1], cycle[0])
                dep_edges = list(G.edges())
            except Exception:
                dep_edges = []
                G = nx.DiGraph()
                G.add_nodes_from(range(len(task_list)))

        # 计算关键路径长度 (考虑任务执行时间)
        cp_length = 0.0
        total_work = sum(task_durations)
        dag_depth = 1

        if G.number_of_edges() > 0 and nx.is_directed_acyclic_graph(G):
            # 关键路径：最长加权路径
            cp_dict = {}
            depth_dict = {}
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
            # 无依赖，所有任务独立 → 关键路径 = 最长单任务
            cp_length = max(task_durations) if task_durations else total_work
            dag_depth = 1

        # 并行度指标
        parallelism_ratio = total_work / cp_length if cp_length > 0 else 1.0
        serialization_ratio = dag_depth / len(task_list) if len(task_list) > 0 else 1.0
        dependency_density = len(dep_edges) / len(task_list) if len(task_list) > 0 else 0

        # 资源定义
        set_seed(SEED + pid)  # 确保每个工作流资源配置一致
        resources = []
        for j in range(NUM_RESOURCES):
            resources.append({
                "id": j,
                "cpu_capacity": random.choice([8, 16]),
                "memory_capacity": random.choice([16, 32]),
                "speed_factor": round(random.uniform(0.8, 1.2), 2),
            })

        # 原始 makespan
        original_makespan = 0.0
        if pd.notna(proc["start_time"]) and pd.notna(proc["end_time"]):
            try:
                original_makespan = (
                    pd.to_datetime(proc["end_time"]) - pd.to_datetime(proc["start_time"])
                ).total_seconds()
            except Exception:
                pass

        all_workflows.append({
            "name": str(proc["name"]),
            "tasks": task_list,
            "resources": resources,
            "dependencies": dep_edges,
            "original_makespan": original_makespan,
            "num_tasks": len(task_list),
            "num_deps": len(dep_edges),
            # 并行度指标
            "parallelism_ratio": parallelism_ratio,
            "serialization_ratio": serialization_ratio,
            "dependency_density": dependency_density,
            "dag_depth": dag_depth,
            "critical_path_length": cp_length,
            "total_work": total_work,
        })

    conn.close()
    logger.info(f"成功分析 {len(all_workflows)} 个工作流")

    # ─── 按并行度分组 ───
    # 高并行度：parallelism_ratio >= 2.0 且 serialization_ratio <= 0.5
    # 低并行度：serialization_ratio >= 0.7 且 parallelism_ratio < 2.0
    high_par = [w for w in all_workflows
                if w["parallelism_ratio"] >= 2.0 and w["serialization_ratio"] <= 0.5]
    low_par = [w for w in all_workflows
               if w["serialization_ratio"] >= 0.7 and w["parallelism_ratio"] < 2.0]

    # 排序并去除极端异常值
    high_par.sort(key=lambda w: -w["parallelism_ratio"])
    low_par.sort(key=lambda w: w["serialization_ratio"])

    need_per_group = WORKFLOWS_PER_GROUP_TRAIN + WORKFLOWS_PER_GROUP_TEST

    logger.info(f"高并行度候选: {len(high_par)} 个 (需要 {need_per_group} 个)")
    logger.info(f"低并行度候选: {len(low_par)} 个 (需要 {need_per_group} 个)")

    # 如果候选不足，放宽条件
    if len(high_par) < need_per_group:
        logger.info("高并行度候选不足，放宽条件...")
        high_par = [w for w in all_workflows
                    if w["parallelism_ratio"] >= 1.5 or w["dependency_density"] < 0.5]
        high_par.sort(key=lambda w: -w["parallelism_ratio"])
        logger.info(f"  放宽后: {len(high_par)} 个")

    if len(low_par) < need_per_group:
        logger.info("低并行度候选不足，放宽条件...")
        low_par = [w for w in all_workflows
                   if w["serialization_ratio"] >= 0.6 and w["dependency_density"] >= 0.8]
        low_par.sort(key=lambda w: -w["serialization_ratio"])
        logger.info(f"  放宽后: {len(low_par)} 个")

    high_par = high_par[:need_per_group]
    low_par = low_par[:need_per_group]

    # 打印分组统计
    for label, group in [("高并行度", high_par), ("低并行度", low_par)]:
        if not group:
            continue
        avg_pr = np.mean([w["parallelism_ratio"] for w in group])
        avg_sr = np.mean([w["serialization_ratio"] for w in group])
        avg_dd = np.mean([w["dependency_density"] for w in group])
        avg_tasks = np.mean([w["num_tasks"] for w in group])
        avg_depth = np.mean([w["dag_depth"] for w in group])
        logger.info(f"  [{label}] {len(group)} 个: 并行度={avg_pr:.2f}, "
                     f"串行度={avg_sr:.3f}, 依赖密度={avg_dd:.2f}, "
                     f"平均任务数={avg_tasks:.1f}, 平均深度={avg_depth:.1f}")

    return high_par, low_par


def make_env(workflow: Dict) -> EnhancedWorkflowSimulator:
    return EnhancedWorkflowSimulator(
        tasks=workflow["tasks"],
        resources=workflow["resources"],
        dependencies=workflow["dependencies"],
        use_dag_aware=True,
        use_critical_path_priority=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 启发式基线
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HeuristicScheduler:
    def __init__(self, name, num_resources):
        self.name = name
        self.num_resources = num_resources
        self.counter = 0
    def reset(self): self.counter = 0
    def select_action(self, env): raise NotImplementedError

class RandomScheduler(HeuristicScheduler):
    def __init__(self, n): super().__init__("Random", n)
    def select_action(self, env): return random.randint(0, self.num_resources - 1)

class RoundRobinScheduler(HeuristicScheduler):
    def __init__(self, n): super().__init__("RoundRobin", n)
    def select_action(self, env):
        a = self.counter % self.num_resources; self.counter += 1; return a

class ShortestJobFirst(HeuristicScheduler):
    def __init__(self, n): super().__init__("SJF", n)
    def select_action(self, env):
        loads = [env.resource_states[r["id"]].available_time for r in env.resources]
        return int(np.argmin(loads))

class EarliestFinishTime(HeuristicScheduler):
    def __init__(self, n): super().__init__("EFT", n)
    def select_action(self, env):
        if not env.ready_tasks: return 0
        task_id = env.ready_tasks[0]
        task = next(t for t in env.tasks if t["id"] == task_id)
        dur = task.get("duration", 1.0)
        finish_times = [env.resource_states[r["id"]].available_time + dur for r in env.resources]
        return int(np.argmin(finish_times))

class CriticalPathFirst(HeuristicScheduler):
    def __init__(self, n): super().__init__("CPOP", n)
    def select_action(self, env):
        if not env.ready_tasks: return 0
        task_id = env.ready_tasks[0]
        is_critical = task_id in getattr(env, "critical_path_set", set())
        if is_critical:
            loads = [env.resource_states[r["id"]].available_time for r in env.resources]
            return int(np.argmin(loads))
        else:
            a = self.counter % self.num_resources; self.counter += 1; return a


def evaluate_heuristic(scheduler, workflow, num_episodes=1):
    makespans, utils, balances = [], [], []
    for _ in range(num_episodes):
        env = make_env(workflow)
        scheduler.reset()
        state = env.reset()
        done = False
        steps = 0
        max_steps = len(workflow["tasks"]) * 3
        while not done and steps < max_steps:
            action = scheduler.select_action(env)
            state, reward, done, info = env.step(action)
            steps += 1
        result = env.get_scheduling_result()
        makespans.append(result["makespan"])
        utils.append(result["resource_utilization"])
        balances.append(result["load_balance"])
    return {
        "makespan": np.mean(makespans),
        "utilization": np.mean(utils),
        "load_balance": np.mean(balances),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GA-HPO FE-IDDQN (紧凑状态 + Dueling DQN + 专家知识)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compact_state(state):
    """128 维紧凑状态"""
    tf = state.get("task_features", np.zeros((1, 19)))
    rf = state.get("resource_features", np.zeros((1, 11)))
    gf = state.get("global_features", np.zeros(11))

    gf_flat = gf.flatten()
    if len(gf_flat) < 11: gf_flat = np.concatenate([gf_flat, np.zeros(11 - len(gf_flat))])
    gf_flat = gf_flat[:11]

    task_mean = tf.mean(axis=0)
    task_std = tf.std(axis=0)

    ready_mask = tf[:, 4] > 0.5
    if ready_mask.any():
        current_task = tf[np.where(ready_mask)[0][0]]
    else:
        current_task = tf.mean(axis=0)

    rf_flat = rf.flatten()
    if len(rf_flat) < 55: rf_flat = np.concatenate([rf_flat, np.zeros(55 - len(rf_flat))])
    rf_flat = rf_flat[:55]

    cp = state.get("critical_path_mask", np.zeros(tf.shape[0]))
    cp_stats = np.array([cp.sum() / max(1, len(cp)), cp.mean()], dtype=np.float32)

    nd = state.get("node_depths", np.zeros(tf.shape[0]))
    nd_f = nd.astype(np.float32)
    nd_stats = np.array([
        nd_f.mean() / 10.0 if nd_f.size > 0 else 0.0,
        nd_f.std() / 10.0 if nd_f.size > 0 else 0.0,
        nd_f.max() / 10.0 if nd_f.size > 0 else 0.0,
    ], dtype=np.float32)

    compact = np.concatenate([gf_flat, task_mean, task_std, current_task, rf_flat, cp_stats, nd_stats]).astype(np.float32)
    compact = np.clip(compact, -100.0, 100.0)
    np.nan_to_num(compact, nan=0.0, posinf=100.0, neginf=-100.0, copy=False)
    return compact


class RunningNormalizer:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count
    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        std = np.sqrt(np.maximum(self.var, 1e-8)).astype(np.float32)
        return np.clip((x - self.mean.astype(np.float32)) / std, -10.0, 10.0)


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.adv_stream = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, action_dim))
    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.adv_stream(f)
        return v + a - a.mean(dim=-1, keepdim=True)


def _load_balance_bonus(env) -> float:
    loads = [env.resource_states[r["id"]].available_time for r in env.resources]
    if not loads: return 0.0
    max_load = max(loads)
    mean_load = np.mean(loads)
    if max_load == 0: return 1.0
    return 1.0 - (max_load - mean_load) / (max_load + 1e-8)


def train_ga_hpo(workflows_train, workflows_test, logger, label=""):
    """训练 GA-HPO FE-IDDQN"""
    logger.info(f"[GA-HPO FE-IDDQN{label}] 训练 {TRAIN_EPISODES} episodes...")

    action_dim = NUM_RESOURCES
    device = torch.device(DEVICE)

    q_net = DuelingDQN(COMPACT_DIM, action_dim, hidden=256).to(device)
    target_net = DuelingDQN(COMPACT_DIM, action_dim, hidden=256).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)

    normalizer = RunningNormalizer(COMPACT_DIM)

    GAMMA, TAU, BATCH_SIZE, BUFFER_SIZE = 0.99, 0.005, 64, 100000
    WARMUP, TRAIN_FREQ, TARGET_UPDATE = 100, 2, 50
    EPS_START, EPS_END, EPS_DECAY = 0.5, 0.02, 0.998
    N_STEP, GRAD_CLIP = 3, 1.0

    replay_buf = deque(maxlen=BUFFER_SIZE)
    n_step_buf = deque(maxlen=N_STEP)
    priorities = deque(maxlen=BUFFER_SIZE)
    epsilon = EPS_START
    total_steps = 0

    def _add_exp(s, a, r, s2, d):
        n_step_buf.append((s, a, r, s2, d))
        if len(n_step_buf) == N_STEP or d:
            R = 0.0
            for i in reversed(range(len(n_step_buf))):
                R = n_step_buf[i][2] + GAMMA * R * (1.0 - float(n_step_buf[i][4]))
            replay_buf.append((n_step_buf[0][0], n_step_buf[0][1], R,
                               n_step_buf[-1][3], n_step_buf[-1][4], GAMMA ** len(n_step_buf)))
            priorities.append(2.0)
            if d: n_step_buf.clear()

    def _sample():
        p = np.array(list(priorities), dtype=np.float32) ** 0.6
        p /= p.sum()
        idx = np.random.choice(len(replay_buf), BATCH_SIZE, p=p, replace=False)
        w = (len(replay_buf) * p[idx]) ** (-0.4)
        w /= w.max()
        return [replay_buf[i] for i in idx], idx, w

    # SJF 专家预填充
    sjf = ShortestJobFirst(NUM_RESOURCES)
    exp_count = 0
    for wf in workflows_train:
        for _ in range(5):
            env = make_env(wf); sjf.reset(); state = env.reset(); done = False; steps = 0
            while not done and steps < len(wf["tasks"]) * 3:
                raw = _compact_state(state); normalizer.update(raw); norm = normalizer.normalize(raw)
                action = sjf.select_action(env)
                next_state, reward, done, _ = env.step(action)
                shaped = reward + _load_balance_bonus(env) * 2.0
                next_norm = normalizer.normalize(_compact_state(next_state))
                _add_exp(norm, action, shaped, next_norm, done)
                state = next_state; steps += 1; exp_count += 1
    logger.info(f"    SJF 预填充 {exp_count} 条经验")

    # 在线训练
    t0 = time.time()
    train_rewards = []
    for ep in range(TRAIN_EPISODES):
        wf = workflows_train[ep % len(workflows_train)]
        env = make_env(wf); state = env.reset(); done = False; steps = 0; ep_reward = 0.0
        max_steps = len(wf["tasks"]) * 3

        while not done and steps < max_steps:
            raw = _compact_state(state); normalizer.update(raw); norm = normalizer.normalize(raw)
            if random.random() < epsilon:
                action = (random.randint(0, action_dim - 1)
                          if random.random() < 0.5
                          else int(np.argmin([env.resource_states[r["id"]].available_time for r in env.resources])))
            else:
                with torch.no_grad():
                    action = q_net(torch.FloatTensor(norm).unsqueeze(0).to(device)).argmax(dim=-1).item()
            next_state, reward, done, _ = env.step(action)
            shaped = reward + _load_balance_bonus(env) * 2.0
            next_norm = normalizer.normalize(_compact_state(next_state))
            _add_exp(norm, action, shaped, next_norm, done)
            total_steps += 1

            if total_steps >= WARMUP and total_steps % TRAIN_FREQ == 0 and len(replay_buf) >= BATCH_SIZE:
                batch, bi, isw = _sample()
                s_b = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
                a_b = torch.LongTensor([b[1] for b in batch]).to(device)
                r_b = torch.FloatTensor([b[2] for b in batch]).to(device)
                s2_b = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
                d_b = torch.BoolTensor([b[4] for b in batch]).to(device)
                g_b = torch.FloatTensor([b[5] for b in batch]).to(device)
                w_b = torch.FloatTensor(isw).to(device)
                q_vals = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_acts = q_net(s2_b).argmax(dim=1)
                    next_q = target_net(s2_b).gather(1, next_acts.unsqueeze(1)).squeeze(1)
                    target_q = r_b + g_b * next_q * (~d_b)
                td_errors = torch.abs(q_vals - target_q).detach()
                loss = (w_b * F.smooth_l1_loss(q_vals, target_q, reduction='none')).mean()
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP); optimizer.step()
                for idx_i, td in zip(bi, td_errors.cpu().numpy()):
                    priorities[idx_i] = float(td) + 1e-6
            if total_steps % TARGET_UPDATE == 0:
                for tp, qp in zip(target_net.parameters(), q_net.parameters()):
                    tp.data.copy_(TAU * qp.data + (1 - TAU) * tp.data)
            ep_reward += shaped; steps += 1; state = next_state
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        train_rewards.append(ep_reward)
        if (ep + 1) % 200 == 0:
            logger.info(f"    ep {ep+1}/{TRAIN_EPISODES}  reward={np.mean(train_rewards[-200:]):.2f}  eps={epsilon:.3f}")

    train_time = time.time() - t0
    logger.info(f"    训练完成, 耗时 {train_time:.1f}s")

    # 测试
    q_net.eval()
    test_results = []
    for wf in workflows_test:
        env = make_env(wf)
        makespans, utils, bals = [], [], []
        for _ in range(EVAL_EPISODES):
            state = env.reset(); done = False; steps = 0
            while not done and steps < len(wf["tasks"]) * 3:
                norm = normalizer.normalize(_compact_state(state))
                with torch.no_grad():
                    action = q_net(torch.FloatTensor(norm).unsqueeze(0).to(device)).argmax(dim=-1).item()
                state, _, done, _ = env.step(action); steps += 1
            r = env.get_scheduling_result()
            makespans.append(r["makespan"]); utils.append(r["resource_utilization"]); bals.append(r["load_balance"])
        test_results.append({
            "name": wf["name"], "num_tasks": wf["num_tasks"], "num_deps": wf["num_deps"],
            "parallelism_ratio": wf["parallelism_ratio"],
            "serialization_ratio": wf["serialization_ratio"],
            "dag_depth": wf["dag_depth"],
            "original_makespan": wf["original_makespan"],
            "makespan": float(np.mean(makespans)), "utilization": float(np.mean(utils)),
            "load_balance": float(np.mean(bals)),
        })
    return {"algorithm": "GA-HPO FE-IDDQN", "train_time": train_time, "test_results": test_results}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 原始 FE-IDDQN (DualStreamNetwork)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _pad_features(arr, max_rows):
    if arr is None: return None
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    n, d = arr.shape
    if n >= max_rows: return arr[:max_rows]
    padded = np.zeros((max_rows, d), dtype=arr.dtype)
    padded[:n] = arr
    return padded


class OriginalAttentionModule(nn.Module):
    def __init__(self, input_dim, attention_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, attention_dim // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(attention_dim // 2, input_dim))
    def forward(self, x):
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm(x + attn_out)
        ff_out = self.feed_forward(x)
        return self.layer_norm(x + ff_out)


class OriginalTaskStream(nn.Module):
    def __init__(self, input_dim, hidden_dims, attention_dim, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.input_embedding = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU(),
                nn.Dropout(dropout_rate), nn.LayerNorm(hidden_dims[i + 1])))
        self.attention = OriginalAttentionModule(hidden_dims[-1], attention_dim, num_heads)
        self.output_dim = hidden_dims[-1]
    def forward(self, x):
        batch_size, num_tasks, _ = x.shape
        x = x.view(-1, self.input_dim)
        x = F.relu(self.input_embedding(x))
        for layer in self.hidden_layers: x = layer(x)
        x = x.view(batch_size, num_tasks, -1)
        return self.attention(x)


class OriginalResourceStream(nn.Module):
    def __init__(self, input_dim, hidden_dims, attention_dim, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.input_embedding = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU(),
                nn.Dropout(dropout_rate), nn.LayerNorm(hidden_dims[i + 1])))
        self.attention = OriginalAttentionModule(hidden_dims[-1], attention_dim, num_heads)
        self.output_dim = hidden_dims[-1]
    def forward(self, x):
        batch_size, num_resources, _ = x.shape
        x = x.view(-1, self.input_dim)
        x = F.relu(self.input_embedding(x))
        for layer in self.hidden_layers: x = layer(x)
        x = x.view(batch_size, num_resources, -1)
        return self.attention(x)


class OriginalFeatureFusion(nn.Module):
    def __init__(self, task_dim, resource_dim, fusion_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.task_projection = nn.Linear(task_dim, fusion_dim)
        self.resource_projection = nn.Linear(resource_dim, fusion_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4, dropout=dropout_rate, batch_first=True)
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, output_dim))
        self.layer_norm = nn.LayerNorm(fusion_dim)
    def forward(self, task_features, resource_features):
        task_proj = self.task_projection(task_features)
        resource_proj = self.resource_projection(resource_features)
        task_attended, _ = self.cross_attention(task_proj, resource_proj, resource_proj)
        task_attended = self.layer_norm(task_proj + task_attended)
        resource_attended, _ = self.cross_attention(resource_proj, task_proj, task_proj)
        resource_attended = self.layer_norm(resource_proj + resource_attended)
        task_global = torch.mean(task_attended, dim=1)
        resource_global = torch.mean(resource_attended, dim=1)
        return self.fusion_network(torch.cat([task_global, resource_global], dim=1))


class OriginalDualStreamDQN(nn.Module):
    def __init__(self, task_input_dim, resource_input_dim, action_dim,
                 task_hidden_dims=(256, 128), resource_hidden_dims=(256, 128),
                 fusion_dim=256, attention_dim=128, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.task_stream = OriginalTaskStream(task_input_dim, list(task_hidden_dims), attention_dim, num_heads, dropout_rate)
        self.resource_stream = OriginalResourceStream(resource_input_dim, list(resource_hidden_dims), attention_dim, num_heads, dropout_rate)
        self.feature_fusion = OriginalFeatureFusion(task_hidden_dims[-1], resource_hidden_dims[-1], fusion_dim, action_dim, dropout_rate)
    def forward(self, task_features, resource_features):
        task_features = F.dropout(task_features, p=0.2, training=self.training)
        resource_features = F.dropout(resource_features, p=0.2, training=self.training)
        task_out = self.task_stream(task_features)
        resource_out = self.resource_stream(resource_features)
        return self.feature_fusion(task_out, resource_out)


def train_original(workflows_train, workflows_test, logger, label=""):
    """训练原始 FE-IDDQN"""
    logger.info(f"[原始 FE-IDDQN{label}] 训练 {TRAIN_EPISODES} episodes...")

    action_dim = NUM_RESOURCES
    device = torch.device(DEVICE)
    MAX_TASKS_PAD = 60
    MAX_RES_PAD = NUM_RESOURCES

    test_env = make_env(workflows_train[0])
    test_state = test_env.reset()
    task_feat_dim = test_state["task_features"].shape[1]
    res_feat_dim = test_state["resource_features"].shape[1]

    q_net = OriginalDualStreamDQN(task_feat_dim, res_feat_dim, action_dim).to(device)
    target_net = OriginalDualStreamDQN(task_feat_dim, res_feat_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=3e-5, weight_decay=1e-5)

    GAMMA, TAU = 0.99, 0.005
    BATCH_SIZE, BUFFER_SIZE = 32, 10000
    WARMUP, TRAIN_FREQ, TARGET_UPDATE = 200, 4, 100
    EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.998
    GRAD_CLIP = 1.0
    PER_ALPHA, PER_BETA, PER_BETA_INC = 0.6, 0.4, 0.001

    replay_buf = deque(maxlen=BUFFER_SIZE)
    priorities = deque(maxlen=BUFFER_SIZE)
    epsilon = EPS_START
    total_steps = 0
    per_beta = PER_BETA

    def _prepare_state(state):
        tf = _pad_features(state["task_features"], MAX_TASKS_PAD)
        rf = _pad_features(state["resource_features"], MAX_RES_PAD)
        return tf, rf

    def _sample():
        nonlocal per_beta
        per_beta = min(1.0, per_beta + PER_BETA_INC)
        p = np.array(list(priorities), dtype=np.float32) ** PER_ALPHA
        p /= p.sum()
        idx = np.random.choice(len(replay_buf), BATCH_SIZE, p=p, replace=False)
        w = (len(replay_buf) * p[idx]) ** (-per_beta)
        w /= w.max()
        return [replay_buf[i] for i in idx], idx, w

    t0 = time.time()
    train_rewards = []

    for ep in range(TRAIN_EPISODES):
        wf = workflows_train[ep % len(workflows_train)]
        env = make_env(wf); state = env.reset(); done = False; steps = 0; ep_reward = 0.0
        max_steps = len(wf["tasks"]) * 3

        while not done and steps < max_steps:
            s_tf, s_rf = _prepare_state(state)
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.FloatTensor(s_tf).unsqueeze(0).to(device),
                                   torch.FloatTensor(s_rf).unsqueeze(0).to(device))
                    action = q_vals.argmax(dim=-1).item()
            next_state, reward, done, _ = env.step(action)
            s2_tf, s2_rf = _prepare_state(next_state)
            replay_buf.append((s_tf, s_rf, action, reward, s2_tf, s2_rf, done))
            priorities.append(2.0)
            total_steps += 1

            if total_steps >= WARMUP and total_steps % TRAIN_FREQ == 0 and len(replay_buf) >= BATCH_SIZE:
                batch, bi, isw = _sample()
                tf_b = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
                rf_b = torch.FloatTensor(np.array([b[1] for b in batch])).to(device)
                a_b = torch.LongTensor([b[2] for b in batch]).to(device)
                r_b = torch.FloatTensor([b[3] for b in batch]).to(device)
                tf2_b = torch.FloatTensor(np.array([b[4] for b in batch])).to(device)
                rf2_b = torch.FloatTensor(np.array([b[5] for b in batch])).to(device)
                d_b = torch.BoolTensor([b[6] for b in batch]).to(device)
                w_b = torch.FloatTensor(isw).to(device)
                q_vals = q_net(tf_b, rf_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_acts = q_net(tf2_b, rf2_b).argmax(dim=1)
                    next_q = target_net(tf2_b, rf2_b).gather(1, next_acts.unsqueeze(1)).squeeze(1)
                    target_q = r_b + GAMMA * next_q * (~d_b)
                td_errors = torch.abs(q_vals - target_q).detach()
                loss = (w_b * F.smooth_l1_loss(q_vals, target_q, reduction='none')).mean()
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP); optimizer.step()
                for idx_i, td in zip(bi, td_errors.cpu().numpy()):
                    priorities[idx_i] = float(td) + 1e-6
            if total_steps % TARGET_UPDATE == 0:
                for tp, qp in zip(target_net.parameters(), q_net.parameters()):
                    tp.data.copy_(TAU * qp.data + (1 - TAU) * tp.data)
            ep_reward += reward; steps += 1; state = next_state
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        train_rewards.append(ep_reward)
        if (ep + 1) % 200 == 0:
            logger.info(f"    ep {ep+1}/{TRAIN_EPISODES}  reward={np.mean(train_rewards[-200:]):.2f}  eps={epsilon:.3f}")

    train_time = time.time() - t0
    logger.info(f"    训练完成, 耗时 {train_time:.1f}s")

    # 测试
    q_net.eval()
    test_results = []
    for wf in workflows_test:
        env = make_env(wf)
        makespans, utils, bals = [], [], []
        for _ in range(EVAL_EPISODES):
            state = env.reset(); done = False; steps = 0
            while not done and steps < len(wf["tasks"]) * 3:
                s_tf, s_rf = _prepare_state(state)
                with torch.no_grad():
                    q_vals = q_net(torch.FloatTensor(s_tf).unsqueeze(0).to(device),
                                   torch.FloatTensor(s_rf).unsqueeze(0).to(device))
                    action = q_vals.argmax(dim=-1).item()
                state, _, done, _ = env.step(action); steps += 1
            r = env.get_scheduling_result()
            makespans.append(r["makespan"]); utils.append(r["resource_utilization"]); bals.append(r["load_balance"])
        test_results.append({
            "name": wf["name"], "num_tasks": wf["num_tasks"], "num_deps": wf["num_deps"],
            "parallelism_ratio": wf["parallelism_ratio"],
            "serialization_ratio": wf["serialization_ratio"],
            "dag_depth": wf["dag_depth"],
            "original_makespan": wf["original_makespan"],
            "makespan": float(np.mean(makespans)), "utilization": float(np.mean(utils)),
            "load_balance": float(np.mean(bals)),
        })
    return {"algorithm": "Orig FE-IDDQN", "train_time": train_time, "test_results": test_results}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_group_experiment(group_name, train_wfs, test_wfs, logger):
    """对一组工作流运行完整的对比实验"""
    logger.info("")
    logger.info("=" * 90)
    logger.info(f"    [{group_name}] 实验: {len(train_wfs)} 训练 + {len(test_wfs)} 测试")
    logger.info("=" * 90)

    # 打印工作流详情
    logger.info(f"  测试集工作流:")
    for i, wf in enumerate(test_wfs):
        logger.info(f"    {i+1}. {wf['name'][:50]:52s} tasks={wf['num_tasks']:>3} deps={wf['num_deps']:>3} "
                     f"par_ratio={wf['parallelism_ratio']:.2f} ser_ratio={wf['serialization_ratio']:.3f} "
                     f"depth={wf['dag_depth']:>3}")

    all_results = []

    # 1. 启发式基线
    logger.info(f"\n  [基线评估]")
    schedulers = [RandomScheduler(NUM_RESOURCES), RoundRobinScheduler(NUM_RESOURCES),
                  ShortestJobFirst(NUM_RESOURCES), EarliestFinishTime(NUM_RESOURCES),
                  CriticalPathFirst(NUM_RESOURCES)]
    for sched in schedulers:
        test_results = []
        for wf in test_wfs:
            r = evaluate_heuristic(sched, wf, num_episodes=EVAL_EPISODES)
            test_results.append({
                "name": wf["name"], "num_tasks": wf["num_tasks"], "num_deps": wf["num_deps"],
                "parallelism_ratio": wf["parallelism_ratio"],
                "serialization_ratio": wf["serialization_ratio"],
                "dag_depth": wf["dag_depth"],
                "original_makespan": wf["original_makespan"], **r})
        all_results.append({"algorithm": sched.name, "train_time": 0.0, "test_results": test_results})
        avg_ms = np.mean([r["makespan"] for r in test_results])
        logger.info(f"    {sched.name:12s}  makespan={avg_ms:.2f}")

    # 2. 原始 FE-IDDQN
    set_seed(SEED)
    orig_result = train_original(train_wfs, test_wfs, logger, f" ({group_name})")
    all_results.append(orig_result)
    avg_ms = np.mean([r["makespan"] for r in orig_result["test_results"]])
    logger.info(f"    Orig FE-IDDQN  makespan={avg_ms:.2f}  (train {orig_result['train_time']:.1f}s)")

    # 3. GA-HPO FE-IDDQN
    set_seed(SEED)
    gahpo_result = train_ga_hpo(train_wfs, test_wfs, logger, f" ({group_name})")
    all_results.append(gahpo_result)
    avg_ms = np.mean([r["makespan"] for r in gahpo_result["test_results"]])
    logger.info(f"    GA-HPO FE-IDDQN  makespan={avg_ms:.2f}  (train {gahpo_result['train_time']:.1f}s)")

    return all_results


def print_comparison(group_name, all_results, test_wfs, logger):
    """打印一组实验的对比结果"""
    logger.info("")
    logger.info("=" * 110)
    logger.info(f"    [{group_name}] 对比总表")
    logger.info("=" * 110)

    header = f"  {'Algorithm':<22s} | {'Makespan':>10s} | {'Std':>10s} | {'Util':>7s} | {'LoadBal':>7s} | {'Train':>8s}"
    logger.info(header)
    logger.info("  " + "-" * 85)

    for res in all_results:
        alg = res["algorithm"]
        ms_list = [r["makespan"] for r in res["test_results"]]
        ut_list = [r["utilization"] for r in res["test_results"]]
        bl_list = [r["load_balance"] for r in res["test_results"]]
        logger.info(f"  {alg:<22s} | {np.mean(ms_list):>10.2f} | {np.std(ms_list):>10.2f} | "
                     f"{np.mean(ut_list):>7.4f} | {np.mean(bl_list):>7.4f} | {res.get('train_time', 0):>7.1f}s")

    # 逐工作流对比
    orig = next((r for r in all_results if "Orig" in r["algorithm"]), None)
    gahpo = next((r for r in all_results if "GA-HPO" in r["algorithm"]), None)
    sjf = next((r for r in all_results if r["algorithm"] == "SJF"), None)

    if orig and gahpo and sjf:
        logger.info("")
        logger.info(f"  {'#':<3} {'name':32s} {'tasks':>5} {'deps':>5} {'par_r':>6} | {'SJF':>10} {'Orig':>10} {'GA-HPO':>10} {'Improve':>8} {'Winner':<10}")
        logger.info("  " + "-" * 115)

        ga_wins, orig_wins, tie = 0, 0, 0
        ga_beats_sjf = 0

        for i in range(len(test_wfs)):
            wf = test_wfs[i]
            sjf_ms = sjf["test_results"][i]["makespan"]
            orig_ms = orig["test_results"][i]["makespan"]
            gahpo_ms = gahpo["test_results"][i]["makespan"]
            improvement = (orig_ms - gahpo_ms) / orig_ms * 100 if orig_ms > 0 else 0

            if gahpo_ms < orig_ms * 0.99:
                winner = "GA-HPO"; ga_wins += 1
            elif orig_ms < gahpo_ms * 0.99:
                winner = "Orig"; orig_wins += 1
            else:
                winner = "Tie"; tie += 1
            if gahpo_ms <= sjf_ms * 1.01:
                ga_beats_sjf += 1

            name = wf["name"][:30]
            logger.info(f"  {i+1:<3} {name:32s} {wf['num_tasks']:>5} {wf['num_deps']:>5} "
                        f"{wf['parallelism_ratio']:>6.2f} | {sjf_ms:>10.1f} {orig_ms:>10.1f} "
                        f"{gahpo_ms:>10.1f} {improvement:>+7.1f}% {winner:<10}")

        n = len(test_wfs)
        orig_avg = np.mean([orig["test_results"][i]["makespan"] for i in range(n)])
        gahpo_avg = np.mean([gahpo["test_results"][i]["makespan"] for i in range(n)])
        sjf_avg = np.mean([sjf["test_results"][i]["makespan"] for i in range(n)])

        logger.info("")
        logger.info(f"  Summary: GA-HPO wins {ga_wins}/{n}, Orig wins {orig_wins}/{n}, Tie {tie}/{n}")
        logger.info(f"  GA-HPO matches/beats SJF: {ga_beats_sjf}/{n}")
        logger.info(f"  GA-HPO vs Orig: {(orig_avg - gahpo_avg) / orig_avg * 100:+.1f}%")
        logger.info(f"  GA-HPO vs SJF:  {(gahpo_avg - sjf_avg) / sjf_avg * 100:+.1f}%")
        logger.info(f"  Orig vs SJF:    {(orig_avg - sjf_avg) / sjf_avg * 100:+.1f}%")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_DIR / f"compare_{ts}.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("ParallelismCompare")

    set_seed(SEED)
    logger.info("=" * 90)
    logger.info("  按工作流并行度分组的 GA-HPO FE-IDDQN vs 原始 FE-IDDQN 对比实验")
    logger.info("=" * 90)
    logger.info(f"  训练: {TRAIN_EPISODES} ep, 评估: {EVAL_EPISODES} ep/wf, 资源: {NUM_RESOURCES}")
    logger.info(f"  每组: {WORKFLOWS_PER_GROUP_TRAIN} 训练 + {WORKFLOWS_PER_GROUP_TEST} 测试")

    # 1. 从 DB 加载并分类工作流
    high_par_all, low_par_all = load_and_classify_workflows(logger)

    if len(high_par_all) < WORKFLOWS_PER_GROUP_TRAIN + WORKFLOWS_PER_GROUP_TEST:
        logger.warning(f"高并行度工作流不足: {len(high_par_all)} < {WORKFLOWS_PER_GROUP_TRAIN + WORKFLOWS_PER_GROUP_TEST}")
    if len(low_par_all) < WORKFLOWS_PER_GROUP_TRAIN + WORKFLOWS_PER_GROUP_TEST:
        logger.warning(f"低并行度工作流不足: {len(low_par_all)} < {WORKFLOWS_PER_GROUP_TRAIN + WORKFLOWS_PER_GROUP_TEST}")

    # 分训练/测试
    set_seed(SEED)
    random.shuffle(high_par_all)
    random.shuffle(low_par_all)

    high_train = high_par_all[:WORKFLOWS_PER_GROUP_TRAIN]
    high_test = high_par_all[WORKFLOWS_PER_GROUP_TRAIN:WORKFLOWS_PER_GROUP_TRAIN + WORKFLOWS_PER_GROUP_TEST]
    low_train = low_par_all[:WORKFLOWS_PER_GROUP_TRAIN]
    low_test = low_par_all[WORKFLOWS_PER_GROUP_TRAIN:WORKFLOWS_PER_GROUP_TRAIN + WORKFLOWS_PER_GROUP_TEST]

    logger.info(f"\n  高并行度: 训练 {len(high_train)}, 测试 {len(high_test)}")
    logger.info(f"  低并行度: 训练 {len(low_train)}, 测试 {len(low_test)}")

    save_data = {"config": {
        "seed": SEED, "train_episodes": TRAIN_EPISODES, "eval_episodes": EVAL_EPISODES,
        "num_resources": NUM_RESOURCES, "per_group_train": WORKFLOWS_PER_GROUP_TRAIN,
        "per_group_test": WORKFLOWS_PER_GROUP_TEST,
    }, "groups": {}}

    # 2. 分别在两组上运行实验
    for group_name, train_wfs, test_wfs in [
        ("HIGH_PARALLELISM", high_train, high_test),
        ("LOW_PARALLELISM", low_train, low_test),
    ]:
        if not train_wfs or not test_wfs:
            logger.warning(f"  [{group_name}] 数据不足，跳过")
            continue

        all_results = run_group_experiment(group_name, train_wfs, test_wfs, logger)
        print_comparison(group_name, all_results, test_wfs, logger)

        save_data["groups"][group_name] = {
            "train_count": len(train_wfs),
            "test_count": len(test_wfs),
            "avg_parallelism_ratio": float(np.mean([w["parallelism_ratio"] for w in test_wfs])),
            "avg_serialization_ratio": float(np.mean([w["serialization_ratio"] for w in test_wfs])),
            "results": [{
                "algorithm": r["algorithm"],
                "avg_makespan": float(np.mean([t["makespan"] for t in r["test_results"]])),
                "std_makespan": float(np.std([t["makespan"] for t in r["test_results"]])),
                "avg_utilization": float(np.mean([t["utilization"] for t in r["test_results"]])),
                "avg_load_balance": float(np.mean([t["load_balance"] for t in r["test_results"]])),
                "train_time": r.get("train_time", 0),
            } for r in all_results],
            "per_workflow": [{
                "name": test_wfs[i]["name"],
                "num_tasks": test_wfs[i]["num_tasks"],
                "num_deps": test_wfs[i]["num_deps"],
                "parallelism_ratio": test_wfs[i]["parallelism_ratio"],
                "serialization_ratio": test_wfs[i]["serialization_ratio"],
                "algorithm_results": {r["algorithm"]: r["test_results"][i]["makespan"] for r in all_results},
            } for i in range(len(test_wfs))],
        }

    # 3. 跨组对比摘要
    logger.info("")
    logger.info("=" * 110)
    logger.info("                     跨组对比摘要: 高并行度 vs 低并行度")
    logger.info("=" * 110)

    for alg_name in ["SJF", "Orig FE-IDDQN", "GA-HPO FE-IDDQN"]:
        vals = {}
        for gname in ["HIGH_PARALLELISM", "LOW_PARALLELISM"]:
            if gname not in save_data["groups"]:
                continue
            for r in save_data["groups"][gname]["results"]:
                if r["algorithm"] == alg_name:
                    vals[gname] = r["avg_makespan"]
        if len(vals) == 2:
            logger.info(f"  {alg_name:<22s}: HIGH={vals['HIGH_PARALLELISM']:.2f}, LOW={vals['LOW_PARALLELISM']:.2f}")

    # 4. 保存结果
    out_path = OUTPUT_DIR / f"summary_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
