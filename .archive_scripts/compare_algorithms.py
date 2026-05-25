#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDS-PPO vs GA-HPO FE-IDDQN 生产环境工作流数据详细对比实验

从本地 MySQL 加载真实 DolphinScheduler/WhaleScheduler 工作流数据，
将其转换为 EnhancedWorkflowSimulator 格式，
然后分别训练两种 RL 算法 + 启发式基线，
在测试集上评估 makespan、资源利用率、负载均衡等指标。

用法:
    python compare_algorithms.py
"""

import json
import logging
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import copy
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pymysql
import pandas as pd
import networkx as nx

# ─── 项目内部导入 ─── #
from environment.enhanced_workflow_simulator import EnhancedWorkflowSimulator

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 全局配置
# ──────────────────────────────────────────────

SEED = 42
DEVICE = "cpu"

# 训练参数
TRAIN_EPISODES = 1000        # 训练 episode 数
EVAL_EPISODES = 5            # 每次评估时的 episode 数
NUM_RESOURCES = 5            # 资源（机器）数量

# 工作流采样
NUM_TRAIN_WORKFLOWS = 30     # 训练用工作流数
NUM_TEST_WORKFLOWS = 20      # 测试用工作流数
WORKFLOW_TASK_MIN = 8        # 最少任务数
WORKFLOW_TASK_MAX = 60       # 最多任务数

# 特征维度（固定pad大小）
MAX_TASKS_PAD = 60           # 任务维度pad上限
MAX_RES_PAD = NUM_RESOURCES  # 资源维度pad上限

OUTPUT_DIR = Path("comparison_results")


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pad_features(arr: np.ndarray, max_rows: int) -> np.ndarray:
    """将 [N, D] 数组 pad/truncate 到 [max_rows, D]"""
    if arr is None:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n, d = arr.shape
    if n >= max_rows:
        return arr[:max_rows]
    padded = np.zeros((max_rows, d), dtype=arr.dtype)
    padded[:n] = arr
    return padded


def pad_adj(arr: np.ndarray, max_n: int) -> np.ndarray:
    """将邻接矩阵 pad 到 [max_n, max_n]"""
    if arr is None:
        return None
    n = arr.shape[0]
    if n >= max_n:
        return arr[:max_n, :max_n]
    padded = np.zeros((max_n, max_n), dtype=arr.dtype)
    padded[:n, :n] = arr
    return padded


def setup_logging():
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
    return logging.getLogger("Compare")


# ──────────────────────────────────────────────
# 1. 从 MySQL 提取真实工作流
# ──────────────────────────────────────────────

def load_workflows_from_db(logger) -> List[Dict[str, Any]]:
    """
    从本地 MySQL (whalesb) 读取 DolphinScheduler 工作流，
    转换为 EnhancedWorkflowSimulator 需要的格式:
      {
        'name': str,
        'tasks': [{'id', 'duration', 'cpu_req', 'memory_req', 'priority', 'task_type'}, ...],
        'resources': [{'id', 'cpu_capacity', 'memory_capacity', 'speed_factor'}, ...],
        'dependencies': [(pre, post), ...],
        'original_makespan': float   # 历史真实 makespan
      }
    """
    logger.info("连接 MySQL 数据库 (localhost/whalesb)...")
    conn = pymysql.connect(
        host="localhost", user="root", password="",
        database="whalesb", port=3306, charset="utf8mb4",
    )

    # --- 1) 找出有足够任务的成功进程实例 ---
    query_processes = f"""
        SELECT pi.id, pi.name, pi.process_definition_code,
               pi.start_time, pi.end_time,
               COUNT(ti.id) AS task_count
        FROM t_ds_process_instance pi
        JOIN t_ds_task_instance ti ON ti.process_instance_id = pi.id
        WHERE pi.state = 7 AND ti.state = 7
        GROUP BY pi.id
        HAVING task_count BETWEEN {WORKFLOW_TASK_MIN} AND {WORKFLOW_TASK_MAX}
        ORDER BY RAND({SEED})
        LIMIT {NUM_TRAIN_WORKFLOWS + NUM_TEST_WORKFLOWS + 20}
    """
    processes_df = pd.read_sql(query_processes, conn)
    logger.info(f"找到 {len(processes_df)} 个符合条件的工作流实例")

    workflows: List[Dict] = []
    used_def_codes = set()  # 避免重复定义

    for _, proc in processes_df.iterrows():
        pid = int(proc["id"])
        def_code = int(proc["process_definition_code"])

        # 去重：同一 definition 只取一个实例
        if def_code in used_def_codes:
            continue
        used_def_codes.add(def_code)

        # --- 2) 获取该实例的所有成功任务 ---
        tasks_df = pd.read_sql(f"""
            SELECT id, name, task_type, task_code,
                   task_instance_priority, retry_times,
                   start_time, end_time, host
            FROM t_ds_task_instance
            WHERE process_instance_id = {pid} AND state = 7
            ORDER BY start_time
        """, conn)

        if len(tasks_df) < WORKFLOW_TASK_MIN:
            continue

        # --- 3) 获取依赖关系 ---
        deps_df = pd.read_sql(f"""
            SELECT pre_task_code, post_task_code
            FROM t_ds_process_task_relation
            WHERE process_definition_code = {def_code}
              AND pre_task_code != 0
        """, conn)

        # --- 4) 构建任务列表 ---
        task_code_to_idx = {}
        task_list = []
        for idx, (_, t) in enumerate(tasks_df.iterrows()):
            code = int(t["task_code"]) if pd.notna(t.get("task_code")) else t["id"]
            task_code_to_idx[code] = idx

            # 计算真实执行时长 (秒)
            duration = 30.0
            if pd.notna(t["start_time"]) and pd.notna(t["end_time"]):
                try:
                    dur = (pd.to_datetime(t["end_time"]) - pd.to_datetime(t["start_time"])).total_seconds()
                    if dur > 0:
                        duration = dur
                except Exception:
                    pass

            # 按任务类型估算资源需求
            tt = str(t.get("task_type", "SHELL"))
            cpu_map = {"SQL": 1, "SHELL": 1, "PYTHON": 2, "JAVA": 2, "SPARK": 4, "FLINK": 4, "HTTP": 1}
            mem_map = {"SQL": 2, "SHELL": 1, "PYTHON": 4, "JAVA": 4, "SPARK": 8, "FLINK": 8, "HTTP": 1}
            priority = int(t.get("task_instance_priority", 0)) if pd.notna(t.get("task_instance_priority")) else 0

            task_list.append({
                "id": idx,
                "duration": duration,
                "cpu_req": cpu_map.get(tt, 1),
                "memory_req": mem_map.get(tt, 2),
                "priority": priority,
                "task_type": tt,
            })

        # --- 5) 构建依赖 ---
        dep_edges = []
        for _, d in deps_df.iterrows():
            pre_code = int(d["pre_task_code"])
            post_code = int(d["post_task_code"])
            if pre_code in task_code_to_idx and post_code in task_code_to_idx:
                dep_edges.append((task_code_to_idx[pre_code], task_code_to_idx[post_code]))

        # 验证无环
        G = nx.DiGraph()
        G.add_nodes_from(range(len(task_list)))
        G.add_edges_from(dep_edges)
        if not nx.is_directed_acyclic_graph(G):
            # 移除回边使其成为 DAG
            try:
                cycles = list(nx.simple_cycles(G))
                for cycle in cycles:
                    if len(cycle) > 1:
                        G.remove_edge(cycle[-1], cycle[0])
                dep_edges = list(G.edges())
            except Exception:
                dep_edges = []

        # --- 6) 构建资源列表 ---
        resources = []
        for j in range(NUM_RESOURCES):
            resources.append({
                "id": j,
                "cpu_capacity": random.choice([8, 16]),
                "memory_capacity": random.choice([16, 32]),
                "speed_factor": round(random.uniform(0.8, 1.2), 2),
            })

        # --- 7) 计算历史真实 makespan ---
        original_makespan = 0.0
        if pd.notna(proc["start_time"]) and pd.notna(proc["end_time"]):
            try:
                original_makespan = (
                    pd.to_datetime(proc["end_time"]) - pd.to_datetime(proc["start_time"])
                ).total_seconds()
            except Exception:
                pass

        workflows.append({
            "name": str(proc["name"]),
            "tasks": task_list,
            "resources": resources,
            "dependencies": dep_edges,
            "original_makespan": original_makespan,
            "num_tasks": len(task_list),
            "num_deps": len(dep_edges),
        })

        if len(workflows) >= NUM_TRAIN_WORKFLOWS + NUM_TEST_WORKFLOWS:
            break

    conn.close()
    logger.info(f"成功提取 {len(workflows)} 个工作流 (任务数 {WORKFLOW_TASK_MIN}-{WORKFLOW_TASK_MAX})")
    return workflows


# ──────────────────────────────────────────────
# 2. 创建模拟环境
# ──────────────────────────────────────────────

def make_env(workflow: Dict) -> EnhancedWorkflowSimulator:
    return EnhancedWorkflowSimulator(
        tasks=workflow["tasks"],
        resources=workflow["resources"],
        dependencies=workflow["dependencies"],
        use_dag_aware=True,
        use_critical_path_priority=True,
    )


# ──────────────────────────────────────────────
# 3. 启发式基线
# ──────────────────────────────────────────────

class HeuristicScheduler:
    """统一接口的启发式调度器"""
    def __init__(self, name: str, num_resources: int):
        self.name = name
        self.num_resources = num_resources
        self.counter = 0

    def reset(self):
        self.counter = 0

    def select_action(self, env: EnhancedWorkflowSimulator) -> int:
        raise NotImplementedError


class RandomScheduler(HeuristicScheduler):
    def __init__(self, n): super().__init__("Random", n)
    def select_action(self, env): return random.randint(0, self.num_resources - 1)


class RoundRobinScheduler(HeuristicScheduler):
    def __init__(self, n): super().__init__("RoundRobin", n)
    def select_action(self, env):
        a = self.counter % self.num_resources
        self.counter += 1
        return a


class ShortestJobFirst(HeuristicScheduler):
    """选择当前最空闲的资源"""
    def __init__(self, n): super().__init__("SJF", n)
    def select_action(self, env):
        loads = [env.resource_states[r["id"]].available_time for r in env.resources]
        return int(np.argmin(loads))


class EarliestFinishTime(HeuristicScheduler):
    """EFT: 选择使当前任务最早完成的资源"""
    def __init__(self, n): super().__init__("EFT", n)
    def select_action(self, env):
        if not env.ready_tasks:
            return 0
        task_id = env.ready_tasks[0]
        task = next(t for t in env.tasks if t["id"] == task_id)
        dur = task.get("duration", 1.0)
        finish_times = []
        for r in env.resources:
            avail = env.resource_states[r["id"]].available_time
            finish_times.append(avail + dur)
        return int(np.argmin(finish_times))


class CriticalPathFirst(HeuristicScheduler):
    """CPOP-like: 关键路径任务优先分配到最快资源"""
    def __init__(self, n): super().__init__("CPOP", n)
    def select_action(self, env):
        if not env.ready_tasks:
            return 0
        task_id = env.ready_tasks[0]
        is_critical = task_id in getattr(env, "critical_path_set", set())
        if is_critical:
            # 分配到最快（最空闲）的资源
            loads = [env.resource_states[r["id"]].available_time for r in env.resources]
            return int(np.argmin(loads))
        else:
            # 非关键路径：轮询
            a = self.counter % self.num_resources
            self.counter += 1
            return a


def evaluate_heuristic(scheduler: HeuristicScheduler,
                       workflow: Dict,
                       num_episodes: int = 1) -> Dict[str, float]:
    """评估启发式调度器"""
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


# ──────────────────────────────────────────────
# 4. 紧凑状态表示 & RL 网络
# ──────────────────────────────────────────────

# 紧凑状态维度（固定，不随工作流大小变化）
# global(11) + task_mean(19) + task_std(19) + current_task(19)
# + resource_flat(5*11=55) + cp_stats(2) + depth_stats(3) = 128
COMPACT_DIM = 128


def _compact_state(state):
    """
    固定维度的紧凑状态表示 (128 维).
    通过统计量聚合可变长度的任务特征,
    使状态维度不随工作流大小变化.
    """
    tf = state.get("task_features", np.zeros((1, 19)))
    rf = state.get("resource_features", np.zeros((1, 11)))
    gf = state.get("global_features", np.zeros(11))

    # (1) 全局特征 [11]
    gf_flat = gf.flatten()
    if len(gf_flat) < 11:
        gf_flat = np.concatenate([gf_flat, np.zeros(11 - len(gf_flat))])
    gf_flat = gf_flat[:11]

    # (2) 任务统计量 [19 + 19 = 38]
    task_mean = tf.mean(axis=0)   # 每维特征的均值
    task_std  = tf.std(axis=0)    # 每维特征的标准差

    # (3) 当前待调度任务的特征 [19]
    # 在 DAG-aware 模式下, ready_tasks[0] 就是当前要调度的任务
    # 该任务在 task_features 中标记为 ready=1 (column 4)
    ready_mask = tf[:, 4] > 0.5
    if ready_mask.any():
        # 取第一个就绪任务的特征
        ready_idx = np.where(ready_mask)[0]
        current_task = tf[ready_idx[0]]
    else:
        current_task = tf.mean(axis=0)  # 兜底

    # (4) 资源特征 [55]  — 5 个资源 × 11 维
    rf_flat = rf.flatten()
    if len(rf_flat) < 55:
        rf_flat = np.concatenate([rf_flat, np.zeros(55 - len(rf_flat))])
    rf_flat = rf_flat[:55]

    # (5) 关键路径统计 [2]
    cp = state.get("critical_path_mask", np.zeros(tf.shape[0]))
    cp_stats = np.array([cp.sum() / max(1, len(cp)), cp.mean()], dtype=np.float32)

    # (6) 节点深度统计 [3]
    nd = state.get("node_depths", np.zeros(tf.shape[0]))
    nd_f = nd.astype(np.float32)
    nd_stats = np.array([
        nd_f.mean() / 10.0 if nd_f.size > 0 else 0.0,
        nd_f.std() / 10.0  if nd_f.size > 0 else 0.0,
        nd_f.max() / 10.0  if nd_f.size > 0 else 0.0,
    ], dtype=np.float32)

    compact = np.concatenate([
        gf_flat,       # 11
        task_mean,     # 19
        task_std,      # 19
        current_task,  # 19
        rf_flat,       # 55
        cp_stats,      # 2
        nd_stats,      # 3
    ]).astype(np.float32)

    # 安全 clip
    compact = np.clip(compact, -100.0, 100.0)
    np.nan_to_num(compact, nan=0.0, posinf=100.0, neginf=-100.0, copy=False)

    return compact


class RunningNormalizer:
    """Welford 在线均值/方差归一化"""
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape, dtype=np.float64)
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
        normed = (x - self.mean.astype(np.float32)) / std
        return np.clip(normed, -10.0, 10.0)


class MLPActorCritic(nn.Module):
    """PPO Actor-Critic (3层MLP + 正交初始化)"""
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor  = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, action_dim))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)


class MLPDQN(nn.Module):
    """Dueling DQN (特征提取 + value/advantage 流)"""
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.adv_stream   = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, action_dim))

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.adv_stream(f)
        return v + a - a.mean(dim=-1, keepdim=True)


def _load_balance_bonus(env) -> float:
    """计算负载均衡奖励加成（鼓励分散任务到不同资源）"""
    loads = [env.resource_states[r["id"]].available_time for r in env.resources]
    if not loads:
        return 0.0
    max_load = max(loads)
    mean_load = np.mean(loads)
    if max_load == 0:
        return 1.0
    # 负载越均衡 bonus 越高 (0~1)
    return 1.0 - (max_load - mean_load) / (max_load + 1e-8)


# ──────────────────────────────────────────────
# 5. SJF 专家示范 & 行为克隆
# ──────────────────────────────────────────────

def _generate_sjf_demos(workflows, normalizer, num_per_wf=3):
    """
    用 SJF 启发式为每个工作流生成专家示范轨迹.
    返回 (states, actions) 对列表.
    """
    sjf = ShortestJobFirst(NUM_RESOURCES)
    all_states, all_actions = [], []
    for wf in workflows:
        for _ in range(num_per_wf):
            env = make_env(wf)
            sjf.reset()
            state = env.reset()
            done = False
            steps = 0
            max_steps = len(wf["tasks"]) * 3
            while not done and steps < max_steps:
                raw = _compact_state(state)
                normalizer.update(raw)
                norm = normalizer.normalize(raw)
                action = sjf.select_action(env)
                all_states.append(norm)
                all_actions.append(action)
                state, _, done, _ = env.step(action)
                steps += 1
    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.int64)


def _behavior_clone(policy, optimizer, states, actions, device, epochs=20, logger=None):
    """
    行为克隆: 用 SJF 示范预训练 PPO 策略网络.
    """
    s_t = torch.FloatTensor(states).to(device)
    a_t = torch.LongTensor(actions).to(device)
    n = len(states)
    batch_size = min(256, n)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        num_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits, _ = policy(s_t[idx])
            loss = F.cross_entropy(logits, a_t[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        if logger and (epoch + 1) % 5 == 0:
            logger.info(f"    BC epoch {epoch+1}/{epochs}  loss={total_loss/max(1,num_batches):.4f}")


# ──────────────────────────────────────────────
# 6. PPO 训练 & 评估  (紧凑状态 + 行为克隆热启动)
# ──────────────────────────────────────────────

def train_and_eval_ppo(workflows_train: List[Dict],
                       workflows_test: List[Dict],
                       logger) -> Dict[str, Any]:
    """训练 GDS-PPO 并在测试集评估"""
    logger.info("=" * 60)
    logger.info("训练 GDS-PPO (紧凑128维状态 + BC热启动)")
    logger.info("=" * 60)

    action_dim = NUM_RESOURCES
    device = torch.device(DEVICE)

    policy = MLPActorCritic(COMPACT_DIM, action_dim, hidden=256).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    logger.info(f"  state_dim={COMPACT_DIM}, action_dim={action_dim}")
    logger.info(f"  PPO params: {sum(p.numel() for p in policy.parameters()):,}")

    normalizer = RunningNormalizer(COMPACT_DIM)

    # ---- 阶段1: SJF行为克隆热启动 ----
    logger.info("  [阶段1] SJF行为克隆预训练...")
    bc_states, bc_actions = _generate_sjf_demos(workflows_train, normalizer, num_per_wf=5)
    logger.info(f"    生成 {len(bc_states)} 条SJF示范样本")
    _behavior_clone(policy, optimizer, bc_states, bc_actions, device, epochs=30, logger=logger)
    logger.info("  [阶段1] 行为克隆完成")

    # ---- 阶段2: PPO-Clip + GAE 在线微调 ----
    logger.info(f"  [阶段2] PPO在线训练 {TRAIN_EPISODES} episodes...")
    GAMMA, GAE_LAMBDA = 0.99, 0.95
    EPS_CLIP = 0.2
    K_EPOCHS = 4
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5

    t0 = time.time()
    train_rewards = []

    for ep in range(TRAIN_EPISODES):
        wf = workflows_train[ep % len(workflows_train)]
        env = make_env(wf)
        state = env.reset()
        done = False
        steps = 0
        max_steps = len(wf["tasks"]) * 3

        states_buf, actions_buf, rewards_buf = [], [], []
        log_probs_buf, values_buf, dones_buf = [], [], []

        while not done and steps < max_steps:
            raw = _compact_state(state)
            normalizer.update(raw)
            norm = normalizer.normalize(raw)
            st = torch.FloatTensor(norm).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, val = policy(st)
                logits = torch.clamp(logits, -20, 20)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                lp = dist.log_prob(action)

            next_state, reward, done, info = env.step(action.item())

            # 加入负载均衡奖励
            lb_bonus = _load_balance_bonus(env)
            shaped_reward = reward + lb_bonus * 2.0
            reward_scaled = np.clip(shaped_reward / 10.0, -10.0, 10.0)

            states_buf.append(norm)
            actions_buf.append(action.item())
            rewards_buf.append(reward_scaled)
            log_probs_buf.append(lp.item())
            values_buf.append(val.squeeze().item())
            dones_buf.append(done)

            state = next_state
            steps += 1

        ep_reward = sum(rewards_buf) * 10.0
        train_rewards.append(ep_reward)

        if len(states_buf) < 2:
            continue

        # GAE
        last_raw = _compact_state(state)
        last_norm = normalizer.normalize(last_raw)
        with torch.no_grad():
            _, last_val = policy(torch.FloatTensor(last_norm).unsqueeze(0).to(device))
            last_v = last_val.squeeze().item()

        n = len(rewards_buf)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_v = (last_v if t == n - 1 else values_buf[t + 1]) * (1.0 - float(dones_buf[min(t + 1, n - 1)]))
            if t == n - 1:
                next_v = last_v * (1.0 - float(done))
            delta = rewards_buf[t] + GAMMA * next_v - values_buf[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1.0 - float(dones_buf[t])) * gae
            advantages[t] = gae
        returns = advantages + np.array(values_buf, dtype=np.float32)

        # PPO 更新
        s_t = torch.FloatTensor(np.array(states_buf)).to(device)
        a_t = torch.LongTensor(actions_buf).to(device)
        old_lp_t = torch.FloatTensor(log_probs_buf).to(device)
        adv_t = torch.FloatTensor(advantages).to(device)
        ret_t = torch.FloatTensor(returns).to(device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(K_EPOCHS):
            logits, vals = policy(s_t)
            logits = torch.clamp(logits, -20, 20)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(a_t)
            entropy = dist.entropy().mean()
            vals = vals.squeeze(-1)

            ratio = torch.exp(new_lp - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(vals, ret_t)
            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        if (ep + 1) % 100 == 0:
            logger.info(f"  PPO ep {ep+1}/{TRAIN_EPISODES}  reward={np.mean(train_rewards[-100:]):.2f}")

    train_time = time.time() - t0
    logger.info(f"  PPO 训练完成, 耗时 {train_time:.1f}s")

    # ---- 测试 ----
    policy.eval()
    test_results = []
    for wf in workflows_test:
        env = make_env(wf)
        makespans, utils, bals = [], [], []
        for _ in range(EVAL_EPISODES):
            state = env.reset()
            done = False
            steps = 0
            max_steps = len(wf["tasks"]) * 3
            while not done and steps < max_steps:
                norm = normalizer.normalize(_compact_state(state))
                with torch.no_grad():
                    logits, _ = policy(torch.FloatTensor(norm).unsqueeze(0).to(device))
                    action = logits.argmax(dim=-1).item()
                state, _, done, _ = env.step(action)
                steps += 1
            r = env.get_scheduling_result()
            makespans.append(r["makespan"])
            utils.append(r["resource_utilization"])
            bals.append(r["load_balance"])
        test_results.append({
            "name": wf["name"],
            "num_tasks": wf["num_tasks"],
            "num_deps": wf["num_deps"],
            "original_makespan": wf["original_makespan"],
            "makespan": float(np.mean(makespans)),
            "utilization": float(np.mean(utils)),
            "load_balance": float(np.mean(bals)),
        })

    return {
        "algorithm": "GDS-PPO",
        "train_time": train_time,
        "train_rewards": [float(r) for r in train_rewards],
        "test_results": test_results,
    }


# ──────────────────────────────────────────────
# 7. FE-IDDQN 训练 & 评估 (紧凑状态 + 专家缓冲)
# ──────────────────────────────────────────────

def train_and_eval_fe_iddqn(workflows_train: List[Dict],
                            workflows_test: List[Dict],
                            logger) -> Dict[str, Any]:
    """训练 GA-HPO FE-IDDQN 并在测试集评估
       (Dueling DQN + PER + N-step + 专家经验预填充)"""
    logger.info("=" * 60)
    logger.info("训练 GA-HPO FE-IDDQN (+ 专家预填充)")
    logger.info("=" * 60)

    action_dim = NUM_RESOURCES
    device = torch.device(DEVICE)

    q_net = MLPDQN(COMPACT_DIM, action_dim, hidden=256).to(device)
    target_net = MLPDQN(COMPACT_DIM, action_dim, hidden=256).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)
    logger.info(f"  state_dim={COMPACT_DIM}, action_dim={action_dim}")
    logger.info(f"  FE-IDDQN params: {sum(p.numel() for p in q_net.parameters()):,}")

    normalizer = RunningNormalizer(COMPACT_DIM)

    # 超参数
    GAMMA = 0.99
    TAU = 0.005
    BATCH_SIZE = 64
    BUFFER_SIZE = 100000
    WARMUP = 100          # 经过专家预填充后, 减小 warmup
    TRAIN_FREQ = 2
    TARGET_UPDATE = 50
    EPS_START, EPS_END, EPS_DECAY = 0.5, 0.02, 0.998
    N_STEP = 3
    GRAD_CLIP = 1.0

    from collections import deque
    replay_buf = deque(maxlen=BUFFER_SIZE)
    n_step_buf = deque(maxlen=N_STEP)
    priorities = deque(maxlen=BUFFER_SIZE)
    epsilon = EPS_START
    total_steps = 0

    def _add_experience(s, a, r, s2, d):
        n_step_buf.append((s, a, r, s2, d))
        if len(n_step_buf) == N_STEP or d:
            R = 0.0
            for i in reversed(range(len(n_step_buf))):
                R = n_step_buf[i][2] + GAMMA * R * (1.0 - float(n_step_buf[i][4]))
            s0 = n_step_buf[0][0]
            a0 = n_step_buf[0][1]
            sn = n_step_buf[-1][3]
            dn = n_step_buf[-1][4]
            gamma_n = GAMMA ** len(n_step_buf)
            replay_buf.append((s0, a0, R, sn, dn, gamma_n))
            priorities.append(2.0)  # 高初始优先级
            if d:
                n_step_buf.clear()

    def _sample_batch():
        p = np.array(list(priorities), dtype=np.float32)
        p = p ** 0.6
        p = p / p.sum()
        indices = np.random.choice(len(replay_buf), BATCH_SIZE, p=p, replace=False)
        weights = (len(replay_buf) * p[indices]) ** (-0.4)
        weights = weights / weights.max()
        batch = [replay_buf[i] for i in indices]
        return batch, indices, weights

    # ---- 阶段1: SJF专家经验预填充 ----
    logger.info("  [阶段1] SJF专家经验预填充 replay buffer...")
    sjf = ShortestJobFirst(NUM_RESOURCES)
    expert_count = 0
    for wf in workflows_train:
        for _ in range(5):   # 每个工作流5条示范
            env = make_env(wf)
            sjf.reset()
            state = env.reset()
            done = False
            steps = 0
            max_steps = len(wf["tasks"]) * 3
            while not done and steps < max_steps:
                raw = _compact_state(state)
                normalizer.update(raw)
                norm = normalizer.normalize(raw)
                action = sjf.select_action(env)
                next_state, reward, done, info = env.step(action)
                lb_bonus = _load_balance_bonus(env)
                shaped_reward = reward + lb_bonus * 2.0
                next_raw = _compact_state(next_state)
                next_norm = normalizer.normalize(next_raw)
                _add_experience(norm, action, shaped_reward, next_norm, done)
                state = next_state
                steps += 1
                expert_count += 1
    logger.info(f"    预填充 {expert_count} 条专家经验, buffer size={len(replay_buf)}")

    # ---- 阶段2: Double DQN + PER + N-step 在线训练 ----
    logger.info(f"  [阶段2] FE-IDDQN 在线训练 {TRAIN_EPISODES} episodes...")
    t0 = time.time()
    train_rewards = []

    for ep in range(TRAIN_EPISODES):
        wf = workflows_train[ep % len(workflows_train)]
        env = make_env(wf)
        state = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0
        max_steps = len(wf["tasks"]) * 3

        while not done and steps < max_steps:
            raw = _compact_state(state)
            normalizer.update(raw)
            norm = normalizer.normalize(raw)

            # ε-greedy (有偏探索: 随机时偏向SJF)
            if random.random() < epsilon:
                # 50% 纯随机, 50% 用SJF
                if random.random() < 0.5:
                    action = random.randint(0, action_dim - 1)
                else:
                    loads = [env.resource_states[r["id"]].available_time for r in env.resources]
                    action = int(np.argmin(loads))
            else:
                with torch.no_grad():
                    qv = q_net(torch.FloatTensor(norm).unsqueeze(0).to(device))
                    action = qv.argmax(dim=-1).item()

            next_state, reward, done, info = env.step(action)
            lb_bonus = _load_balance_bonus(env)
            shaped_reward = reward + lb_bonus * 2.0

            next_raw = _compact_state(next_state)
            next_norm = normalizer.normalize(next_raw)
            _add_experience(norm, action, shaped_reward, next_norm, done)

            total_steps += 1

            # 训练
            if total_steps >= WARMUP and total_steps % TRAIN_FREQ == 0 and len(replay_buf) >= BATCH_SIZE:
                batch, bi, isw = _sample_batch()
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

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP)
                optimizer.step()

                for idx, td in zip(bi, td_errors.cpu().numpy()):
                    priorities[idx] = float(td) + 1e-6

            # 软更新 target
            if total_steps % TARGET_UPDATE == 0:
                for tp, qp in zip(target_net.parameters(), q_net.parameters()):
                    tp.data.copy_(TAU * qp.data + (1 - TAU) * tp.data)

            ep_reward += shaped_reward
            steps += 1
            state = next_state

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        train_rewards.append(ep_reward)

        if (ep + 1) % 100 == 0:
            logger.info(f"  FE-IDDQN ep {ep+1}/{TRAIN_EPISODES}  reward={np.mean(train_rewards[-100:]):.2f}  eps={epsilon:.3f}")

    train_time = time.time() - t0
    logger.info(f"  FE-IDDQN 训练完成, 耗时 {train_time:.1f}s")

    # ---- 测试 ----
    q_net.eval()
    test_results = []
    for wf in workflows_test:
        env = make_env(wf)
        makespans, utils, bals = [], [], []
        for _ in range(EVAL_EPISODES):
            state = env.reset()
            done = False
            steps = 0
            max_steps = len(wf["tasks"]) * 3
            while not done and steps < max_steps:
                norm = normalizer.normalize(_compact_state(state))
                with torch.no_grad():
                    qv = q_net(torch.FloatTensor(norm).unsqueeze(0).to(device))
                    action = qv.argmax(dim=-1).item()
                state, _, done, _ = env.step(action)
                steps += 1
            r = env.get_scheduling_result()
            makespans.append(r["makespan"])
            utils.append(r["resource_utilization"])
            bals.append(r["load_balance"])
        test_results.append({
            "name": wf["name"],
            "num_tasks": wf["num_tasks"],
            "num_deps": wf["num_deps"],
            "original_makespan": wf["original_makespan"],
            "makespan": float(np.mean(makespans)),
            "utilization": float(np.mean(utils)),
            "load_balance": float(np.mean(bals)),
        })

    return {
        "algorithm": "GA-HPO FE-IDDQN",
        "train_time": train_time,
        "train_rewards": [float(r) for r in train_rewards],
        "test_results": test_results,
    }


# ──────────────────────────────────────────────
# 6. 评估所有启发式基线
# ──────────────────────────────────────────────

def evaluate_all_heuristics(workflows_test: List[Dict],
                            logger) -> List[Dict[str, Any]]:
    logger.info("=" * 60)
    logger.info("评估启发式基线")
    logger.info("=" * 60)

    schedulers = [
        RandomScheduler(NUM_RESOURCES),
        RoundRobinScheduler(NUM_RESOURCES),
        ShortestJobFirst(NUM_RESOURCES),
        EarliestFinishTime(NUM_RESOURCES),
        CriticalPathFirst(NUM_RESOURCES),
    ]

    all_results = []
    for sched in schedulers:
        test_results = []
        for wf in workflows_test:
            r = evaluate_heuristic(sched, wf, num_episodes=EVAL_EPISODES)
            test_results.append({
                "name": wf["name"],
                "num_tasks": wf["num_tasks"],
                "num_deps": wf["num_deps"],
                "original_makespan": wf["original_makespan"],
                **r,
            })
        all_results.append({
            "algorithm": sched.name,
            "train_time": 0.0,
            "test_results": test_results,
        })
        avg_ms = np.mean([r["makespan"] for r in test_results])
        avg_ut = np.mean([r["utilization"] for r in test_results])
        logger.info(f"  {sched.name:12s}  makespan={avg_ms:.2f}  util={avg_ut:.4f}")

    return all_results


# ──────────────────────────────────────────────
# 7. 汇总 & 输出
# ──────────────────────────────────────────────

def print_comparison_table(all_algo_results: List[Dict], logger):
    """打印对比总结表"""
    logger.info("")
    logger.info("=" * 100)
    logger.info("                         详 细 对 比 结 果 总 表")
    logger.info("=" * 100)

    header = f"{'算法':<20s} | {'Makespan':>10s} | {'Std':>8s} | {'利用率':>8s} | {'负载均衡':>8s} | {'训练时间':>8s} | {'vs历史':>8s}"
    logger.info(header)
    logger.info("-" * 100)

    # 以EFT作为基准
    eft_makespans = None
    for res in all_algo_results:
        if res["algorithm"] == "EFT":
            eft_makespans = [r["makespan"] for r in res["test_results"]]
            break

    rows = []
    for res in all_algo_results:
        ms_list = [r["makespan"] for r in res["test_results"]]
        ut_list = [r["utilization"] for r in res["test_results"]]
        bl_list = [r["load_balance"] for r in res["test_results"]]
        orig_list = [r["original_makespan"] for r in res["test_results"]]

        avg_ms = np.mean(ms_list)
        std_ms = np.std(ms_list)
        avg_ut = np.mean(ut_list)
        avg_bl = np.mean(bl_list)
        train_t = res.get("train_time", 0)

        # 对比历史 makespan
        improvements = []
        for ms, orig in zip(ms_list, orig_list):
            if orig > 0:
                improvements.append((orig - ms) / orig * 100)
        avg_imp = np.mean(improvements) if improvements else 0.0

        row_str = (
            f"{res['algorithm']:<20s} | "
            f"{avg_ms:>10.2f} | "
            f"{std_ms:>8.2f} | "
            f"{avg_ut:>8.4f} | "
            f"{avg_bl:>8.4f} | "
            f"{train_t:>7.1f}s | "
            f"{avg_imp:>+7.1f}%"
        )
        logger.info(row_str)

        rows.append({
            "algorithm": res["algorithm"],
            "avg_makespan": float(avg_ms),
            "std_makespan": float(std_ms),
            "avg_utilization": float(avg_ut),
            "avg_load_balance": float(avg_bl),
            "train_time": float(train_t),
            "avg_improvement_vs_history": float(avg_imp),
        })

    logger.info("=" * 100)

    # RL vs 最佳启发式
    rl_names = {"GDS-PPO", "GA-HPO FE-IDDQN"}
    heuristic_best_ms = min(
        (r["avg_makespan"] for r in rows if r["algorithm"] not in rl_names),
        default=float("inf")
    )
    for r in rows:
        if r["algorithm"] in rl_names:
            diff = (heuristic_best_ms - r["avg_makespan"]) / heuristic_best_ms * 100
            logger.info(f"  {r['algorithm']} vs 最佳启发式: makespan {'改进' if diff > 0 else '差距'} {abs(diff):.2f}%")

    return rows


def print_per_workflow_detail(all_algo_results: List[Dict], logger):
    """按工作流打印各算法详细结果"""
    logger.info("")
    logger.info("=" * 120)
    logger.info("                      各 工 作 流 详 细 结 果")
    logger.info("=" * 120)

    # 获取工作流列表
    wf_names = [r["name"] for r in all_algo_results[0]["test_results"]]

    for i, name in enumerate(wf_names):
        short_name = name[:50] if len(name) > 50 else name
        num_tasks = all_algo_results[0]["test_results"][i]["num_tasks"]
        num_deps = all_algo_results[0]["test_results"][i]["num_deps"]
        orig_ms = all_algo_results[0]["test_results"][i]["original_makespan"]

        logger.info(f"\n  工作流: {short_name}")
        logger.info(f"  任务数: {num_tasks}, 依赖数: {num_deps}, 历史makespan: {orig_ms:.1f}s")
        logger.info(f"  {'算法':<20s} {'Makespan':>10s} {'利用率':>10s} {'负载均衡':>10s}")

        for res in all_algo_results:
            tr = res["test_results"][i]
            logger.info(
                f"  {res['algorithm']:<20s} "
                f"{tr['makespan']:>10.2f} "
                f"{tr['utilization']:>10.4f} "
                f"{tr['load_balance']:>10.4f}"
            )

    logger.info("")


# ──────────────────────────────────────────────
# 8. 主函数
# ──────────────────────────────────────────────

def main():
    set_seed(SEED)
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("  GDS-PPO vs GA-HPO FE-IDDQN 生产环境对比实验")
    logger.info(f"  训练集: {NUM_TRAIN_WORKFLOWS} 工作流")
    logger.info(f"  测试集: {NUM_TEST_WORKFLOWS} 工作流")
    logger.info(f"  训练episodes: {TRAIN_EPISODES}")
    logger.info(f"  任务范围: {WORKFLOW_TASK_MIN}-{WORKFLOW_TASK_MAX}")
    logger.info("=" * 60)

    # --- 加载数据 ---
    workflows = load_workflows_from_db(logger)
    if len(workflows) < NUM_TRAIN_WORKFLOWS + NUM_TEST_WORKFLOWS:
        logger.warning(f"工作流不足，实际 {len(workflows)} 个")
        split = max(1, len(workflows) * 3 // 5)
    else:
        split = NUM_TRAIN_WORKFLOWS

    workflows_train = workflows[:split]
    workflows_test = workflows[split:]
    logger.info(f"训练集 {len(workflows_train)} 个, 测试集 {len(workflows_test)} 个")

    # 数据集统计
    for label, wfs in [("训练集", workflows_train), ("测试集", workflows_test)]:
        tasks_counts = [w["num_tasks"] for w in wfs]
        deps_counts = [w["num_deps"] for w in wfs]
        logger.info(f"  {label}: 任务数 {np.mean(tasks_counts):.1f}±{np.std(tasks_counts):.1f}, "
                     f"依赖数 {np.mean(deps_counts):.1f}±{np.std(deps_counts):.1f}")

    # --- 评估启发式基线 ---
    heuristic_results = evaluate_all_heuristics(workflows_test, logger)

    # --- 训练 & 评估 GDS-PPO ---
    ppo_results = train_and_eval_ppo(workflows_train, workflows_test, logger)

    # --- 训练 & 评估 FE-IDDQN ---
    iddqn_results = train_and_eval_fe_iddqn(workflows_train, workflows_test, logger)

    # --- 汇总 ---
    all_results = heuristic_results + [ppo_results, iddqn_results]

    summary_rows = print_comparison_table(all_results, logger)
    print_per_workflow_detail(all_results, logger)

    # --- 保存 ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(OUTPUT_DIR / f"summary_{ts}.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "seed": SEED,
                "train_episodes": TRAIN_EPISODES,
                "eval_episodes": EVAL_EPISODES,
                "num_resources": NUM_RESOURCES,
                "train_workflows": len(workflows_train),
                "test_workflows": len(workflows_test),
            },
            "summary": summary_rows,
            "all_results": all_results,
        }, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n结果已保存到 {OUTPUT_DIR}/")
    logger.info("实验完成！")


if __name__ == "__main__":
    main()
