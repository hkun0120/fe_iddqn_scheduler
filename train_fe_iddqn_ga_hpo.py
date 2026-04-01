#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA-HPO FE-IDDQN 训练系统 — 完整pipeline

将 GA-HPO 混合优化框架 (Zhou et al., Sensors 2025) 应用于 FE-IDDQN:
  Phase 1 (可选): GA 搜索最优网络架构
  Phase 2 (可选): Optuna 搜索最优 DQN 超参数
  Phase 3: 使用最优配置进行完整 FE-IDDQN 训练 + 评估

用法:
  python train_fe_iddqn_ga_hpo.py                        # 直接训练 (默认参数)
  python train_fe_iddqn_ga_hpo.py --mode full             # 完整pipeline (GA + HPO + 训练)
  python train_fe_iddqn_ga_hpo.py --mode train_only       # 仅训练 (跳过GA/HPO)
  python train_fe_iddqn_ga_hpo.py --mode ga_search        # 仅GA架构搜索
  python train_fe_iddqn_ga_hpo.py --mode hpo              # 仅HPO
"""

import argparse
import json
import logging
import os
import sys
import time
import random
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch

# ─── 项目内部导入 ─── #
from models.enhanced_fe_iddqn import EnhancedFE_IDDQN, EnhancedFE_IDDQN_Config
from models.ga_optimizer import GAArchitectureOptimizer, GAConfig
from models.dqn_hpo_optimizer import DQNHPOptimizer, DQNHPOConfig
from environment.enhanced_workflow_simulator import EnhancedWorkflowSimulator
from environment.historical_replay_simulator import HistoricalReplaySimulator as LogReplaySimulator
from baselines.traditional_schedulers import FIFOScheduler, SJFScheduler, HEFTScheduler


# ─────────────────── 工具函数 ─────────────────── #

def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path) -> logging.Logger:
    """配置日志"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"fe_iddqn_ga_hpo_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger('FE_IDDQN_GA_HPO')


# ─────────── 合成工作流生成 ─────────── #

def generate_synthetic_workflow(
        num_tasks: int = 20, num_resources: int = 5,
        dag_density: float = 0.3, seed: int = None
) -> Tuple[List[Dict], List[Dict], List[Tuple[int, int]]]:
    """生成合成DAG工作流"""
    rng = np.random.RandomState(seed)

    tasks = []
    for i in range(num_tasks):
        tasks.append({
            'id': i,
            'duration': rng.uniform(1.0, 20.0),
            'cpu_req': rng.choice([1, 2, 4]),
            'memory_req': rng.choice([1, 2, 4, 8]),
            'priority': rng.randint(0, 3),
        })

    resources = []
    for j in range(num_resources):
        resources.append({
            'id': j,
            'cpu_capacity': rng.choice([4, 8, 16]),
            'memory_capacity': rng.choice([8, 16, 32]),
            'speed_factor': rng.uniform(0.8, 1.5),
        })

    dependencies = []
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if rng.random() < dag_density:
                dependencies.append((i, j))

    return tasks, resources, dependencies


def make_env(num_tasks=20, num_resources=5, seed=42):
    """创建合成环境"""
    tasks, resources, deps = generate_synthetic_workflow(
        num_tasks=num_tasks, num_resources=num_resources, seed=seed)
    return EnhancedWorkflowSimulator(tasks, resources, deps)


def _normalize_state_for_agent(state: Any,
                               task_input_dim: int,
                               resource_input_dim: int
                               ) -> Tuple[Dict[str, np.ndarray], np.ndarray,
                                          np.ndarray, Optional[np.ndarray],
                                          Optional[np.ndarray], Optional[np.ndarray]]:
    """将不同环境的状态统一为agent可消费格式"""
    if isinstance(state, dict):
        task_feats = np.asarray(
            state.get('task_features', np.zeros((1, task_input_dim), dtype=np.float32)),
            dtype=np.float32
        )
        res_feats = np.asarray(
            state.get('resource_features', np.zeros((1, resource_input_dim), dtype=np.float32)),
            dtype=np.float32
        )

        if task_feats.ndim == 3 and task_feats.shape[0] == 1:
            task_feats = task_feats[0]
        if res_feats.ndim == 3 and res_feats.shape[0] == 1:
            res_feats = res_feats[0]

        global_feats = np.asarray(state.get('global_features', np.array([], dtype=np.float32)),
                                  dtype=np.float32)
        agent_state = {
            'task_features': task_feats,
            'resource_features': res_feats,
            'global_features': global_feats
        }
        return (
            agent_state,
            task_feats,
            res_feats,
            state.get('adj_matrix', None),
            state.get('node_depths', None),
            state.get('critical_path_mask', None)
        )

    if isinstance(state, (tuple, list)) and len(state) >= 2:
        task_feats = np.asarray(state[0], dtype=np.float32)
        res_feats = np.asarray(state[1], dtype=np.float32)

        if task_feats.ndim == 3 and task_feats.shape[0] == 1:
            task_feats = task_feats[0]
        if res_feats.ndim == 3 and res_feats.shape[0] == 1:
            res_feats = res_feats[0]

        agent_state = {
            'task_features': task_feats,
            'resource_features': res_feats,
            'global_features': np.array([], dtype=np.float32)
        }
        return agent_state, task_feats, res_feats, None, None, None

    raise TypeError(f"Unsupported state type: {type(state)}")


def _extract_env_metrics(env: Any, info: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """统一提取环境指标"""
    info = info or {}

    if hasattr(env, 'get_scheduling_result'):
        result = env.get_scheduling_result()
        return {
            'makespan': float(result.get('makespan', info.get('makespan', 0.0))),
            'resource_utilization': float(result.get('resource_utilization', info.get('utilization', 0.0))),
            'load_balance': float(result.get('load_balance', info.get('load_balance', 0.0)))
        }

    makespan = float(env.get_makespan()) if hasattr(env, 'get_makespan') else float(info.get('makespan', 0.0))
    utilization = float(env.get_resource_utilization()) if hasattr(env, 'get_resource_utilization') else float(info.get('utilization', 0.0))
    load_balance = float(env.get_load_balance_score()) if hasattr(env, 'get_load_balance_score') else float(info.get('load_balance', 0.0))

    return {
        'makespan': makespan,
        'resource_utilization': utilization,
        'load_balance': load_balance
    }


def _get_valid_action_count(env: Any, res_feats: np.ndarray, fallback: int) -> int:
    """获取当前状态下有效动作数，避免动作别名映射。"""
    if hasattr(env, 'get_valid_action_count'):
        try:
            return max(1, int(env.get_valid_action_count()))
        except Exception:
            pass

    if isinstance(res_feats, np.ndarray) and res_feats.ndim >= 2:
        # 资源特征中全0行通常表示padding资源，不计入有效动作
        non_zero_rows = int(np.sum(np.any(np.abs(res_feats) > 1e-8, axis=1)))
        if non_zero_rows > 0:
            return max(1, non_zero_rows)

    if hasattr(env, 'num_resources'):
        try:
            return max(1, int(getattr(env, 'num_resources')))
        except Exception:
            pass

    return max(1, int(fallback))


def _compute_training_reward(agent: EnhancedFE_IDDQN,
                             env: Any,
                             env_reward: float,
                             info: Dict[str, Any]) -> float:
    """融合环境奖励与增强奖励，确保 reward_calculator 在训练主环生效。"""
    if not hasattr(agent, 'reward_calculator'):
        return float(env_reward)

    try:
        host = info.get('host')
        resource = {}
        if host and hasattr(env, 'available_resources'):
            resource = dict(getattr(env, 'available_resources', {}).get(host, {}))

        task = {
            'cpu_req': float(info.get('cpu_req', 0.0)),
            'memory_req': float(info.get('memory_req', 0.0)),
            'criticality_score': 1.0,
            'is_critical_path': False,
            'earliest_start_time': float(info.get('start_time', 0.0)),
        }
        scheduler_state = {
            'current_makespan': float(info.get('current_makespan', 0.0)),
            'total_tasks': int(info.get('total_tasks', 1)),
            'completed_tasks': int(info.get('completed_tasks', 0)),
            'resource_loads': list(info.get('resource_loads', [])),
            'num_resources': int(info.get('num_resources', 1)),
            'concurrent_tasks_at_time': {},
        }
        shaped_reward, _ = agent.reward_calculator.calculate_reward(
            task=task,
            resource=resource,
            start_time=float(info.get('start_time', 0.0)),
            end_time=float(info.get('end_time', info.get('start_time', 0.0))),
            scheduler_state=scheduler_state,
        )

        # 以环境奖励为主，增强奖励为辅，降低训练分布突变风险
        alpha = 0.7
        return float(alpha * env_reward + (1.0 - alpha) * shaped_reward)
    except Exception:
        return float(env_reward)


def _reset_env_and_get_state(env: Any) -> Any:
    """兼容不同环境reset返回风格，统一返回state"""
    state = env.reset()
    if state is None and hasattr(env, 'get_state'):
        state = env.get_state()
    return state


def _read_first_existing_csv(base_dir: Path, candidate_names: List[str]) -> pd.DataFrame:
    """按候选文件名顺序读取首个存在的CSV"""
    for name in candidate_names:
        candidate = base_dir / name
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError(f"No CSV found in {base_dir} for candidates: {candidate_names}")


def load_replay_dataframes(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """加载真实日志数据并返回DataFrame字典"""
    return {
        'process_definition': _read_first_existing_csv(data_dir, [
            't_ds_process_definition.csv',
            'oceanbase_t_ds_process_definition.csv',
            '__B_t_ds_process_definition.csv'
        ]),
        'process_instance': _read_first_existing_csv(data_dir, [
            't_ds_process_instance.csv',
            'gaussdb_t_ds_process_instance_a.csv',
            'Commercial_B_t_ds_process_instance.csv',
            '__B_t_ds_process_instance.csv'
        ]),
        'task_definition': _read_first_existing_csv(data_dir, [
            't_ds_task_definition.csv',
            'oceanbase_t_ds_task_definition.csv',
            '__B_t_ds_task_definition.csv'
        ]),
        'task_instance': _read_first_existing_csv(data_dir, [
            't_ds_task_instance.csv',
            'gaussdb_t_ds_task_instance_a.csv',
            'Commercial_B_t_ds_task_instance.csv',
            '__B_t_ds_task_instance.csv'
        ]),
        'process_task_relation': _read_first_existing_csv(data_dir, [
            't_ds_process_task_relation.csv',
            'oceanbase_t_ds_process_task_relation.csv',
            '__B_t_ds_process_task_relation.csv'
        ])
    }


def make_replay_envs(data_dir: Path,
                     train_ratio: float,
                     logger: logging.Logger
                     ) -> Tuple[LogReplaySimulator, LogReplaySimulator, Dict[str, Any]]:
    """创建训练/验证回放环境"""
    data = load_replay_dataframes(data_dir)

    process_instances = data['process_instance'].copy()
    task_instances = data['task_instance'].copy()
    task_definitions = data['task_definition'].copy()
    process_task_relations = data['process_task_relation'].copy()

    # 兼容不同来源CSV字段命名，适配 HistoricalReplaySimulator 预期列
    if 'process_definition_code' not in process_instances.columns:
        if 'process_definition_id' in process_instances.columns:
            process_instances['process_definition_code'] = process_instances['process_definition_id']
        else:
            process_instances['process_definition_code'] = 0

    if 'process_definition_code' not in process_task_relations.columns:
        if 'process_definition_id' in process_task_relations.columns:
            process_task_relations['process_definition_code'] = process_task_relations['process_definition_id']
        else:
            process_task_relations['process_definition_code'] = 0

    if 'process_definition_code' not in task_instances.columns:
        proc_code_map = process_instances[['id', 'process_definition_code']].drop_duplicates()
        task_instances = task_instances.merge(
            proc_code_map,
            left_on='process_instance_id',
            right_on='id',
            how='left',
            suffixes=('', '_proc')
        )
        if 'id_proc' in task_instances.columns:
            task_instances = task_instances.drop(columns=['id_proc'])

    if 'host' not in task_instances.columns:
        if 'worker_group' in task_instances.columns:
            task_instances['host'] = task_instances['worker_group'].fillna('default_host')
        else:
            task_instances['host'] = 'default_host'

    processes_with_tasks = set(task_instances['process_instance_id'].unique())
    successful = process_instances[
        (process_instances['state'] == 7) &
        (process_instances['id'].isin(processes_with_tasks))
    ].sort_values('start_time').reset_index(drop=True)

    if successful.empty:
        raise ValueError('No successful process instances with tasks found in replay data.')

    split_idx = int(len(successful) * train_ratio)
    split_idx = max(1, min(split_idx, len(successful) - 1)) if len(successful) > 1 else 1

    train_ids = set(successful.iloc[:split_idx]['id'].tolist())
    val_ids = set(successful.iloc[split_idx:]['id'].tolist()) if len(successful) > 1 else train_ids

    if not val_ids:
        val_ids = train_ids

    train_process_df = process_instances[process_instances['id'].isin(train_ids)].copy()
    train_task_df = task_instances[task_instances['process_instance_id'].isin(train_ids)].copy()

    val_process_df = process_instances[process_instances['id'].isin(val_ids)].copy()
    val_task_df = task_instances[task_instances['process_instance_id'].isin(val_ids)].copy()

    train_env = LogReplaySimulator(
        process_instances=train_process_df,
        task_instances=train_task_df,
        task_definitions=task_definitions,
        process_task_relations=process_task_relations
    )
    val_env = LogReplaySimulator(
        process_instances=val_process_df,
        task_instances=val_task_df,
        task_definitions=task_definitions,
        process_task_relations=process_task_relations
    )

    meta = {
        'data_dir': str(data_dir),
        'total_successful_processes': int(len(successful)),
        'train_processes': int(len(train_ids)),
        'val_processes': int(len(val_ids)),
        'train_tasks': int(len(train_task_df)),
        'val_tasks': int(len(val_task_df))
    }
    logger.info(f"Replay data loaded: {meta}")
    return train_env, val_env, meta


def _extract_replay_snapshot_for_baselines(env: LogReplaySimulator) -> Tuple[List[Dict], List[Dict], List[Tuple[int, int]]]:
    """从回放环境中抽取单个流程快照用于传统算法公平对比"""
    if not hasattr(env, 'current_process_tasks') or env.current_process_tasks is None or env.current_process_tasks.empty:
        env.reset()

    task_df = env.current_process_tasks
    tasks: List[Dict] = []
    code_to_task_id: Dict[Any, int] = {}

    for _, row in task_df.iterrows():
        task_id = int(row['id'])
        task_code = row.get('task_code', row.get('task_definition_code', None))
        if pd.notna(task_code):
            code_to_task_id[str(task_code)] = task_id

        tasks.append({
            'id': task_id,
            'name': row.get('name', f'task_{task_id}'),
            'duration': float(env._estimate_task_duration(row)),
            'cpu_req': float(env._estimate_task_cpu_requirement(row)),
            'memory_req': float(env._estimate_task_memory_requirement(row)),
            'priority': float(row.get('task_instance_priority', 0) or 0),
            'task_type': row.get('task_type', 'SHELL')
        })

    resources: List[Dict] = []
    for host, r in env.available_resources.items():
        resources.append({
            'id': host,
            'name': host,
            'cpu_capacity': float(r.get('cpu_capacity', 4.0)),
            'memory_capacity': float(r.get('memory_capacity', 8.0))
        })

    dependencies: List[Tuple[int, int]] = []
    for dep in getattr(env, 'current_process_dependencies', []) or []:
        pre_code = dep.get('pre_task_code')
        post_code = dep.get('post_task_code')
        if pd.notna(pre_code) and pd.notna(post_code):
            pre_id = code_to_task_id.get(str(pre_code))
            post_id = code_to_task_id.get(str(post_code))
            if pre_id is not None and post_id is not None:
                dependencies.append((pre_id, post_id))

    return tasks, resources, dependencies


def _clone_replay_env(env: LogReplaySimulator) -> LogReplaySimulator:
    """克隆回放环境（用于多算法多次公平评估）"""
    return LogReplaySimulator(
        process_instances=env.process_instances.copy(),
        task_instances=env.task_instances.copy(),
        task_definitions=env.task_definitions.copy(),
        process_task_relations=env.process_task_relations.copy(),
    )


def run_replay_baseline_comparison(val_env: LogReplaySimulator,
                                   logger: logging.Logger,
                                   num_episodes: int = 10) -> Dict[str, Dict[str, float]]:
    """在回放数据上运行传统基线并汇总为论文表格用指标"""
    schedulers = {
        'FIFO': FIFOScheduler(),
        'SJF': SJFScheduler(),
        'HEFT': HEFTScheduler(),
    }

    comparison: Dict[str, Dict[str, float]] = {}
    for name, scheduler in schedulers.items():
        makespans: List[float] = []
        utilizations: List[float] = []

        for _ in range(num_episodes):
            env = _clone_replay_env(val_env)
            env.reset()
            tasks, resources, dependencies = _extract_replay_snapshot_for_baselines(env)

            if not tasks or not resources:
                continue

            result = scheduler.schedule(tasks, resources, dependencies)
            makespan = result.get('makespan', float('inf'))
            if not isinstance(makespan, (int, float)) or not math.isfinite(makespan):
                continue

            util = result.get('resource_utilization', 0.0)
            makespans.append(float(makespan))
            utilizations.append(float(util) if isinstance(util, (int, float)) else 0.0)

        comparison[name] = {
            'makespan': float(np.mean(makespans)) if makespans else float('inf'),
            'makespan_std': float(np.std(makespans)) if makespans else 0.0,
            'utilization': float(np.mean(utilizations)) if utilizations else 0.0,
            'episodes': len(makespans)
        }
        logger.info(f"Replay baseline {name}: {comparison[name]}")

    return comparison


def generate_paper_tables(output_dir: Path,
                          fe_result: Dict[str, float],
                          baseline_results: Dict[str, Dict[str, float]],
                          logger: logging.Logger):
    """自动生成论文可直接引用的结果表"""
    rows = []
    rows.append({
        'Algorithm': 'FE-IDDQN',
        'Avg Makespan': fe_result.get('makespan', 0.0),
        'Std Makespan': fe_result.get('makespan_std', 0.0),
        'Avg Utilization': fe_result.get('utilization', 0.0),
        'Episodes': fe_result.get('episodes', 0)
    })

    for name, m in baseline_results.items():
        rows.append({
            'Algorithm': name,
            'Avg Makespan': m.get('makespan', 0.0),
            'Std Makespan': m.get('makespan_std', 0.0),
            'Avg Utilization': m.get('utilization', 0.0),
            'Episodes': m.get('episodes', 0)
        })

    df = pd.DataFrame(rows)
    if not df.empty and 'Avg Makespan' in df.columns:
        df = df.sort_values('Avg Makespan', ascending=True).reset_index(drop=True)

    csv_path = output_dir / 'paper_results_table.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 论文友好的Markdown表
    md_path = output_dir / 'paper_results_table.md'
    md_lines = [
        '| Algorithm | Avg Makespan | Std Makespan | Avg Utilization | Episodes |',
        '|---|---:|---:|---:|---:|'
    ]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['Algorithm']} | {row['Avg Makespan']:.4f} | {row['Std Makespan']:.4f} | {row['Avg Utilization']:.4f} | {int(row['Episodes'])} |"
        )

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')

    # 计算FE-IDDQN相对提升
    improvements = {}
    fe_makespan = float(fe_result.get('makespan', 0.0))
    for name, m in baseline_results.items():
        base_ms = float(m.get('makespan', 0.0))
        if base_ms > 0 and math.isfinite(base_ms):
            improvements[name] = {
                'makespan_improvement_ratio': (base_ms - fe_makespan) / base_ms
            }

    with open(output_dir / 'paper_improvements.json', 'w', encoding='utf-8') as f:
        json.dump(improvements, f, indent=2, ensure_ascii=False)

    logger.info(f"Paper tables generated: {csv_path}, {md_path}")


# ─────────── 评估函数 ─────────── #

def evaluate_dqn_agent(agent: EnhancedFE_IDDQN,
                       env: EnhancedWorkflowSimulator,
                       num_episodes: int = 10,
                       ) -> Dict[str, float]:
    """
    评估 FE-IDDQN agent 性能

    Returns:
        {'makespan': ..., 'utilization': ..., 'load_balance': ...}
    """
    makespans = []
    utilizations = []
    balances = []

    for _ in range(num_episodes):
        state = _reset_env_and_get_state(env)
        done = False
        last_info = {}

        while not done:
            _, task_feats, res_feats, adj, node_depths, critical_mask = \
                _normalize_state_for_agent(
                    state,
                    task_input_dim=agent.task_input_dim,
                    resource_input_dim=agent.resource_input_dim
                )

            action = agent.select_action(
                task_feats, res_feats,
                adj_matrix=adj,
                node_depths=node_depths,
                critical_path_mask=critical_mask,
                valid_action_count=_get_valid_action_count(env, res_feats, agent.action_dim),
                training=False)

            state, reward, done, info = env.step(action)
            last_info = info

        result = _extract_env_metrics(env, last_info)
        makespans.append(result['makespan'])
        utilizations.append(result['resource_utilization'])
        balances.append(result['load_balance'])

    return {
        'makespan': float(np.mean(makespans)),
        'makespan_std': float(np.std(makespans)),
        'utilization': float(np.mean(utilizations)),
        'load_balance': float(np.mean(balances)),
        'episodes': int(num_episodes),
    }


# ─────────── FE-IDDQN 训练循环 ─────────── #

def train_dqn(agent: EnhancedFE_IDDQN,
              env: EnhancedWorkflowSimulator,
              config: EnhancedFE_IDDQN_Config,
              output_dir: Path,
              logger: logging.Logger,
              val_env: Optional[EnhancedWorkflowSimulator] = None,
              ) -> Dict[str, Any]:
    """
    FE-IDDQN 训练主循环 (off-policy)

    Returns:
        训练结果字典
    """
    logger.info("=" * 60)
    logger.info("FE-IDDQN (GA-HPO Enhanced) Training Start")
    logger.info(f"  Episodes:          {config.max_episodes}")
    logger.info(f"  Max steps/ep:      {config.max_steps_per_episode}")
    logger.info(f"  Batch size:        {config.batch_size}")
    logger.info(f"  LR:                {config.learning_rate}")
    logger.info(f"  Target update:     {config.target_update_freq}")
    logger.info(f"  N-step:            {config.n_step}")
    logger.info(f"  Device:            {agent.device}")
    logger.info("=" * 60)

    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    best_makespan = float('inf')
    training_log: List[Dict] = []
    total_steps = 0

    for episode in range(config.max_episodes):
        ep_start = time.time()
        state = _reset_env_and_get_state(env)
        ep_reward = 0.0
        ep_steps = 0
        last_info: Dict[str, Any] = {}

        done = False
        while not done and ep_steps < config.max_steps_per_episode:
            state_for_store, task_feats, res_feats, adj, node_depths, critical_mask = \
                _normalize_state_for_agent(
                    state,
                    task_input_dim=agent.task_input_dim,
                    resource_input_dim=agent.resource_input_dim
                )

            # 选择动作
            action = agent.select_action(
                task_feats, res_feats,
                adj_matrix=adj,
                node_depths=node_depths,
                critical_path_mask=critical_mask,
                valid_action_count=_get_valid_action_count(env, res_feats, agent.action_dim),
                training=True)

            # 环境交互
            next_state, env_reward, done, info = env.step(action)
            last_info = info
            reward = _compute_training_reward(agent, env, env_reward, info)

            next_state_for_store, _, _, _, _, _ = _normalize_state_for_agent(
                next_state,
                task_input_dim=agent.task_input_dim,
                resource_input_dim=agent.resource_input_dim
            )

            # 存储经验
            agent.store_experience(state_for_store, action, reward,
                                   next_state_for_store, done, info)

            # DQN训练步骤 (off-policy: 每 train_freq 步训练一次)
            if total_steps % config.train_freq == 0:
                train_result = agent.train_step()

            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            state = next_state

        # Episode 结束回调
        ep_metrics = _extract_env_metrics(env, last_info)
        agent.on_episode_end(ep_reward, {
            'makespan': ep_metrics.get('makespan', 0),
            'utilization': ep_metrics.get('resource_utilization', 0),
        })

        ep_time = time.time() - ep_start

        # ── 验证评估 ──
        val_result = None
        if val_env is not None and (episode + 1) % 10 == 0:
            val_result = evaluate_dqn_agent(agent, val_env, num_episodes=5)
            if val_result['makespan'] < best_makespan:
                best_makespan = val_result['makespan']
                agent.save(str(models_dir / 'best_model.pt'))
                logger.info(f"  ** New best makespan: {best_makespan:.2f}")

        # 记录
        exploration_stats = agent.exploration.get_stats()
        log_entry = {
            'episode': episode,
            'ep_reward': ep_reward,
            'ep_steps': ep_steps,
            'total_steps': total_steps,
            'time': ep_time,
            'epsilon': exploration_stats.get('epsilon', 0),
            'avg_loss': np.mean(agent.training_losses[-100:]) if agent.training_losses else 0,
        }
        if val_result:
            log_entry['val_makespan'] = val_result['makespan']
            log_entry['val_utilization'] = val_result['utilization']
        training_log.append(log_entry)

        if (episode + 1) % 5 == 0:
            logger.info(
                f"Ep {episode + 1}/{config.max_episodes} | "
                f"reward={ep_reward:.2f} | "
                f"steps={ep_steps} | "
                f"eps={exploration_stats.get('epsilon', 0):.3f} | "
                f"loss={log_entry['avg_loss']:.4f} | "
                f"t={ep_time:.1f}s")

        # 定期保存
        if (episode + 1) % 50 == 0:
            agent.save(str(models_dir / f'checkpoint_ep{episode + 1}.pt'))

    # 保存最终模型 + 日志
    agent.save(str(models_dir / 'final_model.pt'))
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2, default=str)

    logger.info("Training complete!")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Best makespan: {best_makespan:.2f}")

    return {
        'training_log': training_log,
        'best_makespan': best_makespan,
        'total_steps': total_steps,
    }


# ─────────── Phase 1: GA 架构搜索 ─────────── #

def run_ga_search(task_input_dim: int, resource_input_dim: int,
                  action_dim: int,
                  env: EnhancedWorkflowSimulator,
                  output_dir: Path, logger: logging.Logger,
                  ga_config: Optional[GAConfig] = None
                  ) -> Dict[str, Any]:
    """执行GA网络架构搜索 (复用 ga_optimizer.py)"""
    logger.info("=" * 60)
    logger.info("Phase 1: GA Architecture Search (for FE-IDDQN)")
    logger.info("=" * 60)

    ga_cfg = ga_config or GAConfig()
    optimizer = GAArchitectureOptimizer(ga_cfg)

    def fitness_fn(network_structure: Dict) -> Dict[str, float]:
        """适应度函数: 构建 FE-IDDQN → 短期训练 → 评估"""
        config = EnhancedFE_IDDQN_Config(
            hidden_dim=network_structure.get('hidden_dim', 256),
            fusion_dim=network_structure.get('fusion_dim', 256),
            num_transformer_layers=network_structure.get(
                'num_transformer_layers', 2),
            num_heads=network_structure.get('num_heads', 4),
            dropout=network_structure.get('dropout', 0.1),
            use_gnn=network_structure.get('use_gnn', True),
            max_episodes=ga_cfg.eval_episodes,
            batch_size=64,
            warmup_steps=50,
            device='cpu',
        )

        try:
            agent = EnhancedFE_IDDQN(
                task_input_dim, resource_input_dim,
                action_dim, config)

            # 短期训练
            for ep in range(ga_cfg.eval_episodes):
                state = _reset_env_and_get_state(env)
                done = False
                steps = 0
                while not done and steps < 200:
                    state_for_store, tf, rf, adj, nd, cm = \
                        _normalize_state_for_agent(
                            state,
                            task_input_dim=task_input_dim,
                            resource_input_dim=resource_input_dim
                        )

                    action = agent.select_action(
                        tf, rf, adj_matrix=adj,
                        node_depths=nd, critical_path_mask=cm,
                        valid_action_count=_get_valid_action_count(env, rf, action_dim),
                        training=True)
                    next_state, env_reward, done, info = env.step(action)
                    reward = _compute_training_reward(agent, env, env_reward, info)

                    next_state_for_store, _, _, _, _, _ = _normalize_state_for_agent(
                        next_state,
                        task_input_dim=task_input_dim,
                        resource_input_dim=resource_input_dim
                    )
                    agent.store_experience(
                        state_for_store, action, reward,
                        next_state_for_store, done, info)

                    if steps % 4 == 0:
                        agent.train_step()

                    state = next_state if not done else _reset_env_and_get_state(env)
                    steps += 1

            # 评估
            result = evaluate_dqn_agent(agent, env, num_episodes=5)
            param_count = sum(
                p.numel() for p in agent.q_network.parameters())
            result['params'] = param_count
            return result

        except Exception as e:
            logger.warning(f"GA fitness eval failed: {e}")
            return {'makespan': 1e6, 'utilization': 0, 'params': 1e8}

    best_structure = optimizer.search(fitness_fn)

    summary = optimizer.get_search_summary()
    with open(output_dir / 'ga_search_result.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"GA best structure: {best_structure}")
    return best_structure


# ─────────── Phase 2: Optuna HPO ─────────── #

def run_hpo(task_input_dim: int, resource_input_dim: int,
            action_dim: int,
            env: EnhancedWorkflowSimulator,
            output_dir: Path, logger: logging.Logger,
            network_structure: Optional[Dict] = None,
            hpo_config: Optional[DQNHPOConfig] = None
            ) -> Dict[str, Any]:
    """执行Optuna超参数搜索 (DQN版本)"""
    logger.info("=" * 60)
    logger.info("Phase 2: Optuna HPO (for FE-IDDQN)")
    logger.info("=" * 60)

    hpo_cfg = hpo_config or DQNHPOConfig(n_trials=30, timeout=1800)
    optimizer = DQNHPOptimizer(hpo_cfg)

    def objective_fn(params: Dict[str, Any]) -> Dict[str, float]:
        """目标函数: 构建 FE-IDDQN → 短期训练 → 评估"""
        ns = params.get('network_structure', network_structure) or {}

        config = EnhancedFE_IDDQN_Config(
            hidden_dim=ns.get('hidden_dim', 256),
            fusion_dim=ns.get('fusion_dim', 256),
            num_transformer_layers=ns.get('num_transformer_layers', 2),
            num_heads=ns.get('num_heads', 4),
            use_gnn=ns.get('use_gnn', True),
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            tau=params['tau'],
            epsilon_decay=params['epsilon_decay'],
            n_step=params['n_step'],
            use_n_step=params['n_step'] > 1,
            batch_size=params['batch_size'],
            replay_buffer_size=params['replay_buffer_size'],
            target_update_freq=params['target_update_freq'],
            per_alpha=params['per_alpha'],
            per_beta_start=params['per_beta_start'],
            gradient_clip=params['gradient_clip'],
            max_episodes=hpo_cfg.eval_episodes,
            warmup_steps=50,
            device='cpu',
        )

        agent = EnhancedFE_IDDQN(
            task_input_dim, resource_input_dim,
            action_dim, config, )

        # 短期训练
        for ep in range(hpo_cfg.eval_episodes):
            state = _reset_env_and_get_state(env)
            done = False
            steps = 0
            while not done and steps < 200:
                state_for_store, tf, rf, adj, nd, cm = _normalize_state_for_agent(
                    state,
                    task_input_dim=task_input_dim,
                    resource_input_dim=resource_input_dim
                )

                action = agent.select_action(
                    tf, rf, adj_matrix=adj,
                    node_depths=nd, critical_path_mask=cm,
                    valid_action_count=_get_valid_action_count(env, rf, action_dim),
                    training=True)
                next_state, env_reward, done, info = env.step(action)
                reward = _compute_training_reward(agent, env, env_reward, info)

                next_state_for_store, _, _, _, _, _ = _normalize_state_for_agent(
                    next_state,
                    task_input_dim=task_input_dim,
                    resource_input_dim=resource_input_dim
                )
                agent.store_experience(
                    state_for_store, action, reward,
                    next_state_for_store, done, info)

                if steps % 4 == 0:
                    agent.train_step()

                state = next_state if not done else _reset_env_and_get_state(env)
                steps += 1

        return evaluate_dqn_agent(agent, env, num_episodes=5)

    best_config = optimizer.optimize(objective_fn, network_structure)

    summary = optimizer.get_search_summary()
    with open(output_dir / 'hpo_result.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    importances = optimizer.get_param_importances()
    if importances:
        logger.info("Parameter importances:")
        for k, v in importances.items():
            logger.info(f"  {k}: {v:.4f}")

    logger.info(f"HPO best config: {best_config}")
    return best_config


# ─────────── Main ─────────── #

def main():
    parser = argparse.ArgumentParser(
        description='GA-HPO FE-IDDQN Training System')
    parser.add_argument('--mode', type=str, default='train_only',
                        choices=['full', 'train_only', 'ga_search', 'hpo'],
                        help='运行模式')
    parser.add_argument('--output_dir', type=str,
                        default='results/fe_iddqn_ga_hpo',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env_type', type=str, default='synthetic',
                        choices=['synthetic', 'replay'],
                        help='环境类型: synthetic|replay')
    parser.add_argument('--replay_data_dir', type=str, default='data/raw_data',
                        help='回放数据目录 (包含 process/task CSV)')
    parser.add_argument('--replay_train_ratio', type=float, default=0.8,
                        help='回放训练集比例')
    parser.add_argument('--paper_eval_episodes', type=int, default=10,
                        help='论文表格基线评估回合数（仅replay模式）')
    parser.add_argument('--final_eval_episodes', type=int, default=20,
                        help='最终评估回合数')
    parser.add_argument('--num_tasks', type=int, default=20)
    parser.add_argument('--num_resources', type=int, default=5)
    parser.add_argument('--max_episodes', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--no_gnn', action='store_true',
                        help='禁用GNN')
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    logger = setup_logging(output_dir)

    logger.info(f"GA-HPO FE-IDDQN Training System | Mode: {args.mode}")
    logger.info(f"Output: {output_dir}")

    # ── 创建环境 ──
    replay_meta = None
    if args.env_type == 'replay':
        replay_data_dir = Path(args.replay_data_dir)
        train_env, val_env, replay_meta = make_replay_envs(
            replay_data_dir,
            train_ratio=args.replay_train_ratio,
            logger=logger
        )
    else:
        train_env = make_env(args.num_tasks, args.num_resources, seed=args.seed)
        val_env = make_env(args.num_tasks, args.num_resources, seed=args.seed + 1)

    # 获取维度
    state = _reset_env_and_get_state(train_env)
    _, task_feats, res_feats, _, _, _ = _normalize_state_for_agent(
        state,
        task_input_dim=16,
        resource_input_dim=7
    )
    task_input_dim = task_feats.shape[-1] if len(task_feats.shape) >= 2 else 16
    resource_input_dim = res_feats.shape[-1] if len(res_feats.shape) >= 2 else 7
    action_dim = int(max(
        getattr(train_env, 'num_resources', args.num_resources) or args.num_resources,
        res_feats.shape[0] if len(res_feats.shape) >= 2 else args.num_resources
    ))
    action_dim = max(action_dim, 1)

    logger.info(f"Env dims: task={task_input_dim}, resource={resource_input_dim}, "
                f"action={action_dim}")

    # ── 保存配置 ──
    config_info = vars(args)
    config_info['task_input_dim'] = task_input_dim
    config_info['resource_input_dim'] = resource_input_dim
    config_info['action_dim'] = action_dim
    if replay_meta:
        config_info['replay_meta'] = replay_meta
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_info, f, indent=2)

    network_structure = None
    best_hpo_config = None

    # ── Phase 1: GA架构搜索 ──
    if args.mode in ('full', 'ga_search'):
        network_structure = run_ga_search(
            task_input_dim, resource_input_dim, action_dim,
            train_env, output_dir, logger)

    # ── Phase 2: Optuna HPO ──
    if args.mode in ('full', 'hpo'):
        best_hpo_config = run_hpo(
            task_input_dim, resource_input_dim, action_dim,
            train_env, output_dir, logger,
            network_structure=network_structure)

    # ── Phase 3: 完整训练 ──
    if args.mode in ('full', 'train_only'):
        dqn_config = EnhancedFE_IDDQN_Config(
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_step=args.n_step,
            use_n_step=args.n_step > 1,
            max_episodes=args.max_episodes,
            use_gnn=not args.no_gnn,
            device=args.device,
        )

        # 如果有HPO结果，覆盖默认值
        if best_hpo_config:
            dqn_config.learning_rate = best_hpo_config.get(
                'learning_rate', dqn_config.learning_rate)
            dqn_config.gamma = best_hpo_config.get(
                'gamma', dqn_config.gamma)
            dqn_config.tau = best_hpo_config.get(
                'tau', dqn_config.tau)
            dqn_config.epsilon_decay = best_hpo_config.get(
                'epsilon_decay', dqn_config.epsilon_decay)
            dqn_config.n_step = best_hpo_config.get(
                'n_step', dqn_config.n_step)
            dqn_config.use_n_step = dqn_config.n_step > 1
            dqn_config.batch_size = best_hpo_config.get(
                'batch_size', dqn_config.batch_size)
            dqn_config.replay_buffer_size = best_hpo_config.get(
                'replay_buffer_size', dqn_config.replay_buffer_size)
            dqn_config.target_update_freq = best_hpo_config.get(
                'target_update_freq', dqn_config.target_update_freq)
            dqn_config.per_alpha = best_hpo_config.get(
                'per_alpha', dqn_config.per_alpha)
            dqn_config.per_beta_start = best_hpo_config.get(
                'per_beta_start', dqn_config.per_beta_start)
            dqn_config.gradient_clip = best_hpo_config.get(
                'gradient_clip', dqn_config.gradient_clip)

        # 如果有GA结构结果
        if network_structure:
            dqn_config.hidden_dim = network_structure.get(
                'hidden_dim', dqn_config.hidden_dim)
            dqn_config.fusion_dim = network_structure.get(
                'fusion_dim', dqn_config.fusion_dim)
            dqn_config.num_transformer_layers = network_structure.get(
                'num_transformer_layers',
                dqn_config.num_transformer_layers)
            dqn_config.num_heads = network_structure.get(
                'num_heads', dqn_config.num_heads)

        # 创建Agent
        agent = EnhancedFE_IDDQN(
            task_input_dim, resource_input_dim, action_dim,
            dqn_config)

        logger.info(f"Network params: "
                     f"{sum(p.numel() for p in agent.q_network.parameters()):,}")

        # 训练
        result = train_dqn(
            agent, train_env, dqn_config, output_dir, logger,
            val_env=val_env)

        # 最终评估
        logger.info("=" * 60)
        logger.info("Final Evaluation")
        logger.info("=" * 60)

        final_eval = evaluate_dqn_agent(
            agent, val_env, num_episodes=args.final_eval_episodes)
        logger.info(f"  Makespan:     {final_eval['makespan']:.2f} "
                     f"± {final_eval['makespan_std']:.2f}")
        logger.info(f"  Utilization:  {final_eval['utilization']:.4f}")
        logger.info(f"  Load Balance: {final_eval['load_balance']:.4f}")

        with open(output_dir / 'final_eval.json', 'w') as f:
            json.dump(final_eval, f, indent=2, default=str)

        # replay模式自动生成论文表格（含传统基线）
        if args.env_type == 'replay':
            baseline_results = run_replay_baseline_comparison(
                val_env=val_env,
                logger=logger,
                num_episodes=args.paper_eval_episodes
            )
            generate_paper_tables(
                output_dir=output_dir,
                fe_result=final_eval,
                baseline_results=baseline_results,
                logger=logger
            )

    logger.info("Done!")


if __name__ == '__main__':
    main()
