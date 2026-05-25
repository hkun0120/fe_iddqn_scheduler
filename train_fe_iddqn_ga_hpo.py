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
import traceback
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

    # Python 3.7 does not support logging.basicConfig(force=True).
    # Clear existing handlers explicitly so repeated seed runs still reconfigure logging.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger('FE_IDDQN_GA_HPO')


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """写入JSON，确保目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def update_run_status(output_dir: Path,
                      stage: str,
                      status: str,
                      **extra: Any) -> Dict[str, Any]:
    """记录当前运行阶段，便于定位中断点。"""
    payload: Dict[str, Any] = {
        'stage': stage,
        'status': status,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
    }
    payload.update(extra)
    _write_json(output_dir / 'run_status.json', payload)
    return payload


def record_failure(output_dir: Path,
                   stage: str,
                   exc: BaseException,
                   seed: Optional[int] = None) -> Dict[str, Any]:
    """持久化失败信息，避免只留下空目录和开头日志。"""
    failure_info: Dict[str, Any] = {
        'status': 'failed',
        'stage': stage,
        'error_type': type(exc).__name__,
        'error_message': str(exc),
        'traceback': traceback.format_exc(),
        'failed_at': datetime.now().isoformat(timespec='seconds'),
    }
    if seed is not None:
        failure_info['seed'] = seed

    _write_json(output_dir / 'failure_info.json', failure_info)
    update_run_status(
        output_dir,
        stage=stage,
        status='failed',
        seed=seed,
        error_type=failure_info['error_type'],
        error_message=failure_info['error_message'],
        failure_file='failure_info.json',
    )
    return failure_info


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
                               resource_input_dim: int,
                               disable_fe: bool = False
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

        # Apply ablation for feature engineering: truncate to primitive features
        if disable_fe:
            # Task prim features: duration, cpu_req, mem_req, priority (4 dims)
            if task_feats.shape[-1] > 4:
                task_feats = task_feats[..., :4]
            # Resource prim features: cpu_cap, mem_cap, processing_power (3 dims)
            if res_feats.shape[-1] > 3:
                res_feats = res_feats[..., :3]

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

        # Apply ablation for feature engineering: truncate to primitive features
        if disable_fe:
            if task_feats.shape[-1] > 4:
                task_feats = task_feats[..., :4]
            if res_feats.shape[-1] > 3:
                res_feats = res_feats[..., :3]

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


def _read_first_existing_csv(base_dir: Path,
                             candidate_names: List[str],
                             logger: Optional[logging.Logger] = None,
                             logical_name: str = 'csv') -> pd.DataFrame:
    """按候选文件名顺序读取首个存在的CSV"""
    for name in candidate_names:
        candidate = base_dir / name
        if candidate.exists():
            if logger:
                logger.info(f"Loading replay CSV '{logical_name}' from {candidate}")
            df = pd.read_csv(candidate)
            if logger:
                logger.info(f"Loaded replay CSV '{logical_name}': shape={df.shape}")
            return df
    raise FileNotFoundError(f"No CSV found in {base_dir} for candidates: {candidate_names}")


def load_replay_dataframes(data_dir: Path,
                           logger: Optional[logging.Logger] = None) -> Dict[str, pd.DataFrame]:
    """加载真实日志数据并返回DataFrame字典"""
    return {
        'process_definition': _read_first_existing_csv(data_dir, [
            'Commercial_t_ds_process_definition.csv',
            'Commercial_B_t_ds_process_definition.csv',
            't_ds_process_definition.csv',
            'oceanbase_t_ds_process_definition.csv',
            '__B_t_ds_process_definition.csv'
        ], logger=logger, logical_name='process_definition'),
        'process_instance': _read_first_existing_csv(data_dir, [
            'Commercial_t_ds_process_instance.csv',
            'Commercial_B_t_ds_process_instance.csv',
            't_ds_process_instance.csv',
            'gaussdb_t_ds_process_instance_a.csv',
            '__B_t_ds_process_instance.csv'
        ], logger=logger, logical_name='process_instance'),
        'task_definition': _read_first_existing_csv(data_dir, [
            'Commercial_t_ds_task_definition.csv',
            'Commercial_B_t_ds_task_definition.csv',
            't_ds_task_definition.csv',
            'oceanbase_t_ds_task_definition.csv',
            '__B_t_ds_task_definition.csv'
        ], logger=logger, logical_name='task_definition'),
        'task_instance': _read_first_existing_csv(data_dir, [
            'Commercial_t_ds_task_instance.csv',
            'Commercial_B_t_ds_task_instance.csv',
            't_ds_task_instance.csv',
            'gaussdb_t_ds_task_instance_a.csv',
            '__B_t_ds_task_instance.csv'
        ], logger=logger, logical_name='task_instance'),
        'process_task_relation': _read_first_existing_csv(data_dir, [
            'Commercial_t_ds_process_task_relation.csv',
            'Commercial_B_t_ds_process_task_relation.csv',
            't_ds_process_task_relation.csv',
            'oceanbase_t_ds_process_task_relation.csv',
            '__B_t_ds_process_task_relation.csv'
        ], logger=logger, logical_name='process_task_relation')
    }


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    """存在则读取JSON文件，不存在或解析失败返回None。"""
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _read_process_ids_csv(path: Path,
                          logger: Optional[logging.Logger] = None) -> List[Any]:
    """从split文件读取流程ID，支持 process_id/id 或首列。"""
    df = pd.read_csv(path)
    if df.empty:
        return []

    if 'process_id' in df.columns:
        column = 'process_id'
    elif 'id' in df.columns:
        column = 'id'
    else:
        column = df.columns[0]

    ids = [
        LogReplaySimulator._normalize_process_id(value)
        for value in df[column].tolist()
    ]
    ids = [value for value in ids if value is not None]
    if logger:
        logger.info("Loaded %d workflow ids from %s", len(ids), path)
    return ids


SPLIT_METADATA_COLUMNS = [
    'task_count',
    'duration_seconds',
    'dag_node_count',
    'dag_edge_count',
    'dag_depth',
    'dag_width',
    'dag_density',
    'parallelism_ratio',
    'dag_complexity_score',
    'workflow_size',
    'duration_bin',
    'dag_complexity_bin',
    'workflow_stratum',
    'size_duration_stratum',
    'size_complexity_stratum',
    'duration_complexity_stratum',
    'balanced_workflow_stratum',
]


def _read_split_metadata_csv(path: Path,
                             logger: Optional[logging.Logger] = None
                             ) -> Optional[pd.DataFrame]:
    """读取带分层字段的split CSV；普通id文件会被安全忽略。"""
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if df.empty or 'process_id' not in df.columns:
        return None

    metadata_cols = [
        column for column in SPLIT_METADATA_COLUMNS
        if column in df.columns
    ]
    if not metadata_cols:
        return None

    out = df[['process_id', *metadata_cols]].copy()
    out['_normalized_process_id'] = out['process_id'].map(
        LogReplaySimulator._normalize_process_id
    )
    out = out[out['_normalized_process_id'].notna()]
    out = out.drop_duplicates('_normalized_process_id')
    if logger:
        logger.info(
            "Loaded workflow metadata for %d ids from %s",
            len(out),
            path
        )
    return out


def _load_split_metadata(split_dir: Optional[Path],
                         split_name: str,
                         explicit_path: Optional[Path],
                         logger: logging.Logger
                         ) -> Optional[pd.DataFrame]:
    """读取split分层元数据；优先使用显式文件，其次使用 *_data.csv。"""
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(explicit_path)
    if split_dir is not None:
        candidates.append(split_dir / f'{split_name}_data.csv')

    for candidate in candidates:
        metadata = _read_split_metadata_csv(candidate, logger=logger)
        if metadata is not None:
            return metadata
    return None


def _merge_split_metadata(process_df: pd.DataFrame,
                          metadata_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """把任务数、DAG复杂度、时长分层等split字段合并到process表。"""
    if metadata_df is None or metadata_df.empty or process_df.empty:
        return process_df

    out = process_df.copy()
    out['_normalized_process_id'] = out['id'].map(
        LogReplaySimulator._normalize_process_id
    )

    metadata_cols = [
        column for column in SPLIT_METADATA_COLUMNS
        if column in metadata_df.columns and column not in out.columns
    ]
    if not metadata_cols:
        return out.drop(columns=['_normalized_process_id'], errors='ignore')

    merge_df = metadata_df[
        ['_normalized_process_id', *metadata_cols]
    ].drop_duplicates('_normalized_process_id')
    out = out.merge(merge_df, on='_normalized_process_id', how='left')
    return out.drop(columns=['_normalized_process_id'], errors='ignore')


def _resolve_split_dir(data_dir: Path, split_dir: Optional[Path]) -> Optional[Path]:
    if split_dir is not None:
        return split_dir
    default_split_dir = data_dir / 'splits'
    return default_split_dir if default_split_dir.exists() else None


def _load_split_process_ids(split_dir: Optional[Path],
                            split_name: str,
                            explicit_path: Optional[Path],
                            logger: logging.Logger) -> Optional[List[Any]]:
    """读取固定split中的流程ID；显式路径优先。"""
    if explicit_path:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Workflow id file not found: {explicit_path}")
        return _read_process_ids_csv(explicit_path, logger=logger)

    if split_dir is None:
        return None

    candidates = [
        split_dir / f'{split_name}_process_ids.csv',
        split_dir / f'{split_name}_data.csv',
    ]
    for candidate in candidates:
        if candidate.exists():
            return _read_process_ids_csv(candidate, logger=logger)
    return None


def _subset_replay_frames_by_process_ids(process_instances: pd.DataFrame,
                                         task_instances: pd.DataFrame,
                                         process_ids: List[Any]
                                         ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Any]]:
    """按归一化流程ID过滤process/task表，并保留传入ID顺序。"""
    normalized_ids = LogReplaySimulator._normalize_process_id_list(process_ids) or []
    selected_id_set = set(normalized_ids)
    if not selected_id_set:
        return process_instances.head(0).copy(), task_instances.head(0).copy(), []

    proc_mask = process_instances['id'].map(
        LogReplaySimulator._normalize_process_id
    ).isin(selected_id_set)
    task_mask = task_instances['process_instance_id'].map(
        LogReplaySimulator._normalize_process_id
    ).isin(selected_id_set)

    process_df = process_instances[proc_mask].copy()
    if 'state' in process_df.columns:
        process_df = process_df[process_df['state'] == 7].copy()
    task_df = task_instances[task_mask].copy()

    existing = set(
        process_df['id'].map(LogReplaySimulator._normalize_process_id).tolist()
    )
    task_df = task_df[
        task_df['process_instance_id'].map(
            LogReplaySimulator._normalize_process_id
        ).isin(existing)
    ].copy()
    ordered_existing_ids = [
        process_id for process_id in normalized_ids
        if process_id in existing
    ]
    return process_df, task_df, ordered_existing_ids


def make_replay_envs(data_dir: Path,
                     train_ratio: float,
                     logger: logging.Logger,
                     split_dir: Optional[Path] = None,
                     train_process_ids_path: Optional[Path] = None,
                     val_process_ids_path: Optional[Path] = None,
                     test_process_ids_path: Optional[Path] = None,
                     eval_split: str = 'val',
                     train_workflows_per_episode: int = 1,
                     eval_workflows_per_episode: int = 5,
                     ) -> Tuple[LogReplaySimulator, LogReplaySimulator,
                                LogReplaySimulator, Dict[str, Any]]:
    """创建训练、验证和最终评估回放环境。"""
    logger.info(f"Preparing replay environments from {data_dir}")
    data = load_replay_dataframes(data_dir, logger=logger)

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

    resolved_split_dir = _resolve_split_dir(data_dir, split_dir)
    if resolved_split_dir:
        logger.info("Replay split dir: %s", resolved_split_dir)

    train_ids = _load_split_process_ids(
        resolved_split_dir,
        'train',
        train_process_ids_path,
        logger
    )
    val_ids = _load_split_process_ids(
        resolved_split_dir,
        'val',
        val_process_ids_path,
        logger
    )
    test_ids = _load_split_process_ids(
        resolved_split_dir,
        'test',
        test_process_ids_path,
        logger
    )
    train_metadata = _load_split_metadata(
        resolved_split_dir,
        'train',
        train_process_ids_path,
        logger
    )
    val_metadata = _load_split_metadata(
        resolved_split_dir,
        'val',
        val_process_ids_path,
        logger
    )
    test_metadata = _load_split_metadata(
        resolved_split_dir,
        'test',
        test_process_ids_path,
        logger
    )

    split_source = 'explicit_or_split_files' if train_ids else 'chronological_train_ratio'
    if not train_ids:
        split_idx = int(len(successful) * train_ratio)
        split_idx = max(1, min(split_idx, len(successful) - 1)) if len(successful) > 1 else 1
        train_ids = successful.iloc[:split_idx]['id'].tolist()
        val_ids = successful.iloc[split_idx:]['id'].tolist() if len(successful) > 1 else train_ids
        test_ids = None
        logger.warning(
            "No replay workflow split files found; falling back to chronological train_ratio split"
        )

    train_process_df, train_task_df, train_ids_ordered = _subset_replay_frames_by_process_ids(
        process_instances,
        task_instances,
        train_ids or []
    )
    val_process_df, val_task_df, val_ids_ordered = _subset_replay_frames_by_process_ids(
        process_instances,
        task_instances,
        val_ids or []
    )
    test_process_df, test_task_df, test_ids_ordered = _subset_replay_frames_by_process_ids(
        process_instances,
        task_instances,
        test_ids or []
    )

    if not train_ids_ordered:
        raise ValueError('No train workflow ids matched successful replay processes.')
    if not val_ids_ordered:
        logger.warning("No val workflow ids matched successful replay processes; using train ids for validation.")
        val_process_df = train_process_df.copy()
        val_task_df = train_task_df.copy()
        val_ids_ordered = list(train_ids_ordered)

    train_process_df = _merge_split_metadata(train_process_df, train_metadata)
    val_process_df = _merge_split_metadata(val_process_df, val_metadata)
    test_process_df = _merge_split_metadata(test_process_df, test_metadata)

    if eval_split == 'test' and test_ids_ordered:
        final_process_df = test_process_df
        final_task_df = test_task_df
        final_ids_ordered = test_ids_ordered
        final_split_name = 'test'
    else:
        if eval_split == 'test':
            logger.warning("Requested test eval split but no test ids matched; falling back to val.")
        final_process_df = val_process_df.copy()
        final_task_df = val_task_df.copy()
        final_ids_ordered = list(val_ids_ordered)
        final_split_name = 'val'

    logger.info("Creating replay training environment")
    train_env = LogReplaySimulator(
        process_instances=train_process_df,
        task_instances=train_task_df,
        task_definitions=task_definitions,
        process_task_relations=process_task_relations,
        episode_process_ids=train_ids_ordered,
        episode_window_size=train_workflows_per_episode,
        episode_window_stride=train_workflows_per_episode,
        shuffle_episode_processes=True,
        episode_seed=42,
    )
    logger.info("Creating replay validation environment")
    val_env = LogReplaySimulator(
        process_instances=val_process_df,
        task_instances=val_task_df,
        task_definitions=task_definitions,
        process_task_relations=process_task_relations,
        episode_process_ids=val_ids_ordered,
        episode_window_size=eval_workflows_per_episode,
        episode_window_stride=eval_workflows_per_episode,
        shuffle_episode_processes=False,
        episode_seed=42,
    )
    logger.info("Creating replay final evaluation environment (%s split)", final_split_name)
    final_env = LogReplaySimulator(
        process_instances=final_process_df,
        task_instances=final_task_df,
        task_definitions=task_definitions,
        process_task_relations=process_task_relations,
        episode_process_ids=final_ids_ordered,
        episode_window_size=eval_workflows_per_episode,
        episode_window_stride=eval_workflows_per_episode,
        shuffle_episode_processes=False,
        episode_seed=42,
    )

    meta = {
        'data_dir': str(data_dir),
        'split_dir': str(resolved_split_dir) if resolved_split_dir else None,
        'split_source': split_source,
        'eval_split': final_split_name,
        'total_successful_processes': int(len(successful)),
        'train_processes': int(len(train_ids_ordered)),
        'val_processes': int(len(val_ids_ordered)),
        'test_processes': int(len(test_ids_ordered)),
        'eval_processes': int(len(final_ids_ordered)),
        'train_tasks': int(len(train_task_df)),
        'val_tasks': int(len(val_task_df)),
        'test_tasks': int(len(test_task_df)),
        'eval_tasks': int(len(final_task_df)),
        'train_workflows_per_episode': int(train_workflows_per_episode or 0),
        'eval_workflows_per_episode': int(eval_workflows_per_episode or 0),
        'train_process_id_preview': train_ids_ordered[:10],
        'val_process_id_preview': val_ids_ordered[:10],
        'eval_process_id_preview': final_ids_ordered[:10],
    }
    logger.info(f"Replay data loaded: {meta}")
    return train_env, val_env, final_env, meta


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
        episode_process_ids=list(getattr(env, 'episode_process_ids', []) or []),
        episode_window_size=getattr(env, 'episode_window_size', None),
        episode_window_stride=getattr(env, 'episode_window_stride', None),
        shuffle_episode_processes=getattr(env, 'shuffle_episode_processes', False),
        episode_seed=getattr(env, 'episode_seed', 42),
    )


def _task_counts_by_process(task_instances: pd.DataFrame) -> Dict[Any, int]:
    """按归一化流程ID统计任务数。"""
    if task_instances.empty or 'process_instance_id' not in task_instances.columns:
        return {}
    normalized = task_instances['process_instance_id'].map(
        LogReplaySimulator._normalize_process_id
    )
    return normalized.value_counts().astype(int).to_dict()


def _select_stratified_process_pool(process_pool: pd.DataFrame,
                                    task_instances: pd.DataFrame,
                                    max_processes: int,
                                    max_tasks: Optional[int],
                                    seed: int,
                                    strata_column: str,
                                    logger: logging.Logger) -> pd.DataFrame:
    """从验证集按分层字段轮转抽样，避免GA/HPO只看到简单短workflow。"""
    if process_pool.empty:
        return process_pool

    pool = process_pool.copy().reset_index(drop=True)
    pool['_normalized_process_id'] = pool['id'].map(
        LogReplaySimulator._normalize_process_id
    )
    task_counts = _task_counts_by_process(task_instances)
    if 'task_count' in pool.columns:
        pool['_search_task_count'] = pd.to_numeric(
            pool['task_count'], errors='coerce'
        )
    else:
        pool['_search_task_count'] = np.nan
    pool['_search_task_count'] = pool['_search_task_count'].fillna(
        pool['_normalized_process_id'].map(task_counts)
    ).fillna(0).astype(int)

    max_processes = max(1, int(max_processes))
    task_cap = None if max_tasks is None or int(max_tasks) <= 0 else int(max_tasks)

    if strata_column not in pool.columns or pool[strata_column].isna().all():
        selected = pool.sample(
            n=min(max_processes, len(pool)),
            random_state=seed
        ).reset_index(drop=True)
    else:
        pool['_search_stratum'] = pool[strata_column].fillna('unknown').astype(str)
        strata = sorted(pool['_search_stratum'].unique().tolist())
        grouped = {
            stratum: pool[pool['_search_stratum'] == stratum]
            .sample(frac=1.0, random_state=seed + idx)
            .reset_index(drop=True)
            for idx, stratum in enumerate(strata)
        }
        cursors = {stratum: 0 for stratum in strata}
        selected_rows: List[pd.Series] = []
        selected_ids = set()
        selected_task_count = 0

        while len(selected_rows) < max_processes:
            progressed = False
            for stratum in strata:
                group = grouped[stratum]
                while cursors[stratum] < len(group):
                    row = group.iloc[cursors[stratum]]
                    cursors[stratum] += 1
                    process_id = row['_normalized_process_id']
                    if process_id in selected_ids:
                        continue
                    task_count = int(row.get('_search_task_count', 0) or 0)
                    if task_cap is not None and selected_rows and selected_task_count + task_count > task_cap:
                        continue
                    if task_cap is not None and not selected_rows and task_count > task_cap:
                        continue
                    selected_rows.append(row)
                    selected_ids.add(process_id)
                    selected_task_count += task_count
                    progressed = True
                    break
                if len(selected_rows) >= max_processes:
                    break
            if not progressed:
                break

        if not selected_rows:
            selected = pool.sample(
                n=min(max_processes, len(pool)),
                random_state=seed
            ).reset_index(drop=True)
            if task_cap is not None:
                selected['_cum_tasks'] = selected['_search_task_count'].cumsum()
                selected = selected[selected['_cum_tasks'] <= task_cap]
                if selected.empty:
                    selected = pool.sort_values('_search_task_count').head(1)
                selected = selected.drop(columns=['_cum_tasks'], errors='ignore')
        else:
            selected = pd.DataFrame(selected_rows).reset_index(drop=True)

    if task_cap is not None and len(selected) > 1:
        total_tasks = int(selected['_search_task_count'].sum())
        if total_tasks > task_cap:
            kept_rows = []
            kept_tasks = 0
            for _, row in selected.iterrows():
                task_count = int(row.get('_search_task_count', 0) or 0)
                if kept_rows and kept_tasks + task_count > task_cap:
                    continue
                kept_rows.append(row)
                kept_tasks += task_count
            selected = pd.DataFrame(kept_rows).reset_index(drop=True)

    if '_search_stratum' in selected.columns:
        dist = selected['_search_stratum'].value_counts().to_dict()
        logger.info("Search replay stratum distribution: %s", dist)

    return selected.drop(
        columns=['_normalized_process_id', '_search_task_count', '_search_stratum'],
        errors='ignore'
    ).reset_index(drop=True)


def _compact_replay_env_factory(env: LogReplaySimulator,
                                max_processes: int,
                                max_tasks: int,
                                seed: int,
                                strata_column: str,
                                logger: logging.Logger):
    """
    为GA/HPO短评估创建轻量replay环境工厂。

    真实数据集可能包含百万级流程和千万级任务，不能在每个GA个体里
    deepcopy完整环境。这里仅保留当前采样窗口中的少量流程和对应关系表。
    """
    if not hasattr(env, 'successful_processes') or env.successful_processes is None:
        env.reset()

    if getattr(env, 'episode_process_ids', None):
        process_ids = set(env.episode_process_ids)
        proc_mask = (
            (env.process_instances['state'] == 7) &
            env.process_instances['id'].map(
                LogReplaySimulator._normalize_process_id
            ).isin(process_ids)
        )
        process_pool = env.process_instances[proc_mask].sort_values('start_time').copy()
    else:
        process_pool = env.successful_processes.copy()

    if process_pool.empty:
        process_pool = env.process_instances.head(max(1, max_processes)).copy()

    max_processes = max(1, int(max_processes))
    task_cap = None if int(max_tasks or 0) <= 0 else int(max_tasks)
    process_pool = _select_stratified_process_pool(
        process_pool=process_pool,
        task_instances=env.task_instances,
        max_processes=max_processes,
        max_tasks=task_cap,
        seed=seed,
        strata_column=strata_column,
        logger=logger,
    )

    task_pool = env.task_instances[
        env.task_instances['process_instance_id'].isin(process_pool['id'])
    ].copy()

    proc_codes = set(process_pool.get('process_definition_code', pd.Series(dtype=object)).dropna().tolist())
    relation_pool = env.process_task_relations
    if proc_codes and 'process_definition_code' in relation_pool.columns:
        relation_pool = relation_pool[relation_pool['process_definition_code'].isin(proc_codes)].copy()
    else:
        relation_pool = relation_pool.head(0).copy()

    task_def_pool = env.task_definitions
    task_codes = set()
    for col in ('task_code', 'task_definition_code'):
        if col in task_pool.columns:
            task_codes.update(task_pool[col].dropna().tolist())
    for col in ('code', 'task_code', 'task_definition_code'):
        if task_codes and col in task_def_pool.columns:
            task_def_pool = task_def_pool[task_def_pool[col].isin(task_codes)].copy()
            break
    else:
        task_def_pool = task_def_pool.copy(deep=False)

    logger.info(
        "Search replay subset: processes=%d, tasks=%d, relations=%d, task_defs=%d",
        len(process_pool), len(task_pool), len(relation_pool), len(task_def_pool)
    )

    def _factory() -> LogReplaySimulator:
        return LogReplaySimulator(
            process_instances=process_pool.copy(deep=False),
            task_instances=task_pool.copy(deep=False),
            task_definitions=task_def_pool.copy(deep=False),
            process_task_relations=relation_pool.copy(deep=False),
            episode_process_ids=process_pool['id'].tolist(),
            episode_window_size=None,
            shuffle_episode_processes=False,
            episode_seed=seed,
        )

    return _factory


def _make_search_env_factory(env: Any,
                             train_args: Optional[Any],
                             logger: logging.Logger):
    """返回GA/HPO评估用环境工厂，避免共享状态和大对象深拷贝。"""
    if isinstance(env, LogReplaySimulator):
        max_processes = getattr(train_args, 'search_max_processes', 8) if train_args else 8
        max_tasks = getattr(train_args, 'search_max_tasks', 120) if train_args else 120
        seed = getattr(train_args, 'seed', 42) if train_args else 42
        strata_column = (
            getattr(train_args, 'search_strata_column', 'balanced_workflow_stratum')
            if train_args else 'balanced_workflow_stratum'
        )
        return _compact_replay_env_factory(
            env, max_processes, max_tasks, seed, strata_column, logger
        )

    import copy

    def _factory():
        return copy.deepcopy(env)

    return _factory


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
                       max_steps_per_episode: Optional[int] = None,
                       return_details: bool = False,
                       ) -> Dict[str, float]:
    """
    评估 FE-IDDQN agent 性能

    Returns:
        {'makespan': ..., 'utilization': ..., 'load_balance': ...}
    """
    makespans = []
    utilizations = []
    balances = []
    episode_details: List[Dict[str, Any]] = []
    truncated_episodes = 0

    for episode in range(num_episodes):
        state = _reset_env_and_get_state(env)
        done = False
        last_info = {}
        steps = 0
        truncated = False

        while not done:
            _, task_feats, res_feats, adj, node_depths, critical_mask = \
                _normalize_state_for_agent(
                    state,
                    task_input_dim=agent.task_input_dim,
                    resource_input_dim=agent.resource_input_dim,
                    disable_fe=not agent.config.use_feature_engineering
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
            steps += 1
            if max_steps_per_episode and steps >= max_steps_per_episode and not done:
                truncated = True
                truncated_episodes += 1
                break

        result = _extract_env_metrics(env, last_info)
        makespans.append(result['makespan'])
        utilizations.append(result['resource_utilization'])
        balances.append(result['load_balance'])
        if return_details:
            episode_details.append({
                'episode': int(episode),
                'steps': int(steps),
                'truncated': bool(truncated),
                'makespan': float(result['makespan']),
                'utilization': float(result['resource_utilization']),
                'load_balance': float(result['load_balance']),
            })

    summary = {
        'makespan': float(np.mean(makespans)),
        'makespan_std': float(np.std(makespans)),
        'utilization': float(np.mean(utilizations)),
        'load_balance': float(np.mean(balances)),
        'episodes': int(num_episodes),
        'truncated_episodes': int(truncated_episodes),
    }
    if return_details:
        summary['episode_details'] = episode_details
    return summary


def _process_metadata_for_id(env: LogReplaySimulator, process_id: Any) -> Dict[str, Any]:
    """读取单个workflow的split元数据，用于per-workflow评估明细。"""
    normalized_id = LogReplaySimulator._normalize_process_id(process_id)
    process_df = env.process_instances.copy()
    process_df['_normalized_process_id'] = process_df['id'].map(
        LogReplaySimulator._normalize_process_id
    )
    matched = process_df[process_df['_normalized_process_id'] == normalized_id]
    if matched.empty:
        return {'process_id': normalized_id}

    row = matched.iloc[0]
    metadata = {
        'process_id': normalized_id,
        'process_name': row.get('name', row.get('process_name', '')),
        'process_definition_code': row.get('process_definition_code', ''),
    }
    for column in SPLIT_METADATA_COLUMNS:
        if column in row.index:
            metadata[column] = row.get(column)
    if 'task_count' not in metadata or pd.isna(metadata.get('task_count')):
        metadata['task_count'] = int(
            env.task_instances[
                env.task_instances['process_instance_id'].map(
                    LogReplaySimulator._normalize_process_id
                ) == normalized_id
            ].shape[0]
        )
    return metadata


def _clone_replay_env_for_process_ids(env: LogReplaySimulator,
                                      process_ids: List[Any],
                                      episode_seed: int = 42) -> LogReplaySimulator:
    """基于一组workflow id创建轻量评估环境。"""
    normalized_ids = LogReplaySimulator._normalize_process_id_list(process_ids) or []
    id_set = set(normalized_ids)
    process_df = env.process_instances[
        env.process_instances['id'].map(
            LogReplaySimulator._normalize_process_id
        ).isin(id_set)
    ].copy()
    task_df = env.task_instances[
        env.task_instances['process_instance_id'].map(
            LogReplaySimulator._normalize_process_id
        ).isin(id_set)
    ].copy()
    return LogReplaySimulator(
        process_instances=process_df,
        task_instances=task_df,
        task_definitions=env.task_definitions.copy(deep=False),
        process_task_relations=env.process_task_relations.copy(deep=False),
        episode_process_ids=normalized_ids,
        episode_window_size=1,
        episode_window_stride=1,
        shuffle_episode_processes=False,
        episode_seed=episode_seed,
    )


def _build_stratified_eval_summary(records_df: pd.DataFrame) -> pd.DataFrame:
    """生成按任务数、时长、DAG复杂度等维度的论文分层表。"""
    rows: List[Dict[str, Any]] = []
    strata_columns = [
        'workflow_size',
        'duration_bin',
        'dag_complexity_bin',
        'workflow_stratum',
        'balanced_workflow_stratum',
    ]
    for column in strata_columns:
        if column not in records_df.columns:
            continue
        for value, group in records_df.groupby(column, dropna=True):
            if group.empty:
                continue
            rows.append({
                'stratum_type': column,
                'stratum_value': value,
                'workflows': int(len(group)),
                'makespan_mean': float(group['makespan'].mean()),
                'makespan_std': float(group['makespan'].std(ddof=1)) if len(group) > 1 else 0.0,
                'utilization_mean': float(group['utilization'].mean()),
                'utilization_std': float(group['utilization'].std(ddof=1)) if len(group) > 1 else 0.0,
                'load_balance_mean': float(group['load_balance'].mean()),
                'load_balance_std': float(group['load_balance'].std(ddof=1)) if len(group) > 1 else 0.0,
                'truncated_workflows': int(group.get('truncated', pd.Series(dtype=bool)).sum()),
            })
    return pd.DataFrame(rows)


def evaluate_replay_workflows(agent: EnhancedFE_IDDQN,
                              env: LogReplaySimulator,
                              output_dir: Path,
                              logger: logging.Logger,
                              max_steps_per_workflow: Optional[int] = None,
                              ) -> Dict[str, Any]:
    """逐workflow覆盖完整评估split，并写出明细和分层汇总。"""
    process_ids = list(getattr(env, 'episode_process_ids', []) or [])
    if not process_ids:
        raise ValueError('Full replay evaluation requires explicit episode_process_ids.')

    records: List[Dict[str, Any]] = []
    for idx, process_id in enumerate(process_ids):
        single_env = _clone_replay_env_for_process_ids(
            env,
            [process_id],
            episode_seed=idx,
        )
        result = evaluate_dqn_agent(
            agent,
            single_env,
            num_episodes=1,
            max_steps_per_episode=max_steps_per_workflow,
            return_details=True,
        )
        detail = result.get('episode_details', [{}])[0]
        metadata = _process_metadata_for_id(env, process_id)
        records.append({
            **metadata,
            'makespan': float(result['makespan']),
            'utilization': float(result['utilization']),
            'load_balance': float(result['load_balance']),
            'steps': int(detail.get('steps', 0)),
            'truncated': bool(detail.get('truncated', False)),
        })

        if (idx + 1) % 50 == 0 or idx + 1 == len(process_ids):
            logger.info(
                "Full replay evaluation progress: %d/%d workflows",
                idx + 1,
                len(process_ids)
            )

    records_df = pd.DataFrame(records)
    workflow_csv = output_dir / 'final_eval_workflows.csv'
    records_df.to_csv(workflow_csv, index=False)

    strata_df = _build_stratified_eval_summary(records_df)
    strata_csv = output_dir / 'final_eval_strata.csv'
    strata_df.to_csv(strata_csv, index=False)

    makespans = records_df['makespan'].astype(float)
    utilizations = records_df['utilization'].astype(float)
    balances = records_df['load_balance'].astype(float)
    return {
        'makespan': float(makespans.mean()),
        'makespan_std': float(makespans.std(ddof=0)),
        'utilization': float(utilizations.mean()),
        'load_balance': float(balances.mean()),
        'episodes': int(len(records_df)),
        'workflows': int(len(records_df)),
        'eval_unit': 'workflow',
        'truncated_episodes': int(records_df['truncated'].sum()),
        'workflow_detail_csv': str(workflow_csv),
        'strata_summary_csv': str(strata_csv),
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
                    resource_input_dim=agent.resource_input_dim,
                    disable_fe=not agent.config.use_feature_engineering
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
                resource_input_dim=agent.resource_input_dim,
                disable_fe=not agent.config.use_feature_engineering
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
        val_eval_interval = int(getattr(config, 'val_eval_interval', 10) or 0)
        val_eval_episodes = int(getattr(config, 'val_eval_episodes', 5) or 5)
        if (
            val_env is not None
            and val_eval_interval > 0
            and (episode + 1) % val_eval_interval == 0
        ):
            val_result = evaluate_dqn_agent(agent, val_env, num_episodes=val_eval_episodes)
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
                  ga_config: Optional[GAConfig] = None,
                  train_args: Optional[Any] = None
                  ) -> Dict[str, Any]:
    """执行GA网络架构搜索 (复用 ga_optimizer.py)"""
    logger.info("=" * 60)
    logger.info("Phase 1: GA Architecture Search (for FE-IDDQN)")
    logger.info("=" * 60)

    ga_cfg = ga_config or GAConfig()
    optimizer = GAArchitectureOptimizer(ga_cfg)
    env_factory = _make_search_env_factory(env, train_args, logger)
    max_eval_steps = int(getattr(train_args, 'search_max_steps', 200) or 200)
    search_eval_episodes = int(getattr(train_args, 'search_eval_episodes', 3) or 3)

    def fitness_fn(network_structure: Dict) -> Dict[str, float]:
        """适应度函数: 构建 FE-IDDQN → 短期训练 → 评估"""
        local_env = env_factory()
        config = EnhancedFE_IDDQN_Config(
            hidden_dim=network_structure.get('hidden_dim', 256),
            fusion_dim=network_structure.get('fusion_dim', 256),
            num_transformer_layers=network_structure.get(
                'num_transformer_layers', 2),
            num_heads=network_structure.get('num_heads', 4),
            dropout=network_structure.get('dropout', 0.1),
            use_gnn=network_structure.get('use_gnn', True),
            max_episodes=ga_cfg.eval_episodes,
            batch_size=min(64, max(16, getattr(train_args, 'batch_size', 64) if train_args else 64)),
            warmup_steps=50,
            device='cpu',
        )
        if train_args:
            config.use_feature_engineering = not train_args.disable_fe
            config.use_per = not train_args.disable_per
            config.use_n_step = (config.n_step > 1) and not train_args.disable_nstep

        try:
            agent = EnhancedFE_IDDQN(
                task_input_dim, resource_input_dim,
                action_dim, config)

            # 短期训练
            for ep in range(ga_cfg.eval_episodes):
                state = _reset_env_and_get_state(local_env)
                done = False
                steps = 0
                while not done and steps < max_eval_steps:
                    state_for_store, tf, rf, adj, nd, cm = \
                        _normalize_state_for_agent(
                            state,
                            task_input_dim=task_input_dim,
                            resource_input_dim=resource_input_dim,
                            disable_fe=not config.use_feature_engineering
                        )

                    action = agent.select_action(
                        tf, rf, adj_matrix=adj,
                        node_depths=nd, critical_path_mask=cm,
                        valid_action_count=_get_valid_action_count(local_env, rf, action_dim),
                        training=True)
                    next_state, env_reward, done, info = local_env.step(action)
                    reward = _compute_training_reward(agent, local_env, env_reward, info)

                    next_state_for_store, _, _, _, _, _ = _normalize_state_for_agent(
                        next_state,
                        task_input_dim=task_input_dim,
                        resource_input_dim=resource_input_dim,
                        disable_fe=not config.use_feature_engineering
                    )
                    agent.store_experience(
                        state_for_store, action, reward,
                        next_state_for_store, done, info)

                    if steps % 4 == 0:
                        agent.train_step()

                    state = next_state if not done else _reset_env_and_get_state(local_env)
                    steps += 1

            # 评估
            result = evaluate_dqn_agent(agent, local_env, num_episodes=search_eval_episodes)
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
            hpo_config: Optional[DQNHPOConfig] = None,
            train_args: Optional[Any] = None
            ) -> Dict[str, Any]:
    """执行Optuna超参数搜索 (DQN版本)"""
    logger.info("=" * 60)
    logger.info("Phase 2: Optuna HPO (for FE-IDDQN)")
    logger.info("=" * 60)

    hpo_cfg = hpo_config or DQNHPOConfig(n_trials=30, timeout=1800)
    optimizer = DQNHPOptimizer(hpo_cfg)
    env_factory = _make_search_env_factory(env, train_args, logger)
    max_eval_steps = int(getattr(train_args, 'search_max_steps', 200) or 200)
    search_eval_episodes = int(getattr(train_args, 'search_eval_episodes', 3) or 3)

    def objective_fn(params: Dict[str, Any]) -> Dict[str, float]:
        """目标函数: 构建 FE-IDDQN → 短期训练 → 评估"""
        ns = params.get('network_structure', network_structure) or {}
        local_env = env_factory()

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
        if train_args:
            config.use_feature_engineering = not train_args.disable_fe
            config.use_per = not train_args.disable_per
            config.use_n_step = (config.n_step > 1) and not train_args.disable_nstep

        agent = EnhancedFE_IDDQN(
            task_input_dim, resource_input_dim,
            action_dim, config, )

        # 短期训练
        for ep in range(hpo_cfg.eval_episodes):
            state = _reset_env_and_get_state(local_env)
            done = False
            steps = 0
            while not done and steps < max_eval_steps:
                state_for_store, tf, rf, adj, nd, cm = _normalize_state_for_agent(
                    state,
                    task_input_dim=task_input_dim,
                    resource_input_dim=resource_input_dim,
                    disable_fe=not config.use_feature_engineering
                )

                action = agent.select_action(
                    tf, rf, adj_matrix=adj,
                    node_depths=nd, critical_path_mask=cm,
                    valid_action_count=_get_valid_action_count(local_env, rf, action_dim),
                    training=True)
                next_state, env_reward, done, info = local_env.step(action)
                reward = _compute_training_reward(agent, local_env, env_reward, info)

                next_state_for_store, _, _, _, _, _ = _normalize_state_for_agent(
                    next_state,
                    task_input_dim=task_input_dim,
                    resource_input_dim=resource_input_dim,
                    disable_fe=not config.use_feature_engineering
                )
                agent.store_experience(
                    state_for_store, action, reward,
                    next_state_for_store, done, info)

                if steps % 4 == 0:
                    agent.train_step()

                state = next_state if not done else _reset_env_and_get_state(local_env)
                steps += 1

        return evaluate_dqn_agent(agent, local_env, num_episodes=search_eval_episodes)

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
    parser.add_argument('--workflow_split_dir', type=str, default=None,
                        help='固定workflow split目录；默认自动使用 replay_data_dir/splits')
    parser.add_argument('--train_process_ids', type=str, default=None,
                        help='训练workflow/process id CSV，优先级高于workflow_split_dir')
    parser.add_argument('--val_process_ids', type=str, default=None,
                        help='验证workflow/process id CSV，优先级高于workflow_split_dir')
    parser.add_argument('--test_process_ids', type=str, default=None,
                        help='测试workflow/process id CSV，优先级高于workflow_split_dir')
    parser.add_argument('--eval_split', type=str, default='val',
                        choices=['val', 'test'],
                        help='最终评估使用val或test workflow ids')
    parser.add_argument('--full_test_eval', action='store_true',
                        help='replay最终评估逐个workflow覆盖完整eval split，并输出明细CSV')
    parser.add_argument('--full_eval_max_steps', type=int,
                        default=int(os.getenv('FULL_EVAL_MAX_STEPS', '0')),
                        help='完整逐workflow评估的单workflow最大step数，0表示不限制')
    parser.add_argument('--train_workflows_per_episode', type=int,
                        default=int(os.getenv('TRAIN_WORKFLOWS_PER_EPISODE', '1')),
                        help='replay训练每个episode固定窗口中的workflow数量，0表示使用全部训练ids')
    parser.add_argument('--eval_workflows_per_episode', type=int,
                        default=int(os.getenv('EVAL_WORKFLOWS_PER_EPISODE', '5')),
                        help='replay评估每个episode固定窗口中的workflow数量，0表示使用全部评估ids')
    parser.add_argument('--paper_eval_episodes', type=int, default=10,
                        help='论文表格基线评估回合数（仅replay模式）')
    parser.add_argument('--final_eval_episodes', type=int, default=20,
                        help='最终评估回合数')
    parser.add_argument('--num_tasks', type=int, default=20)
    parser.add_argument('--num_resources', type=int, default=5)
    parser.add_argument('--max_episodes', type=int, default=500)
    parser.add_argument('--max_steps_per_episode', type=int,
                        default=int(os.getenv('MAX_STEPS_PER_EPISODE', '500')),
                        help='训练每个episode最大调度步数')
    parser.add_argument('--val_eval_interval', type=int,
                        default=int(os.getenv('VAL_EVAL_INTERVAL', '10')),
                        help='训练中每隔多少episode做一次验证，0表示关闭中间验证')
    parser.add_argument('--val_eval_episodes', type=int,
                        default=int(os.getenv('VAL_EVAL_EPISODES', '5')),
                        help='训练中每次验证的episode数')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--skip_baselines', action='store_true',
                        help='replay模式下跳过传统基线表生成，加快消融实验')

    # GA/HPO搜索预算。默认保持接近原实验设置；消融批跑可通过命令行或环境变量调小。
    parser.add_argument('--ga_population_size', type=int,
                        default=int(os.getenv('GA_POPULATION_SIZE', '12')))
    parser.add_argument('--ga_generations', type=int,
                        default=int(os.getenv('GA_GENERATIONS', '10')))
    parser.add_argument('--ga_eval_episodes', type=int,
                        default=int(os.getenv('GA_EVAL_EPISODES', '20')))
    parser.add_argument('--ga_max_workers', type=int,
                        default=int(os.getenv('GA_MAX_WORKERS', '1')),
                        help='GA个体并发评估数；replay大数据建议保持1')
    parser.add_argument('--hpo_trials', type=int,
                        default=int(os.getenv('HPO_TRIALS', '30')))
    parser.add_argument('--hpo_timeout', type=int,
                        default=int(os.getenv('HPO_TIMEOUT', '1800')))
    parser.add_argument('--hpo_eval_episodes', type=int,
                        default=int(os.getenv('HPO_EVAL_EPISODES', '30')))
    parser.add_argument('--search_max_processes', type=int,
                        default=int(os.getenv('SEARCH_MAX_PROCESSES', '8')),
                        help='GA/HPO短评估使用的最大replay流程数')
    parser.add_argument('--search_max_tasks', type=int,
                        default=int(os.getenv('SEARCH_MAX_TASKS', '120')),
                        help='GA/HPO短评估使用的最大replay任务数')
    parser.add_argument('--search_max_steps', type=int,
                        default=int(os.getenv('SEARCH_MAX_STEPS', '200')),
                        help='GA/HPO每个短评估episode最大step数')
    parser.add_argument('--search_eval_episodes', type=int,
                        default=int(os.getenv('SEARCH_EVAL_EPISODES', '3')),
                        help='GA/HPO候选配置评估episode数')
    parser.add_argument('--search_split', type=str,
                        default=os.getenv('SEARCH_SPLIT', 'val'),
                        choices=['train', 'val'],
                        help='GA/HPO搜索使用train或val split；论文实验建议val')
    parser.add_argument('--search_strata_column', type=str,
                        default=os.getenv('SEARCH_STRATA_COLUMN', 'balanced_workflow_stratum'),
                        help='GA/HPO短评估分层抽样字段')
    parser.add_argument('--torch_num_threads', type=int,
                        default=int(os.getenv('TORCH_NUM_THREADS', '0')),
                        help='限制每个训练进程的PyTorch CPU线程数，0表示不改')

    # Ablation Flags
    parser.add_argument('--disable_fe', action='store_true',
                        help='禁用特征工程(降级维度)')
    parser.add_argument('--disable_per', action='store_true',
                        help='禁用优先经验回放')
    parser.add_argument('--disable_nstep', action='store_true',
                        help='禁用N-step回报')

    parser.add_argument('--no_gnn', action='store_true',
                        help='禁用GNN')
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
        try:
            torch.set_num_interop_threads(max(1, min(args.torch_num_threads, 4)))
        except RuntimeError:
            pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    logger = setup_logging(output_dir)
    current_stage = 'startup'
    update_run_status(
        output_dir,
        stage=current_stage,
        status='running',
        seed=args.seed,
        mode=args.mode,
        env_type=args.env_type,
        pid=os.getpid(),
    )

    try:
        logger.info(f"GA-HPO FE-IDDQN Training System | Mode: {args.mode}")
        logger.info(f"Output: {output_dir}")

        # ── 创建环境 ──
        replay_meta = None
        current_stage = 'create_environment'
        update_run_status(output_dir, stage=current_stage, status='running', seed=args.seed)
        if args.env_type == 'replay':
            replay_data_dir = Path(args.replay_data_dir)
            train_env, val_env, final_env, replay_meta = make_replay_envs(
                replay_data_dir,
                train_ratio=args.replay_train_ratio,
                logger=logger,
                split_dir=Path(args.workflow_split_dir) if args.workflow_split_dir else None,
                train_process_ids_path=Path(args.train_process_ids) if args.train_process_ids else None,
                val_process_ids_path=Path(args.val_process_ids) if args.val_process_ids else None,
                test_process_ids_path=Path(args.test_process_ids) if args.test_process_ids else None,
                eval_split=args.eval_split,
                train_workflows_per_episode=args.train_workflows_per_episode,
                eval_workflows_per_episode=args.eval_workflows_per_episode,
            )
        else:
            train_env = make_env(args.num_tasks, args.num_resources, seed=args.seed)
            val_env = make_env(args.num_tasks, args.num_resources, seed=args.seed + 1)
            final_env = val_env

        # 获取维度
        current_stage = 'infer_environment_dimensions'
        update_run_status(output_dir, stage=current_stage, status='running', seed=args.seed)
        state = _reset_env_and_get_state(train_env)
        _, task_feats, res_feats, _, _, _ = _normalize_state_for_agent(
            state,
            task_input_dim=16,
            resource_input_dim=7,
            disable_fe=args.disable_fe
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
        _write_json(output_dir / 'config.json', config_info)

        network_structure = None
        best_hpo_config = None
        search_env = val_env if args.search_split == 'val' else train_env

        # ── Phase 1: GA架构搜索 ──
        if args.mode in ('full', 'ga_search'):
            current_stage = 'ga_search'
            update_run_status(output_dir, stage=current_stage, status='running', seed=args.seed)
            ga_cache = _load_json_if_exists(output_dir / 'ga_search_result.json')
            if args.mode == 'full' and ga_cache and isinstance(ga_cache.get('best_genome'), dict):
                network_structure = ga_cache['best_genome']
                logger.info(f"Reusing existing GA result from {output_dir / 'ga_search_result.json'}")
            else:
                ga_config = GAConfig(
                    population_size=args.ga_population_size,
                    num_generations=args.ga_generations,
                    eval_episodes=args.ga_eval_episodes,
                    max_workers=args.ga_max_workers,
                    seed=args.seed,
                )
                network_structure = run_ga_search(
                    task_input_dim, resource_input_dim, action_dim,
                    search_env, output_dir, logger,
                    ga_config=ga_config, train_args=args)

        # ── Phase 2: Optuna HPO ──
        if args.mode in ('full', 'hpo'):
            current_stage = 'hpo'
            update_run_status(output_dir, stage=current_stage, status='running', seed=args.seed)
            hpo_cache = _load_json_if_exists(output_dir / 'hpo_result.json')
            if args.mode == 'full' and hpo_cache and isinstance(hpo_cache.get('best_params'), dict):
                best_hpo_config = dict(hpo_cache['best_params'])
                if network_structure:
                    best_hpo_config['network_structure'] = network_structure
                logger.info(f"Reusing existing HPO result from {output_dir / 'hpo_result.json'}")
            else:
                hpo_config = DQNHPOConfig(
                    n_trials=args.hpo_trials,
                    timeout=args.hpo_timeout,
                    eval_episodes=args.hpo_eval_episodes,
                    seed=args.seed,
                    study_name=f"fe_iddqn_hpo_seed_{args.seed}",
                )
                best_hpo_config = run_hpo(
                    task_input_dim, resource_input_dim, action_dim,
                    search_env, output_dir, logger,
                    network_structure=network_structure,
                    hpo_config=hpo_config, train_args=args)

        # ── Phase 3: 完整训练 ──
        if args.mode in ('full', 'train_only'):
            current_stage = 'prepare_training'
            update_run_status(output_dir, stage=current_stage, status='running', seed=args.seed)
            dqn_config = EnhancedFE_IDDQN_Config(
                learning_rate=args.lr,
                batch_size=args.batch_size,
                n_step=args.n_step,
                use_n_step=(args.n_step > 1) and not args.disable_nstep,
                use_per=not args.disable_per,
                use_feature_engineering=not args.disable_fe,
                max_episodes=args.max_episodes,
                max_steps_per_episode=args.max_steps_per_episode,
                use_gnn=not args.no_gnn,
                device=args.device,
            )
            dqn_config.val_eval_interval = args.val_eval_interval
            dqn_config.val_eval_episodes = args.val_eval_episodes

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
                dqn_config.use_n_step = (dqn_config.n_step > 1) and not args.disable_nstep
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
                dqn_config.dropout = network_structure.get(
                    'dropout', dqn_config.dropout)
                dqn_config.use_gnn = (
                    network_structure.get('use_gnn', dqn_config.use_gnn)
                    and not args.no_gnn
                )

            # 创建Agent
            current_stage = 'create_agent'
            update_run_status(output_dir, stage=current_stage, status='running', seed=args.seed)
            agent = EnhancedFE_IDDQN(
                task_input_dim, resource_input_dim, action_dim,
                dqn_config)

            logger.info(f"Network params: "
                        f"{sum(p.numel() for p in agent.q_network.parameters()):,}")

            # 训练
            current_stage = 'training'
            update_run_status(output_dir, stage=current_stage, status='running', seed=args.seed)
            train_dqn(
                agent, train_env, dqn_config, output_dir, logger,
                val_env=val_env)

            # 最终评估
            current_stage = 'final_evaluation'
            update_run_status(output_dir, stage=current_stage, status='running', seed=args.seed)
            logger.info("=" * 60)
            logger.info("Final Evaluation")
            logger.info("=" * 60)

            if args.env_type == 'replay' and args.full_test_eval:
                final_eval = evaluate_replay_workflows(
                    agent,
                    final_env,
                    output_dir,
                    logger,
                    max_steps_per_workflow=(
                        args.full_eval_max_steps
                        if args.full_eval_max_steps > 0 else None
                    ),
                )
            else:
                final_eval = evaluate_dqn_agent(
                    agent,
                    final_env,
                    num_episodes=args.final_eval_episodes,
                    max_steps_per_episode=(
                        args.full_eval_max_steps
                        if args.full_eval_max_steps > 0 else None
                    ),
                )
            logger.info(f"  Makespan:     {final_eval['makespan']:.2f} "
                        f"± {final_eval['makespan_std']:.2f}")
            logger.info(f"  Utilization:  {final_eval['utilization']:.4f}")
            logger.info(f"  Load Balance: {final_eval['load_balance']:.4f}")

            _write_json(output_dir / 'final_eval.json', final_eval)

            # replay模式自动生成论文表格（含传统基线）
            if args.env_type == 'replay' and not args.skip_baselines and args.paper_eval_episodes > 0:
                current_stage = 'baseline_comparison'
                update_run_status(output_dir, stage=current_stage, status='running', seed=args.seed)
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

        current_stage = 'completed'
        update_run_status(output_dir, stage=current_stage, status='completed', seed=args.seed)
        logger.info("Done!")
    except KeyboardInterrupt as exc:
        failure_info = record_failure(output_dir, current_stage, exc, seed=args.seed)
        logger.error(f"Run interrupted at stage '{current_stage}': {failure_info['error_message']}")
        raise
    except Exception as exc:
        failure_info = record_failure(output_dir, current_stage, exc, seed=args.seed)
        logger.exception(f"Run failed at stage '{current_stage}'")
        sys.exit(1)


if __name__ == '__main__':
    main()
