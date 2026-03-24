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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch

# ─── 项目内部导入 ─── #
from models.enhanced_fe_iddqn import EnhancedFE_IDDQN, EnhancedFE_IDDQN_Config
from models.ga_optimizer import GAArchitectureOptimizer, GAConfig
from models.dqn_hpo_optimizer import DQNHPOptimizer, DQNHPOConfig
from environment.enhanced_workflow_simulator import EnhancedWorkflowSimulator


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
        state = env.reset()
        done = False

        while not done:
            task_feats = state.get('task_features',
                                   np.zeros((1, agent.task_input_dim)))
            res_feats = state.get('resource_features',
                                  np.zeros((1, agent.resource_input_dim)))
            adj = state.get('adj_matrix', None)
            node_depths = state.get('node_depths', None)
            critical_mask = state.get('critical_path_mask', None)

            action = agent.select_action(
                task_feats, res_feats,
                adj_matrix=adj,
                node_depths=node_depths,
                critical_path_mask=critical_mask,
                training=False)

            state, reward, done, info = env.step(action)

        result = env.get_scheduling_result()
        makespans.append(result['makespan'])
        utilizations.append(result['resource_utilization'])
        balances.append(result['load_balance'])

    return {
        'makespan': np.mean(makespans),
        'makespan_std': np.std(makespans),
        'utilization': np.mean(utilizations),
        'load_balance': np.mean(balances),
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
        state = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        done = False
        while not done and ep_steps < config.max_steps_per_episode:
            task_feats = state.get('task_features',
                                   np.zeros((1, agent.task_input_dim)))
            res_feats = state.get('resource_features',
                                  np.zeros((1, agent.resource_input_dim)))
            adj = state.get('adj_matrix', None)
            node_depths = state.get('node_depths', None)
            critical_mask = state.get('critical_path_mask', None)

            # 选择动作
            action = agent.select_action(
                task_feats, res_feats,
                adj_matrix=adj,
                node_depths=node_depths,
                critical_path_mask=critical_mask,
                training=True)

            # 环境交互
            next_state, reward, done, info = env.step(action)

            # 存储经验
            agent.store_experience(state, action, reward, next_state, done, info)

            # DQN训练步骤 (off-policy: 每 train_freq 步训练一次)
            if total_steps % config.train_freq == 0:
                train_result = agent.train_step()

            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            state = next_state

        # Episode 结束回调
        agent.on_episode_end(ep_reward, {
            'makespan': info.get('makespan', 0),
            'utilization': info.get('utilization', 0),
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
                state = env.reset()
                done = False
                steps = 0
                while not done and steps < 200:
                    tf = state.get('task_features',
                                   np.zeros((1, task_input_dim)))
                    rf = state.get('resource_features',
                                   np.zeros((1, resource_input_dim)))
                    adj = state.get('adj_matrix', None)
                    nd = state.get('node_depths', None)
                    cm = state.get('critical_path_mask', None)

                    action = agent.select_action(
                        tf, rf, adj_matrix=adj,
                        node_depths=nd, critical_path_mask=cm,
                        training=True)
                    next_state, reward, done, info = env.step(action)
                    agent.store_experience(
                        state, action, reward, next_state, done, info)

                    if steps % 4 == 0:
                        agent.train_step()

                    state = next_state if not done else env.reset()
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
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < 200:
                tf = state.get('task_features',
                               np.zeros((1, task_input_dim)))
                rf = state.get('resource_features',
                               np.zeros((1, resource_input_dim)))
                adj = state.get('adj_matrix', None)
                nd = state.get('node_depths', None)
                cm = state.get('critical_path_mask', None)

                action = agent.select_action(
                    tf, rf, adj_matrix=adj,
                    node_depths=nd, critical_path_mask=cm,
                    training=True)
                next_state, reward, done, info = env.step(action)
                agent.store_experience(
                    state, action, reward, next_state, done, info)

                if steps % 4 == 0:
                    agent.train_step()

                state = next_state if not done else env.reset()
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
    train_env = make_env(args.num_tasks, args.num_resources, seed=args.seed)
    val_env = make_env(args.num_tasks, args.num_resources, seed=args.seed + 1)

    # 获取维度
    state = train_env.reset()
    task_feats = state.get('task_features', np.zeros((1, 16)))
    res_feats = state.get('resource_features', np.zeros((1, 7)))
    task_input_dim = task_feats.shape[-1] if len(task_feats.shape) >= 2 else 16
    resource_input_dim = res_feats.shape[-1] if len(res_feats.shape) >= 2 else 7
    action_dim = args.num_resources

    logger.info(f"Env dims: task={task_input_dim}, resource={resource_input_dim}, "
                f"action={action_dim}")

    # ── 保存配置 ──
    config_info = vars(args)
    config_info['task_input_dim'] = task_input_dim
    config_info['resource_input_dim'] = resource_input_dim
    config_info['action_dim'] = action_dim
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

        final_eval = evaluate_dqn_agent(agent, val_env, num_episodes=20)
        logger.info(f"  Makespan:     {final_eval['makespan']:.2f} "
                     f"± {final_eval['makespan_std']:.2f}")
        logger.info(f"  Utilization:  {final_eval['utilization']:.4f}")
        logger.info(f"  Load Balance: {final_eval['load_balance']:.4f}")

        with open(output_dir / 'final_eval.json', 'w') as f:
            json.dump(final_eval, f, indent=2, default=str)

    logger.info("Done!")


if __name__ == '__main__':
    main()
