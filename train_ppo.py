#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDS-PPO 训练系统 — 完整pipeline

流程:
  Phase 1 (可选): GA搜索最优网络架构
  Phase 2 (可选): Optuna搜索最优PPO超参数
  Phase 3: 使用最优配置进行完整PPO训练 + 评估

用法:
  python train_ppo.py                       # 直接训练 (默认参数)
  python train_ppo.py --mode full           # 完整pipeline (GA + HPO + 训练)
  python train_ppo.py --mode train_only     # 仅训练 (跳过GA/HPO)
  python train_ppo.py --mode ga_search      # 仅GA架构搜索
  python train_ppo.py --mode hpo            # 仅HPO
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
from models.gds_ppo import GDS_PPO, GDS_PPO_Config, DualStreamActorCritic
from models.rollout_buffer import RolloutBuffer
from models.ga_optimizer import GAArchitectureOptimizer, GAConfig
from models.hpo_optimizer import OptunaHPO, HPOConfig
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
    log_file = log_dir / f"gds_ppo_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger('GDS_PPO')


# ─────────── 合成工作流生成 (用于测试/demo) ─────────── #

def generate_synthetic_workflow(
        num_tasks: int = 20, num_resources: int = 5,
        dag_density: float = 0.3, seed: int = None
) -> Tuple[List[Dict], List[Dict], List[Tuple[int, int]]]:
    """
    生成合成DAG工作流, 可用于训练和测试
    
    Args:
        num_tasks: 任务数量
        num_resources: 资源数量
        dag_density: DAG边密度
        seed: 随机种子
    
    Returns:
        tasks, resources, dependencies
    """
    rng = np.random.RandomState(seed)
    
    # 任务
    tasks = []
    for i in range(num_tasks):
        tasks.append({
            'id': i,
            'duration': rng.uniform(1.0, 20.0),
            'cpu_req': rng.choice([1, 2, 4]),
            'memory_req': rng.choice([1, 2, 4, 8]),
            'priority': rng.randint(0, 3),
        })
    
    # 资源
    resources = []
    for j in range(num_resources):
        resources.append({
            'id': j,
            'cpu_capacity': rng.choice([4, 8, 16]),
            'memory_capacity': rng.choice([8, 16, 32]),
            'speed_factor': rng.uniform(0.8, 1.5),
        })
    
    # DAG依赖 (只允许 i → j where i < j, 保证无环)
    dependencies = []
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if rng.random() < dag_density:
                dependencies.append((i, j))
    
    return tasks, resources, dependencies


# ─────────── 评估函数 ─────────── #

def evaluate_agent(agent: GDS_PPO,
                   env: EnhancedWorkflowSimulator,
                   num_episodes: int = 10,
                   deterministic: bool = True
                   ) -> Dict[str, float]:
    """
    评估PPO agent性能
    
    Returns:
        {'makespan': ..., 'utilization': ..., 'load_balance': ...}
    """
    makespans = []
    utilizations = []
    balances = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # 提取特征
            task_feats = state.get('task_features',
                                   np.zeros((1, agent.task_input_dim)))
            res_feats = state.get('resource_features',
                                  np.zeros((1, agent.resource_input_dim)))
            adj = state.get('adj_matrix', None)
            mask = state.get('action_mask', None)
            
            action, _, _ = agent.select_action(
                task_feats, res_feats, adj, mask,
                deterministic=deterministic)
            
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


# ─────────── PPO 训练循环 ─────────── #

def train_ppo(agent: GDS_PPO,
              env: EnhancedWorkflowSimulator,
              config: GDS_PPO_Config,
              output_dir: Path,
              logger: logging.Logger,
              val_env: Optional[EnhancedWorkflowSimulator] = None,
              ) -> Dict[str, Any]:
    """
    PPO训练主循环
    
    Returns:
        训练结果字典
    """
    logger.info("="*60)
    logger.info("GDS-PPO Training Start")
    logger.info(f"  Episodes:     {config.max_episodes}")
    logger.info(f"  Rollout steps:{config.rollout_steps}")
    logger.info(f"  Batch size:   {config.batch_size}")
    logger.info(f"  K epochs:     {config.k_epochs}")
    logger.info(f"  LR:           {config.learning_rate}")
    logger.info(f"  Device:       {agent.device}")
    logger.info("="*60)
    
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    best_makespan = float('inf')
    training_log: List[Dict] = []
    
    total_steps = 0
    
    for episode in range(config.max_episodes):
        ep_start = time.time()
        
        # ── 收集轨迹 (Rollout Phase) ──
        state = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        
        for step in range(config.rollout_steps):
            task_feats = state.get('task_features',
                                   np.zeros((1, agent.task_input_dim)))
            res_feats = state.get('resource_features',
                                  np.zeros((1, agent.resource_input_dim)))
            adj = state.get('adj_matrix', None)
            mask = state.get('action_mask', None)
            
            # 选择动作
            action, log_prob, value = agent.select_action(
                task_feats, res_feats, adj, mask,
                deterministic=False)
            
            # 环境交互
            next_state, reward, done, info = env.step(action)
            
            # 构造flat state供缓冲区存储
            flat_state = np.concatenate([
                task_feats.flatten(), res_feats.flatten()])
            
            # 存储转移
            agent.store_transition(
                state=flat_state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                task_features=task_feats,
                resource_features=res_feats,
                adj_matrix=adj,
                action_mask=mask,
            )
            
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            
            if done:
                # Episode结束, 重置环境继续收集
                state = env.reset()
            else:
                state = next_state
        
        # ── GAE计算 ──
        # 获取最后一个状态的value估计
        last_task_feats = state.get('task_features',
                                    np.zeros((1, agent.task_input_dim)))
        last_res_feats = state.get('resource_features',
                                   np.zeros((1, agent.resource_input_dim)))
        with torch.no_grad():
            _, last_val = agent.policy(
                torch.FloatTensor(last_task_feats).unsqueeze(0).to(agent.device),
                torch.FloatTensor(last_res_feats).unsqueeze(0).to(agent.device))
        last_value = last_val.squeeze().item()
        
        agent.compute_gae(last_value, done)
        
        # ── PPO更新 ──
        update_stats = agent.update()
        agent.episode_count += 1
        
        ep_time = time.time() - ep_start
        
        # ── 验证评估 ──
        val_result = None
        if val_env is not None and (episode + 1) % 10 == 0:
            val_result = evaluate_agent(agent, val_env, num_episodes=5)
            
            if val_result['makespan'] < best_makespan:
                best_makespan = val_result['makespan']
                agent.save(str(models_dir / 'best_model.pt'))
                logger.info(
                    f"  ** New best makespan: {best_makespan:.2f}")
        
        # 记录
        log_entry = {
            'episode': episode,
            'ep_reward': ep_reward,
            'ep_steps': ep_steps,
            'total_steps': total_steps,
            'time': ep_time,
            **update_stats,
        }
        if val_result:
            log_entry['val_makespan'] = val_result['makespan']
            log_entry['val_utilization'] = val_result['utilization']
        training_log.append(log_entry)
        
        # 日志输出
        if (episode + 1) % 5 == 0:
            logger.info(
                f"Ep {episode+1}/{config.max_episodes} | "
                f"reward={ep_reward:.2f} | "
                f"p_loss={update_stats['policy_loss']:.4f} | "
                f"v_loss={update_stats['value_loss']:.4f} | "
                f"ent={update_stats['entropy']:.4f} | "
                f"kl={update_stats['approx_kl']:.4f} | "
                f"clip={update_stats['clip_fraction']:.3f} | "
                f"t={ep_time:.1f}s")
        
        # 定期保存
        if (episode + 1) % 50 == 0:
            agent.save(str(models_dir / f'checkpoint_ep{episode+1}.pt'))
    
    # 保存最终模型 + 训练日志
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
    """执行GA网络架构搜索"""
    logger.info("="*60)
    logger.info("Phase 1: GA Architecture Search")
    logger.info("="*60)
    
    ga_cfg = ga_config or GAConfig()
    optimizer = GAArchitectureOptimizer(ga_cfg)
    
    def fitness_fn(network_structure: Dict) -> Dict[str, float]:
        """适应度函数: 构建PPO → 短期训练 → 评估"""
        config = GDS_PPO_Config(
            hidden_dim=network_structure.get('hidden_dim', 256),
            fusion_dim=network_structure.get('fusion_dim', 256),
            num_transformer_layers=network_structure.get(
                'num_transformer_layers', 2),
            num_heads=network_structure.get('num_heads', 4),
            dropout=network_structure.get('dropout', 0.1),
            use_gnn=network_structure.get('use_gnn', True),
            max_episodes=ga_cfg.eval_episodes,
            rollout_steps=512,
            batch_size=64,
            k_epochs=3,
        )
        
        try:
            agent = GDS_PPO(
                task_input_dim, resource_input_dim,
                action_dim, config, network_structure)
            
            # 短期训练
            for _ in range(ga_cfg.eval_episodes):
                state = env.reset()
                done = False
                steps = 0
                while not done and steps < 200:
                    task_feats = state.get('task_features',
                                           np.zeros((1, task_input_dim)))
                    res_feats = state.get('resource_features',
                                          np.zeros((1, resource_input_dim)))
                    adj = state.get('adj_matrix', None)
                    mask = state.get('action_mask', None)
                    
                    action, log_prob, value = agent.select_action(
                        task_feats, res_feats, adj, mask)
                    
                    next_state, reward, done, info = env.step(action)
                    
                    flat_state = np.concatenate([
                        task_feats.flatten(), res_feats.flatten()])
                    agent.store_transition(
                        flat_state, action, reward, value, log_prob, done,
                        task_features=task_feats,
                        resource_features=res_feats)
                    
                    state = next_state if not done else env.reset()
                    steps += 1
                
                if agent.rollout_buffer.pos > 0:
                    agent.compute_gae(0.0, True)
                    agent.update()
            
            # 评估
            result = evaluate_agent(agent, env, num_episodes=5)
            
            # 参数量
            param_count = sum(
                p.numel() for p in agent.policy.parameters())
            result['params'] = param_count
            
            return result
            
        except Exception as e:
            logger.warning(f"GA fitness eval failed: {e}")
            return {'makespan': 1e6, 'utilization': 0, 'params': 1e8}
    
    best_structure = optimizer.search(fitness_fn)
    
    # 保存搜索结果
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
            hpo_config: Optional[HPOConfig] = None
            ) -> Dict[str, Any]:
    """执行Optuna超参数搜索"""
    logger.info("="*60)
    logger.info("Phase 2: Optuna Hyperparameter Optimization")
    logger.info("="*60)
    
    hpo_cfg = hpo_config or HPOConfig(n_trials=30, timeout=1800)
    optimizer = OptunaHPO(hpo_cfg)
    
    def objective_fn(params: Dict[str, Any]) -> Dict[str, float]:
        """目标函数: 构建PPO → 短期训练 → 评估"""
        ns = params.get('network_structure', network_structure) or {}
        
        config = GDS_PPO_Config(
            hidden_dim=ns.get('hidden_dim', 256),
            fusion_dim=ns.get('fusion_dim', 256),
            num_transformer_layers=ns.get('num_transformer_layers', 2),
            num_heads=ns.get('num_heads', 4),
            use_gnn=ns.get('use_gnn', True),
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            eps_clip=params['eps_clip'],
            k_epochs=params['k_epochs'],
            gae_lambda=params['gae_lambda'],
            batch_size=params['batch_size'],
            entropy_coef=params['entropy_coef'],
            value_loss_coef=params['value_loss_coef'],
            max_episodes=hpo_cfg.eval_episodes,
            rollout_steps=512,
        )
        
        agent = GDS_PPO(
            task_input_dim, resource_input_dim,
            action_dim, config, ns if ns else None)
        
        # 短期训练
        for _ in range(hpo_cfg.eval_episodes):
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < 200:
                task_feats = state.get('task_features',
                                       np.zeros((1, task_input_dim)))
                res_feats = state.get('resource_features',
                                      np.zeros((1, resource_input_dim)))
                adj = state.get('adj_matrix', None)
                mask = state.get('action_mask', None)
                
                action, log_prob, value = agent.select_action(
                    task_feats, res_feats, adj, mask)
                
                next_state, reward, done, info = env.step(action)
                
                flat_state = np.concatenate([
                    task_feats.flatten(), res_feats.flatten()])
                agent.store_transition(
                    flat_state, action, reward, value, log_prob, done,
                    task_features=task_feats,
                    resource_features=res_feats)
                
                state = next_state if not done else env.reset()
                steps += 1
            
            if agent.rollout_buffer.pos > 0:
                agent.compute_gae(0.0, True)
                agent.update()
        
        return evaluate_agent(agent, env, num_episodes=5)
    
    best_config = optimizer.optimize(objective_fn, network_structure)
    
    # 保存HPO结果
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

def make_env(num_tasks=20, num_resources=5, seed=42):
    """创建合成环境 (用于测试)"""
    tasks, resources, deps = generate_synthetic_workflow(
        num_tasks=num_tasks, num_resources=num_resources, seed=seed)
    return EnhancedWorkflowSimulator(tasks, resources, deps)


def main():
    parser = argparse.ArgumentParser(
        description='GDS-PPO Training System')
    parser.add_argument('--mode', type=str, default='train_only',
                        choices=['full', 'train_only', 'ga_search', 'hpo'],
                        help='运行模式')
    parser.add_argument('--output_dir', type=str,
                        default='results/gds_ppo',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_tasks', type=int, default=20)
    parser.add_argument('--num_resources', type=int, default=5)
    parser.add_argument('--max_episodes', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--k_epochs', type=int, default=10)
    parser.add_argument('--rollout_steps', type=int, default=2048)
    parser.add_argument('--no_gnn', action='store_true',
                        help='禁用GNN')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # 基础设置
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(args.seed)
    logger = setup_logging(output_dir)
    
    logger.info(f"GDS-PPO Training System | Mode: {args.mode}")
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
        # 构建配置
        ppo_config = GDS_PPO_Config(
            learning_rate=args.lr,
            batch_size=args.batch_size,
            k_epochs=args.k_epochs,
            rollout_steps=args.rollout_steps,
            max_episodes=args.max_episodes,
            use_gnn=not args.no_gnn,
            device=args.device,
        )
        
        # 如果有HPO结果，覆盖默认值
        if best_hpo_config:
            ppo_config.learning_rate = best_hpo_config.get(
                'learning_rate', ppo_config.learning_rate)
            ppo_config.gamma = best_hpo_config.get(
                'gamma', ppo_config.gamma)
            ppo_config.eps_clip = best_hpo_config.get(
                'eps_clip', ppo_config.eps_clip)
            ppo_config.k_epochs = best_hpo_config.get(
                'k_epochs', ppo_config.k_epochs)
            ppo_config.gae_lambda = best_hpo_config.get(
                'gae_lambda', ppo_config.gae_lambda)
            ppo_config.batch_size = best_hpo_config.get(
                'batch_size', ppo_config.batch_size)
            ppo_config.entropy_coef = best_hpo_config.get(
                'entropy_coef', ppo_config.entropy_coef)
            ppo_config.value_loss_coef = best_hpo_config.get(
                'value_loss_coef', ppo_config.value_loss_coef)
        
        # 如果有GA结构结果
        if network_structure:
            ppo_config.hidden_dim = network_structure.get(
                'hidden_dim', ppo_config.hidden_dim)
            ppo_config.fusion_dim = network_structure.get(
                'fusion_dim', ppo_config.fusion_dim)
            ppo_config.num_transformer_layers = network_structure.get(
                'num_transformer_layers',
                ppo_config.num_transformer_layers)
            ppo_config.num_heads = network_structure.get(
                'num_heads', ppo_config.num_heads)
        
        # 创建Agent
        agent = GDS_PPO(
            task_input_dim, resource_input_dim, action_dim,
            ppo_config, network_structure)
        
        logger.info(f"Network params: "
                     f"{sum(p.numel() for p in agent.policy.parameters()):,}")
        
        # 训练
        result = train_ppo(
            agent, train_env, ppo_config, output_dir, logger,
            val_env=val_env)
        
        # 最终评估
        logger.info("="*60)
        logger.info("Final Evaluation")
        logger.info("="*60)
        
        final_eval = evaluate_agent(agent, val_env, num_episodes=20)
        logger.info(f"  Makespan:     {final_eval['makespan']:.2f} "
                     f"± {final_eval['makespan_std']:.2f}")
        logger.info(f"  Utilization:  {final_eval['utilization']:.4f}")
        logger.info(f"  Load Balance: {final_eval['load_balance']:.4f}")
        
        with open(output_dir / 'final_eval.json', 'w') as f:
            json.dump(final_eval, f, indent=2, default=str)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
