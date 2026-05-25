#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用与训练相同的环境进行评估
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import logging
from typing import Dict
from environment.enhanced_workflow_simulator import EnhancedWorkflowSimulator
from models.enhanced_fe_iddqn import EnhancedFE_IDDQN, EnhancedFE_IDDQN_Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_training_environment():
    """创建与训练时完全相同的环境"""
    # 这是 main_enhanced.py 中的 create_sample_environment() 的完全复制
    tasks = [
        {'id': 0, 'name': 'task_0', 'duration': 10, 'resource_req': 2},
        {'id': 1, 'name': 'task_1', 'duration': 15, 'resource_req': 3},
        {'id': 2, 'name': 'task_2', 'duration': 20, 'resource_req': 4},
        {'id': 3, 'name': 'task_3', 'duration': 30, 'resource_req': 6},
        {'id': 4, 'name': 'task_4', 'duration': 10, 'resource_req': 2},
        {'id': 5, 'name': 'task_5', 'duration': 12, 'resource_req': 3},
        {'id': 6, 'name': 'task_6', 'duration': 8, 'resource_req': 2},
        {'id': 7, 'name': 'task_7', 'duration': 5, 'resource_req': 1},
    ]
    
    dependencies = [
        (0, 1), (0, 5),
        (1, 2), (5, 6),
        (2, 3), (6, 3),
        (3, 4), (4, 7)
    ]
    
    resources = [
        {'id': 0, 'name': 'worker_0', 'capacity': 4, 'speed': 1.0},
        {'id': 1, 'name': 'worker_1', 'capacity': 6, 'speed': 1.2},
        {'id': 2, 'name': 'worker_2', 'capacity': 8, 'speed': 0.9},
        {'id': 3, 'name': 'worker_3', 'capacity': 4, 'speed': 1.1},
        {'id': 4, 'name': 'worker_4', 'capacity': 10, 'speed': 2.0},
        {'id': 5, 'name': 'worker_5', 'capacity': 12, 'speed': 0.8},
    ]
    
    return tasks, resources, dependencies


def schedule_with_algorithm(env: EnhancedWorkflowSimulator, algorithm: str) -> Dict:
    """使用指定算法调度"""
    state = env.reset()
    total_reward = 0
    step = 0
    current_resource = 0
    
    while not env.is_done() and step < 500:
        if algorithm == "FIFO":
            action = 0
        elif algorithm == "RoundRobin":
            action = current_resource % env.num_resources
            current_resource += 1
        elif algorithm == "Random":
            action = np.random.randint(0, env.num_resources)
        elif algorithm == "SJF":
            loads = [env.resource_states[r['id']].total_busy_time for r in env.resources]
            action = int(np.argmin(loads))
        else:
            action = 0
        
        _, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        if done:
            break
    
    return {
        'makespan': env.get_makespan(),
        'utilization': env.get_resource_utilization(),
        'load_balance': env.get_load_balance_score(),
        'total_reward': total_reward
    }


def schedule_with_model(env: EnhancedWorkflowSimulator, agent, task_dim: int, resource_dim: int) -> Dict:
    """使用训练好的模型调度"""
    state = env.reset()
    total_reward = 0
    step = 0
    
    while not env.is_done() and step < 500:
        task_features = state.get('task_features', np.zeros((8, task_dim)))
        resource_features = state.get('resource_features', np.zeros((6, resource_dim)))
        
        # 确保维度匹配
        if task_features.shape[1] != task_dim:
            new_task = np.zeros((task_features.shape[0], task_dim))
            min_dim = min(task_features.shape[1], task_dim)
            new_task[:, :min_dim] = task_features[:, :min_dim]
            task_features = new_task
        
        if resource_features.shape[1] != resource_dim:
            new_res = np.zeros((resource_features.shape[0], resource_dim))
            min_dim = min(resource_features.shape[1], resource_dim)
            new_res[:, :min_dim] = resource_features[:, :min_dim]
            resource_features = new_res
        
        action = agent.select_action(
            task_features=task_features,
            resource_features=resource_features,
            training=False
        )
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        step += 1
        if done:
            break
    
    return {
        'makespan': env.get_makespan(),
        'utilization': env.get_resource_utilization(),
        'load_balance': env.get_load_balance_score(),
        'total_reward': total_reward
    }


def main():
    logger.info("=" * 70)
    logger.info("      使用训练环境进行公平评估")
    logger.info("=" * 70)
    
    # 获取训练环境配置
    tasks, resources, dependencies = create_training_environment()
    
    logger.info(f"环境配置: {len(tasks)} 任务, {len(resources)} 资源, {len(dependencies)} 依赖")
    logger.info(f"任务时长: {[t['duration'] for t in tasks]}")
    logger.info(f"关键路径: 0→1→2→3→4→7 = {10+15+20+30+10+5} = 90 (理论最优)")
    
    # 加载模型
    model_path = "checkpoints/enhanced_fe_iddqn.pt"
    agent = None
    task_dim = 19
    resource_dim = 11
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        task_dim = checkpoint['q_network_state']['task_stream.input_projection.weight'].shape[1]
        resource_dim = checkpoint['q_network_state']['resource_stream.input_projection.weight'].shape[1]
        
        config = EnhancedFE_IDDQN_Config()
        agent = EnhancedFE_IDDQN(
            task_input_dim=task_dim,
            resource_input_dim=resource_dim,
            action_dim=6,
            config=config
        )
        agent.q_network.load_state_dict(checkpoint['q_network_state'])
        agent.q_network.eval()
        logger.info(f"模型加载成功，维度: task={task_dim}, resource={resource_dim}")
    
    # 运行评估
    algorithms = ["Random", "FIFO", "RoundRobin", "SJF"]
    num_episodes = 50
    
    results = {alg: {'makespan': [], 'utilization': [], 'reward': []} for alg in algorithms}
    results['FE-IDDQN'] = {'makespan': [], 'utilization': [], 'reward': []}
    
    logger.info(f"\n运行 {num_episodes} 次评估...")
    
    for ep in range(num_episodes):
        # 传统算法
        for alg in algorithms:
            env = EnhancedWorkflowSimulator(tasks, resources, dependencies)
            result = schedule_with_algorithm(env, alg)
            results[alg]['makespan'].append(result['makespan'])
            results[alg]['utilization'].append(result['utilization'])
            results[alg]['reward'].append(result['total_reward'])
        
        # FE-IDDQN
        if agent is not None:
            env = EnhancedWorkflowSimulator(tasks, resources, dependencies)
            result = schedule_with_model(env, agent, task_dim, resource_dim)
            results['FE-IDDQN']['makespan'].append(result['makespan'])
            results['FE-IDDQN']['utilization'].append(result['utilization'])
            results['FE-IDDQN']['reward'].append(result['total_reward'])
    
    # 打印结果
    print("\n" + "=" * 80)
    print("                    评估结果（使用训练环境）")
    print("=" * 80)
    print(f"理论最优 Makespan: 90 (关键路径长度)")
    print("=" * 80)
    print(f"{'算法':<15} | {'Makespan':>12} | {'利用率':>12} | {'总奖励':>12}")
    print("-" * 80)
    
    for name in ["Random", "FIFO", "RoundRobin", "SJF", "FE-IDDQN"]:
        if name in results and results[name]['makespan']:
            avg_ms = np.mean(results[name]['makespan'])
            avg_util = np.mean(results[name]['utilization'])
            avg_reward = np.mean(results[name]['reward'])
            print(f"{name:<15} | {avg_ms:>12.2f} | {avg_util:>11.2%} | {avg_reward:>12.2f}")
    
    print("=" * 80)
    
    # 分析 FE-IDDQN
    if 'FE-IDDQN' in results and results['FE-IDDQN']['makespan']:
        fe_ms = np.mean(results['FE-IDDQN']['makespan'])
        random_ms = np.mean(results['Random']['makespan'])
        best_baseline_ms = min(np.mean(results[alg]['makespan']) for alg in algorithms)
        
        print(f"\n📊 FE-IDDQN 分析:")
        print(f"  • 与随机对比: {((random_ms - fe_ms) / random_ms * 100):+.2f}%")
        print(f"  • 与最佳基线对比: {((best_baseline_ms - fe_ms) / best_baseline_ms * 100):+.2f}%")
        print(f"  • 距离理论最优: {((fe_ms - 90) / 90 * 100):+.2f}%")
        
        # 检查每次运行的结果是否相同
        unique_makespans = set(results['FE-IDDQN']['makespan'])
        print(f"  • Makespan 分布: {unique_makespans}")
        if len(unique_makespans) == 1:
            print(f"  ⚠️ 模型输出确定性策略（每次结果相同）")


if __name__ == "__main__":
    main()
