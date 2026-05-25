#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试未训练的 FE-IDDQN 算法 vs 传统调度算法
验证网络架构本身是否有优势
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


def create_test_environment():
    """创建测试环境"""
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


class BaselineScheduler:
    """基线调度器"""
    def __init__(self, name: str):
        self.name = name
        self.current_resource = 0
    
    def schedule(self, env: EnhancedWorkflowSimulator) -> Dict:
        """执行调度"""
        state = env.reset()
        total_reward = 0
        step = 0
        
        while not env.is_done() and step < 500:
            if self.name == "Random":
                action = np.random.randint(0, env.num_resources)
            elif self.name == "FIFO":
                action = 0
            elif self.name == "RoundRobin":
                action = self.current_resource % env.num_resources
                self.current_resource += 1
            elif self.name == "SJF":
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
            'total_reward': total_reward,
            'steps': step
        }


class UntrainedFEIDDQN:
    """未训练的 FE-IDDQN（随机初始化网络）"""
    def __init__(self, exploration_mode: str = 'greedy'):
        self.name = f"FE-IDDQN-{exploration_mode}"
        self.exploration_mode = exploration_mode  # 'greedy', 'random', 'epsilon_0.1'
        self.agent = None
        self.task_dim = None
        self.resource_dim = None
    
    def initialize(self, task_dim: int, resource_dim: int, action_dim: int):
        """初始化网络（不加载训练权重）"""
        config = EnhancedFE_IDDQN_Config()
        self.agent = EnhancedFE_IDDQN(
            task_input_dim=task_dim,
            resource_input_dim=resource_dim,
            action_dim=action_dim,
            config=config
        )
        self.agent.q_network.eval()
        self.task_dim = task_dim
        self.resource_dim = resource_dim
        logger.info(f"初始化 {self.name}，维度: task={task_dim}, resource={resource_dim}")
    
    def schedule(self, env: EnhancedWorkflowSimulator) -> Dict:
        """使用未训练的网络进行调度"""
        state = env.reset()
        total_reward = 0
        step = 0
        
        while not env.is_done() and step < 500:
            task_features = state.get('task_features', np.zeros((8, self.task_dim)))
            resource_features = state.get('resource_features', np.zeros((6, self.resource_dim)))
            
            # 确保维度匹配
            if task_features.shape[1] != self.task_dim:
                new_task = np.zeros((task_features.shape[0], self.task_dim))
                min_dim = min(task_features.shape[1], self.task_dim)
                new_task[:, :min_dim] = task_features[:, :min_dim]
                task_features = new_task
            
            if resource_features.shape[1] != self.resource_dim:
                new_res = np.zeros((resource_features.shape[0], self.resource_dim))
                min_dim = min(resource_features.shape[1], self.resource_dim)
                new_res[:, :min_dim] = resource_features[:, :min_dim]
                resource_features = new_res
            
            # 选择动作
            if self.exploration_mode == 'random':
                action = np.random.randint(0, env.num_resources)
            elif self.exploration_mode == 'epsilon_0.1':
                if np.random.rand() < 0.1:
                    action = np.random.randint(0, env.num_resources)
                else:
                    action = self.agent.select_action(
                        task_features=task_features,
                        resource_features=resource_features,
                        training=False
                    )
            else:  # greedy
                action = self.agent.select_action(
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
            'total_reward': total_reward,
            'steps': step
        }


def main():
    print("\n" + "=" * 80)
    print("           未训练 FE-IDDQN 算法 vs 传统调度算法对比")
    print("=" * 80)
    print("问题：不训练模型，直接用 FE-IDDQN 网络架构能否比传统算法好？")
    print("=" * 80)
    
    # 创建环境
    tasks, resources, dependencies = create_test_environment()
    logger.info(f"环境: {len(tasks)} 任务, {len(resources)} 资源")
    logger.info(f"理论最优 Makespan: 90 (关键路径 0→1→2→3→4→7)")
    
    # 初始化调度器
    schedulers = [
        BaselineScheduler("Random"),
        BaselineScheduler("FIFO"),
        BaselineScheduler("RoundRobin"),
        BaselineScheduler("SJF"),
    ]
    
    # 添加未训练的 FE-IDDQN（不同探索策略）
    untrained_greedy = UntrainedFEIDDQN('greedy')
    untrained_greedy.initialize(task_dim=19, resource_dim=11, action_dim=6)
    schedulers.append(untrained_greedy)
    
    untrained_epsilon = UntrainedFEIDDQN('epsilon_0.1')
    untrained_epsilon.initialize(task_dim=19, resource_dim=11, action_dim=6)
    schedulers.append(untrained_epsilon)
    
    # 如果有训练好的模型，也加载进来对比
    model_path = "checkpoints/enhanced_fe_iddqn.pt"
    trained_agent = None
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        task_dim = checkpoint['q_network_state']['task_stream.input_projection.weight'].shape[1]
        resource_dim = checkpoint['q_network_state']['resource_stream.input_projection.weight'].shape[1]
        
        config = EnhancedFE_IDDQN_Config()
        trained_agent = EnhancedFE_IDDQN(
            task_input_dim=task_dim,
            resource_input_dim=resource_dim,
            action_dim=6,
            config=config
        )
        trained_agent.q_network.load_state_dict(checkpoint['q_network_state'])
        trained_agent.q_network.eval()
        logger.info("已加载训练好的模型用于对比")
    
    # 运行评估
    num_episodes = 100
    logger.info(f"\n开始评估，每个算法运行 {num_episodes} 次...")
    
    results = {s.name: {'makespan': [], 'utilization': [], 'reward': [], 'steps': []} 
               for s in schedulers}
    
    if trained_agent is not None:
        results['FE-IDDQN-Trained'] = {'makespan': [], 'utilization': [], 'reward': [], 'steps': []}
    
    for ep in range(num_episodes):
        if (ep + 1) % 20 == 0:
            logger.info(f"  进度: {ep + 1}/{num_episodes}")
        
        # 基线算法 + 未训练 FE-IDDQN
        for scheduler in schedulers:
            env = EnhancedWorkflowSimulator(tasks, resources, dependencies)
            result = scheduler.schedule(env)
            results[scheduler.name]['makespan'].append(result['makespan'])
            results[scheduler.name]['utilization'].append(result['utilization'])
            results[scheduler.name]['reward'].append(result['total_reward'])
            results[scheduler.name]['steps'].append(result['steps'])
        
        # 训练好的模型
        if trained_agent is not None:
            env = EnhancedWorkflowSimulator(tasks, resources, dependencies)
            state = env.reset()
            total_reward = 0
            step = 0
            
            while not env.is_done() and step < 500:
                task_features = state.get('task_features', np.zeros((8, task_dim)))
                resource_features = state.get('resource_features', np.zeros((6, resource_dim)))
                
                if task_features.shape[1] != task_dim:
                    new_task = np.zeros((task_features.shape[0], task_dim))
                    new_task[:, :min(task_features.shape[1], task_dim)] = \
                        task_features[:, :min(task_features.shape[1], task_dim)]
                    task_features = new_task
                
                if resource_features.shape[1] != resource_dim:
                    new_res = np.zeros((resource_features.shape[0], resource_dim))
                    new_res[:, :min(resource_features.shape[1], resource_dim)] = \
                        resource_features[:, :min(resource_features.shape[1], resource_dim)]
                    resource_features = new_res
                
                action = trained_agent.select_action(
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
            
            results['FE-IDDQN-Trained']['makespan'].append(env.get_makespan())
            results['FE-IDDQN-Trained']['utilization'].append(env.get_resource_utilization())
            results['FE-IDDQN-Trained']['reward'].append(total_reward)
            results['FE-IDDQN-Trained']['steps'].append(step)
    
    # 打印结果
    print("\n" + "=" * 95)
    print("                                评估结果")
    print("=" * 95)
    print(f"{'算法':<25} | {'Makespan':>12} | {'利用率':>10} | {'负载均衡':>10} | {'奖励':>10} | {'步数':>8}")
    print(f"{'':25} | {'(越小越好)':>12} | {'(越大越好)':>10} | {'(越大越好)':>10} | {'(越大越好)':>10} | {'':>8}")
    print("-" * 95)
    
    summary = []
    for name in ["Random", "FIFO", "RoundRobin", "SJF", 
                 "FE-IDDQN-greedy", "FE-IDDQN-epsilon_0.1", "FE-IDDQN-Trained"]:
        if name in results and results[name]['makespan']:
            avg_ms = np.mean(results[name]['makespan'])
            std_ms = np.std(results[name]['makespan'])
            avg_util = np.mean(results[name]['utilization'])
            avg_lb = np.mean([r.get('load_balance', 0) for r in results[name].get('load_balance', [])])
            avg_reward = np.mean(results[name]['reward'])
            avg_steps = np.mean(results[name]['steps'])
            
            print(f"{name:<25} | {avg_ms:>9.2f}±{std_ms:>3.1f} | {avg_util:>9.2%} | {avg_lb:>9.2%} | {avg_reward:>10.2f} | {avg_steps:>8.1f}")
            
            summary.append({
                'name': name,
                'makespan': avg_ms,
                'std': std_ms,
                'utilization': avg_util,
                'reward': avg_reward
            })
    
    print("=" * 95)
    
    # 分析
    best = min(summary, key=lambda x: x['makespan'])
    print(f"\n🏆 最佳 Makespan: {best['name']} ({best['makespan']:.2f})")
    print(f"📏 理论最优: 90.00 (关键路径长度)")
    
    print("\n📊 关键发现:")
    
    # 对比未训练 vs 传统算法
    untrained_greedy_ms = next((s['makespan'] for s in summary if s['name'] == 'FE-IDDQN-greedy'), None)
    random_ms = next((s['makespan'] for s in summary if s['name'] == 'Random'), None)
    rr_ms = next((s['makespan'] for s in summary if s['name'] == 'RoundRobin'), None)
    
    if untrained_greedy_ms and random_ms:
        improvement = (random_ms - untrained_greedy_ms) / random_ms * 100
        print(f"  • 未训练 FE-IDDQN (greedy) vs Random: {improvement:+.2f}%")
    
    if untrained_greedy_ms and rr_ms:
        improvement = (rr_ms - untrained_greedy_ms) / rr_ms * 100
        print(f"  • 未训练 FE-IDDQN (greedy) vs RoundRobin: {improvement:+.2f}%")
    
    # 对比训练前后
    if 'FE-IDDQN-Trained' in results and untrained_greedy_ms:
        trained_ms = next((s['makespan'] for s in summary if s['name'] == 'FE-IDDQN-Trained'), None)
        if trained_ms:
            improvement = (untrained_greedy_ms - trained_ms) / untrained_greedy_ms * 100
            print(f"  • 训练后改进: {improvement:+.2f}%")
            if improvement <= 0:
                print(f"    ⚠️ 训练反而让性能变差了！")
    
    # 检查未训练网络的行为
    untrained_std = next((s['std'] for s in summary if s['name'] == 'FE-IDDQN-greedy'), None)
    if untrained_std is not None:
        if untrained_std < 0.1:
            print(f"\n  ⚠️ 未训练 FE-IDDQN 输出几乎是确定性的（std={untrained_std:.3f}）")
            print(f"     → 随机初始化的网络已经有了固定的偏好")
        else:
            print(f"\n  ✓ 未训练 FE-IDDQN 有一定随机性（std={untrained_std:.3f}）")


if __name__ == "__main__":
    main()
