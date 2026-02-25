#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本 - 对比FE-IDDQN与传统调度算法
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import logging
from typing import Dict, List
from environment.enhanced_workflow_simulator import EnhancedWorkflowSimulator
from models.enhanced_fe_iddqn import EnhancedFE_IDDQN, EnhancedFE_IDDQN_Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_environment(num_tasks: int = 8, num_resources: int = 6):
    """创建测试环境"""
    # 创建任务
    np.random.seed(42)
    tasks = []
    for i in range(num_tasks):
        tasks.append({
            'id': i,
            'name': f'task_{i}',
            'duration': np.random.randint(5, 30),
            'resource_req': np.random.randint(1, 5),
            'cpu_req': np.random.randint(1, 4),
            'memory_req': np.random.randint(1, 8)
        })
    
    # 创建依赖关系（随机DAG）
    dependencies = []
    for i in range(1, num_tasks):
        # 每个任务随机依赖之前的1-2个任务
        num_deps = min(np.random.randint(1, 3), i)
        deps = np.random.choice(range(i), num_deps, replace=False)
        for dep in deps:
            dependencies.append((dep, i))
    
    # 创建资源
    resources = []
    for i in range(num_resources):
        resources.append({
            'id': i,
            'name': f'node_{i}',
            'capacity': np.random.randint(6, 12),
            'cpu_capacity': np.random.randint(4, 16),
            'memory_capacity': np.random.randint(8, 32)
        })
    
    return tasks, resources, dependencies


class FIFOScheduler:
    """FIFO调度器"""
    def __init__(self):
        self.name = "FIFO"
    
    def schedule(self, env: EnhancedWorkflowSimulator) -> Dict:
        """按FIFO顺序调度"""
        state = env.reset()
        total_reward = 0
        step = 0
        
        while not env.is_done() and step < 500:
            # FIFO: 总是选择第一个可用资源
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


class RoundRobinScheduler:
    """轮询调度器"""
    def __init__(self):
        self.name = "RoundRobin"
        self.current_resource = 0
    
    def schedule(self, env: EnhancedWorkflowSimulator) -> Dict:
        """轮询分配资源"""
        state = env.reset()
        total_reward = 0
        step = 0
        num_resources = env.num_resources
        
        while not env.is_done() and step < 500:
            action = self.current_resource % num_resources
            self.current_resource += 1
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


class RandomScheduler:
    """随机调度器"""
    def __init__(self):
        self.name = "Random"
    
    def schedule(self, env: EnhancedWorkflowSimulator) -> Dict:
        """随机分配资源"""
        state = env.reset()
        total_reward = 0
        step = 0
        
        while not env.is_done() and step < 500:
            action = np.random.randint(0, env.num_resources)
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


class ShortestJobFirstScheduler:
    """最短任务优先调度器"""
    def __init__(self):
        self.name = "SJF"
    
    def schedule(self, env: EnhancedWorkflowSimulator) -> Dict:
        """选择负载最小的资源"""
        state = env.reset()
        total_reward = 0
        step = 0
        
        while not env.is_done() and step < 500:
            # 选择当前负载最小的资源
            loads = [env.resource_states[r['id']].total_busy_time 
                    for r in env.resources]
            action = int(np.argmin(loads))
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


class FEIDDQNScheduler:
    """FE-IDDQN调度器"""
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.name = "FE-IDDQN"
        self.device = device
        self.model_path = model_path
        self.agent = None
        self.task_input_dim = 64
        self.resource_input_dim = 32
        
    def load_model(self, task_input_dim: int = 64, resource_input_dim: int = 32, 
                   action_dim: int = 6):
        """加载训练好的模型"""
        self.task_input_dim = task_input_dim
        self.resource_input_dim = resource_input_dim
        
        config = EnhancedFE_IDDQN_Config()
        self.agent = EnhancedFE_IDDQN(
            task_input_dim=task_input_dim,
            resource_input_dim=resource_input_dim,
            action_dim=action_dim,
            config=config
        )
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.agent.q_network.load_state_dict(checkpoint['q_network_state'])
        self.agent.q_network.eval()
        logger.info(f"模型已从 {self.model_path} 加载")
    
    def schedule(self, env: EnhancedWorkflowSimulator) -> Dict:
        """使用训练好的模型调度"""
        state = env.reset()
        total_reward = 0
        step = 0
        
        while not env.is_done() and step < 500:
            # 提取特征并调整维度
            task_features = state.get('task_features', np.zeros((8, self.task_input_dim)))
            resource_features = state.get('resource_features', np.zeros((6, self.resource_input_dim)))
            
            # 确保维度正确
            if task_features.shape[1] != self.task_input_dim:
                # 调整维度
                new_task = np.zeros((task_features.shape[0], self.task_input_dim))
                new_task[:, :min(task_features.shape[1], self.task_input_dim)] = \
                    task_features[:, :min(task_features.shape[1], self.task_input_dim)]
                task_features = new_task
            
            if resource_features.shape[1] != self.resource_input_dim:
                new_res = np.zeros((resource_features.shape[0], self.resource_input_dim))
                new_res[:, :min(resource_features.shape[1], self.resource_input_dim)] = \
                    resource_features[:, :min(resource_features.shape[1], self.resource_input_dim)]
                resource_features = new_res
            
            # 选择动作（不探索）
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
            'total_reward': total_reward
        }


def run_comparison(num_episodes: int = 50):
    """运行对比实验"""
    logger.info("=" * 70)
    logger.info("                    模型对比评估实验")
    logger.info("=" * 70)
    
    # 创建测试环境
    tasks, resources, dependencies = create_test_environment(num_tasks=8, num_resources=6)
    
    # 初始化调度器
    schedulers = [
        RandomScheduler(),
        FIFOScheduler(),
        RoundRobinScheduler(),
        ShortestJobFirstScheduler(),
    ]
    
    # 尝试加载FE-IDDQN模型
    model_path = "checkpoints/enhanced_fe_iddqn.pt"
    if os.path.exists(model_path):
        # 从checkpoint获取实际维度
        checkpoint = torch.load(model_path, map_location='cpu')
        # 从权重shape推断维度
        task_dim = checkpoint['q_network_state']['task_stream.input_projection.weight'].shape[1]
        resource_dim = checkpoint['q_network_state']['resource_stream.input_projection.weight'].shape[1]
        logger.info(f"检测到模型维度: task_dim={task_dim}, resource_dim={resource_dim}")
        
        fe_iddqn = FEIDDQNScheduler(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        fe_iddqn.load_model(task_input_dim=task_dim, resource_input_dim=resource_dim, action_dim=6)
        schedulers.append(fe_iddqn)
    else:
        logger.warning(f"模型文件 {model_path} 不存在，跳过FE-IDDQN评估")
    
    # 运行实验
    results = {s.name: {'makespan': [], 'utilization': [], 'load_balance': [], 'reward': []} 
               for s in schedulers}
    
    for episode in range(num_episodes):
        if (episode + 1) % 10 == 0:
            logger.info(f"评估进度: {episode + 1}/{num_episodes}")
        
        for scheduler in schedulers:
            # 创建新环境
            env = EnhancedWorkflowSimulator(tasks, resources, dependencies)
            
            # 运行调度
            result = scheduler.schedule(env)
            
            # 记录结果
            results[scheduler.name]['makespan'].append(result['makespan'])
            results[scheduler.name]['utilization'].append(result['utilization'])
            results[scheduler.name]['load_balance'].append(result['load_balance'])
            results[scheduler.name]['reward'].append(result['total_reward'])
    
    # 打印结果
    logger.info("\n" + "=" * 70)
    logger.info("                         评估结果")
    logger.info("=" * 70)
    
    print("\n" + "=" * 90)
    print(f"{'算法':<15} | {'Makespan':>12} | {'利用率':>12} | {'负载均衡':>12} | {'总奖励':>12}")
    print(f"{'':15} | {'(越小越好)':>12} | {'(越大越好)':>12} | {'(越大越好)':>12} | {'(越大越好)':>12}")
    print("=" * 90)
    
    summary = []
    for name, data in results.items():
        avg_makespan = np.mean(data['makespan'])
        avg_util = np.mean(data['utilization'])
        avg_lb = np.mean(data['load_balance'])
        avg_reward = np.mean(data['reward'])
        
        print(f"{name:<15} | {avg_makespan:>12.2f} | {avg_util:>12.2%} | {avg_lb:>12.2%} | {avg_reward:>12.2f}")
        
        summary.append({
            'name': name,
            'makespan': avg_makespan,
            'utilization': avg_util,
            'load_balance': avg_lb,
            'reward': avg_reward
        })
    
    print("=" * 90)
    
    # 找出最佳算法
    best_makespan = min(summary, key=lambda x: x['makespan'])
    best_util = max(summary, key=lambda x: x['utilization'])
    best_reward = max(summary, key=lambda x: x['reward'])
    
    print("\n📊 最佳表现:")
    print(f"  • 最小 Makespan: {best_makespan['name']} ({best_makespan['makespan']:.2f})")
    print(f"  • 最高利用率:   {best_util['name']} ({best_util['utilization']:.2%})")
    print(f"  • 最高奖励:     {best_reward['name']} ({best_reward['reward']:.2f})")
    
    # FE-IDDQN相对于Random的改进
    if 'FE-IDDQN' in results and 'Random' in results:
        fe_makespan = np.mean(results['FE-IDDQN']['makespan'])
        random_makespan = np.mean(results['Random']['makespan'])
        improvement = (random_makespan - fe_makespan) / random_makespan * 100
        
        print(f"\n🎯 FE-IDDQN vs Random:")
        print(f"  • Makespan 改进: {improvement:.2f}%")
        print(f"  • 利用率提升: {(np.mean(results['FE-IDDQN']['utilization']) - np.mean(results['Random']['utilization'])):.2%}")
    
    return results


if __name__ == "__main__":
    results = run_comparison(num_episodes=50)
