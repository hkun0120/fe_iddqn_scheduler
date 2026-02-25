#!/usr/bin/env python3
"""
FE-IDDQN训练和评估系统 - 主目录版本
按6:2:2比例划分数据集，训练模型，并与启发式算法对比评估
"""

import json
import pandas as pd
import numpy as np
import logging
import sys
import os
import pickle
import torch
import networkx as nx
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import random

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

from config.config import Config
from config.hyperparameters import Hyperparameters
from models.fe_iddqn import FE_IDDQN
from environment.historical_replay_simulator import HistoricalReplaySimulator
from data.mysql_data_loader import MySQLDataLoader


class HeuristicScheduler:
    """启发式调度算法基类"""
    
    def __init__(self, name, num_resources=5):
        self.name = name
        self.num_resources = num_resources
        self.resource_available_time = {i: 0 for i in range(num_resources)}
        self.counter = 0  # for round robin
    
    def reset(self):
        self.resource_available_time = {i: 0 for i in range(self.num_resources)}
        self.counter = 0
    
    def select_action(self, task_features, resource_features, **kwargs):
        """根据策略选择资源"""
        raise NotImplementedError


class FIFOScheduler(HeuristicScheduler):
    """FIFO调度 - 总是选择第一个可用资源"""
    def __init__(self, num_resources=5):
        super().__init__("FIFO", num_resources)
    
    def select_action(self, task_features, resource_features, **kwargs):
        return 0


class RoundRobinScheduler(HeuristicScheduler):
    """轮询调度"""
    def __init__(self, num_resources=5):
        super().__init__("RoundRobin", num_resources)
    
    def select_action(self, task_features, resource_features, **kwargs):
        action = self.counter % self.num_resources
        self.counter += 1
        return action


class SJFScheduler(HeuristicScheduler):
    """最短作业优先 - 选择当前负载最低的资源"""
    def __init__(self, num_resources=5):
        super().__init__("SJF", num_resources)
    
    def select_action(self, task_features, resource_features, **kwargs):
        # resource_features: [num_resources, feature_dim]
        # 假设第一个特征是可用时间或负载
        if isinstance(resource_features, np.ndarray) and len(resource_features.shape) >= 2:
            loads = resource_features[:, 0]  # 第一个特征作为负载指标
            return int(np.argmin(loads))
        return 0


class EFTScheduler(HeuristicScheduler):
    """最早完成时间调度"""
    def __init__(self, num_resources=5):
        super().__init__("EFT", num_resources)
    
    def select_action(self, task_features, resource_features, **kwargs):
        # 选择能让任务最早完成的资源
        if isinstance(resource_features, np.ndarray) and len(resource_features.shape) >= 2:
            # 假设resource_features[:, 0]是资源可用时间
            available_times = resource_features[:, 0]
            # task_features中包含任务时长信息
            if isinstance(task_features, np.ndarray) and len(task_features.shape) >= 2:
                # 获取当前任务的预计执行时间
                task_duration = task_features[0, 0] if task_features.shape[0] > 0 else 1
                finish_times = available_times + task_duration
                return int(np.argmin(finish_times))
        return 0


class RandomScheduler(HeuristicScheduler):
    """随机调度"""
    def __init__(self, num_resources=5):
        super().__init__("Random", num_resources)
    
    def select_action(self, task_features, resource_features, **kwargs):
        return random.randint(0, self.num_resources - 1)

class FEIDDQNTrainer:
    """FE-IDDQN训练器"""
    
    def __init__(self, output_dir="fe_iddqn_training_system"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.logs_dir = self.output_dir / "logs"
        self.models_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        self.data_dir = self.output_dir / "data"
        
        for dir_path in [self.logs_dir, self.models_dir, self.results_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据加载器
        self.data_loader = MySQLDataLoader(
            host='localhost',
            user='root',
            password='',
            database='whalesb',
            port=3306
        )
        
        # 训练参数
        self.task_input_dim = 16
        self.resource_input_dim = 7
        # 动作空间与HistoricalReplaySimulator.MAX_RESOURCES保持一致
        self.action_dim = 5
        
        # 训练历史
        self.training_history = {
            'train_rewards': [],
            'train_makespans': [],
            'val_rewards': [],
            'val_makespans': [],
            'test_rewards': [],
            'test_makespans': []
        }
    
    def setup_logging(self):
        """设置日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def load_and_split_data(self):
        """加载数据并按6:2:2比例划分"""
        self.logger.info("正在加载数据...")
        
        # 加载数据
        data = self.data_loader.load_all_data()
        if not data:
            raise Exception("数据加载失败")
        
        self.logger.info("数据加载成功")
        
        # 获取成功的进程实例
        successful_processes = data['process_instance'][data['process_instance']['state'] == 7]
        self.logger.info(f"找到 {len(successful_processes)} 个成功的进程")
        
        # 按任务数量分类工作流大小
        process_stats = []
        for _, process in successful_processes.iterrows():
            process_id = process['id']
            task_count = len(data['task_instance'][data['task_instance']['process_instance_id'] == process_id])
            
            process_stats.append({
                'process_id': process_id,
                'task_count': task_count,
                'process_name': process['name'],
                'start_time': process['start_time'],
                'end_time': process['end_time']
            })
        
        process_df = pd.DataFrame(process_stats)
        
        # 按任务数量分类：小(1-10), 中(11-30), 大(31+)
        process_df['workflow_size'] = pd.cut(
            process_df['task_count'], 
            bins=[0, 10, 30, float('inf')], 
            labels=['small', 'medium', 'large']
        )
        
        self.logger.info(f"工作流大小分布:")
        self.logger.info(f"  小工作流 (1-10任务): {len(process_df[process_df['workflow_size'] == 'small'])} 个")
        self.logger.info(f"  中工作流 (11-30任务): {len(process_df[process_df['workflow_size'] == 'medium'])} 个")
        self.logger.info(f"  大工作流 (31+任务): {len(process_df[process_df['workflow_size'] == 'large'])} 个")
        
        # 分层抽样划分数据集
        train_data, temp_data = train_test_split(
            process_df, test_size=0.4, random_state=42, stratify=process_df['workflow_size']
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42, stratify=temp_data['workflow_size']
        )
        
        self.logger.info(f"数据集划分完成:")
        self.logger.info(f"  训练集: {len(train_data)} 个进程")
        self.logger.info(f"  验证集: {len(val_data)} 个进程")
        self.logger.info(f"  测试集: {len(test_data)} 个进程")
        
        # 保存数据集信息
        dataset_info = {
            'total_processes': len(process_df),
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'workflow_size_distribution': process_df['workflow_size'].value_counts().to_dict(),
            'train_workflow_distribution': train_data['workflow_size'].value_counts().to_dict(),
            'val_workflow_distribution': val_data['workflow_size'].value_counts().to_dict(),
            'test_workflow_distribution': test_data['workflow_size'].value_counts().to_dict()
        }
        
        with open(self.data_dir / 'dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存数据集
        train_data.to_csv(self.data_dir / 'train_data.csv', index=False)
        val_data.to_csv(self.data_dir / 'val_data.csv', index=False)
        test_data.to_csv(self.data_dir / 'test_data.csv', index=False)
        
        return data, train_data, val_data, test_data, process_df
    
    def create_simulator_for_process(self, data, process_id):
        """为特定进程创建仿真器"""
        process_instance = data['process_instance'][data['process_instance']['id'] == process_id]
        task_instances = data['task_instance'][data['task_instance']['process_instance_id'] == process_id]
        successful_tasks = task_instances[task_instances['state'] == 7]
        
        if len(successful_tasks) == 0:
            return None
        
        filtered_data = {
            'process_instance': process_instance,
            'task_instance': successful_tasks,
            'task_definition': data['task_definition'],
            'process_task_relation': data['process_task_relation']
        }
        
        return HistoricalReplaySimulator(
            filtered_data['process_instance'],
            filtered_data['task_instance'],
            filtered_data['task_definition'],
            filtered_data['process_task_relation']
        )
    
    def calculate_reward(self, simulator, task, host, action):
        """计算改进的奖励"""
        reward = 0.0
        
        # 1. 基础奖励：成功调度
        reward += 1.0
        
        # 2. 基于makespan的奖励
        current_makespan = simulator.get_makespan()
        if not hasattr(self, 'baseline_makespan'):
            self.baseline_makespan = current_makespan
        
        # 相对makespan改进奖励
        makespan_improvement = (self.baseline_makespan - current_makespan) / self.baseline_makespan
        reward += makespan_improvement * 10.0
        
        # 3. 资源利用率奖励
        resource = simulator.available_resources[host]
        cpu_util = resource['cpu_used'] / resource['cpu_capacity']
        memory_util = resource['memory_used'] / resource['memory_capacity']
        
        # 平衡的资源利用率获得更高奖励
        balance_score = 1.0 - abs(cpu_util - memory_util)
        reward += balance_score * 2.0
        
        # 4. 负载均衡奖励
        all_cpu_utils = [r['cpu_used']/r['cpu_capacity'] for r in simulator.available_resources.values()]
        load_balance = 1.0 - np.std(all_cpu_utils) if len(all_cpu_utils) > 1 else 1.0
        reward += load_balance * 1.0
        
        # 5. 任务优先级奖励
        priority = task.get('task_instance_priority', 0)
        if priority > 0:
            reward += priority * 0.1
        
        return reward
    
    def train_episode(self, data, process_id, agent, is_training=True):
        """训练一个episode"""
        simulator = self.create_simulator_for_process(data, process_id)
        if simulator is None:
            return 0.0, 0.0
        
        simulator.reset()
        episode_reward = 0.0
        step_count = 0
        max_steps = 1000
        
        while not simulator.is_done() and step_count < max_steps:
            state = simulator.get_state()
            if state is None:
                break
            
            task_features, resource_features = state
            graph_adj = simulator.get_graph_adj()
            action = agent.select_action(task_features, resource_features, graph_adj=graph_adj)
            
            # 执行动作
            next_state, old_reward, done, info = simulator.step(action)
            next_graph_adj = simulator.get_graph_adj()
            
            # 使用改进的奖励计算
            learning_reward = old_reward
            if len(simulator.available_resources) > action:
                selected_host = list(simulator.available_resources.keys())[action]
                current_task = simulator.current_process_tasks.iloc[simulator.current_task_idx-1] if simulator.current_task_idx > 0 else None
                if current_task is not None:
                    new_reward = self.calculate_reward(simulator, current_task, selected_host, action)
                    learning_reward = new_reward
                    episode_reward += new_reward
                else:
                    episode_reward += old_reward
            else:
                episode_reward += old_reward

            # 存储经验（包含DAG邻接矩阵，用于Graph Transformer）
            agent.store_experience(state, action, learning_reward, next_state, done, graph_adj=graph_adj, next_graph_adj=next_graph_adj)

            # 更新当前state
            state = next_state
            
            step_count += 1
            
            if done:
                break
        
        # 训练网络
        if is_training and len(agent.replay_buffer) >= 64:
            loss = agent.train()
        
        return episode_reward, simulator.get_makespan()
    
    def train_model(self, data, train_data, val_data, n_epochs=50):
        """训练模型"""
        self.logger.info(f"开始训练模型，共{n_epochs}个epoch...")
        
        # 创建智能体
        agent = FE_IDDQN(
            self.task_input_dim,
            self.resource_input_dim,
            self.action_dim,
            max_tasks=5,
            max_resources=5,
            enable_graph_encoder=True,
        )
        
        # 获取算法参数
        algorithm_params = Hyperparameters.get_algorithm_params('FE_IDDQN')
        
        best_val_makespan = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # 训练阶段
            train_rewards = []
            train_makespans = []
            
            # 随机选择训练进程
            train_processes = train_data.sample(n=min(20, len(train_data)), random_state=epoch)
            
            for _, process in train_processes.iterrows():
                reward, makespan = self.train_episode(data, process['process_id'], agent, is_training=True)
                train_rewards.append(reward)
                train_makespans.append(makespan)
            
            # 验证阶段
            val_rewards = []
            val_makespans = []
            
            for _, process in val_data.iterrows():
                reward, makespan = self.train_episode(data, process['process_id'], agent, is_training=False)
                val_rewards.append(reward)
                val_makespans.append(makespan)
            
            # 记录训练历史
            self.training_history['train_rewards'].append(np.mean(train_rewards))
            self.training_history['train_makespans'].append(np.mean(train_makespans))
            self.training_history['val_rewards'].append(np.mean(val_rewards))
            self.training_history['val_makespans'].append(np.mean(val_makespans))
            
            # 早停检查
            current_val_makespan = np.mean(val_makespans)
            if current_val_makespan < best_val_makespan:
                best_val_makespan = current_val_makespan
                patience_counter = 0
                # 保存最佳模型
                self.save_model(agent, epoch, is_best=True)
            else:
                patience_counter += 1
            
            # 每5个epoch输出一次统计信息
            if (epoch + 1) % 5 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{n_epochs}:")
                self.logger.info(f"  训练 - 平均奖励: {np.mean(train_rewards):.2f}, 平均Makespan: {np.mean(train_makespans):.2f}")
                self.logger.info(f"  验证 - 平均奖励: {np.mean(val_rewards):.2f}, 平均Makespan: {np.mean(val_makespans):.2f}")
                self.logger.info(f"  最佳验证Makespan: {best_val_makespan:.2f}")
            
            # 早停
            if patience_counter >= patience:
                self.logger.info(f"早停触发，在第{epoch + 1}个epoch停止训练")
                break
            
            # 更新目标网络和探索参数
            if epoch % 5 == 0:
                agent.update_target_network()
            agent.update_exploration_params()
        
        # 保存最终模型
        self.save_model(agent, epoch, is_best=False)
        
        # 保存训练历史
        self.save_training_history()
        
        return agent
    
    def save_model(self, agent, epoch, is_best=False):
        """保存模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            model_path = self.models_dir / f"fe_iddqn_best_model_{timestamp}.pkl"
        else:
            model_path = self.models_dir / f"fe_iddqn_epoch_{epoch}_{timestamp}.pkl"
        
        # 保存模型状态
        model_state = {
            'q_network_state': agent.q_network.state_dict(),
            'target_network_state': agent.target_network.state_dict(),
            'optimizer_state': agent.optimizer.state_dict(),
            'replay_buffer': agent.replay_buffer,
            'epoch': epoch,
            'timestamp': timestamp
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_state, f)
        
        self.logger.info(f"模型已保存到: {model_path}")
    
    def save_training_history(self):
        """保存训练历史"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = self.results_dir / f"training_history_{timestamp}.json"
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"训练历史已保存到: {history_path}")
    
    def evaluate_model(self, data, test_data, agent):
        """评估模型"""
        self.logger.info("开始评估模型...")
        
        # 按工作流大小分组评估
        evaluation_results = {}
        
        for size in ['small', 'medium', 'large']:
            size_data = test_data[test_data['workflow_size'] == size]
            if len(size_data) == 0:
                continue
            
            self.logger.info(f"评估{size}工作流 ({len(size_data)}个进程)...")
            
            size_rewards = []
            size_makespans = []
            size_resource_utilizations = []
            
            for _, process in size_data.iterrows():
                simulator = self.create_simulator_for_process(data, process['process_id'])
                if simulator is None:
                    continue
                
                simulator.reset()
                episode_reward = 0.0
                step_count = 0
                max_steps = 1000
                
                while not simulator.is_done() and step_count < max_steps:
                    state = simulator.get_state()
                    if state is None:
                        break
                    
                    task_features, resource_features = state
                    graph_adj = simulator.get_graph_adj()
                    action = agent.select_action(task_features, resource_features, graph_adj=graph_adj)
                    
                    state, reward, done, info = simulator.step(action)
                    episode_reward += reward
                    step_count += 1
                    
                    if done:
                        break
                
                size_rewards.append(episode_reward)
                size_makespans.append(simulator.get_makespan())
                size_resource_utilizations.append(simulator.get_resource_utilization())
            
            evaluation_results[size] = {
                'process_count': len(size_data),
                'avg_reward': np.mean(size_rewards),
                'std_reward': np.std(size_rewards),
                'avg_makespan': np.mean(size_makespans),
                'std_makespan': np.std(size_makespans),
                'avg_resource_utilization': np.mean(size_resource_utilizations),
                'std_resource_utilization': np.std(size_resource_utilizations),
                'rewards': size_rewards,
                'makespans': size_makespans,
                'resource_utilizations': size_resource_utilizations
            }
        
        # 保存评估结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_path = self.results_dir / f"evaluation_results_{timestamp}.json"
        
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"评估结果已保存到: {eval_path}")
        
        return evaluation_results
    
    def create_evaluation_plots(self, evaluation_results):
        """创建评估图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        sizes = list(evaluation_results.keys())
        makespans = [evaluation_results[size]['avg_makespan'] for size in sizes]
        makespan_stds = [evaluation_results[size]['std_makespan'] for size in sizes]
        resource_utils = [evaluation_results[size]['avg_resource_utilization'] for size in sizes]
        resource_std = [evaluation_results[size]['std_resource_utilization'] for size in sizes]
        
        # Makespan对比
        ax1.bar(sizes, makespans, yerr=makespan_stds, capsize=5, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax1.set_title('不同工作流大小的Makespan对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Makespan (秒)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 资源利用率对比
        ax2.bar(sizes, resource_utils, yerr=resource_std, capsize=5, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax2.set_title('不同工作流大小的资源利用率对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('资源利用率', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Makespan分布
        for i, size in enumerate(sizes):
            ax3.hist(evaluation_results[size]['makespans'], alpha=0.7, label=size, bins=20)
        ax3.set_title('Makespan分布', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Makespan (秒)', fontsize=12)
        ax3.set_ylabel('频次', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 资源利用率分布
        for i, size in enumerate(sizes):
            ax4.hist(evaluation_results[size]['resource_utilizations'], alpha=0.7, label=size, bins=20)
        ax4.set_title('资源利用率分布', fontsize=14, fontweight='bold')
        ax4.set_xlabel('资源利用率', fontsize=12)
        ax4.set_ylabel('频次', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.results_dir / f"evaluation_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"评估图表已保存到: {plot_path}")
    
    def evaluate_with_scheduler(self, data, process_id, scheduler):
        """使用指定调度器评估单个进程"""
        simulator = self.create_simulator_for_process(data, process_id)
        if simulator is None:
            return None, None, None
        
        simulator.reset()
        if hasattr(scheduler, 'reset'):
            scheduler.reset()
        
        episode_reward = 0.0
        step_count = 0
        max_steps = 1000
        
        while not simulator.is_done() and step_count < max_steps:
            state = simulator.get_state()
            if state is None:
                break
            
            task_features, resource_features = state
            graph_adj = simulator.get_graph_adj()
            
            # 使用调度器选择动作
            if hasattr(scheduler, 'select_action'):
                if isinstance(scheduler, FE_IDDQN):
                    action = scheduler.select_action(task_features, resource_features, graph_adj=graph_adj)
                else:
                    action = scheduler.select_action(task_features, resource_features, graph_adj=graph_adj)
            else:
                action = 0
            
            state, reward, done, info = simulator.step(action)
            episode_reward += reward
            step_count += 1
            
            if done:
                break
        
        return simulator.get_makespan(), simulator.get_resource_utilization(), episode_reward
    
    def compare_with_heuristics(self, data, test_data, trained_agent):
        """与启发式算法对比评估"""
        self.logger.info("="*80)
        self.logger.info("开始与启发式算法对比评估...")
        self.logger.info("="*80)
        
        # 创建启发式调度器
        heuristic_schedulers = [
            FIFOScheduler(num_resources=self.action_dim),
            RoundRobinScheduler(num_resources=self.action_dim),
            SJFScheduler(num_resources=self.action_dim),
            EFTScheduler(num_resources=self.action_dim),
            RandomScheduler(num_resources=self.action_dim),
        ]
        
        # 所有调度器（包括FE-IDDQN）
        all_schedulers = {
            'FE-IDDQN': trained_agent,
            **{s.name: s for s in heuristic_schedulers}
        }
        
        # 存储结果
        comparison_results = {name: {'makespans': [], 'utilizations': [], 'rewards': []} 
                            for name in all_schedulers.keys()}
        
        # 按工作流大小分组
        size_comparison = {size: {name: {'makespans': [], 'utilizations': []} 
                                  for name in all_schedulers.keys()}
                          for size in ['small', 'medium', 'large']}
        
        total_processes = len(test_data)
        self.logger.info(f"评估 {total_processes} 个测试工作流...")
        
        for idx, (_, process) in enumerate(test_data.iterrows()):
            process_id = process['process_id']
            workflow_size = process['workflow_size']
            
            if (idx + 1) % 10 == 0:
                self.logger.info(f"进度: {idx + 1}/{total_processes}")
            
            for name, scheduler in all_schedulers.items():
                makespan, utilization, reward = self.evaluate_with_scheduler(data, process_id, scheduler)
                
                if makespan is not None:
                    comparison_results[name]['makespans'].append(makespan)
                    comparison_results[name]['utilizations'].append(utilization)
                    comparison_results[name]['rewards'].append(reward)
                    
                    if workflow_size in size_comparison:
                        size_comparison[workflow_size][name]['makespans'].append(makespan)
                        size_comparison[workflow_size][name]['utilizations'].append(utilization)
        
        # 计算统计结果
        summary = {}
        for name in all_schedulers.keys():
            makespans = comparison_results[name]['makespans']
            utilizations = comparison_results[name]['utilizations']
            
            if len(makespans) > 0:
                summary[name] = {
                    'avg_makespan': np.mean(makespans),
                    'std_makespan': np.std(makespans),
                    'avg_utilization': np.mean(utilizations),
                    'std_utilization': np.std(utilizations),
                    'total_makespan': np.sum(makespans),
                    'count': len(makespans)
                }
        
        # 计算相对于FIFO的提升
        if 'FIFO' in summary:
            fifo_makespan = summary['FIFO']['avg_makespan']
            for name in summary:
                improvement = (fifo_makespan - summary[name]['avg_makespan']) / fifo_makespan * 100
                summary[name]['improvement_vs_fifo'] = improvement
        
        # 输出结果
        self.logger.info("\n" + "="*80)
        self.logger.info("算法对比结果 (测试集)")
        self.logger.info("="*80)
        
        # 按平均makespan排序
        sorted_algorithms = sorted(summary.items(), key=lambda x: x[1]['avg_makespan'])
        
        self.logger.info(f"{'算法':<15} {'平均Makespan':>15} {'标准差':>12} {'资源利用率':>12} {'vs FIFO提升':>12}")
        self.logger.info("-"*70)
        
        for name, stats in sorted_algorithms:
            improvement = stats.get('improvement_vs_fifo', 0)
            self.logger.info(f"{name:<15} {stats['avg_makespan']:>15.2f} {stats['std_makespan']:>12.2f} "
                           f"{stats['avg_utilization']:>11.2%} {improvement:>11.1f}%")
        
        # 按工作流大小输出结果
        self.logger.info("\n" + "="*80)
        self.logger.info("按工作流大小分组的结果")
        self.logger.info("="*80)
        
        for size in ['small', 'medium', 'large']:
            size_data = size_comparison[size]
            if not any(len(size_data[name]['makespans']) > 0 for name in size_data):
                continue
            
            self.logger.info(f"\n{size.upper()} 工作流:")
            self.logger.info(f"{'算法':<15} {'平均Makespan':>15} {'样本数':>10}")
            self.logger.info("-"*45)
            
            for name in all_schedulers.keys():
                makespans = size_data[name]['makespans']
                if len(makespans) > 0:
                    self.logger.info(f"{name:<15} {np.mean(makespans):>15.2f} {len(makespans):>10}")
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.results_dir / f"algorithm_comparison_{timestamp}.json"
        
        save_results = {
            'summary': summary,
            'size_comparison': {
                size: {name: {'avg_makespan': np.mean(data['makespans']) if len(data['makespans']) > 0 else 0,
                             'count': len(data['makespans'])}
                      for name, data in size_data.items()}
                for size, size_data in size_comparison.items()
            }
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"\n对比结果已保存到: {result_path}")
        
        # 创建对比图表
        self.create_comparison_plots(summary, size_comparison)
        
        return summary, size_comparison
    
    def create_comparison_plots(self, summary, size_comparison):
        """创建算法对比图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        algorithms = list(summary.keys())
        makespans = [summary[alg]['avg_makespan'] for alg in algorithms]
        makespan_stds = [summary[alg]['std_makespan'] for alg in algorithms]
        utilizations = [summary[alg]['avg_utilization'] for alg in algorithms]
        
        # 颜色设置 - FE-IDDQN用红色突出
        colors = ['#FF6B6B' if alg == 'FE-IDDQN' else '#4ECDC4' for alg in algorithms]
        
        # 1. Makespan对比
        ax1 = axes[0, 0]
        bars = ax1.bar(algorithms, makespans, yerr=makespan_stds, capsize=5, color=colors, alpha=0.8)
        ax1.set_title('算法Makespan对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('平均Makespan (秒)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上标注数值
        for bar, val in zip(bars, makespans):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 资源利用率对比
        ax2 = axes[0, 1]
        bars = ax2.bar(algorithms, utilizations, color=colors, alpha=0.8)
        ax2.set_title('算法资源利用率对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('平均资源利用率', fontsize=12)
        ax2.set_ylim(0, max(utilizations) * 1.2)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, utilizations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.2%}', ha='center', va='bottom', fontsize=9)
        
        # 3. 相对于FIFO的提升
        ax3 = axes[1, 0]
        improvements = [summary[alg].get('improvement_vs_fifo', 0) for alg in algorithms]
        colors_imp = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvements]
        bars = ax3.bar(algorithms, improvements, color=colors_imp, alpha=0.8)
        ax3.set_title('相对于FIFO的Makespan提升', fontsize=14, fontweight='bold')
        ax3.set_ylabel('提升百分比 (%)', fontsize=12)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, improvements):
            y_pos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 1.5
            ax3.text(bar.get_x() + bar.get_width()/2, y_pos, 
                    f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
        
        # 4. 按工作流大小的对比
        ax4 = axes[1, 1]
        sizes = ['small', 'medium', 'large']
        x = np.arange(len(sizes))
        width = 0.12
        
        for i, alg in enumerate(algorithms):
            alg_makespans = []
            for size in sizes:
                data = size_comparison[size][alg]['makespans']
                alg_makespans.append(np.mean(data) if len(data) > 0 else 0)
            
            color = '#FF6B6B' if alg == 'FE-IDDQN' else plt.cm.Set2(i / len(algorithms))
            ax4.bar(x + i * width, alg_makespans, width, label=alg, alpha=0.8, color=color)
        
        ax4.set_title('按工作流大小的Makespan对比', fontsize=14, fontweight='bold')
        ax4.set_ylabel('平均Makespan (秒)', fontsize=12)
        ax4.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax4.set_xticklabels(['小(1-10任务)', '中(11-30任务)', '大(31+任务)'])
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.results_dir / f"algorithm_comparison_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"对比图表已保存到: {plot_path}")
    
    def load_model(self, model_path):
        """加载保存的模型"""
        self.logger.info(f"加载模型: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_state = pickle.load(f)
        
        # 创建智能体
        agent = FE_IDDQN(
            self.task_input_dim,
            self.resource_input_dim,
            self.action_dim,
            max_tasks=5,
            max_resources=5,
            enable_graph_encoder=True,
        )
        
        # 加载权重
        agent.q_network.load_state_dict(model_state['q_network_state'])
        agent.target_network.load_state_dict(model_state['target_network_state'])
        
        self.logger.info("模型加载成功")
        return agent
    
    def run_training_pipeline(self, n_epochs=30, skip_training=False, model_path=None):
        """运行完整的训练流程"""
        try:
            # 1. 加载和划分数据
            data, train_data, val_data, test_data, process_df = self.load_and_split_data()
            
            if skip_training and model_path:
                # 直接加载模型
                agent = self.load_model(model_path)
            else:
                # 2. 训练模型
                agent = self.train_model(data, train_data, val_data, n_epochs=n_epochs)
            
            # 3. 评估FE-IDDQN模型
            self.logger.info("\n" + "="*80)
            self.logger.info("FE-IDDQN模型单独评估")
            self.logger.info("="*80)
            evaluation_results = self.evaluate_model(data, test_data, agent)
            
            # 4. 创建评估图表
            self.create_evaluation_plots(evaluation_results)
            
            # 5. 与启发式算法对比
            comparison_summary, size_comparison = self.compare_with_heuristics(data, test_data, agent)
            
            # 6. 输出最终总结
            self.logger.info("\n" + "="*80)
            self.logger.info("最终总结")
            self.logger.info("="*80)
            
            # FE-IDDQN vs 最佳启发式算法
            fe_iddqn_makespan = comparison_summary['FE-IDDQN']['avg_makespan']
            best_heuristic = min(
                [(name, stats['avg_makespan']) for name, stats in comparison_summary.items() if name != 'FE-IDDQN'],
                key=lambda x: x[1]
            )
            
            self.logger.info(f"FE-IDDQN 平均Makespan: {fe_iddqn_makespan:.2f}秒")
            self.logger.info(f"最佳启发式算法 ({best_heuristic[0]}): {best_heuristic[1]:.2f}秒")
            
            improvement = (best_heuristic[1] - fe_iddqn_makespan) / best_heuristic[1] * 100
            if improvement > 0:
                self.logger.info(f"FE-IDDQN 相比最佳启发式算法提升: {improvement:.2f}%")
            else:
                self.logger.info(f"FE-IDDQN 相比最佳启发式算法落后: {-improvement:.2f}%")
            
            self.logger.info(f"\n所有结果已保存到: {self.output_dir}")
            
            return agent, comparison_summary
            
        except Exception as e:
            self.logger.error(f"训练流程失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    """主函数"""
    print("="*80)
    print("FE-IDDQN训练和评估系统 (含启发式算法对比)")
    print("="*80)
    
    import argparse
    parser = argparse.ArgumentParser(description='FE-IDDQN训练和评估')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--skip-training', action='store_true', help='跳过训练，直接加载模型')
    parser.add_argument('--model-path', type=str, default=None, help='模型路径')
    parser.add_argument('--output-dir', type=str, default='fe_iddqn_training_system', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = FEIDDQNTrainer(output_dir=args.output_dir)
    
    # 运行训练流程
    trainer.run_training_pipeline(
        n_epochs=args.epochs,
        skip_training=args.skip_training,
        model_path=args.model_path
    )


if __name__ == "__main__":
    main()
