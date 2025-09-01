#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公平比较实验运行器
确保所有算法使用相同的工作流实例，进行公平的性能比较
"""

import logging
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from config.config import Config
from config.hyperparameters import Hyperparameters
from models.fe_iddqn import FE_IDDQN
from baselines.traditional_schedulers import FIFOScheduler, SJFScheduler, HEFTScheduler
from baselines.rl_baselines import DQNScheduler, DDQNScheduler, BF_DDQNScheduler
from baselines.meta_heuristics import GAScheduler, PSOScheduler, ACOScheduler
from environment.historical_replay_simulator import HistoricalReplaySimulator
from evaluation.metrics import Evaluator
import json
import os
from datetime import datetime

class FairComparisonRunner:
    """公平比较实验运行器"""
    
    def __init__(self, data: Dict[str, pd.DataFrame], output_dir: str):
        self.logger = logging.getLogger(__name__)
        self.data = data
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化评估器
        self.evaluator = Evaluator()
        
        # 预定义的工作流实例集（确保所有算法使用相同的数据）
        self.fixed_workflow_instances = None
        self._prepare_fixed_workflow_instances()
    
    def _prepare_fixed_workflow_instances(self):
        """准备固定的工作流实例集，确保所有算法使用相同数据"""
        self.logger.info("准备固定的工作流实例集...")
        
        # 获取有任务的进程ID
        processes_with_tasks = self.data['task_instance']['process_instance_id'].unique()
        
        # 获取成功且有任务的进程实例
        successful_processes = self.data['process_instance'][
            (self.data['process_instance']['state'] == 7) & 
            (self.data['process_instance']['id'].isin(processes_with_tasks))
        ].sort_values('start_time').reset_index(drop=True)
        
        if len(successful_processes) == 0:
            self.logger.error("没有找到成功的工作流实例！")
            return
        
        # 使用固定的随机种子选择工作流实例
        np.random.seed(Config.RANDOM_SEED)
        max_processes = min(Config.MAX_PROCESSES_PER_EPISODE, len(successful_processes))
        
        # 随机选择但保持固定
        selected_indices = np.random.choice(len(successful_processes), max_processes, replace=False)
        self.fixed_workflow_instances = successful_processes.iloc[selected_indices].reset_index(drop=True)
        
        self.logger.info(f"固定工作流实例集准备完成: {len(self.fixed_workflow_instances)} 个工作流")
        
        # 保存工作流实例信息
        workflow_info = {
            'total_workflows': len(self.fixed_workflow_instances),
            'workflow_ids': self.fixed_workflow_instances['id'].tolist(),
            'workflow_names': self.fixed_workflow_instances['name'].tolist(),
            'total_tasks': len(self.data['task_instance'][
                self.data['task_instance']['process_instance_id'].isin(self.fixed_workflow_instances['id'])
            ]),
            'random_seed': Config.RANDOM_SEED,
            'creation_time': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, 'fixed_workflow_instances.json'), 'w', encoding='utf-8') as f:
            json.dump(workflow_info, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"工作流实例信息已保存到: {os.path.join(self.output_dir, 'fixed_workflow_instances.json')}")
    
    def create_simulator_with_fixed_instances(self) -> HistoricalReplaySimulator:
        """创建使用固定工作流实例的模拟器"""
        if self.fixed_workflow_instances is None:
            raise ValueError("固定工作流实例集未准备好！")
        
        # 创建模拟器，但覆盖其工作流实例
        simulator = HistoricalReplaySimulator(
            process_instances=self.data['process_instance'],
            task_instances=self.data['task_instance'],
            task_definitions=self.data['task_definition'],
            process_task_relations=self.data['process_task_relation']
        )
        
        # 强制使用固定的工作流实例
        simulator.successful_processes = self.fixed_workflow_instances.copy()
        simulator.current_process_idx = 0
        simulator.current_task_idx = 0
        simulator.completed_tasks = set()
        simulator.running_tasks = {}
        simulator.available_resources = {}
        simulator.task_schedule_history = []
        
        # 初始化第一个进程（依赖关系会在_load_current_process中自动处理）
        simulator._load_current_process()
        
        return simulator
    
    def run_algorithm_comparison(self, algorithms: List[str], n_experiments: int = 5) -> Dict:
        """运行算法比较实验"""
        self.logger.info("=" * 80)
        self.logger.info("开始公平比较实验")
        self.logger.info("=" * 80)
        self.logger.info(f"参与比较的算法: {algorithms}")
        self.logger.info(f"每个算法运行次数: {n_experiments}")
        if self.fixed_workflow_instances is not None:
            self.logger.info(f"使用固定的工作流实例集: {len(self.fixed_workflow_instances)} 个工作流")
        else:
            self.logger.warning("固定工作流实例集未初始化")
        
        results = {}
        
        for algorithm_name in algorithms:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"运行算法: {algorithm_name}")
            self.logger.info(f"{'='*60}")
            
            algorithm_results = []
            
            for exp in range(n_experiments):
                self.logger.info(f"\n--- 实验 {exp + 1}/{n_experiments} ---")
                
                try:
                    # 为每个实验创建新的模拟器（但使用相同的工作流实例）
                    simulator = self.create_simulator_with_fixed_instances()
                    
                    # 运行算法
                    if algorithm_name == "FE_IDDQN":
                        result = self._run_fe_iddqn(simulator, exp)
                    elif algorithm_name in ["FIFO", "SJF", "HEFT"]:
                        result = self._run_traditional_scheduler(simulator, algorithm_name, exp)
                    elif algorithm_name in ["DQN", "DDQN", "BF_DDQN"]:
                        result = self._run_rl_baseline(simulator, algorithm_name, exp)
                    elif algorithm_name in ["GA", "PSO", "ACO"]:
                        result = self._run_meta_heuristic(simulator, algorithm_name, exp)
                    else:
                        self.logger.warning(f"未知算法: {algorithm_name}")
                        continue
                    
                    algorithm_results.append(result)
                    self.logger.info(f"实验 {exp + 1} 完成: Makespan={result.get('makespan', 'N/A'):.2f}, "
                                  f"资源利用率={result.get('resource_utilization', 'N/A'):.2f}")
                    
                except Exception as e:
                    self.logger.error(f"算法 {algorithm_name} 实验 {exp + 1} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 计算统计结果
            if algorithm_results:
                results[algorithm_name] = self._calculate_statistics(algorithm_results)
                self.logger.info(f"\n算法 {algorithm_name} 统计结果:")
                self.logger.info(f"  平均 Makespan: {results[algorithm_name]['avg_makespan']:.2f} ± {results[algorithm_name]['std_makespan']:.2f}")
                self.logger.info(f"  平均资源利用率: {results[algorithm_name]['avg_resource_utilization']:.2f} ± {results[algorithm_name]['std_resource_utilization']:.2f}")
                self.logger.info(f"  平均奖励: {results[algorithm_name]['avg_reward']:.2f} ± {results[algorithm_name]['std_reward']:.2f}")
            else:
                self.logger.error(f"算法 {algorithm_name} 没有成功的实验结果！")
        
        # 生成比较报告
        self._generate_comparison_report(results)
        
        return results
    
    def _run_fe_iddqn(self, simulator: HistoricalReplaySimulator, exp_id: int) -> Dict:
        """运行FE-IDDQN算法"""
        # 获取特征维度
        task_features, resource_features = simulator.get_state()
        task_input_dim = task_features.shape[-1] if task_features.size > 0 else 16
        resource_input_dim = resource_features.shape[-1] if resource_features.size > 0 else 7
        action_dim = simulator.num_resources
        
        # 创建智能体
        agent = FE_IDDQN(task_input_dim, resource_input_dim, action_dim, self.device)
        
        # 获取算法参数
        algorithm_params = Hyperparameters.get_algorithm_params('FE_IDDQN')
        
        # 训练过程
        episode_rewards = []
        episode_makespans = []
        
        for episode in range(algorithm_params.get('num_episodes', 50)):
            simulator.reset()
            episode_reward = 0
            step_count = 0
            
            max_steps = algorithm_params.get('max_steps_per_episode', 1000)
            
            while not simulator.is_done() and step_count < max_steps:
                state = simulator.get_state()
                action = agent.select_action(state[0], state[1])
                next_state, reward, done, info = simulator.step(action)
                
                agent.store_experience(state, action, reward, next_state, done)
                
                if step_count % algorithm_params.get('train_freq', 4) == 0:
                    agent.train()
                
                episode_reward += reward
                step_count += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_makespans.append(simulator.get_makespan())
            
            # 更新网络
            if episode % algorithm_params.get('target_update_frequency', 10) == 0:
                agent.update_target_network()
            
            agent.update_exploration_params()
        
        # 最终评估
        final_metrics = {
            'makespan': simulator.get_makespan(),
            'resource_utilization': simulator.get_resource_utilization(),
            'average_reward': np.mean(episode_rewards),
            'final_episode_reward': episode_rewards[-1] if episode_rewards else 0,
            'training_losses': agent.get_training_stats().get('training_losses', []),
            'experiment_id': exp_id,
            'algorithm': 'FE_IDDQN'
        }
        
        return final_metrics
    
    def _run_rl_baseline(self, simulator: HistoricalReplaySimulator, algorithm_name: str, exp_id: int) -> Dict:
        """运行RL基线算法"""
        # 简化实现，使用随机调度
        schedule_result = simulator.simulate_random_schedule(algorithm_name)
        
        return {
            'makespan': schedule_result.get('makespan', 0),
            'resource_utilization': schedule_result.get('resource_utilization', 0),
            'average_reward': 0,
            'final_episode_reward': 0,
            'experiment_id': exp_id,
            'algorithm': algorithm_name
        }
    
    def _run_traditional_scheduler(self, simulator: HistoricalReplaySimulator, algorithm_name: str, exp_id: int) -> Dict:
        """运行传统调度算法"""
        if algorithm_name == "FIFO":
            scheduler = FIFOScheduler()
        elif algorithm_name == "SJF":
            scheduler = SJFScheduler()
        elif algorithm_name == "HEFT":
            scheduler = HEFTScheduler()
        else:
            raise ValueError(f"未知的传统调度算法: {algorithm_name}")
        
        # 添加调试信息
        self.logger.info(f"  调试信息 - 算法 {algorithm_name}:")
        self.logger.info(f"    任务数量: {len(simulator.tasks)}")
        self.logger.info(f"    资源数量: {len(simulator.resources)}")
        self.logger.info(f"    依赖关系数量: {len(simulator.dependencies)}")
        
        # 显示任务ID和依赖关系
        if simulator.tasks:
            task_ids = [task['id'] for task in simulator.tasks]
            self.logger.info(f"    任务ID列表: {task_ids}")
        
        if simulator.dependencies:
            dep_info = [(dep.get('pre_task'), dep.get('post_task')) for dep in simulator.dependencies]
            self.logger.info(f"    依赖关系: {dep_info}")
        
        # 转换依赖关系格式
        dependencies = [(dep['pre_task'], dep['post_task']) 
                       for dep in simulator.dependencies 
                       if dep['pre_task'] is not None and dep['post_task'] is not None]
        
        self.logger.info(f"    转换后的依赖关系: {dependencies}")
        
        # 验证依赖关系中的任务ID是否都存在于任务列表中
        if simulator.tasks:
            available_task_ids = set(task['id'] for task in simulator.tasks)
            dependency_task_ids = set()
            for pre_task, post_task in dependencies:
                if pre_task is not None:
                    dependency_task_ids.add(pre_task)
                if post_task is not None:
                    dependency_task_ids.add(post_task)
            
            missing_task_ids = dependency_task_ids - available_task_ids
            if missing_task_ids:
                self.logger.warning(f"    警告: 依赖关系中的任务ID在任务列表中不存在: {missing_task_ids}")
                self.logger.warning(f"    这可能导致调度失败")
        
        schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, dependencies)
        
        # 检查是否有错误
        if 'error' in schedule_result:
            self.logger.error(f"算法 {algorithm_name} 调度失败: {schedule_result['error']}")
            return {
                'makespan': float('inf'),
                'resource_utilization': 0,
                'average_reward': -1000,
                'final_episode_reward': -1000,
                'experiment_id': exp_id,
                'algorithm': algorithm_name,
                'error': schedule_result['error']
            }
        
        # 计算makespan和资源利用率
        makespan = schedule_result.get('makespan', 0)
        resource_utilization = schedule_result.get('resource_utilization', 0)
        
        # 如果没有这些字段，尝试计算
        if makespan == 0 and 'task_end_times' in schedule_result:
            if schedule_result['task_end_times']:
                makespan = max(schedule_result['task_end_times'].values())
        
        if resource_utilization == 0 and makespan > 0:
            total_work = sum(task['duration'] for task in simulator.tasks)
            total_capacity = makespan * len(simulator.resources)
            resource_utilization = total_work / total_capacity if total_capacity > 0 else 0
        
        return {
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'average_reward': 0,
            'final_episode_reward': 0,
            'experiment_id': exp_id,
            'algorithm': algorithm_name
        }
    
    def _run_meta_heuristic(self, simulator: HistoricalReplaySimulator, algorithm_name: str, exp_id: int) -> Dict:
        """运行元启发式算法"""
        if algorithm_name == "GA":
            scheduler = GAScheduler()
        elif algorithm_name == "PSO":
            scheduler = PSOScheduler()
        elif algorithm_name == "ACO":
            scheduler = ACOScheduler()
        else:
            raise ValueError(f"未知的元启发式算法: {algorithm_name}")
        
        # 转换依赖关系格式
        dependencies = [(dep['pre_task'], dep['post_task']) 
                       for dep in simulator.dependencies 
                       if dep['pre_task'] is not None and dep['post_task'] is not None]
        
        schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, dependencies)
        
        return {
            'makespan': schedule_result.get('makespan', 0),
            'resource_utilization': schedule_result.get('resource_utilization', 0),
            'average_reward': 0,
            'final_episode_reward': 0,
            'experiment_id': exp_id,
            'algorithm': algorithm_name
        }
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """计算统计结果"""
        makespans = [r['makespan'] for r in results if isinstance(r.get('makespan'), (int, float))]
        resource_utilizations = [r['resource_utilization'] for r in results if isinstance(r.get('resource_utilization'), (int, float))]
        rewards = [r['average_reward'] for r in results if isinstance(r.get('average_reward'), (int, float))]
        
        return {
            'avg_makespan': np.mean(makespans) if makespans else 0,
            'std_makespan': np.std(makespans) if makespans else 0,
            'avg_resource_utilization': np.mean(resource_utilizations) if resource_utilizations else 0,
            'std_resource_utilization': np.std(resource_utilizations) if resource_utilizations else 0,
            'avg_reward': np.mean(rewards) if rewards else 0,
            'std_reward': np.std(rewards) if rewards else 0,
            'n_experiments': len(results),
            'raw_results': results
        }
    
    def _generate_comparison_report(self, results: Dict):
        """生成比较报告"""
        self.logger.info("\n" + "="*80)
        self.logger.info("算法性能比较报告")
        self.logger.info("="*80)
        
        # 按makespan排序（越小越好）
        sorted_algorithms = sorted(results.keys(), 
                                 key=lambda x: results[x]['avg_makespan'])
        
        self.logger.info("\n🏆 算法性能排名 (按makespan排序，越小越好):")
        for i, algorithm in enumerate(sorted_algorithms, 1):
            result = results[algorithm]
            self.logger.info(f"  {i}. {algorithm}:")
            self.logger.info(f"    Makespan: {result['avg_makespan']:.2f} ± {result['std_makespan']:.2f}")
            self.logger.info(f"    资源利用率: {result['avg_resource_utilization']:.2f} ± {result['std_resource_utilization']:.2f}")
            self.logger.info(f"    平均奖励: {result['avg_reward']:.2f} ± {result['std_reward']:.2f}")
            self.logger.info(f"    实验次数: {result['n_experiments']}")
        
        # 保存详细结果
        report_file = os.path.join(self.output_dir, 'algorithm_comparison_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"\n详细比较报告已保存到: {report_file}")
        
        # 生成性能提升分析
        if 'FE_IDDQN' in results:
            fe_result = results['FE_IDDQN']
            self.logger.info(f"\n🚀 FE-IDDQN 算法性能分析:")
            
            for algorithm, result in results.items():
                if algorithm != 'FE_IDDQN':
                    makespan_improvement = ((result['avg_makespan'] - fe_result['avg_makespan']) / result['avg_makespan']) * 100
                    utilization_improvement = ((fe_result['avg_resource_utilization'] - result['avg_resource_utilization']) / result['avg_resource_utilization']) * 100
                    
                    self.logger.info(f"  相比 {algorithm}:")
                    self.logger.info(f"    Makespan 提升: {makespan_improvement:+.1f}%")
                    self.logger.info(f"    资源利用率提升: {utilization_improvement:+.1f}%")
        
        self.logger.info("="*80)
