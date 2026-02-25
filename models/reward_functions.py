# -*- coding: utf-8 -*-
"""
增强版奖励函数模块 - 多目标奖励设计
包含关键路径奖励、并行度奖励、负载均衡、等待时间惩罚等
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """奖励函数配置"""
    # 基础权重
    makespan_weight: float = 0.35
    resource_utilization_weight: float = 0.20
    load_balance_weight: float = 0.15
    parallelism_weight: float = 0.15
    critical_path_weight: float = 0.10
    waiting_time_weight: float = 0.05
    
    # 惩罚权重
    dependency_violation_penalty: float = -100.0
    resource_overflow_penalty: float = -50.0
    idle_time_penalty: float = -5.0
    
    # 奖励缩放
    reward_scale: float = 1.0
    reward_clip: Tuple[float, float] = (-100.0, 100.0)
    
    # 归一化参数
    normalize_rewards: bool = True
    
    # 稀疏奖励选项
    use_sparse_reward: bool = False
    sparse_reward_threshold: float = 0.8  # makespan改善阈值


class EnhancedRewardCalculator:
    """增强版奖励计算器"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
        # 统计信息用于归一化
        self.makespan_history = []
        self.utilization_history = []
        self.reward_history = []
        
        # 基准值
        self.baseline_makespan = None
        self.optimal_makespan_estimate = None
        
    def set_baseline(self, baseline_makespan: float, 
                    optimal_estimate: Optional[float] = None):
        """设置基准makespan（如使用HEFT等启发式算法的结果）"""
        self.baseline_makespan = baseline_makespan
        self.optimal_makespan_estimate = optimal_estimate or baseline_makespan * 0.8
        
    def calculate_reward(self, 
                        task: Dict[str, Any],
                        resource: Dict[str, Any],
                        start_time: float,
                        end_time: float,
                        scheduler_state: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        计算综合奖励
        
        Args:
            task: 当前调度的任务信息
            resource: 分配的资源信息
            start_time: 任务开始时间
            end_time: 任务结束时间
            scheduler_state: 调度器状态（包含DAG、已完成任务等）
            
        Returns:
            总奖励值和各组件奖励的字典
        """
        reward_components = {}
        
        # 1. Makespan改善奖励
        makespan_reward = self._calculate_makespan_reward(
            end_time, scheduler_state
        )
        reward_components['makespan'] = makespan_reward
        
        # 2. 资源利用率奖励
        utilization_reward = self._calculate_utilization_reward(
            task, resource
        )
        reward_components['utilization'] = utilization_reward
        
        # 3. 负载均衡奖励
        load_balance_reward = self._calculate_load_balance_reward(
            scheduler_state
        )
        reward_components['load_balance'] = load_balance_reward
        
        # 4. 并行度奖励
        parallelism_reward = self._calculate_parallelism_reward(
            start_time, end_time, scheduler_state
        )
        reward_components['parallelism'] = parallelism_reward
        
        # 5. 关键路径奖励
        critical_path_reward = self._calculate_critical_path_reward(
            task, scheduler_state
        )
        reward_components['critical_path'] = critical_path_reward
        
        # 6. 等待时间惩罚
        waiting_penalty = self._calculate_waiting_time_penalty(
            task, start_time, scheduler_state
        )
        reward_components['waiting_penalty'] = waiting_penalty
        
        # 7. 依赖违规惩罚
        dependency_penalty = self._calculate_dependency_penalty(
            task, start_time, scheduler_state
        )
        reward_components['dependency_penalty'] = dependency_penalty
        
        # 计算加权总奖励
        total_reward = (
            self.config.makespan_weight * makespan_reward +
            self.config.resource_utilization_weight * utilization_reward +
            self.config.load_balance_weight * load_balance_reward +
            self.config.parallelism_weight * parallelism_reward +
            self.config.critical_path_weight * critical_path_reward +
            self.config.waiting_time_weight * waiting_penalty +
            dependency_penalty  # 依赖违规不加权，直接惩罚
        )
        
        # 奖励缩放
        total_reward *= self.config.reward_scale
        
        # 奖励裁剪
        total_reward = np.clip(
            total_reward, 
            self.config.reward_clip[0], 
            self.config.reward_clip[1]
        )
        
        # 更新历史
        self.reward_history.append(total_reward)
        
        return total_reward, reward_components
    
    def _calculate_makespan_reward(self, end_time: float, 
                                   scheduler_state: Dict) -> float:
        """计算makespan改善奖励"""
        current_makespan = scheduler_state.get('current_makespan', end_time)
        total_tasks = scheduler_state.get('total_tasks', 1)
        completed_tasks = scheduler_state.get('completed_tasks', 0)
        
        # 进度奖励：每完成一个任务的基础奖励
        progress_reward = 10.0
        
        # 时间效率奖励
        if self.baseline_makespan is not None and self.baseline_makespan > 0:
            # 相对于基准的改善
            improvement = (self.baseline_makespan - current_makespan) / self.baseline_makespan
            improvement_reward = improvement * 50.0
        else:
            # 如果没有基准，使用时间惩罚
            improvement_reward = -end_time * 0.01
        
        # 完成奖励（所有任务完成时的额外奖励）
        completion_bonus = 0.0
        if completed_tasks == total_tasks:
            if self.baseline_makespan is not None:
                # 优于基准给予额外奖励
                if current_makespan < self.baseline_makespan:
                    completion_bonus = 50.0 * (1 - current_makespan / self.baseline_makespan)
            else:
                completion_bonus = 20.0
        
        return progress_reward + improvement_reward + completion_bonus
    
    def _calculate_utilization_reward(self, task: Dict, resource: Dict) -> float:
        """计算资源利用率奖励"""
        cpu_capacity = resource.get('cpu_capacity', 1)
        memory_capacity = resource.get('memory_capacity', 1)
        cpu_req = task.get('cpu_req', 0)
        memory_req = task.get('memory_req', 0)
        
        # 资源匹配度（避免过度浪费）
        cpu_utilization = cpu_req / cpu_capacity if cpu_capacity > 0 else 0
        memory_utilization = memory_req / memory_capacity if memory_capacity > 0 else 0
        
        # 理想利用率在0.6-0.9之间
        def utilization_score(util):
            if util < 0.3:
                return util * 2  # 利用率太低，惩罚
            elif util <= 0.9:
                return 1.0  # 理想范围
            else:
                return 1.0 - (util - 0.9) * 2  # 过度使用，惩罚
        
        cpu_score = utilization_score(cpu_utilization)
        memory_score = utilization_score(memory_utilization)
        
        return (cpu_score + memory_score) * 10.0
    
    def _calculate_load_balance_reward(self, scheduler_state: Dict) -> float:
        """计算负载均衡奖励"""
        resource_loads = scheduler_state.get('resource_loads', [])
        
        if not resource_loads or len(resource_loads) < 2:
            return 0.0
        
        # 计算负载标准差
        loads = np.array(resource_loads)
        mean_load = np.mean(loads)
        
        if mean_load == 0:
            return 10.0  # 空载状态
        
        # 使用变异系数（CV）评估负载均衡
        cv = np.std(loads) / mean_load
        
        # CV越小越均衡，给予更高奖励
        balance_score = max(0, 1 - cv) * 20.0
        
        return balance_score
    
    def _calculate_parallelism_reward(self, start_time: float, end_time: float,
                                      scheduler_state: Dict) -> float:
        """计算并行度奖励"""
        concurrent_tasks = scheduler_state.get('concurrent_tasks_at_time', {})
        num_resources = scheduler_state.get('num_resources', 1)
        
        # 计算当前时间段的并行任务数
        current_parallelism = 0
        for t in range(int(start_time), int(end_time) + 1):
            current_parallelism = max(
                current_parallelism, 
                concurrent_tasks.get(t, 0)
            )
        
        # 并行度评分：实际并行数 / 资源数
        parallelism_ratio = current_parallelism / num_resources if num_resources > 0 else 0
        
        # 鼓励高并行度
        parallelism_reward = parallelism_ratio * 15.0
        
        return parallelism_reward
    
    def _calculate_critical_path_reward(self, task: Dict, 
                                        scheduler_state: Dict) -> float:
        """计算关键路径奖励"""
        is_critical = task.get('is_critical_path', False)
        criticality_score = task.get('criticality_score', 0.5)
        
        if is_critical:
            # 关键路径任务得到优先调度给予奖励
            return 20.0 * criticality_score
        else:
            # 非关键路径任务正常奖励
            return 5.0 * criticality_score
    
    def _calculate_waiting_time_penalty(self, task: Dict, start_time: float,
                                        scheduler_state: Dict) -> float:
        """计算等待时间惩罚"""
        # 任务最早可开始时间（依赖完成后）
        earliest_start = task.get('earliest_start_time', 0)
        
        # 等待时间
        waiting_time = max(0, start_time - earliest_start)
        
        # 等待时间惩罚
        waiting_penalty = -waiting_time * 0.5
        
        return waiting_penalty
    
    def _calculate_dependency_penalty(self, task: Dict, start_time: float,
                                      scheduler_state: Dict) -> float:
        """计算依赖违规惩罚"""
        dependencies = task.get('dependencies', [])
        task_end_times = scheduler_state.get('task_end_times', {})
        
        penalty = 0.0
        for dep_task_id in dependencies:
            if dep_task_id in task_end_times:
                dep_end_time = task_end_times[dep_task_id]
                if start_time < dep_end_time:
                    # 依赖违规：严重惩罚
                    penalty += self.config.dependency_violation_penalty
        
        return penalty
    
    def calculate_episode_reward(self, episode_stats: Dict) -> Tuple[float, Dict]:
        """计算整个episode的奖励（用于稀疏奖励设置）"""
        final_makespan = episode_stats.get('makespan', float('inf'))
        total_utilization = episode_stats.get('avg_utilization', 0)
        load_balance_score = episode_stats.get('load_balance', 0)
        
        reward_components = {}
        
        # Makespan奖励
        if self.baseline_makespan is not None:
            makespan_improvement = (self.baseline_makespan - final_makespan) / self.baseline_makespan
            makespan_reward = makespan_improvement * 100.0
        else:
            makespan_reward = -final_makespan * 0.1
        reward_components['makespan'] = makespan_reward
        
        # 利用率奖励
        utilization_reward = total_utilization * 50.0
        reward_components['utilization'] = utilization_reward
        
        # 负载均衡奖励
        balance_reward = load_balance_score * 30.0
        reward_components['load_balance'] = balance_reward
        
        total_reward = makespan_reward + utilization_reward + balance_reward
        
        return total_reward, reward_components


class AdaptiveRewardShaper:
    """自适应奖励塑形器"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.potential_values = {}
        self.gamma = 0.99
        
    def calculate_potential(self, state: Dict) -> float:
        """计算状态势函数值"""
        completed_tasks = state.get('completed_tasks', 0)
        total_tasks = state.get('total_tasks', 1)
        current_makespan = state.get('current_makespan', 0)
        
        # 进度势
        progress = completed_tasks / total_tasks
        progress_potential = progress * 100
        
        # makespan势（越小越好）
        if current_makespan > 0:
            makespan_potential = -current_makespan * 0.1
        else:
            makespan_potential = 0
        
        return progress_potential + makespan_potential
    
    def shape_reward(self, reward: float, 
                    current_state: Dict, 
                    next_state: Dict) -> float:
        """
        基于势函数的奖励塑形
        F(s, s') = gamma * Phi(s') - Phi(s)
        """
        current_potential = self.calculate_potential(current_state)
        next_potential = self.calculate_potential(next_state)
        
        shaping = self.gamma * next_potential - current_potential
        shaped_reward = reward + shaping
        
        return shaped_reward


class CurriculumRewardScheduler:
    """课程学习奖励调度器"""
    
    def __init__(self, total_episodes: int, 
                 initial_config: Optional[RewardConfig] = None):
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.config = initial_config or RewardConfig()
        
        # 课程阶段配置
        self.curriculum_stages = [
            # 阶段1: 关注完成任务（简单目标）
            {
                'makespan_weight': 0.2,
                'resource_utilization_weight': 0.1,
                'load_balance_weight': 0.1,
                'parallelism_weight': 0.1,
                'critical_path_weight': 0.1,
                'waiting_time_weight': 0.4,  # 高权重，鼓励快速完成
            },
            # 阶段2: 关注效率
            {
                'makespan_weight': 0.3,
                'resource_utilization_weight': 0.2,
                'load_balance_weight': 0.15,
                'parallelism_weight': 0.15,
                'critical_path_weight': 0.1,
                'waiting_time_weight': 0.1,
            },
            # 阶段3: 全面优化
            {
                'makespan_weight': 0.35,
                'resource_utilization_weight': 0.20,
                'load_balance_weight': 0.15,
                'parallelism_weight': 0.15,
                'critical_path_weight': 0.10,
                'waiting_time_weight': 0.05,
            }
        ]
        
    def get_current_config(self) -> RewardConfig:
        """获取当前阶段的奖励配置"""
        progress = self.current_episode / self.total_episodes
        
        if progress < 0.3:
            stage_config = self.curriculum_stages[0]
        elif progress < 0.7:
            stage_config = self.curriculum_stages[1]
        else:
            stage_config = self.curriculum_stages[2]
        
        # 更新配置
        for key, value in stage_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        return self.config
    
    def step(self):
        """更新episode计数"""
        self.current_episode += 1
