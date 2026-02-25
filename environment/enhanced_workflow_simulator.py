# -*- coding: utf-8 -*-
"""
增强版工作流模拟器 - 支持DAG感知调度和关键路径优先
"""

import numpy as np
import networkx as nx
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

from data.enhanced_state_encoder import EnhancedStateEncoder, CriticalPathAnalyzer
from models.reward_functions import EnhancedRewardCalculator, RewardConfig


@dataclass
class SchedulingEvent:
    """调度事件"""
    time: float
    event_type: str  # 'task_start', 'task_end', 'resource_free'
    task_id: Optional[int] = None
    resource_id: Optional[int] = None
    
    def __lt__(self, other):
        return self.time < other.time


@dataclass
class TaskState:
    """任务状态"""
    id: int
    status: str  # 'pending', 'ready', 'running', 'completed', 'failed'
    assigned_resource: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    actual_duration: Optional[float] = None


@dataclass
class ResourceState:
    """资源状态"""
    id: int
    status: str  # 'idle', 'busy'
    current_task: Optional[int] = None
    available_time: float = 0.0
    task_history: List[int] = field(default_factory=list)
    total_busy_time: float = 0.0


class EnhancedWorkflowSimulator:
    """增强版工作流调度仿真环境"""
    
    def __init__(self, tasks: List[Dict], resources: List[Dict],
                 dependencies: List[Tuple[int, int]],
                 use_dag_aware: bool = True,
                 use_critical_path_priority: bool = True):
        """
        初始化仿真环境
        
        Args:
            tasks: 任务列表
            resources: 资源列表
            dependencies: 依赖关系 [(前驱任务, 后继任务), ...]
            use_dag_aware: 是否使用DAG感知调度
            use_critical_path_priority: 是否使用关键路径优先
        """
        self.logger = logging.getLogger(__name__)
        
        self.tasks = tasks
        self.resources = resources
        self.dependencies = dependencies
        self.num_tasks = len(tasks)
        self.num_resources = len(resources)
        
        self.use_dag_aware = use_dag_aware
        self.use_critical_path_priority = use_critical_path_priority
        
        # 构建DAG
        self._build_dag()
        
        # 初始化组件
        self.state_encoder = EnhancedStateEncoder()
        self.critical_path_analyzer = CriticalPathAnalyzer()
        self.reward_calculator = EnhancedRewardCalculator(RewardConfig())
        
        # 预计算关键路径信息
        if use_critical_path_priority:
            self._compute_critical_path_info()
        
        # 初始化状态
        self.reset()
    
    def _build_dag(self):
        """构建DAG图"""
        self.dag = nx.DiGraph()
        
        # 添加节点
        for task in self.tasks:
            self.dag.add_node(task['id'], **task)
        
        # 添加边
        for pre_task, post_task in self.dependencies:
            if pre_task in self.dag and post_task in self.dag:
                self.dag.add_edge(pre_task, post_task)
        
        # 验证是否为DAG
        if not nx.is_directed_acyclic_graph(self.dag):
            self.logger.warning("Graph contains cycles! Attempting to remove cycles...")
            # 尝试移除环
            try:
                cycles = list(nx.simple_cycles(self.dag))
                for cycle in cycles:
                    self.dag.remove_edge(cycle[-1], cycle[0])
            except:
                pass
    
    def _compute_critical_path_info(self):
        """预计算关键路径信息"""
        task_durations = {t['id']: t.get('duration', 1.0) for t in self.tasks}
        
        # 找到关键路径
        self.critical_path, self.critical_path_length = \
            self.critical_path_analyzer.find_critical_path(self.dag, task_durations)
        self.critical_path_set = set(self.critical_path)
        
        # 计算关键性评分
        self.criticality_scores = \
            self.critical_path_analyzer.calculate_criticality_scores(self.dag, task_durations)
        
        # 计算节点深度
        self.node_depths = {}
        for node in self.dag.nodes():
            self.node_depths[node] = self._compute_node_depth(node)
    
    def _compute_node_depth(self, node: int) -> int:
        """计算节点深度"""
        predecessors = list(self.dag.predecessors(node))
        if not predecessors:
            return 0
        return 1 + max(self._compute_node_depth(p) for p in predecessors)
    
    def reset(self) -> Dict[str, np.ndarray]:
        """重置仿真环境"""
        self.current_time = 0.0
        
        # 初始化任务状态
        self.task_states = {}
        for task in self.tasks:
            self.task_states[task['id']] = TaskState(
                id=task['id'],
                status='pending'
            )
        
        # 初始化资源状态
        self.resource_states = {}
        for resource in self.resources:
            self.resource_states[resource['id']] = ResourceState(
                id=resource['id'],
                status='idle'
            )
        
        # 事件队列
        self.event_queue = []
        
        # 统计信息
        self.completed_tasks: Set[int] = set()
        self.task_assignments: Dict[int, int] = {}
        self.task_start_times: Dict[int, float] = {}
        self.task_end_times: Dict[int, float] = {}
        
        # 更新就绪任务
        self.ready_tasks = self._get_ready_tasks()
        
        # 返回初始状态
        return self.get_state()
    
    def _get_ready_tasks(self) -> List[int]:
        """获取当前可调度的任务"""
        ready = []
        
        for task in self.tasks:
            task_id = task['id']
            state = self.task_states[task_id]
            
            # 跳过已完成或正在运行的任务
            if state.status in ['running', 'completed']:
                continue
            
            # 检查所有前驱是否已完成
            predecessors = list(self.dag.predecessors(task_id))
            all_deps_completed = all(
                self.task_states[p].status == 'completed'
                for p in predecessors
            )
            
            if all_deps_completed:
                ready.append(task_id)
                state.status = 'ready'
        
        # 如果启用关键路径优先，按关键性评分排序
        if self.use_critical_path_priority and ready:
            ready.sort(
                key=lambda t: self.criticality_scores.get(t, 0),
                reverse=True
            )
        
        return ready
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """获取当前状态"""
        scheduler_state = {
            'completed_tasks': self.completed_tasks,
            'ready_tasks': self.ready_tasks,
            'task_end_times': self.task_end_times,
            'current_time': self.current_time,
            'current_makespan': max(self.task_end_times.values()) if self.task_end_times else 0,
            'total_tasks': self.num_tasks,
            'num_resources': self.num_resources,
            'resource_available_time': {
                r['id']: self.resource_states[r['id']].available_time
                for r in self.resources
            },
            'resource_loads': [
                self.resource_states[r['id']].total_busy_time
                for r in self.resources
            ]
        }
        
        return self.state_encoder.encode_state(
            self.tasks, self.resources, self.dag, scheduler_state
        )
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """
        获取有效的动作列表
        
        Returns:
            有效动作列表 [(task_id, resource_id), ...]
        """
        valid_actions = []
        
        for task_id in self.ready_tasks:
            task = next(t for t in self.tasks if t['id'] == task_id)
            
            for resource in self.resources:
                resource_id = resource['id']
                
                # 检查资源容量是否满足
                if self._can_assign(task, resource):
                    valid_actions.append((task_id, resource_id))
        
        return valid_actions
    
    def _can_assign(self, task: Dict, resource: Dict) -> bool:
        """检查任务是否可以分配到资源"""
        cpu_ok = resource.get('cpu_capacity', 4) >= task.get('cpu_req', 1)
        mem_ok = resource.get('memory_capacity', 8) >= task.get('memory_req', 1)
        return cpu_ok and mem_ok
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        执行一步调度
        
        Args:
            action: 动作索引（编码为 task_idx * num_resources + resource_idx）
            
        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否完成
            info: 额外信息
        """
        # 没有可调度任务
        if not self.ready_tasks:
            self._advance_time()
            done = self.is_done()
            info = {
                'makespan': self.get_makespan() if done else 0,
                'utilization': self.get_resource_utilization() if done else 0,
                'load_balance': self.get_load_balance_score() if done else 0
            }
            return self.get_state(), 0.0, done, info
        
        # 解码动作
        if self.use_dag_aware:
            # DAG感知模式：action直接选择资源
            task_id = self.ready_tasks[0]  # 选择优先级最高的就绪任务
            resource_id = action % self.num_resources
        else:
            # 标准模式
            task_idx = action // self.num_resources
            resource_id = action % self.num_resources
            task_idx = min(task_idx, len(self.ready_tasks) - 1)
            task_id = self.ready_tasks[task_idx]
        
        # 获取任务和资源
        task = next(t for t in self.tasks if t['id'] == task_id)
        resource = self.resources[resource_id]
        
        # 检查资源是否满足要求
        if not self._can_assign(task, resource):
            # 选择一个可用的资源
            for alt_resource in self.resources:
                if self._can_assign(task, alt_resource):
                    resource = alt_resource
                    resource_id = alt_resource['id']
                    break
            else:
                # 没有可用资源，给予惩罚
                return self.get_state(), -10.0, self.is_done(), {'error': 'no_valid_resource'}
        
        # 计算开始时间
        resource_state = self.resource_states[resource_id]
        
        # 考虑依赖完成时间
        predecessors = list(self.dag.predecessors(task_id))
        dep_ready_time = max(
            (self.task_end_times.get(p, 0) for p in predecessors),
            default=0
        )
        
        start_time = max(
            self.current_time,
            resource_state.available_time,
            dep_ready_time
        )
        
        # 计算结束时间
        duration = task.get('duration', 1.0)
        end_time = start_time + duration
        
        # 更新状态
        self.task_states[task_id].status = 'running'
        self.task_states[task_id].assigned_resource = resource_id
        self.task_states[task_id].start_time = start_time
        
        self.task_assignments[task_id] = resource_id
        self.task_start_times[task_id] = start_time
        
        # 添加任务完成事件
        heapq.heappush(
            self.event_queue,
            SchedulingEvent(end_time, 'task_end', task_id, resource_id)
        )
        
        # 更新资源状态
        resource_state.status = 'busy'
        resource_state.current_task = task_id
        resource_state.available_time = end_time
        
        # 处理所有当前时间的事件
        self._process_events()
        
        # 更新就绪任务
        self.ready_tasks = self._get_ready_tasks()
        
        # 计算奖励
        reward = self._calculate_reward(task, resource, start_time, end_time)
        
        # 获取下一状态
        next_state = self.get_state()
        
        # 判断是否完成
        done = self.is_done()
        
        info = {
            'task_id': task_id,
            'resource_id': resource_id,
            'start_time': start_time,
            'end_time': end_time,
            'is_critical': task_id in self.critical_path_set if hasattr(self, 'critical_path_set') else False,
            'makespan': self.get_makespan() if done else 0,
            'utilization': self.get_resource_utilization() if done else 0,
            'load_balance': self.get_load_balance_score() if done else 0
        }
        
        return next_state, reward, done, info
    
    def _process_events(self):
        """处理事件队列"""
        while self.event_queue and self.event_queue[0].time <= self.current_time:
            event = heapq.heappop(self.event_queue)
            
            if event.event_type == 'task_end':
                self._handle_task_end(event)
    
    def _handle_task_end(self, event: SchedulingEvent):
        """处理任务完成事件"""
        task_id = event.task_id
        resource_id = event.resource_id
        
        # 更新任务状态
        self.task_states[task_id].status = 'completed'
        self.task_states[task_id].end_time = event.time
        self.completed_tasks.add(task_id)
        self.task_end_times[task_id] = event.time
        
        # 更新资源状态
        resource_state = self.resource_states[resource_id]
        resource_state.status = 'idle'
        resource_state.current_task = None
        resource_state.task_history.append(task_id)
        resource_state.total_busy_time += (
            event.time - self.task_start_times.get(task_id, 0)
        )
    
    def _advance_time(self):
        """推进时间到下一个事件"""
        if self.event_queue:
            next_event_time = self.event_queue[0].time
            self.current_time = next_event_time
            self._process_events()
            self.ready_tasks = self._get_ready_tasks()
    
    def _calculate_reward(self, task: Dict, resource: Dict,
                         start_time: float, end_time: float) -> float:
        """计算奖励"""
        scheduler_state = {
            'completed_tasks': len(self.completed_tasks),
            'total_tasks': self.num_tasks,
            'current_makespan': max(self.task_end_times.values()) if self.task_end_times else end_time,
            'resource_loads': [
                self.resource_states[r['id']].total_busy_time
                for r in self.resources
            ],
            'num_resources': self.num_resources,
            'task_end_times': self.task_end_times,
            'concurrent_tasks_at_time': {}  # 简化
        }
        
        # 添加关键路径信息
        task_info = task.copy()
        task_info['is_critical_path'] = task['id'] in self.critical_path_set if hasattr(self, 'critical_path_set') else False
        task_info['criticality_score'] = self.criticality_scores.get(task['id'], 0.5) if hasattr(self, 'criticality_scores') else 0.5
        task_info['earliest_start_time'] = start_time
        task_info['dependencies'] = list(self.dag.predecessors(task['id']))
        
        reward, _ = self.reward_calculator.calculate_reward(
            task_info, resource, start_time, end_time, scheduler_state
        )
        
        return reward
    
    def is_done(self) -> bool:
        """检查是否所有任务完成"""
        return len(self.completed_tasks) == self.num_tasks
    
    def get_makespan(self) -> float:
        """获取当前makespan"""
        if not self.task_end_times:
            return 0.0
        return max(self.task_end_times.values())
    
    def get_resource_utilization(self) -> float:
        """获取资源利用率"""
        makespan = self.get_makespan()
        if makespan == 0:
            return 0.0
        
        total_work = sum(t.get('duration', 1.0) for t in self.tasks)
        total_capacity = makespan * self.num_resources
        
        return total_work / total_capacity if total_capacity > 0 else 0.0
    
    def get_load_balance_score(self) -> float:
        """获取负载均衡评分"""
        loads = [
            self.resource_states[r['id']].total_busy_time
            for r in self.resources
        ]
        
        if not loads or max(loads) == 0:
            return 1.0
        
        mean_load = np.mean(loads)
        if mean_load == 0:
            return 1.0
        
        cv = np.std(loads) / mean_load  # 变异系数
        return max(0, 1 - cv)
    
    def get_scheduling_result(self) -> Dict[str, Any]:
        """获取调度结果"""
        return {
            'makespan': self.get_makespan(),
            'resource_utilization': self.get_resource_utilization(),
            'load_balance': self.get_load_balance_score(),
            'task_assignments': self.task_assignments.copy(),
            'task_start_times': self.task_start_times.copy(),
            'task_end_times': self.task_end_times.copy(),
            'completed_tasks': len(self.completed_tasks),
            'total_tasks': self.num_tasks
        }


class HistoricalReplaySimulator(EnhancedWorkflowSimulator):
    """
    历史数据回放模拟器
    使用真实历史数据进行训练
    """
    
    def __init__(self, historical_data: Dict[str, Any], **kwargs):
        """
        Args:
            historical_data: 历史调度数据
        """
        # 从历史数据提取任务和资源信息
        tasks = historical_data.get('tasks', [])
        resources = historical_data.get('resources', [])
        dependencies = historical_data.get('dependencies', [])
        
        super().__init__(tasks, resources, dependencies, **kwargs)
        
        # 存储历史最优解作为基准
        self.historical_makespan = historical_data.get('makespan', None)
        self.historical_assignments = historical_data.get('assignments', {})
        
        # 设置奖励基准
        if self.historical_makespan:
            self.reward_calculator.set_baseline(self.historical_makespan)
    
    def get_improvement_ratio(self) -> float:
        """获取相对于历史结果的改善比率"""
        if self.historical_makespan is None:
            return 0.0
        
        current_makespan = self.get_makespan()
        if current_makespan == 0:
            return 0.0
        
        return (self.historical_makespan - current_makespan) / self.historical_makespan
