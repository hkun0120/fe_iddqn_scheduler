import numpy as np
import networkx as nx
import logging
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    """调度器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def schedule(self, tasks: List[Dict], resources: List[Dict], 
                dependencies: List[Tuple[int, int]]) -> Dict:
        """
        调度任务到资源
        
        Args:
            tasks: 任务列表，每个任务包含id, duration, cpu_req, memory_req等
            resources: 资源列表，每个资源包含id, cpu_capacity, memory_capacity等
            dependencies: 任务依赖关系列表 [(pre_task_id, post_task_id), ...]
            
        Returns:
            调度结果字典，包含task_assignments, makespan, resource_utilization等
        """
        pass

class FIFOScheduler(BaseScheduler):
    """先进先出调度器"""
    
    def __init__(self):
        super().__init__("FIFO")
    
    def schedule(self, tasks: List[Dict], resources: List[Dict], 
                dependencies: List[Tuple[int, int]]) -> Dict:
        """FIFO调度算法"""
        self.logger.info(f"Scheduling {len(tasks)} tasks using FIFO")
        
        # 创建DAG图
        dag = nx.DiGraph()
        for task in tasks:
            dag.add_node(task['id'], **task)
        
        for pre_task, post_task in dependencies:
            dag.add_edge(pre_task, post_task)
        
        # 拓扑排序获取任务执行顺序
        try:
            task_order = list(nx.topological_sort(dag))
        except nx.NetworkXError:
            self.logger.error("DAG contains cycles")
            return {'error': 'DAG contains cycles'}
        
        # 初始化资源状态
        resource_available_time = {res['id']: 0 for res in resources}
        task_assignments = {}
        task_start_times = {}
        task_end_times = {}
        
        # 按FIFO顺序调度任务
        for task_id in task_order:
            task = next(t for t in tasks if t['id'] == task_id)
            
            # 找到最早可用的资源
            best_resource = None
            earliest_start_time = float('inf')
            
            for resource in resources:
                # 检查资源容量是否满足任务需求
                if (resource['cpu_capacity'] >= task['cpu_req'] and 
                    resource['memory_capacity'] >= task['memory_req']):
                    
                    # 计算任务最早开始时间
                    resource_ready_time = resource_available_time[resource['id']]
                    
                    # 考虑依赖任务的完成时间
                    dependency_ready_time = 0
                    for pre_task, post_task in dependencies:
                        if post_task == task_id and pre_task in task_end_times:
                            dependency_ready_time = max(dependency_ready_time, 
                                                      task_end_times[pre_task])
                    
                    start_time = max(resource_ready_time, dependency_ready_time)
                    
                    if start_time < earliest_start_time:
                        earliest_start_time = start_time
                        best_resource = resource
            
            if best_resource is None:
                self.logger.error(f"No suitable resource found for task {task_id}")
                continue
            
            # 分配任务到资源
            task_assignments[task_id] = best_resource['id']
            task_start_times[task_id] = earliest_start_time
            task_end_times[task_id] = earliest_start_time + task['duration']
            
            # 更新资源可用时间
            resource_available_time[best_resource['id']] = task_end_times[task_id]
        
        # 计算调度指标
        makespan = max(task_end_times.values()) if task_end_times else 0
        
        # 计算资源利用率
        total_work = sum(task['duration'] for task in tasks)
        total_capacity = makespan * len(resources)
        resource_utilization = total_work / total_capacity if total_capacity > 0 else 0
        
        return {
            'task_assignments': task_assignments,
            'task_start_times': task_start_times,
            'task_end_times': task_end_times,
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'algorithm': self.name
        }

class SJFScheduler(BaseScheduler):
    """最短作业优先调度器"""
    
    def __init__(self):
        super().__init__("SJF")
    
    def schedule(self, tasks: List[Dict], resources: List[Dict], 
                dependencies: List[Tuple[int, int]]) -> Dict:
        """SJF调度算法"""
        self.logger.info(f"Scheduling {len(tasks)} tasks using SJF")
        
        # 创建DAG图
        dag = nx.DiGraph()
        for task in tasks:
            dag.add_node(task['id'], **task)
        
        for pre_task, post_task in dependencies:
            dag.add_edge(pre_task, post_task)
        
        # 初始化资源状态
        resource_available_time = {res['id']: 0 for res in resources}
        task_assignments = {}
        task_start_times = {}
        task_end_times = {}
        scheduled_tasks = set()
        
        while len(scheduled_tasks) < len(tasks):
            # 找到所有可调度的任务（依赖已满足）
            ready_tasks = []
            for task in tasks:
                if task['id'] not in scheduled_tasks:
                    # 检查依赖是否满足
                    dependencies_satisfied = True
                    for pre_task, post_task in dependencies:
                        if post_task == task['id'] and pre_task not in scheduled_tasks:
                            dependencies_satisfied = False
                            break
                    
                    if dependencies_satisfied:
                        ready_tasks.append(task)
            
            if not ready_tasks:
                self.logger.error("No ready tasks found, possible circular dependency")
                break
            
            # 按持续时间排序（最短优先）
            ready_tasks.sort(key=lambda t: t['duration'])
            
            # 调度最短的任务
            task = ready_tasks[0]
            
            # 找到最早可用的资源
            best_resource = None
            earliest_start_time = float('inf')
            
            for resource in resources:
                if (resource['cpu_capacity'] >= task['cpu_req'] and 
                    resource['memory_capacity'] >= task['memory_req']):
                    
                    resource_ready_time = resource_available_time[resource['id']]
                    
                    # 考虑依赖任务的完成时间
                    dependency_ready_time = 0
                    for pre_task, post_task in dependencies:
                        if post_task == task['id'] and pre_task in task_end_times:
                            dependency_ready_time = max(dependency_ready_time, 
                                                      task_end_times[pre_task])
                    
                    start_time = max(resource_ready_time, dependency_ready_time)
                    
                    if start_time < earliest_start_time:
                        earliest_start_time = start_time
                        best_resource = resource
            
            if best_resource is None:
                self.logger.error(f"No suitable resource found for task {task['id']}")
                scheduled_tasks.add(task['id'])
                continue
            
            # 分配任务到资源
            task_assignments[task['id']] = best_resource['id']
            task_start_times[task['id']] = earliest_start_time
            task_end_times[task['id']] = earliest_start_time + task['duration']
            
            # 更新资源可用时间
            resource_available_time[best_resource['id']] = task_end_times[task['id']]
            scheduled_tasks.add(task['id'])
        
        # 计算调度指标
        makespan = max(task_end_times.values()) if task_end_times else 0
        total_work = sum(task['duration'] for task in tasks)
        total_capacity = makespan * len(resources)
        resource_utilization = total_work / total_capacity if total_capacity > 0 else 0
        
        return {
            'task_assignments': task_assignments,
            'task_start_times': task_start_times,
            'task_end_times': task_end_times,
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'algorithm': self.name
        }

class HEFTScheduler(BaseScheduler):
    """异构最早完成时间调度器"""
    
    def __init__(self):
        super().__init__("HEFT")
    
    def schedule(self, tasks: List[Dict], resources: List[Dict], 
                dependencies: List[Tuple[int, int]]) -> Dict:
        """HEFT调度算法"""
        self.logger.info(f"Scheduling {len(tasks)} tasks using HEFT")
        
        # 创建DAG图
        dag = nx.DiGraph()
        task_dict = {task['id']: task for task in tasks}
        
        for task in tasks:
            dag.add_node(task['id'], **task)
        
        for pre_task, post_task in dependencies:
            dag.add_edge(pre_task, post_task)
        
        # 计算任务优先级（向上排序）
        task_priorities = self._calculate_upward_rank(dag, task_dict, resources)
        
        # 按优先级排序任务
        sorted_tasks = sorted(tasks, key=lambda t: task_priorities[t['id']], reverse=True)
        
        # 初始化资源状态
        resource_available_time = {res['id']: 0 for res in resources}
        task_assignments = {}
        task_start_times = {}
        task_end_times = {}
        
        # 调度每个任务
        for task in sorted_tasks:
            best_resource = None
            best_finish_time = float('inf')
            best_start_time = 0
            
            for resource in resources:
                if (resource['cpu_capacity'] >= task['cpu_req'] and 
                    resource['memory_capacity'] >= task['memory_req']):
                    
                    # 计算任务在该资源上的执行时间
                    execution_time = self._calculate_execution_time(task, resource)
                    
                    # 计算最早开始时间
                    resource_ready_time = resource_available_time[resource['id']]
                    
                    # 考虑依赖任务的完成时间
                    dependency_ready_time = 0
                    for pre_task, post_task in dependencies:
                        if post_task == task['id'] and pre_task in task_end_times:
                            # 考虑通信时间（简化为0）
                            dependency_ready_time = max(dependency_ready_time, 
                                                      task_end_times[pre_task])
                    
                    start_time = max(resource_ready_time, dependency_ready_time)
                    finish_time = start_time + execution_time
                    
                    if finish_time < best_finish_time:
                        best_finish_time = finish_time
                        best_start_time = start_time
                        best_resource = resource
            
            if best_resource is None:
                self.logger.error(f"No suitable resource found for task {task['id']}")
                continue
            
            # 分配任务到资源
            task_assignments[task['id']] = best_resource['id']
            task_start_times[task['id']] = best_start_time
            task_end_times[task['id']] = best_finish_time
            
            # 更新资源可用时间
            resource_available_time[best_resource['id']] = best_finish_time
        
        # 计算调度指标
        makespan = max(task_end_times.values()) if task_end_times else 0
        total_work = sum(task['duration'] for task in tasks)
        total_capacity = makespan * len(resources)
        resource_utilization = total_work / total_capacity if total_capacity > 0 else 0
        
        return {
            'task_assignments': task_assignments,
            'task_start_times': task_start_times,
            'task_end_times': task_end_times,
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'algorithm': self.name
        }
    
    def _calculate_upward_rank(self, dag: nx.DiGraph, task_dict: Dict, 
                              resources: List[Dict]) -> Dict[int, float]:
        """计算任务的向上排序值"""
        upward_rank = {}
        
        # 拓扑排序（逆序）
        topo_order = list(reversed(list(nx.topological_sort(dag))))
        
        for task_id in topo_order:
            task = task_dict[task_id]
            
            # 计算平均执行时间
            avg_execution_time = np.mean([
                self._calculate_execution_time(task, resource) 
                for resource in resources
                if (resource['cpu_capacity'] >= task['cpu_req'] and 
                    resource['memory_capacity'] >= task['memory_req'])
            ])
            
            # 计算后继任务的最大向上排序值
            max_successor_rank = 0
            for successor in dag.successors(task_id):
                if successor in upward_rank:
                    # 通信时间简化为0
                    max_successor_rank = max(max_successor_rank, upward_rank[successor])
            
            upward_rank[task_id] = avg_execution_time + max_successor_rank
        
        return upward_rank
    
    def _calculate_execution_time(self, task: Dict, resource: Dict) -> float:
        """计算任务在资源上的执行时间"""
        # 简化计算：基于CPU需求和资源容量的比例调整执行时间
        cpu_ratio = task['cpu_req'] / resource['cpu_capacity']
        memory_ratio = task['memory_req'] / resource['memory_capacity']
        
        # 使用最大比例作为调整因子
        adjustment_factor = max(cpu_ratio, memory_ratio)
        
        return task['duration'] * adjustment_factor

