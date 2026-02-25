import numpy as np
import networkx as nx
import logging
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

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
    """先进先出调度器 - 多资源并行版本"""
    
    def __init__(self, allow_parallel=True):
        super().__init__("FIFO")
        self.allow_parallel = allow_parallel
    
    def schedule(self, tasks: List[Dict], resources: List[Dict], 
                dependencies: List[Tuple[int, int]]) -> Dict:
        """FIFO调度算法"""
        mode = "串行" if not self.allow_parallel else "并行"
        self.logger.info(f"Scheduling {len(tasks)} tasks using FIFO ({mode})")
        
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
        
        # 如果是串行模式，只使用一个资源
        if not self.allow_parallel:
            return self._schedule_serial(tasks, resources, task_order)
        
        # 并行模式：初始化资源状态
        resource_available_time = {res['id']: 0 for res in resources}
        task_assignments = {}
        task_start_times = {}
        task_end_times = {}
        
        # 按FIFO顺序调度任务
        for task_id in task_order:
            # 添加调试信息和错误处理
            try:
                task = next(t for t in tasks if t['id'] == task_id)
            except StopIteration:
                self.logger.error(f"Task ID {task_id} not found in tasks list!")
                self.logger.error(f"Available task IDs: {[t['id'] for t in tasks]}")
                self.logger.error(f"Task order from DAG: {task_order}")
                self.logger.error(f"Task dependencies: {dependencies}")
                raise ValueError(f"Task ID {task_id} not found in tasks list. Available IDs: {[t['id'] for t in tasks]}")
            
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
        
        # 生成调度序列（用于甘特图）
        schedule = []
        resource_dict = {res['id']: res for res in resources}
        task_dict = {task['id']: task for task in tasks}
        
        for task_id, resource_id in task_assignments.items():
            task = task_dict[task_id]
            resource = resource_dict[resource_id]
            schedule.append({
                'task_id': task_id,
                'task_name': task.get('name', f'task_{task_id}'),
                'resource': resource.get('name', f'resource_{resource_id}'),
                'resource_id': resource_id,
                'start_time': task_start_times[task_id],
                'finish_time': task_end_times[task_id],
                'duration': task['duration']
            })
        
        return {
            'task_assignments': task_assignments,
            'task_start_times': task_start_times,
            'task_end_times': task_end_times,
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'schedule': schedule,
            'algorithm': self.name
        }
    
    def _schedule_serial(self, tasks: List[Dict], resources: List[Dict], 
                        task_order: List[int]) -> Dict:
        """串行FIFO调度 - 单资源单队列"""
        task_dict = {task['id']: task for task in tasks}
        resource = resources[0]  # 只使用第一个资源
        
        current_time = 0
        task_assignments = {}
        task_start_times = {}
        task_end_times = {}
        schedule = []
        
        # 按拓扑顺序串行执行所有任务
        for task_id in task_order:
            task = task_dict[task_id]
            
            # 串行：必须等前一个任务完成
            start_time = current_time
            end_time = start_time + task['duration']
            
            task_assignments[task_id] = resource['id']
            task_start_times[task_id] = start_time
            task_end_times[task_id] = end_time
            
            schedule.append({
                'task_id': task_id,
                'task_name': task.get('name', f'task_{task_id}'),
                'resource': resource.get('name', f'resource_{resource["id"]}'),
                'resource_id': resource['id'],
                'start_time': start_time,
                'finish_time': end_time,
                'duration': task['duration']
            })
            
            current_time = end_time  # 下一个任务必须等这个结束
        
        # 计算指标
        makespan = current_time  # 等于所有任务时间之和
        total_work = sum(task['duration'] for task in tasks)
        total_capacity = makespan * len(resources)
        resource_utilization = total_work / total_capacity if total_capacity > 0 else 0
        
        return {
            'task_assignments': task_assignments,
            'task_start_times': task_start_times,
            'task_end_times': task_end_times,
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'schedule': schedule,
            'algorithm': f"{self.name}_串行"
        }

class SJFScheduler(BaseScheduler):
    """最短作业优先调度器 - 使用随机森林回归模型预测任务执行时间"""
    
    def __init__(self, use_prediction_model=True):
        super().__init__("SJF")
        self.use_prediction_model = use_prediction_model
        self.prediction_model = None
        self.is_trained = False
    
    def train_prediction_model(self, historical_data: List[Dict]):
        """训练随机森林回归模型预测任务执行时间"""
        if not self.use_prediction_model or not historical_data:
            self.logger.info("Skipping prediction model training")
            return
        
        self.logger.info(f"Training Random Forest model with {len(historical_data)} samples")
        
        # 准备训练数据
        X = []
        y = []
        
        for task_data in historical_data:
            # 特征：任务类型、输入数据大小、CPU需求、内存需求
            features = [
                task_data.get('task_type', 0),
                task_data.get('input_size', 0),
                task_data.get('cpu_req', 0),
                task_data.get('memory_req', 0)
            ]
            X.append(features)
            y.append(task_data.get('duration', 0))
        
        X = np.array(X)
        y = np.array(y)
        
        # 训练随机森林模型
        self.prediction_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # 分割训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        self.prediction_model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.prediction_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.logger.info(f"Random Forest model trained - MSE: {mse:.2f}, R²: {r2:.3f}")
        self.is_trained = True
    
    def predict_task_duration(self, task: Dict) -> float:
        """预测任务执行时间"""
        if not self.use_prediction_model or not self.is_trained:
            return task.get('duration', 0)
        
        # 提取特征
        features = [
            task.get('task_type', 0),
            task.get('input_size', 0),
            task.get('cpu_req', 0),
            task.get('memory_req', 0)
        ]
        
        # 预测执行时间
        predicted_duration = self.prediction_model.predict([features])[0]
        return max(predicted_duration, 1.0)  # 确保预测时间至少为1秒
    
    def schedule(self, tasks: List[Dict], resources: List[Dict], 
                dependencies: List[Tuple[int, int]]) -> Dict:
        """SJF调度算法"""
        self.logger.info(f"Scheduling {len(tasks)} tasks using SJF")
        COMM_RATIO = 0.1  # 通信-计算比例（可调）
        
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
            
            # 按预测持续时间排序（最短优先）
            if self.use_prediction_model and self.is_trained:
                # 使用预测模型预测执行时间
                for task in ready_tasks:
                    task['predicted_duration'] = self.predict_task_duration(task)
                ready_tasks.sort(key=lambda t: t.get('predicted_duration', t['duration']))
            else:
                # 使用实际持续时间
                ready_tasks.sort(key=lambda t: t['duration'])
            
            # 调度最短的任务
            task = ready_tasks[0]
            
            # 找到最早可用的资源
            best_resource = None
            earliest_start_time = float('inf')
            best_finish_time = float('inf')
            
            for resource in resources:
                if (resource['cpu_capacity'] >= task['cpu_req'] and 
                    resource['memory_capacity'] >= task['memory_req']):
                    
                    resource_ready_time = resource_available_time[resource['id']]
                    
                    # 考虑依赖任务的完成时间
                    dependency_ready_time = 0
                    for pre_task, post_task in dependencies:
                        if post_task == task['id'] and pre_task in task_end_times:
                            comm = 0.0
                            # 若前驱在不同资源上，加入通信时间（近似）
                            pre_res_id = task_assignments.get(pre_task)
                            if pre_res_id is not None and pre_res_id != resource['id']:
                                pre_dur = tasks[[t['id'] for t in tasks].index(pre_task)]['duration']
                                avg_dur = (pre_dur + task['duration']) / 2.0
                                comm = COMM_RATIO * avg_dur
                            dependency_ready_time = max(dependency_ready_time, task_end_times[pre_task] + comm)
                    
                    start_time = max(resource_ready_time, dependency_ready_time)
                    # 异构：执行时长按资源速度系数缩放
                    speed = float(resource.get('speed_factor', 1.0))
                    finish_time = start_time + task['duration'] * speed
                    if finish_time < best_finish_time:
                        best_finish_time = finish_time
                        earliest_start_time = start_time
                        best_resource = resource
            
            if best_resource is None:
                self.logger.error(f"No suitable resource found for task {task['id']}")
                scheduled_tasks.add(task['id'])
                continue
            
            # 分配任务到资源
            task_assignments[task['id']] = best_resource['id']
            task_start_times[task['id']] = earliest_start_time
            task_end_times[task['id']] = earliest_start_time + task['duration'] * float(best_resource.get('speed_factor', 1.0))
            
            # 更新资源可用时间
            resource_available_time[best_resource['id']] = task_end_times[task['id']]
            scheduled_tasks.add(task['id'])
        
        # 计算调度指标
        makespan = max(task_end_times.values()) if task_end_times else 0
        total_work = sum(task['duration'] for task in tasks)
        total_capacity = makespan * len(resources)
        resource_utilization = total_work / total_capacity if total_capacity > 0 else 0
        
        # 生成调度序列（用于甘特图）
        schedule = []
        resource_dict = {res['id']: res for res in resources}
        task_dict = {task['id']: task for task in tasks}
        
        for task_id, resource_id in task_assignments.items():
            task = task_dict[task_id]
            resource = resource_dict[resource_id]
            schedule.append({
                'task_id': task_id,
                'task_name': task.get('name', f'task_{task_id}'),
                'resource': resource.get('name', f'resource_{resource_id}'),
                'resource_id': resource_id,
                'start_time': task_start_times[task_id],
                'finish_time': task_end_times[task_id],
                'duration': task['duration']
            })
        
        return {
            'task_assignments': task_assignments,
            'task_start_times': task_start_times,
            'task_end_times': task_end_times,
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'schedule': schedule,
            'algorithm': self.name
        }

class HEFTScheduler(BaseScheduler):
    """异构最早完成时间调度器 - 通信计算比率设为0.5"""
    
    def __init__(self, communication_computation_ratio=0.5):
        super().__init__("HEFT")
        self.communication_computation_ratio = communication_computation_ratio
    
    def schedule(self, tasks: List[Dict], resources: List[Dict], 
                dependencies: List[Tuple[int, int]]) -> Dict:
        """HEFT调度算法"""
        self.logger.info(f"Scheduling {len(tasks)} tasks using HEFT")
        COMM_RATIO = 0.1
        
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
                    
                    # 计算任务在该资源上的执行时间（异构缩放）
                    speed = float(resource.get('speed_factor', 1.0))
                    execution_time = self._calculate_execution_time(task, resource) * speed
                    
                    # 计算最早开始时间
                    resource_ready_time = resource_available_time[resource['id']]
                    
                    # 考虑依赖任务的完成时间（含通信）
                    dependency_ready_time = 0
                    for pre_task, post_task in dependencies:
                        if post_task == task['id'] and pre_task in task_end_times:
                            comm = 0.0
                            pre_res = task_assignments.get(pre_task)
                            if pre_res is not None and pre_res != resource['id']:
                                pre_dur = [tt for tt in tasks if tt['id'] == pre_task][0]['duration']
                                avg_dur = (pre_dur + task['duration']) / 2.0
                                comm = COMM_RATIO * avg_dur
                            dependency_ready_time = max(dependency_ready_time, task_end_times[pre_task] + comm)
                    
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
        
        # 生成调度序列（用于甘特图）
        schedule = []
        resource_dict = {res['id']: res for res in resources}
        
        for task_id, resource_id in task_assignments.items():
            task = task_dict[task_id]
            resource = resource_dict[resource_id]
            schedule.append({
                'task_id': task_id,
                'task_name': task.get('name', f'task_{task_id}'),
                'resource': resource.get('name', f'resource_{resource_id}'),
                'resource_id': resource_id,
                'start_time': task_start_times[task_id],
                'finish_time': task_end_times[task_id],
                'duration': task['duration']
            })
        
        return {
            'task_assignments': task_assignments,
            'task_start_times': task_start_times,
            'task_end_times': task_end_times,
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'schedule': schedule,
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
                    # 计算通信时间（基于通信计算比率）
                    communication_time = self._calculate_communication_time(task, task_dict[successor])
                    max_successor_rank = max(max_successor_rank, 
                                           upward_rank[successor] + communication_time)
            
            upward_rank[task_id] = avg_execution_time + max_successor_rank
        
        return upward_rank
    
    def _calculate_execution_time(self, task: Dict, resource: Dict) -> float:
        """
        计算任务在资源上的执行时间
        注意：在真实数据实验中，task['duration']已经是实际执行时间
        不应该再根据资源容量调整，否则会导致结果不准确
        """
        # 直接返回任务的实际持续时间
        return task['duration']
    
    def _calculate_communication_time(self, task1: Dict, task2: Dict) -> float:
        """计算两个任务之间的通信时间"""
        # 基于通信计算比率计算通信时间
        # 通信时间 = 通信计算比率 × 平均执行时间
        avg_duration = (task1['duration'] + task2['duration']) / 2
        communication_time = self.communication_computation_ratio * avg_duration
        return communication_time

