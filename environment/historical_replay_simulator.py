import numpy as np
import pandas as pd
import networkx as nx
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta


class HistoricalReplaySimulator:
    """基于历史数据重放的仿真环境"""
    
    def __init__(self, process_instances: pd.DataFrame, task_instances: pd.DataFrame, 
                 task_definitions: pd.DataFrame, process_task_relations: pd.DataFrame):
        """
        初始化历史重放仿真器
        
        Args:
            process_instances: 进程实例数据
            task_instances: 任务实例数据
            task_definitions: 任务定义数据
            process_task_relations: 进程任务关系数据
        """
        self.logger = logging.getLogger(__name__)
        self.process_instances = process_instances
        self.task_instances = task_instances
        self.task_definitions = task_definitions
        self.process_task_relations = process_task_relations
        
        # 状态映射
        self.state_mapping = {
            0: 'commit_succeeded',
            1: 'running',
            2: 'prepare_to_pause',
            3: 'pause',
            4: 'prepare_to_stop',
            5: 'stop',
            6: 'fail',
            7: 'succeed',
            8: 'need_fault_tolerance',
            9: 'kill',
            10: 'wait_for_thread',
            11: 'wait_for_dependency'
        }
        
        # 初始化仿真状态
        self.reset()
    
    def reset(self):
        """重置仿真环境"""
        self.current_time = 0
        self.current_process_idx = 0
        self.current_task_idx = 0
        self.completed_tasks = set()
        self.running_tasks = {}
        self.available_resources = {}
        self.task_schedule_history = []
        
        # 获取有任务的进程ID
        processes_with_tasks = self.task_instances['process_instance_id'].unique()
        
        # 获取成功且有任务的进程实例，增加处理的进程数量
        self.successful_processes = self.process_instances[
            (self.process_instances['state'] == 7) & 
            (self.process_instances['id'].isin(processes_with_tasks))
        ].sort_values('start_time').reset_index(drop=True)
        
        if len(self.successful_processes) == 0:
            self.logger.warning("No successful process instances with tasks found!")
            return
        
        # 限制处理的进程数量，避免单个episode过长
        max_processes_per_episode = 20  # 每个episode最多处理20个进程
        if len(self.successful_processes) > max_processes_per_episode:
            self.successful_processes = self.successful_processes.head(max_processes_per_episode)
            self.logger.info(f"Limited to {max_processes_per_episode} processes per episode")
        
        self.logger.info(f"Found {len(self.successful_processes)} successful processes with tasks")
        
        # 初始化第一个进程
        self._load_current_process()
    
    def _load_current_process(self):
        """加载当前进程的任务"""
        if self.current_process_idx >= len(self.successful_processes):
            return False
        
        current_process = self.successful_processes.iloc[self.current_process_idx]
        
        # 获取该进程的所有任务实例
        self.current_process_tasks = self.task_instances[
            self.task_instances['process_instance_id'] == current_process['id']
        ].sort_values('start_time').reset_index(drop=True)
        
        # 关键修复：根据依赖关系排序任务
        self.current_process_tasks = self._sort_tasks_by_dependencies(current_process['process_definition_code'])
        
        # 重置资源状态，确保每个新进程开始时资源是干净的
        self.available_resources = {}
        
        # 初始化资源状态
        self._initialize_resources()
        
        self.current_task_idx = 0
        return True
    
    def _initialize_resources(self):
        """初始化资源状态"""
        # 从当前进程的任务中提取主机信息
        hosts = self.current_process_tasks['host'].dropna().unique()
        
        # 如果没有找到主机信息，使用默认主机
        if len(hosts) == 0:
            hosts = ['default_host']
        
        for host in hosts:
            # 估算主机资源容量
            host_tasks = self.current_process_tasks[
                self.current_process_tasks['host'] == host
            ]
            
            # 基于历史数据估算资源容量
            cpu_capacity = self._estimate_host_cpu_capacity(host_tasks)
            memory_capacity = self._estimate_host_memory_capacity(host_tasks)
            
            self.available_resources[host] = {
                'cpu_capacity': cpu_capacity,
                'memory_capacity': memory_capacity,
                'cpu_used': 0,
                'memory_used': 0,
                'execution_time': 0.0,  # 每台机器的独立执行时间
                'task_queue': [],        # 任务队列
                'current_task_end_time': 0.0  # 当前任务的结束时间
            }
    
    def _estimate_host_cpu_capacity(self, host_tasks: pd.DataFrame) -> int:
        """估算主机的CPU容量"""
        if host_tasks.empty:
            return 8
        
        # 基于任务类型和数量估算
        task_types = host_tasks['task_type'].value_counts()
        estimated_cpu = 4  # 基础CPU
        
        # 根据任务类型调整
        if 'SPARK' in task_types or 'FLINK' in task_types:
            estimated_cpu += 4
        if 'JAVA' in task_types:
            estimated_cpu += 2
        if 'PYTHON' in task_types:
            estimated_cpu += 1
        
        return min(estimated_cpu, 16)
    
    def _estimate_host_memory_capacity(self, host_tasks: pd.DataFrame) -> int:
        """估算主机的内存容量"""
        if host_tasks.empty:
            return 16
        
        # 基于任务类型估算
        task_types = host_tasks['task_type'].value_counts()
        estimated_memory = 8  # 基础内存
        
        # 根据任务类型调整
        if 'SPARK' in task_types or 'FLINK' in task_types:
            estimated_memory += 8
        if 'JAVA' in task_types:
            estimated_memory += 4
        if 'PYTHON' in task_types:
            estimated_memory += 2
        
        return min(estimated_memory, 64)
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前状态"""
        # 任务特征 - 现在由分批处理逻辑统一处理
        # 这里先初始化为空，后面会重新构建
        task_features = []
        
        # 资源特征
        resource_features = []
        for host, resource in self.available_resources.items():
            features = [
                resource['cpu_capacity'],
                resource['memory_capacity'],
                resource['cpu_used'],
                resource['memory_used'],
                resource['execution_time'], # 新增执行时间特征
                resource['cpu_capacity'] - resource['cpu_used'],
                resource['memory_capacity'] - resource['memory_used']
            ]
            resource_features.append(features)
        
        # 分批处理参数：每次处理的任务批次大小
        MAX_TASKS = 5  # 每批次处理的任务数量
        MAX_RESOURCES = 5  # 最大资源数量（修复：从3改为5）
        
        # 任务特征标准化 - 现在分批处理已经确保了固定长度
        # 每个任务特征都是16个元素，不需要额外处理
        
        # 分批处理：只处理当前批次的任务，不足时填充空任务
        # 当前批次：从current_task_idx开始的MAX_TASKS个任务
        current_batch_tasks = []
        for i in range(MAX_TASKS):
            task_idx = self.current_task_idx + i
            if task_idx < len(self.current_process_tasks):
                # 获取实际任务
                task = self.current_process_tasks.iloc[task_idx]
                features = self._extract_task_features(task)
                current_batch_tasks.append(features)
            else:
                # 填充空任务
                current_batch_tasks.append([0.0] * 16)
        
        task_features = current_batch_tasks
        
        # 资源特征标准化
        if not resource_features:
            # 创建一个默认资源特征（7个特征）
            default_resource_features = [8.0, 16.0, 0.0, 0.0, 0.0, 8.0, 16.0]
            resource_features = [default_resource_features]
        
        # 填充或截断资源特征到固定长度
        while len(resource_features) < MAX_RESOURCES:
            resource_features.append([8.0, 16.0, 0.0, 0.0, 0.0, 8.0, 16.0])  # 填充默认资源
        if len(resource_features) > MAX_RESOURCES:
            resource_features = resource_features[:MAX_RESOURCES]  # 截断
        
        task_array = np.array(task_features, dtype=np.float32)
        resource_array = np.array(resource_features, dtype=np.float32)
        
        # 确保是3维张量 [batch_size=1, num_items, features]
        # 处理任务特征
        if len(task_array.shape) == 1:
            # 单个任务，需要添加batch和item维度
            task_array = task_array.reshape(1, 1, -1)
        elif len(task_array.shape) == 2:
            # 多个任务，需要添加batch维度
            task_array = task_array.reshape(1, task_array.shape[0], task_array.shape[1])
        
        # 处理资源特征
        if len(resource_array.shape) == 1:
            # 单个资源，需要添加batch和item维度
            resource_array = resource_array.reshape(1, 1, -1)
        elif len(resource_array.shape) == 2:
            # 多个资源，需要添加batch维度
            resource_array = resource_array.reshape(1, resource_array.shape[0], resource_array.shape[1])
        
        return task_array, resource_array
    
    def _extract_task_features(self, task: pd.Series) -> List[float]:
        """提取任务特征"""
        # 任务类型编码
        task_type_encoding = {
            'SQL': [1, 0, 0, 0, 0, 0, 0],
            'SHELL': [0, 1, 0, 0, 0, 0, 0],
            'PYTHON': [0, 0, 1, 0, 0, 0, 0],
            'JAVA': [0, 0, 0, 1, 0, 0, 0],
            'SPARK': [0, 0, 0, 0, 1, 0, 0],
            'FLINK': [0, 0, 0, 0, 0, 1, 0],
            'HTTP': [0, 0, 0, 0, 0, 0, 1]
        }
        
        task_type = task.get('task_type', 'SHELL')
        type_encoding = task_type_encoding.get(task_type, [0, 0, 0, 0, 0, 0, 0])
        
        # 估算任务资源需求
        cpu_req = self._estimate_task_cpu_requirement(task)
        memory_req = self._estimate_task_memory_requirement(task)
        
        # 计算任务持续时间（如果可用）
        duration = 0.0
        if pd.notna(task.get('start_time')) and pd.notna(task.get('end_time')):
            try:
                start_time = pd.to_datetime(task['start_time'])
                end_time = pd.to_datetime(task['end_time'])
                duration = float((end_time - start_time).total_seconds())
            except Exception:
                duration = 30.0  # 默认持续时间
        
        # 优先级
        priority_val = task.get('task_instance_priority', 0)
        try:
            priority = float(priority_val) if pd.notna(priority_val) else 0.0
        except (ValueError, TypeError):
            priority = 0.0
        
        # 重试次数
        retry_val = task.get('retry_times', 0)
        try:
            retry_times = float(retry_val) if pd.notna(retry_val) else 0.0
        except (ValueError, TypeError):
            retry_times = 0.0
        
        # 组合特征
        features = type_encoding + [
            cpu_req,
            memory_req,
            duration,
            priority,
            retry_times,
            self.current_task_idx,  # 任务在队列中的位置
            len(self.current_process_tasks) - self.current_task_idx,  # 剩余任务数
            # 添加缺失的2个特征，使总数达到16个
            float(task.get('process_definition_id', 0)),  # 流程定义ID（有意义：表示工作流类型）
            float(self._calculate_task_dependencies_count(task))  # 任务依赖数量（有意义：影响调度顺序）
        ]
        
        return features
    
    def _estimate_task_cpu_requirement(self, task: pd.Series) -> float:
        """估算任务的CPU需求"""
        task_type = task.get('task_type', 'SHELL')
        
        base_cpu = {
            'SQL': 2.0,
            'SHELL': 1.0,
            'PYTHON': 2.0,
            'JAVA': 3.0,
            'SPARK': 4.0,
            'FLINK': 4.0,
            'HTTP': 1.0
        }.get(task_type, 1.0)
        
        return base_cpu
    
    def _estimate_task_memory_requirement(self, task: pd.Series) -> float:
        """估算任务的内存需求"""
        task_type = task.get('task_type', 'SHELL')
        
        base_memory = {
            'SQL': 1.0,
            'SHELL': 0.5,
            'PYTHON': 2.0,
            'JAVA': 4.0,
            'SPARK': 8.0,
            'FLINK': 8.0,
            'HTTP': 1.0
        }.get(task_type, 1.0)
        
        return base_memory
    
    def _calculate_task_dependencies_count(self, task: pd.Series) -> int:
        """计算任务的依赖数量（前置任务数量）"""
        if self.process_task_relations.empty:
            return 0
        
        # 获取当前任务的task_definition_code
        task_code = task.get('task_definition_code', 0)
        if pd.isna(task_code) or task_code == 0:
            return 0
        
        # 获取当前进程的process_definition_code
        process_code = task.get('process_definition_code', 0)
        if pd.isna(process_code) or process_code == 0:
            return 0
        
        # 计算有多少个前置任务
        dependencies_count = len(
            self.process_task_relations[
                (self.process_task_relations['process_definition_code'] == process_code) &
                (self.process_task_relations['post_task_code'] == task_code) &
                (self.process_task_relations['pre_task_code'].notna())
            ]
        )
        
        return dependencies_count
    
    def step(self, action: int) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict]:
        """执行一个调度动作"""
        if self.current_task_idx >= len(self.current_process_tasks):
            return self.get_state(), 0, True, {}
        
        # 获取当前任务
        current_task = self.current_process_tasks.iloc[self.current_task_idx]
        
        # 获取可用的资源（主机）
        available_hosts = list(self.available_resources.keys())
        if not available_hosts:
            return self.get_state(), -1, True, {}
        
        # 选择资源
        selected_host = available_hosts[action % len(available_hosts)]
        resource = self.available_resources[selected_host]
        
        # 检查资源是否满足需求
        cpu_req = self._estimate_task_cpu_requirement(current_task)
        memory_req = self._estimate_task_memory_requirement(current_task)
        
        # 改进资源检查：允许部分资源重叠，模拟真实环境
        if (resource['cpu_used'] + cpu_req <= resource['cpu_capacity'] * 1.2 and  # 允许20%的CPU超载
            resource['memory_used'] + memory_req <= resource['memory_capacity'] * 1.1):  # 允许10%的内存超载
            
            # 计算奖励
            reward = self._calculate_reward(current_task, selected_host)
            
            # 更新资源状态
            resource['cpu_used'] += cpu_req
            resource['memory_used'] += memory_req
            
            # 计算任务执行时间并更新时间
            if pd.notna(current_task.get('start_time')) and pd.notna(current_task.get('end_time')):
                start_time = pd.to_datetime(current_task['start_time'])
                end_time = pd.to_datetime(current_task['end_time'])
                task_duration = (end_time - start_time).total_seconds()
            else:
                # 使用估算时间
                task_type = current_task.get('task_type', 'SHELL')
                task_duration = {
                    'SQL': 30.0,
                    'SHELL': 10.0,
                    'PYTHON': 60.0,
                    'JAVA': 120.0,
                    'SPARK': 300.0,
                    'FLINK': 300.0,
                    'HTTP': 5.0
                }.get(task_type, 30.0)
            
            # 关键改进：基于真实时间的调度
            if resource['cpu_used'] <= cpu_req and resource['memory_used'] <= memory_req:
                # 资源完全空闲，任务可以立即开始
                resource['execution_time'] += task_duration
                resource['current_task_end_time'] = resource['execution_time']
            else:
                # 资源部分占用，任务需要等待
                # 等待时间 = 当前任务结束时间 - 当前时间
                wait_time = max(0, resource['current_task_end_time'] - resource['execution_time'])
                resource['execution_time'] += wait_time + task_duration
                resource['current_task_end_time'] = resource['execution_time']
            
            # 记录调度历史
            self.task_schedule_history.append({
                'task_id': current_task['id'],
                'task_name': current_task['name'],
                'host': selected_host,
                'action': action,
                'reward': reward,
                'timestamp': self.current_time,
                'duration': task_duration
            })
            
            # 任务完成后立即释放资源（模拟任务执行完成）
            resource['cpu_used'] = max(0, resource['cpu_used'] - cpu_req)
            resource['memory_used'] = max(0, resource['memory_used'] - memory_req)
            
            # 移动到下一个任务
            self.current_task_idx += 1
            
            # 检查是否完成当前进程的所有任务
            if self.current_task_idx >= len(self.current_process_tasks):
                # 移动到下一个进程
                self.current_process_idx += 1
                if not self._load_current_process():
                    # 所有进程都完成了
                    return self.get_state(), reward, True, {}
            
            return self.get_state(), reward, False, {
                'task_scheduled': True,
                'host': selected_host,
                'task_name': current_task['name']
            }
        else:
            # 资源不足，给予负奖励
            return self.get_state(), -1, False, {
                'task_scheduled': False,
                'reason': 'insufficient_resources'
            }
    
    def _calculate_reward(self, task: pd.Series, host: str) -> float:
        """计算调度奖励"""
        reward = 0.0
        
        # 基础奖励：成功调度
        reward += 1.0
        
        # 资源利用率奖励
        resource = self.available_resources[host]
        cpu_utilization = resource['cpu_used'] / resource['cpu_capacity']
        memory_utilization = resource['memory_used'] / resource['memory_capacity']
        
        # 适中的资源利用率获得更高奖励
        if 0.3 <= cpu_utilization <= 0.8 and 0.3 <= memory_utilization <= 0.8:
            reward += 0.5
        
        # 任务类型匹配奖励
        task_type = task.get('task_type', 'SHELL')
        if task_type in ['SPARK', 'FLINK'] and 'spark' in host.lower() or 'flink' in host.lower():
            reward += 0.3
        
        # 优先级奖励
        priority = task.get('task_instance_priority', 0)
        if priority > 0:
            reward += 0.2
        
        return reward
    
    def is_done(self) -> bool:
        """检查是否完成所有任务"""
        return self.current_process_idx >= len(self.successful_processes)
    
    def get_makespan(self) -> float:
        """获取总执行时间（基于每台机器的真实执行时间）"""
        if not self.task_schedule_history:
            return 0.0
        
        # 方法1: 基于每台机器的执行时间（更准确）
        if self.available_resources:
            max_machine_time = 0.0
            for host, resource in self.available_resources.items():
                machine_time = resource.get('execution_time', 0.0)
                max_machine_time = max(max_machine_time, machine_time)
            
            if max_machine_time > 0:
                return max_machine_time
        
        # 方法2: 基于调度历史计算
        if self.task_schedule_history:
            # 找到最后一个任务的完成时间
            max_completion_time = 0.0
            for record in self.task_schedule_history:
                completion_time = record.get('timestamp', 0.0)
                max_completion_time = max(max_completion_time, completion_time)
            return max_completion_time
        
        return 0.0
    
    def get_resource_utilization(self) -> float:
        """获取资源利用率"""
        if not self.available_resources:
            return 0.0
        
        total_utilization = 0.0
        for resource in self.available_resources.values():
            cpu_util = resource['cpu_used'] / resource['cpu_capacity']
            memory_util = resource['memory_used'] / resource['memory_capacity']
            total_utilization += (cpu_util + memory_util) / 2
        
        return total_utilization / len(self.available_resources)
    
    def get_schedule_history(self) -> List[Dict]:
        """获取调度历史"""
        return self.task_schedule_history
    
    def get_current_process_info(self) -> Dict:
        """获取当前进程信息"""
        if self.current_process_idx < len(self.successful_processes):
            process = self.successful_processes.iloc[self.current_process_idx]
            return {
                'process_id': process['id'],
                'process_name': process['name'],
                'total_tasks': len(self.current_process_tasks),
                'completed_tasks': self.current_task_idx,
                'remaining_tasks': len(self.current_process_tasks) - self.current_task_idx
            }
        return {}
    
    @property
    def num_resources(self) -> int:
        """返回可用资源数量"""
        return len(self.available_resources)
    
    def _sort_tasks_by_dependencies(self, process_definition_code: int) -> pd.DataFrame:
        """根据依赖关系对任务进行拓扑排序"""
        if self.process_task_relations.empty:
            return self.current_process_tasks
        
        # 获取当前进程的任务关系
        process_relations = self.process_task_relations[
            self.process_task_relations['process_definition_code'] == process_definition_code
        ]
        
        if process_relations.empty:
            return self.current_process_tasks
        
        # 构建依赖图
        G = nx.DiGraph()
        
        # 添加所有任务节点
        for _, relation in process_relations.iterrows():
            pre_task = relation['pre_task_code']
            post_task = relation['post_task_code']
            
            if pd.notna(pre_task):
                G.add_node(pre_task)
            if pd.notna(post_task):
                G.add_node(post_task)
            
            # 添加依赖边
            if pd.notna(pre_task) and pd.notna(post_task):
                G.add_edge(pre_task, post_task)
        
        # 检查是否有循环依赖
        if not nx.is_directed_acyclic_graph(G):
            self.logger.warning(f"Process {process_definition_code} has circular dependencies, using original order")
            return self.current_process_tasks
        
        try:
            # 拓扑排序
            sorted_tasks = list(nx.topological_sort(G))
            
            # 根据排序结果重新排列任务
            if sorted_tasks:
                # 创建任务ID到索引的映射
                task_id_to_index = {}
                for idx, task_id in enumerate(sorted_tasks):
                    task_id_to_index[task_id] = idx
                
                # 为每个任务添加依赖顺序
                def get_dependency_order(task_row):
                    task_code = task_row.get('task_definition_code', 0)
                    return task_id_to_index.get(task_code, 999999)  # 未知任务放在最后
                
                # 按依赖顺序排序
                sorted_df = self.current_process_tasks.copy()
                sorted_df['dependency_order'] = sorted_df.apply(get_dependency_order, axis=1)
                sorted_df = sorted_df.sort_values('dependency_order').reset_index(drop=True)
                sorted_df = sorted_df.drop('dependency_order', axis=1)
                
                self.logger.info(f"Tasks sorted by dependencies for process {process_definition_code}")
                return sorted_df
                
        except Exception as e:
            self.logger.warning(f"Error in topological sort: {e}, using original order")
        
        return self.current_process_tasks
