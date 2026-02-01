import numpy as np
import pandas as pd
import networkx as nx
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta


class HistoricalReplaySimulator:
    """基于历史数据重放的仿真环境"""

    # RL状态/动作空间的固定上限（需与get_state的padding保持一致）
    MAX_TASKS: int = 5
    MAX_RESOURCES: int = 5
    CPU_OVERLOAD_FACTOR: float = 1.2
    MEM_OVERLOAD_FACTOR: float = 1.1
    
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
        # 增加episode计数器
        if not hasattr(self, 'episode_count'):
            self.episode_count = 0
        else:
            self.episode_count += 1
            
        self.current_time = 0
        self.current_process_idx = 0
        self.current_task_idx = 0
        self.completed_tasks = set()
        self.running_tasks = {}
        self.available_resources = {}
        self.task_schedule_history = []
        
        # 获取有任务的进程ID
        processes_with_tasks = self.task_instances['process_instance_id'].unique()
        
        # 获取成功且有任务的进程实例
        self.successful_processes = self.process_instances[
            (self.process_instances['state'] == 7) & 
            (self.process_instances['id'].isin(processes_with_tasks))
        ].sort_values('start_time').reset_index(drop=True)
        
        if len(self.successful_processes) == 0:
            self.logger.warning("No successful process instances with tasks found!")
            return
        
        # 修复：增加处理的进程数量，允许处理更多数据
        # 可以通过环境变量或配置来调整
        try:
            from config.config import Config
        except ImportError:
            # 如果config.config不存在，使用默认配置
            class Config:
                def __init__(self):
                    self.MAX_PROCESSES_PER_EPISODE = 50
                    self.MAX_TASKS_PER_EPISODE = 200
                    self.RANDOM_SEED = 42
            Config = Config()
        max_processes_per_episode = Config.MAX_PROCESSES_PER_EPISODE
        
        if len(self.successful_processes) > max_processes_per_episode:
            # 训练模式：随机采样，确保数据多样性
            import random
            # 修复：使用episode计数器确保每次采样不同的数据
            episode_count = getattr(self, 'episode_count', 0)
            random.seed(Config.RANDOM_SEED + episode_count)  # 每次使用不同的随机种子
            sampled_indices = random.sample(range(len(self.successful_processes)), max_processes_per_episode)
            self.successful_processes = self.successful_processes.iloc[sampled_indices].reset_index(drop=True)
            self.logger.info(f"Episode {episode_count}: Sampled {max_processes_per_episode} processes from {len(self.process_instances)} total processes")
        
        self.logger.info(f"Processing {len(self.successful_processes)} successful processes with tasks")
        total_tasks = len(self.task_instances[self.task_instances['process_instance_id'].isin(self.successful_processes['id'])])
        self.logger.info(f"Total tasks to process: {total_tasks}")
        
        # 如果任务数量过多，进行智能采样
        if total_tasks > Config.MAX_TASKS_PER_EPISODE:
            self.logger.info(f"Task count {total_tasks} exceeds limit {Config.MAX_TASKS_PER_EPISODE}, performing intelligent sampling...")
            
            # 智能采样策略：优先选择任务数量适中的进程
            process_task_counts = {}
            for _, process in self.successful_processes.iterrows():
                process_tasks = self.task_instances[
                    self.task_instances['process_instance_id'] == process['id']
                ]
                process_task_counts[process['id']] = len(process_tasks)
            
            # 按任务数量排序，优先选择中等规模的进程
            sorted_processes = sorted(process_task_counts.items(), key=lambda x: abs(x[1] - Config.MAX_TASKS_PER_EPISODE // 2))
            
            # 选择进程直到任务数量接近限制
            selected_processes = []
            current_task_count = 0
            target_task_count = Config.MAX_TASKS_PER_EPISODE
            
            for process_id, task_count in sorted_processes:
                if current_task_count + task_count <= target_task_count:
                    selected_processes.append(process_id)
                    current_task_count += task_count
                else:
                    break
            
            # 更新选中的进程
            self.successful_processes = self.successful_processes[
                self.successful_processes['id'].isin(selected_processes)
            ].reset_index(drop=True)
            
            self.logger.info(f"Intelligent sampling: Selected {len(selected_processes)} processes with {current_task_count} tasks")
            self.logger.info(f"Task count reduced from {total_tasks} to {current_task_count}")
        
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
        
        # 获取当前流程的依赖关系
        self.current_process_dependencies = self._get_process_dependencies(current_process['process_definition_code'])
        
        # 根据依赖关系排序任务
        self.current_process_tasks = self._sort_tasks_by_dependencies(current_process['process_definition_code'])
        
        # 关键修复：保留累积的执行时间，只重置资源使用状态
        # 而不是完全重新初始化资源
        self._update_resources_for_new_process()
        
        self.current_task_idx = 0
        return True
    
    def _get_process_dependencies(self, process_definition_code):
        """
        获取指定流程的依赖关系（改进版）
        
        尝试多种方式获取依赖：
        1. 使用process_definition_code
        2. 使用process_definition_id (fallback)
        3. 从任务名称推断（last resort）
        """
        dependency_list = []
        
        # 方式1: 使用process_definition_code
        if pd.notna(process_definition_code) and process_definition_code is not None:
            dependencies = self.process_task_relations[
                self.process_task_relations['process_definition_code'] == process_definition_code
            ]
            
            for _, relation in dependencies.iterrows():
                pre_task = relation.get('pre_task_code')
                post_task = relation.get('post_task_code')
                
                if pd.notna(pre_task) and pd.notna(post_task) and pre_task != post_task:
                    dependency_list.append({
                        'pre_task_code': pre_task,
                        'post_task_code': post_task
                    })
            
            if len(dependency_list) > 0:
                self.logger.info(f"流程 {process_definition_code} 的依赖关系数量: {len(dependency_list)} (通过code)")
                return dependency_list
        
        # 方式2: 使用process_definition_id (fallback)
        if hasattr(self, 'current_process_instance') and len(self.successful_processes) > self.current_process_idx:
            current_process = self.successful_processes.iloc[self.current_process_idx]
            process_def_id = current_process.get('process_definition_id')
            
            if pd.notna(process_def_id):
                # 尝试使用ID查找
                if 'process_definition_id' in self.process_task_relations.columns:
                    dependencies = self.process_task_relations[
                        self.process_task_relations['process_definition_id'] == process_def_id
                    ]
                    
                    for _, relation in dependencies.iterrows():
                        pre_task = relation.get('pre_task_code')
                        post_task = relation.get('post_task_code')
                        
                        if pd.notna(pre_task) and pd.notna(post_task) and pre_task != post_task:
                            dependency_list.append({
                                'pre_task_code': pre_task,
                                'post_task_code': post_task
                            })
                    
                    if len(dependency_list) > 0:
                        self.logger.info(f"流程 ID={process_def_id} 的依赖关系数量: {len(dependency_list)} (通过ID)")
                        return dependency_list
        
        # 方式3: 从任务名称推断基本依赖
        if len(dependency_list) == 0:
            self.logger.warning(f"流程 {process_definition_code} 无法从数据库获取依赖，尝试推断...")
            
            # 基于任务名称的启发式依赖推断
            if hasattr(self, 'current_process_tasks') and len(self.current_process_tasks) > 0:
                # 简单推断：名称包含"开始"的任务应该在其他任务之前
                # 名称包含"结束"、"收尾"的应该在最后
                start_tasks = []
                end_tasks = []
                middle_tasks = []
                
                for _, task in self.current_process_tasks.iterrows():
                    task_code = task.get('task_code', task.get('id'))
                    task_name = task.get('name', '')
                    
                    if '开始' in task_name or 'start' in task_name.lower():
                        start_tasks.append(task_code)
                    elif '结束' in task_name or '收尾' in task_name or 'end' in task_name.lower():
                        end_tasks.append(task_code)
                    else:
                        middle_tasks.append(task_code)
                
                # 创建推断的依赖：开始任务 → 中间任务 → 结束任务
                for start in start_tasks:
                    for middle in middle_tasks[:3]:  # 只连接前几个，避免过度约束
                        if start != middle:
                            dependency_list.append({
                                'pre_task_code': start,
                                'post_task_code': middle
                            })
                
                if len(dependency_list) > 0:
                    self.logger.info(f"推断出 {len(dependency_list)} 条基本依赖关系（基于任务名称）")
        
        if len(dependency_list) == 0:
            self.logger.warning(f"流程 {process_definition_code} 没有找到任何依赖关系")
        
        return dependency_list


    def _update_resources_for_new_process(self):
        """为新进程更新资源状态，保留累积的执行时间"""
        # 从当前进程的任务中提取主机信息
        hosts = self.current_process_tasks['host'].dropna().unique()
        
        # 如果没有找到主机信息，使用默认主机
        if len(hosts) == 0:
            hosts = ['default_host']
        
        # 保存现有的执行时间
        existing_execution_times = {}
        if hasattr(self, 'available_resources'):
            for host, resource in self.available_resources.items():
                existing_execution_times[host] = resource.get('execution_time', 0.0)
        
        # 重新初始化资源，但保留累积的执行时间
        for host in hosts:
            # 估算主机资源容量
            host_tasks = self.current_process_tasks[
                self.current_process_tasks['host'] == host
            ]
            
            # 基于历史数据估算资源容量
            cpu_capacity = self._estimate_host_cpu_capacity(host_tasks)
            memory_capacity = self._estimate_host_memory_capacity(host_tasks)
            
            # 保留累积的执行时间，只重置资源使用状态
            cumulative_execution_time = existing_execution_times.get(host, 0.0)
            
            self.available_resources[host] = {
                'cpu_capacity': cpu_capacity,
                'memory_capacity': memory_capacity,
                'cpu_used': 0,  # 重置资源使用状态
                'memory_used': 0,  # 重置资源使用状态
                'execution_time': cumulative_execution_time,  # 保留累积的执行时间
                'task_queue': [],        # 重置任务队列
                'current_task_end_time': cumulative_execution_time  # 基于累积时间设置
            }
    
    def _initialize_resources(self):
        """初始化资源状态（仅在第一次调用时使用）"""
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
    
    def _estimate_host_cpu_capacity(self, host_tasks: pd.DataFrame) -> float:
        """基于实际任务需求动态估算主机CPU容量"""
        if host_tasks.empty:
            return 2.0  # 更合理的默认值
        
        # 计算所有任务的总CPU需求
        total_cpu_req = 0.0
        for _, task in host_tasks.iterrows():
            total_cpu_req += self._estimate_task_cpu_requirement(task)
        
        # 基于任务需求设置合理的主机容量
        # 允许一定的资源余量，但不浪费
        estimated_cpu = max(total_cpu_req * 1.5, 2.0)
        
        # 根据任务类型进行微调
        task_types = host_tasks['task_type'].value_counts()
        if 'SPARK' in task_types or 'FLINK' in task_types:
            estimated_cpu = max(estimated_cpu, 6.0)  # 大数据任务需要更多资源
        if 'JAVA' in task_types:
            estimated_cpu = max(estimated_cpu, 4.0)  # Java应用需要中等资源
        if 'PYTHON' in task_types:
            estimated_cpu = max(estimated_cpu, 3.0)  # Python脚本需要较少资源
        
        return min(estimated_cpu, 16.0)  # 设置上限避免过度估算
    
    def _estimate_host_memory_capacity(self, host_tasks: pd.DataFrame) -> float:
        """基于实际任务需求动态估算主机内存容量"""
        if host_tasks.empty:
            return 4.0  # 更合理的默认值
        
        # 计算所有任务的总内存需求
        total_memory_req = 0.0
        for _, task in host_tasks.iterrows():
            total_memory_req += self._estimate_task_memory_requirement(task)
        
        # 基于任务需求设置合理的主机容量
        # 允许一定的资源余量，但不浪费
        estimated_memory = max(total_memory_req * 1.5, 4.0)
        
        # 根据任务类型进行微调
        task_types = host_tasks['task_type'].value_counts()
        if 'SPARK' in task_types or 'FLINK' in task_types:
            estimated_memory = max(estimated_memory, 12.0)  # 大数据任务需要更多内存
        if 'JAVA' in task_types:
            estimated_memory = max(estimated_memory, 8.0)   # Java应用需要中等内存
        if 'PYTHON' in task_types:
            estimated_memory = max(estimated_memory, 6.0)   # Python脚本需要较少内存
        
        return min(estimated_memory, 64.0)  # 设置上限避免过度估算
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前状态"""
        # 任务特征 - 现在由分批处理逻辑统一处理
        # 这里先初始化为空，后面会重新构建
        task_features = []
        
        # 资源特征（仅保留RL动作空间可见的前MAX_RESOURCES个资源，确保“看到什么就能选什么”）
        resource_features = []
        for host in self._get_action_hosts():
            resource = self.available_resources[host]
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
        MAX_TASKS = self.MAX_TASKS
        MAX_RESOURCES = self.MAX_RESOURCES
        
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
                try:
                    features = self._extract_task_features(task)
                    # 确保特征数量一致
                    if len(features) != 16:
                        self.logger.warning(f"Task {task_idx} has {len(features)} features, expected 16. Padding with zeros.")
                        # 填充到16个特征
                        while len(features) < 16:
                            features.append(0.0)
                        features = features[:16]  # 截断到16个
                    current_batch_tasks.append(features)
                except Exception as e:
                    self.logger.error(f"Error extracting features for task {task_idx}: {e}")
                    # 使用默认特征
                    current_batch_tasks.append([0.0] * 16)
            else:
                # 使用合理的默认值填充空任务
                # 0值明确表示"这个位置没有任务"
                default_features = [
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 无任务类型
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 无资源需求
                    0.0, 0.0  # 无位置信息
                ]
                current_batch_tasks.append(default_features)
        
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

    def get_graph_adj(self) -> np.ndarray:
        """获取当前任务批次的DAG邻接矩阵（用于Graph Transformer）。

        Returns:
            adj: [1, MAX_TASKS, MAX_TASKS] float32, 1表示存在依赖边(i->j)，含自环。
        """
        max_tasks = self.MAX_TASKS

        # 收集当前批次任务的code（与依赖表中的pre_task_code/post_task_code一致）
        codes = []
        for i in range(max_tasks):
            task_idx = self.current_task_idx + i
            if task_idx < len(getattr(self, 'current_process_tasks', [])):
                task = self.current_process_tasks.iloc[task_idx]
                code = task.get('task_code', task.get('task_definition_code', 0))
                try:
                    codes.append(int(code) if pd.notna(code) else 0)
                except Exception:
                    codes.append(0)
            else:
                codes.append(0)

        code_to_pos = {}
        for idx, c in enumerate(codes):
            if c in (0, None):
                continue
            if c not in code_to_pos:
                code_to_pos[c] = idx
        adj = np.zeros((max_tasks, max_tasks), dtype=np.float32)

        # 自环
        for i in range(max_tasks):
            adj[i, i] = 1.0

        # 依赖边
        if hasattr(self, 'current_process_dependencies') and self.current_process_dependencies:
            for dep in self.current_process_dependencies:
                pre = dep.get('pre_task_code')
                post = dep.get('post_task_code')
                try:
                    pre = int(pre) if pd.notna(pre) else 0
                    post = int(post) if pd.notna(post) else 0
                except Exception:
                    continue
                if pre in code_to_pos and post in code_to_pos:
                    i = code_to_pos[pre]
                    j = code_to_pos[post]
                    adj[i, j] = 1.0

        return adj.reshape(1, max_tasks, max_tasks)
    
    def _extract_task_features(self, task: pd.Series) -> List[float]:
        """提取任务特征，包含更多细粒度信息"""
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
        
        # 智能估算任务资源需求
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
        
        # 任务复杂度分析
        complexity_score = self._calculate_task_complexity_score(task)
        
        # 计算任务依赖数量
        dependency_count = self._calculate_task_dependencies_count(task)
        
        # 组合特征
        features = type_encoding + [
            cpu_req,                    # CPU需求
            memory_req,                 # 内存需求
            duration,                   # 执行时间
            priority,                   # 优先级
            retry_times,                # 重试次数
            complexity_score,           # 复杂度评分
            dependency_count,           # 任务依赖数量
            self.current_task_idx,      # 任务在队列中的位置
            len(self.current_process_tasks) - self.current_task_idx,  # 剩余任务数
        ]
        
        # 确保特征数量为16个
        # 16 = 7(任务类型) + 9(其他特征)
        assert len(features) == 16, f"Expected 16 features, got {len(features)}"
        
        return features
    
    def _calculate_task_complexity_score(self, task: pd.Series) -> float:
        """计算任务的综合复杂度评分"""
        task_type = task.get('task_type', 'SHELL')
        complexity = 1.0
        
        if task_type == 'SQL':
            complexity = self._analyze_sql_complexity(task)
        elif task_type == 'PYTHON':
            complexity = self._analyze_python_complexity(task)
        elif task_type == 'JAVA':
            complexity = self._analyze_java_complexity(task)
        elif task_type in ['SPARK', 'FLINK']:
            complexity = self._analyze_data_scale(task)
        
        # 根据任务持续时间调整复杂度
        if pd.notna(task.get('start_time')) and pd.notna(task.get('end_time')):
            try:
                start_time = pd.to_datetime(task['start_time'])
                end_time = pd.to_datetime(task['end_time'])
                duration = (end_time - start_time).total_seconds()
                
                # 时间越长，复杂度越高
                if duration > 300:  # 5分钟
                    complexity += 0.3
                if duration > 600:  # 10分钟
                    complexity += 0.3
                if duration > 1800:  # 30分钟
                    complexity += 0.5
                if duration > 3600:  # 1小时
                    complexity += 0.8
            except Exception:
                pass
        
        # 根据优先级调整复杂度
        priority = task.get('task_instance_priority', 0)
        if priority > 0:
            complexity += 0.2
        
        # 根据重试次数调整复杂度
        retry_times = task.get('retry_times', 0)
        if retry_times > 0:
            complexity += retry_times * 0.1
        
        return min(complexity, 5.0)  # 最大复杂度5.0
    
    def _estimate_task_cpu_requirement(self, task: pd.Series) -> float:
        """智能估算任务的CPU需求，考虑任务具体特征"""
        task_type = task.get('task_type', 'SHELL')
        
        # 基础CPU需求
        base_cpu = {
            'SQL': 0.2,
            'SHELL': 0.1,
            'PYTHON': 0.2,
            'JAVA': 0.3,
            'SPARK': 1.0,
            'FLINK': 1.0,
            'HTTP': 0.1
        }.get(task_type, 0.1)
        
        # 根据任务具体特征调整CPU需求
        adjusted_cpu = base_cpu
        
        if task_type == 'SQL':
            # SQL任务：根据SQL复杂度调整
            sql_complexity = self._analyze_sql_complexity(task)
            adjusted_cpu = base_cpu * sql_complexity
            
        elif task_type == 'PYTHON':
            # Python任务：根据脚本复杂度调整
            script_complexity = self._analyze_python_complexity(task)
            adjusted_cpu = base_cpu * script_complexity
            
        elif task_type == 'JAVA':
            # Java任务：根据应用类型调整
            app_complexity = self._analyze_java_complexity(task)
            adjusted_cpu = base_cpu * app_complexity
            
        elif task_type in ['SPARK', 'FLINK']:
            # 大数据任务：根据数据规模调整
            data_scale = self._analyze_data_scale(task)
            adjusted_cpu = base_cpu * data_scale
        
        # 考虑任务优先级
        priority = task.get('task_instance_priority', 0)
        if priority > 0:
            adjusted_cpu *= 1.2  # 高优先级任务需要更多资源
        
        # 考虑重试次数
        retry_times = task.get('retry_times', 0)
        if retry_times > 0:
            adjusted_cpu *= (1.0 + retry_times * 0.1)  # 重试任务可能需要更多资源
        
        return min(adjusted_cpu, 16.0)  # 设置上限
    
    def _estimate_task_memory_requirement(self, task: pd.Series) -> float:
        """智能估算任务的内存需求，考虑任务具体特征"""
        task_type = task.get('task_type', 'SHELL')
        
        # 基础内存需求
        base_memory = {
            'SQL': 0.1,
            'SHELL': 0.5,
            'PYTHON': 0.2,
            'JAVA': 0.5,
            'SPARK': 0.8,
            'FLINK': 0.8,
            'HTTP': 0.1
        }.get(task_type, 0.1)
        
        # 根据任务具体特征调整内存需求
        adjusted_memory = base_memory
        
        if task_type == 'SQL':
            # SQL任务：根据查询复杂度调整
            sql_complexity = self._analyze_sql_complexity(task)
            adjusted_memory = base_memory * sql_complexity
            
        elif task_type == 'PYTHON':
            # Python任务：根据数据处理需求调整
            data_processing = self._analyze_python_data_processing(task)
            adjusted_memory = base_memory * data_processing
            
        elif task_type == 'JAVA':
            # Java任务：根据应用内存需求调整
            app_memory = self._analyze_java_memory_usage(task)
            adjusted_memory = base_memory * app_memory
            
        elif task_type in ['SPARK', 'FLINK']:
            # 大数据任务：根据数据规模调整
            data_scale = self._analyze_data_scale(task)
            adjusted_memory = base_memory * data_scale
        
        # 考虑任务优先级
        priority = task.get('task_instance_priority', 0)
        if priority > 0:
            adjusted_memory *= 1.3  # 高优先级任务需要更多内存
        
        return min(adjusted_memory, 64.0)  # 设置上限
    
    def _analyze_sql_complexity(self, task: pd.Series) -> float:
        """分析SQL任务的复杂度"""
        # 尝试从任务名称或描述中提取SQL特征
        task_name = str(task.get('name', '')).upper()
        task_desc = str(task.get('description', '')).upper()
        
        complexity = 1.0  # 基础复杂度
        
        # 根据SQL关键词判断复杂度
        if any(keyword in task_name or keyword in task_desc for keyword in ['JOIN', 'UNION', 'GROUP BY', 'ORDER BY']):
            complexity += 0.5  # 中等复杂度
        
        if any(keyword in task_name or keyword in task_desc for keyword in ['SUBQUERY', 'CTE', 'WINDOW', 'PARTITION']):
            complexity += 0.8  # 高复杂度
        
        if any(keyword in task_name or keyword in task_desc for keyword in ['ANALYTIC', 'LAG', 'LEAD', 'RANK']):
            complexity += 1.0  # 分析函数，很高复杂度
        
        # 根据任务持续时间调整（如果有历史数据）
        if pd.notna(task.get('start_time')) and pd.notna(task.get('end_time')):
            try:
                start_time = pd.to_datetime(task['start_time'])
                end_time = pd.to_datetime(task['end_time'])
                duration = (end_time - start_time).total_seconds()
                
                # 根据执行时间调整复杂度
                if duration > 300:  # 5分钟以上
                    complexity += 0.5
                if duration > 600:  # 10分钟以上
                    complexity += 0.5
                if duration > 1800:  # 30分钟以上
                    complexity += 1.0
            except Exception:
                pass
        
        return max(complexity, 0.5)  # 最小复杂度0.5
    
    def _analyze_python_complexity(self, task: pd.Series) -> float:
        """分析Python任务的复杂度"""
        task_name = str(task.get('name', '')).lower()
        task_desc = str(task.get('description', '')).lower()
        
        complexity = 1.0
        
        # 根据Python库和功能判断复杂度
        if any(lib in task_name or lib in task_desc for lib in ['pandas', 'numpy', 'matplotlib']):
            complexity += 0.3  # 数据处理库
        
        if any(lib in task_name or lib in task_desc for lib in ['sklearn', 'tensorflow', 'pytorch']):
            complexity += 0.8  # 机器学习库
        
        if any(lib in task_name or lib in task_desc for lib in ['requests', 'urllib', 'selenium']):
            complexity += 0.2  # 网络请求库
        
        if any(lib in task_name or lib in task_desc for lib in ['multiprocessing', 'threading', 'asyncio']):
            complexity += 0.5  # 并发处理库
        
        return max(complexity, 0.5)
    
    def _analyze_java_complexity(self, task: pd.Series) -> float:
        """分析Java任务的复杂度"""
        task_name = str(task.get('name', '')).lower()
        task_desc = str(task.get('description', '')).lower()
        
        complexity = 1.0
        
        # 根据Java应用类型判断复杂度
        if any(keyword in task_name or keyword in task_desc for keyword in ['web', 'servlet', 'spring']):
            complexity += 0.3  # Web应用
        
        if any(keyword in task_name or keyword in task_desc for keyword in ['batch', 'job', 'scheduler']):
            complexity += 0.5  # 批处理任务
        
        if any(keyword in task_name or keyword in task_desc for keyword in ['stream', 'kafka', 'rabbitmq']):
            complexity += 0.4  # 流处理
        
        return max(complexity, 0.5)
    
    def _analyze_data_scale(self, task: pd.Series) -> float:
        """分析大数据任务的规模"""
        task_name = str(task.get('name', '')).lower()
        task_desc = str(task.get('description', '')).lower()
        
        scale = 1.0
        
        # 根据数据规模关键词判断
        if any(keyword in task_name or keyword in task_desc for keyword in ['gb', 'gigabyte', 'large']):
            scale += 0.5  # GB级数据
        
        if any(keyword in task_name or keyword in task_desc for keyword in ['tb', 'terabyte', 'huge']):
            scale += 1.0  # TB级数据
        
        if any(keyword in task_name or keyword in task_desc for keyword in ['pb', 'petabyte', 'massive']):
            scale += 2.0  # PB级数据
        
        return max(scale, 0.5)
    
    def _analyze_python_data_processing(self, task: pd.Series) -> float:
        """分析Python任务的数据处理需求"""
        task_name = str(task.get('name', '')).lower()
        task_desc = str(task.get('description', '')).lower()
        
        processing = 1.0
        
        # 根据数据处理类型判断内存需求
        if any(keyword in task_name or keyword in task_desc for keyword in ['csv', 'json', 'xml']):
            processing += 0.2  # 文件处理
        
        if any(keyword in task_name or keyword in task_desc for keyword in ['database', 'sql', 'mongodb']):
            processing += 0.4  # 数据库操作
        
        if any(keyword in task_name or keyword in task_desc for keyword in ['image', 'video', 'audio']):
            processing += 0.6  # 多媒体处理
        
        return max(processing, 0.5)
    
    def _analyze_java_memory_usage(self, task: pd.Series) -> float:
        """分析Java任务的内存使用模式"""
        task_name = str(task.get('name', '')).lower()
        task_desc = str(task.get('description', '')).lower()
        
        memory = 1.0
        
        # 根据Java应用类型判断内存需求
        if any(keyword in task_name or keyword in task_desc for keyword in ['cache', 'redis', 'memory']):
            memory += 0.5  # 缓存应用
        
        if any(keyword in task_name or keyword in task_desc for keyword in ['batch', 'bulk', 'large']):
            memory += 0.4  # 批处理应用
        
        return max(memory, 0.5)
    
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
        # 检查是否还有任务需要调度
        if self.current_task_idx >= len(self.current_process_tasks):
            # 当前进程的所有任务都完成了，尝试切换到下一个进程
            self.current_process_idx += 1
            if not self._load_current_process():
                # 所有进程都完成了
                return self.get_state(), 0, True, {}
            else:
                # 成功切换到下一个进程，继续调度
                pass
        
        # 获取当前任务
        current_task = self.current_process_tasks.iloc[self.current_task_idx]
        
        # 获取可用的资源（主机）
        available_hosts = self._get_action_hosts()
        if not available_hosts:
            return self.get_state(), -1, True, {}

        # 动作越界：直接惩罚并返回mask，避免索引隐式取模导致训练信号混乱
        valid_action_mask = self.get_valid_action_mask(current_task)
        if action < 0 or action >= len(available_hosts):
            return self.get_state(), -0.2, False, {
                'task_scheduled': False,
                'reason': 'invalid_action',
                'valid_action_mask': valid_action_mask
            }
        
        # 选择资源
        selected_host = available_hosts[action]
        resource = self.available_resources[selected_host]
        
        # 检查资源是否满足需求
        cpu_req = self._estimate_task_cpu_requirement(current_task)
        memory_req = self._estimate_task_memory_requirement(current_task)
        
        # 改进资源检查：允许部分资源重叠，模拟真实环境
        if (resource['cpu_used'] + cpu_req <= resource['cpu_capacity'] * self.CPU_OVERLOAD_FACTOR and  # 允许CPU超载
            resource['memory_used'] + memory_req <= resource['memory_capacity'] * self.MEM_OVERLOAD_FACTOR):  # 允许内存超载

            makespan_before = self.get_makespan()
            
            # 计算奖励
            reward = self._calculate_reward(current_task, selected_host)
            
            # 计算任务执行时间
            if pd.notna(current_task.get('start_time')) and pd.notna(current_task.get('end_time')):
                start_time = pd.to_datetime(current_task['start_time'])
                end_time = pd.to_datetime(current_task['end_time'])
                task_duration = (end_time - start_time).total_seconds()
            else:
                # 使用估算时间
                task_type = current_task.get('task_type', 'SHELL')
                task_duration = {
                    'SQL': 3.0,
                    'SHELL': 10.0,
                    'PYTHON': 60.0,
                    'JAVA': 120.0,
                    'SPARK': 300.0,
                    'FLINK': 300.0,
                    'HTTP': 5.0
                }.get(task_type, 10.0)
            
            # 基于每台机器的独立时间线进行调度（单机串行、跨机并行）
            start_time = resource.get('execution_time', 0.0)
            end_time = start_time + task_duration
            resource['execution_time'] = end_time
            resource['current_task_end_time'] = end_time
            
            # 更新当前时间（用于日志/可视化；不作为全局同步时钟）
            self.current_time = max(self.current_time, resource['execution_time'])
            
            # 记录调度历史
            self.task_schedule_history.append({
                'task_id': current_task['id'],
                'task_name': current_task['name'],
                'host': selected_host,
                'action': action,
                'reward': reward,
                'timestamp': start_time,
                'duration': task_duration
            })
            
            # 记录任务完成
            self.completed_tasks.add(current_task['id'])
            
            # 移动到下一个任务
            self.current_task_idx += 1
            
            # 检查当前进程是否完成
            is_current_process_done = (self.current_task_idx >= len(self.current_process_tasks))
            
            # 如果当前进程完成，添加全局makespan奖励/惩罚
            if is_current_process_done:
                final_makespan = self.get_makespan()
                
                # 计算理想makespan（完美并行情况）
                total_duration = sum(self.task_schedule_history[i]['duration'] 
                                    for i in range(len(self.task_schedule_history)))
                ideal_makespan = total_duration / len(self.available_resources)
                
                # 计算并行效率
                if final_makespan > 0 and ideal_makespan > 0:
                    parallel_efficiency = ideal_makespan / final_makespan
                    # parallel_efficiency = 1.0: 完美并行
                    # parallel_efficiency = 0.5: 50%并行度
                    # parallel_efficiency < 0.2: 几乎串行
                    
                    # 并行效率奖励（缩放到小范围，避免奖励爆炸导致训练不稳定）
                    if parallel_efficiency >= 0.7:
                        episode_reward = 2.0
                    elif parallel_efficiency >= 0.5:
                        episode_reward = 1.0
                    elif parallel_efficiency >= 0.3:
                        episode_reward = 0.3
                    else:
                        episode_reward = -1.0
                    
                    reward += episode_reward
                    
                    # 记录用于debugging
                    if not hasattr(self, '_episode_rewards'):
                        self._episode_rewards = []
                    self._episode_rewards.append({
                        'final_makespan': final_makespan,
                        'ideal_makespan': ideal_makespan,
                        'parallel_efficiency': parallel_efficiency,
                        'episode_reward': episode_reward
                    })

            # 形状化：惩罚makespan的增量，让学习信号更密集
            makespan_after = self.get_makespan()
            reward += - (makespan_after - makespan_before) / 100.0

            # 归一化：限制奖励量级，提升训练稳定性
            reward = float(np.clip(reward, -10.0, 10.0) / 10.0)
            
            return self.get_state(), reward, False, {
                'task_scheduled': True,
                'host': selected_host,
                'task_name': current_task['name'],
                'process_done': is_current_process_done,
                'valid_action_mask': valid_action_mask
            }
        else:
            # 资源不足，给予负奖励
            return self.get_state(), -0.2, False, {
                'task_scheduled': False,
                'reason': 'insufficient_resources',
                'valid_action_mask': valid_action_mask
            }

    def _get_action_hosts(self) -> List[str]:
        """获取RL动作空间对应的主机列表（固定上限，确保与state对齐）"""
        if not self.available_resources:
            return []
        return list(self.available_resources.keys())[: self.MAX_RESOURCES]

    def get_valid_action_mask(self, task: Optional[pd.Series] = None) -> List[int]:
        """返回当前任务下的可行动作mask（1=可选，0=不可选）。"""
        hosts = self._get_action_hosts()
        if not hosts:
            return []

        if task is None:
            # 没有任务信息时，默认全部可选
            return [1] * len(hosts)

        cpu_req = self._estimate_task_cpu_requirement(task)
        mem_req = self._estimate_task_memory_requirement(task)
        mask: List[int] = []
        for host in hosts:
            res = self.available_resources[host]
            ok = (
                res['cpu_used'] + cpu_req <= res['cpu_capacity'] * self.CPU_OVERLOAD_FACTOR
                and res['memory_used'] + mem_req <= res['memory_capacity'] * self.MEM_OVERLOAD_FACTOR
            )
            mask.append(1 if ok else 0)
        return mask
    
    def _calculate_reward(self, task: pd.Series, host: str) -> float:
        """
        计算调度奖励 - 基于学术论文设计
        
        总奖励公式：R_t = w1*R_time + w2*R_resource + w3*R_load
        其中：
        - w1=0.6 (时间优化权重)
        - w2=0.3 (资源利用率权重)  
        - w3=0.1 (负载均衡权重)
        
        R_time: 执行时间奖励（包含关键路径、等待时间）
        R_resource: 资源利用率奖励（CPU和内存的分段奖励）
        R_load: 负载均衡奖励（标准差惩罚）
        """
        
        # 获取任务duration
        task_duration = self._estimate_task_duration(task)
        resource = self.available_resources[host]
        
        # ==================== R_time: 时间相关奖励 (权重0.6) ====================
        R_time = 0.0
        
        # 1.1 执行时间奖励：短任务优先完成获得更高奖励
        if task_duration > 0:
            # 归一化任务时长（假设最长任务1000秒）
            normalized_duration = min(task_duration / 1000.0, 1.0)
            # 短任务获得正奖励，长任务获得负奖励
            R_time += (1.0 - normalized_duration) * 2.0
        
        # 1.2 等待时间惩罚：如果资源忙，需要等待
        resource_ready_time = resource.get('current_task_end_time', 0)
        current_time = resource.get('execution_time', 0)
        wait_time = max(0, resource_ready_time - current_time)
        
        if wait_time > 0:
            # 归一化等待时间
            normalized_wait = min(wait_time / 100.0, 1.0)
            R_time -= normalized_wait * 1.5  # 等待时间惩罚
        
        # 1.3 关键路径奖励：优先调度关键路径上的任务
        # 这里简化为：有依赖的任务优先级更高
        num_dependencies = 0
        if hasattr(self, 'current_process_dependencies'):
            task_code = task.get('task_code', task.get('id'))
            for dep in self.current_process_dependencies:
                if dep.get('post_task_code') == task_code:
                    num_dependencies += 1
        
        if num_dependencies > 0:
            R_time += min(num_dependencies * 0.3, 1.0)  # 关键任务奖励
        
        # ==================== R_resource: 资源利用率奖励 (权重0.3) ====================
        R_resource = 0.0
        
        # 2.1 CPU利用率分段奖励
        cpu_util = resource['cpu_used'] / resource['cpu_capacity'] if resource['cpu_capacity'] > 0 else 0
        
        if cpu_util >= 0.7:  # 高利用率 [0.7, 1.0]
            R_cpu = 3.0
        elif cpu_util >= 0.5:  # 中等利用率 [0.5, 0.7)
            R_cpu = 2.0
        elif cpu_util >= 0.3:  # 低利用率 [0.3, 0.5)
            R_cpu = 1.0
        else:  # 极低利用率 [0, 0.3)
            R_cpu = 0.0
        
        # 2.2 内存利用率分段奖励
        memory_util = resource['memory_used'] / resource['memory_capacity'] if resource['memory_capacity'] > 0 else 0
        
        if memory_util >= 0.7:
            R_memory = 3.0
        elif memory_util >= 0.5:
            R_memory = 2.0
        elif memory_util >= 0.3:
            R_memory = 1.0
        else:
            R_memory = 0.0
        
        # 资源综合奖励
        R_resource = (R_cpu + R_memory) / 2.0
        
        # 2.3 资源过载惩罚
        if cpu_util > 1.0 or memory_util > 1.0:
            R_resource -= max(cpu_util, memory_util) - 1.0  # 超载惩罚
        
        # ==================== R_load: 负载均衡奖励 (权重0.3) ====================
        R_load = 0.0
        
        # 3.1 计算所有资源的CPU利用率（基于任务分配后的状态）
        cpu_utils = []
        for res_host, res in self.available_resources.items():
            if res['cpu_capacity'] > 0:
                cpu_utils.append(res['cpu_used'] / res['cpu_capacity'])
        
        # 3.2 计算所有资源的内存利用率
        mem_utils = []
        for res_host, res in self.available_resources.items():
            if res['memory_capacity'] > 0:
                mem_utils.append(res['memory_used'] / res['memory_capacity'])
        
        # 3.3 计算使用中的资源数量
        cpu_active_resources = sum(1 for util in cpu_utils if util > 0.01)  # 利用率>1%认为在使用
        mem_active_resources = sum(1 for util in mem_utils if util > 0.01)
        total_resources = len(cpu_utils)
        
        # 3.4 计算平均利用率
        cpu_avg_util = np.mean(cpu_utils)
        mem_avg_util = np.mean(mem_utils)
        
        # 3.5 资源多样性奖励：使用更多资源获得更高奖励
        cpu_diversity_ratio = cpu_active_resources / total_resources if total_resources > 0 else 0
        mem_diversity_ratio = mem_active_resources / total_resources if total_resources > 0 else 0
        
        cpu_diversity_reward = cpu_diversity_ratio * 3.0  # 最高3.0
        mem_diversity_reward = mem_diversity_ratio * 3.0
        
        # 3.6 负载均衡奖励：只有在有负载的情况下才计算
        if cpu_avg_util > 0.01 and len(cpu_utils) > 1:
            cpu_std = np.std(cpu_utils)
            cpu_balance_reward = max(0, 2.0 - cpu_std * 5.0)
        else:
            cpu_balance_reward = 0.0  # 无负载时不给均衡奖励
        
        if mem_avg_util > 0.01 and len(mem_utils) > 1:
            mem_std = np.std(mem_utils)
            mem_balance_reward = max(0, 2.0 - mem_std * 5.0)
        else:
            mem_balance_reward = 0.0
        
        # 3.7 资源闲置惩罚：如果所有资源都闲置，给予惩罚
        if cpu_avg_util < 0.01:
            cpu_idle_penalty = -2.0  # 闲置惩罚
        else:
            cpu_idle_penalty = 0.0
        
        if mem_avg_util < 0.01:
            mem_idle_penalty = -2.0
        else:
            mem_idle_penalty = 0.0
        
        # 3.8 综合计算
        cpu_reward = cpu_diversity_reward + cpu_balance_reward + cpu_idle_penalty
        mem_reward = mem_diversity_reward + mem_balance_reward + mem_idle_penalty
        
        # 综合CPU和内存的负载均衡
        R_load = (cpu_reward + mem_reward) / 2.0
        
        # ==================== 加权求和 ====================
        # 【改进】调整权重，增加负载均衡（并行度）的权重
        w1, w2, w3 = 0.5, 0.2, 0.3  # 时间0.6->0.5, 资源0.3->0.2, 负载0.1->0.3
        reward = w1 * R_time + w2 * R_resource + w3 * R_load
        
        # ==================== 额外奖励/惩罚 ====================
        # 任务失败惩罚（如果任务状态不是成功）
        task_state = task.get('state', 7)
        if task_state != 7:  # 7表示成功
            reward -= 5.0
        
        # 任务优先级加成
        priority = task.get('task_instance_priority', 0)
        if priority > 2:  # 高优先级任务
            reward += 0.5
        
        # ==================== 资源多样性奖励（防止资源集中）====================
        # 【改进】增强并行调度能力，更积极地奖励资源多样性
        if len(self.task_schedule_history) > 3:  # 从10改为3，更早开始鼓励并行
            # 统计已使用的不同资源数
            used_resources = set(record['host'] for record in self.task_schedule_history)
            total_available = len(self.available_resources)
            
            # 资源多样性比例
            diversity_ratio = len(used_resources) / total_available
            
            # 【改进】更强的惩罚资源集中，从70%降低到50%阈值
            if diversity_ratio < 0.5:
                diversity_penalty = (0.5 - diversity_ratio) * 20.0  # 从10.0增加到20.0
                reward -= diversity_penalty
            
            # 【改进】大幅提高使用新资源的奖励
            if host not in used_resources:
                reward += 10.0  # 从3.0增加到10.0，强烈鼓励使用新资源
            
            # 【新增】额外奖励多资源并行使用
            if diversity_ratio >= 0.8:  # 使用了80%以上的资源
                reward += 15.0  # 高并行度奖励
        
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
        
        # 方法2: 基于调度历史计算（修复：使用timestamp + duration）
        if self.task_schedule_history:
            # 找到最后一个任务的完成时间
            max_completion_time = 0.0
            for record in self.task_schedule_history:
                start_time = record.get('timestamp', 0.0)
                duration = record.get('duration', 0.0)
                completion_time = start_time + duration
                max_completion_time = max(max_completion_time, completion_time)
            return max_completion_time
        
        return 0.0
    
    def get_resource_utilization(self) -> float:
        """
        获取资源利用率（修正版）
        资源利用率 = 实际工作时间 / (Makespan × 所有可用资源数)
        只计算duration > 0的任务，排除0持续时间的判断任务
        """
        if not self.available_resources:
            return 0.0
        
        # 获取makespan
        makespan = self.get_makespan()
        if makespan == 0:
            return 0.0
        
        # 计算实际工作时间（只计算duration > 0的任务）
        total_work = 0.0
        if self.task_schedule_history:
            for record in self.task_schedule_history:
                duration = record.get('duration', 0.0)
                if duration > 0:  # 只计算有实际工作量的任务
                    total_work += duration
        
        # 计算总容量：Makespan × 所有可用资源数
        num_resources = len(self.available_resources)
        total_capacity = makespan * num_resources
        
        # 资源利用率 = 实际工作时间 / 总容量
        utilization = total_work / total_capacity if total_capacity > 0 else 0.0
        return min(utilization, 1.0)  # 限制在0-1之间
    
    def get_resource_efficiency(self) -> Dict:
        """监控资源利用效率"""
        efficiency = {}
        for host, resource in self.available_resources.items():
            cpu_efficiency = resource['cpu_used'] / resource['cpu_capacity'] if resource['cpu_capacity'] > 0 else 0
            memory_efficiency = resource['memory_used'] / resource['memory_capacity'] if resource['memory_capacity'] > 0 else 0
            
            efficiency[host] = {
                'cpu_capacity': resource['cpu_capacity'],
                'cpu_used': resource['cpu_used'],
                'cpu_efficiency': cpu_efficiency,
                'memory_capacity': resource['memory_capacity'],
                'memory_used': resource['memory_used'],
                'memory_efficiency': memory_efficiency,
                'overall_efficiency': (cpu_efficiency + memory_efficiency) / 2,
                'execution_time': resource['execution_time']
            }
        
        return efficiency
    
    def print_resource_status(self):
        """打印当前资源状态（用于调试）"""
        print(f"\n当前资源状态 (Step {getattr(self, 'current_time', 0)}):")
        print(f"进程索引: {self.current_process_idx}, 任务索引: {self.current_task_idx}")
        
        for host, resource in self.available_resources.items():
            cpu_util = resource['cpu_used'] / resource['cpu_capacity'] if resource['cpu_capacity'] > 0 else 0
            mem_util = resource['memory_used'] / resource['memory_capacity'] if resource['memory_capacity'] > 0 else 0
            
            print(f"  {host}:")
            print(f"    CPU: {resource['cpu_used']:.1f}/{resource['cpu_capacity']:.1f} ({cpu_util:.1%})")
            print(f"    Memory: {resource['memory_used']:.1f}/{resource['memory_capacity']:.1f} ({mem_util:.1%})")
            print(f"    执行时间: {resource['execution_time']:.1f}s")
        
        if self.current_task_idx < len(self.current_process_tasks):
            current_task = self.current_process_tasks.iloc[self.current_task_idx]
            print(f"  当前任务: {current_task['name']} (类型: {current_task.get('task_type', 'UNKNOWN')})")
            print(f"  CPU需求: {self._estimate_task_cpu_requirement(current_task):.1f}")
            print(f"  内存需求: {self._estimate_task_memory_requirement(current_task):.1f}")
    
    def get_schedule_history(self) -> List[Dict]:
        """获取调度历史"""
        return self.task_schedule_history
    
    def get_current_process_info(self) -> Dict:
        """获取当前进程信息"""
        if self.current_process_idx < len(self.successful_processes):
            process = self.successful_processes.iloc[self.current_process_idx]
            
            # 计算当前进程中已完成的任务数量
            current_process_task_ids = set(self.current_process_tasks['id'])
            completed_in_current_process = len(self.completed_tasks.intersection(current_process_task_ids))
            
            return {
                'process_id': process['id'],
                'process_name': process['name'],
                'total_tasks': len(self.current_process_tasks),
                'completed_tasks': completed_in_current_process,
                'remaining_tasks': len(self.current_process_tasks) - completed_in_current_process
            }
        return {}
    
    @property
    def num_resources(self) -> int:
        """返回可用资源数量"""
        return len(self._get_action_hosts())
    
    @property
    def tasks(self) -> List[Dict]:
        """为传统算法提供任务列表接口"""
        tasks = []
        for _, task in self.task_instances.iterrows():
            if task['process_instance_id'] in self.successful_processes['id'].values:
                task_dict = {
                    'id': task['id'],
                    'name': task['name'],
                    'duration': self._estimate_task_duration(task),
                    'cpu_req': self._estimate_task_cpu_requirement(task),
                    'memory_req': self._estimate_task_memory_requirement(task),
                    'priority': task.get('task_instance_priority', 0),
                    'task_type': task.get('task_type', 'SHELL')
                }
                tasks.append(task_dict)
        return tasks
    
    @property
    def resources(self) -> List[Dict]:
        """为传统算法提供资源列表接口"""
        resources = []
        for host, resource in self.available_resources.items():
            resource_dict = {
                'id': host,
                'name': host,
                'cpu_capacity': resource['cpu_capacity'],
                'memory_capacity': resource['memory_capacity'],
                'cpu_used': resource['cpu_used'],
                'memory_used': resource['memory_used']
            }
            resources.append(resource_dict)
        return resources
    
    @property
    def dependencies(self) -> List[Dict]:
        """为传统算法提供任务依赖关系接口"""
        # 使用当前进程的依赖关系
        if hasattr(self, 'current_process_dependencies') and self.current_process_dependencies:
            # 转换为传统算法期望的格式
            dependencies = []
            for dep in self.current_process_dependencies:
                dep_dict = {
                    'pre_task': dep['pre_task_code'],
                    'post_task': dep['post_task_code']
                }
                dependencies.append(dep_dict)
            return dependencies
        
        # 如果没有当前进程依赖关系，返回空列表
        return []
    
    def _estimate_task_duration(self, task: pd.Series) -> float:
        """估算任务执行时间，优先使用真实数据"""
        if pd.notna(task.get('start_time')) and pd.notna(task.get('end_time')):
            try:
                start_time = pd.to_datetime(task['start_time'])
                end_time = pd.to_datetime(task['end_time'])
                duration = (end_time - start_time).total_seconds()
                
                # 确保持续时间合理（至少1秒，最多24小时）
                duration = max(1.0, min(duration, 86400.0))
                return duration
            except Exception as e:
                self.logger.warning(f"Error parsing task duration: {e}")
        
        # 如果没有真实时间数据，使用基于任务类型的合理估算
        task_type = task.get('task_type', 'SHELL')
        duration_map = {
            'SQL': 30.0,
            'SHELL': 10.0,
            'PYTHON': 60.0,
            'JAVA': 120.0,
            'SPARK': 300.0,
            'FLINK': 300.0,
            'HTTP': 5.0
        }
        
        base_duration = duration_map.get(task_type, 30.0)
        
        # 根据任务名称或描述调整持续时间
        task_name = str(task.get('name', '')).lower()
        if any(keyword in task_name for keyword in ['large', 'big', 'huge', 'massive']):
            base_duration *= 2.0
        elif any(keyword in task_name for keyword in ['small', 'tiny', 'quick', 'fast']):
            base_duration *= 0.5
        
        return base_duration
    
    def simulate_random_schedule(self, algorithm_name: str) -> Dict:
        """为RL算法提供随机调度接口"""
        # 重置模拟器
        self.reset()
        
        # 随机调度所有任务
        total_reward = 0
        step_count = 0
        max_steps = len(self.task_instances) * 2  # 设置最大步数
        
        while not self.is_done() and step_count < max_steps:
            # 随机选择动作
            import random
            action = random.randint(0, max(0, self.num_resources - 1))
            
            # 执行动作
            state, reward, done, info = self.step(action)
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        # 返回调度结果
        return {
            'algorithm': algorithm_name,
            'metrics': {
                'makespan': self.get_makespan(),
                'resource_utilization': self.get_resource_utilization(),
                'total_reward': total_reward,
                'steps_taken': step_count
            },
            'schedule_history': self.get_schedule_history()
        }
    
    def _sort_tasks_by_dependencies(self, process_definition_code: int) -> pd.DataFrame:
        """根据依赖关系对任务进行拓扑排序"""
        # 使用已经获取的依赖关系
        if not hasattr(self, 'current_process_dependencies') or not self.current_process_dependencies:
            self.logger.warning(f"No dependencies found for process {process_definition_code}, returning original order")
            return self.current_process_tasks
        
        # 构建依赖图
        G = nx.DiGraph()
        
        # 添加所有任务节点
        for _, task in self.current_process_tasks.iterrows():
            task_code = task.get('task_code', task.get('task_definition_code', 0))
            G.add_node(task_code)
        
        # 添加依赖边
        for dep in self.current_process_dependencies:
            pre_task = dep['pre_task_code']
            post_task = dep['post_task_code']
            
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
                # 创建任务代码到索引的映射
                task_code_to_index = {}
                for idx, task_code in enumerate(sorted_tasks):
                    task_code_to_index[task_code] = idx
                
                # 为每个任务添加依赖顺序
                def get_dependency_order(task_row):
                    task_code = task_row.get('task_code', task_row.get('task_definition_code', 0))
                    return task_code_to_index.get(task_code, 999999)  # 未知任务放在最后
                
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
