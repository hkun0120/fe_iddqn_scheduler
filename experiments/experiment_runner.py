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
from environment.workflow_simulator import WorkflowSimulator
from environment.historical_replay_simulator import HistoricalReplaySimulator
from evaluation.metrics import Evaluator


class ExperimentRunner:
    """实验运行器，负责运行不同算法并收集结果"""

    def __init__(self, data: Dict[str, pd.DataFrame], features: pd.DataFrame,
                 output_dir: str, n_experiments: int = Config.N_EXPERIMENTS):
        self.logger = logging.getLogger(__name__)
        self.data = data
        self.features = features
        self.output_dir = output_dir
        self.n_experiments = n_experiments
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化评估器
        self.evaluator = Evaluator()

    def _build_tasks_from_data(self) -> List[Dict]:
        """从真实数据构建任务列表"""
        tasks = []
        
        # 获取任务实例数据
        task_instances = self.data['task_instance']
        task_definitions = self.data['task_definition']
        
        # 合并任务定义和实例
        merged_tasks = pd.merge(
            task_instances,
            task_definitions,
            left_on=['task_code', 'task_definition_version'],
            right_on=['code', 'version'],
            how='left',
            suffixes=('_instance', '_definition')
        )
        
        # 过滤出有效的任务（有开始和结束时间）
        valid_tasks = merged_tasks[
            (merged_tasks['start_time'].notna()) & 
            (merged_tasks['end_time'].notna()) &
            (merged_tasks['state'] == 7)  # 只选择成功的任务
        ].copy()
        
        # 计算任务持续时间
        valid_tasks['duration'] = (
            pd.to_datetime(valid_tasks['end_time']) - 
            pd.to_datetime(valid_tasks['start_time'])
        ).dt.total_seconds()
        
        # 构建任务字典
        for idx, task in valid_tasks.iterrows():
            # 估算资源需求（基于任务类型和历史数据）
            cpu_req = self._estimate_cpu_requirement(task)
            memory_req = self._estimate_memory_requirement(task)
            
            tasks.append({
                "id": task['id_instance'],
                "name": task['name_instance'],
                "task_type": task['task_type'],
                "duration": task['duration'],
                "cpu_req": cpu_req,
                "memory_req": memory_req,
                "submit_time": task['submit_time'],
                "start_time": task['start_time'],
                "end_time": task['end_time'],
                "host": task['host'],
                "worker_group": task['worker_group'],
                "priority": task['task_instance_priority'],
                "retry_times": task['retry_times'],
                "process_instance_id": task['process_instance_id']
            })
        
        return tasks

    def _build_resources_from_data(self) -> List[Dict]:
        """从真实数据构建资源列表"""
        resources = []
        
        # 从任务实例中提取主机信息
        task_instances = self.data['task_instance']
        hosts = task_instances['host'].dropna().unique()
        
        # 基于历史数据估算每个主机的资源容量
        for host in hosts:
            host_tasks = task_instances[task_instances['host'] == host]
            
            # 估算CPU和内存容量（基于历史任务的最大需求）
            cpu_capacity = self._estimate_host_cpu_capacity(host_tasks)
            memory_capacity = self._estimate_host_memory_capacity(host_tasks)
            
            resources.append({
                "id": host,
                "host": host,
                "cpu_capacity": cpu_capacity,
                "memory_capacity": memory_capacity,
                "worker_group": host_tasks['worker_group'].iloc[0] if not host_tasks.empty else 'default'
            })
        
        return resources

    def _build_dependencies_from_data(self) -> List[Tuple[int, int]]:
        """从真实数据构建任务依赖关系"""
        dependencies = []
        
        # 获取进程任务关系数据
        process_task_relations = self.data['process_task_relation']
        task_instances = self.data['task_instance']
        
        # 基于进程实例ID和时间关系推断依赖
        process_instances = self.data['process_instance']
        
        for _, relation in process_task_relations.iterrows():
            # 找到对应的任务实例
            pre_tasks = task_instances[
                (task_instances['task_code'] == relation['pre_task_code']) &
                (task_instances['process_instance_id'] == relation['process_instance_id'])
            ]
            
            post_tasks = task_instances[
                (task_instances['task_code'] == relation['post_task_code']) &
                (task_instances['process_instance_id'] == relation['process_instance_id'])
            ]
            
            for _, pre_task in pre_tasks.iterrows():
                for _, post_task in post_tasks.iterrows():
                    dependencies.append((pre_task['id'], post_task['id']))
        
        return dependencies

    def _estimate_cpu_requirement(self, task: pd.Series) -> int:
        """估算任务的CPU需求"""
        # 基于任务类型和历史数据估算
        task_type = task['task_type']
        
        # 根据任务类型设置基础CPU需求
        base_cpu = {
            'SQL': 2,
            'SHELL': 1,
            'PYTHON': 2,
            'JAVA': 3,
            'SPARK': 4,
            'FLINK': 4,
            'HTTP': 1
        }.get(task_type, 1)
        
        # 根据持续时间调整
        duration = task.get('duration', 60)
        if duration > 300:  # 超过5分钟的任务
            base_cpu = min(base_cpu + 1, 8)
        
        return base_cpu

    def _estimate_memory_requirement(self, task: pd.Series) -> int:
        """估算任务的内存需求"""
        task_type = task['task_type']
        
        # 根据任务类型设置基础内存需求（GB）
        base_memory = {
            'SQL': 1,
            'SHELL': 0.5,
            'PYTHON': 2,
            'JAVA': 4,
            'SPARK': 8,
            'FLINK': 8,
            'HTTP': 1
        }.get(task_type, 1)
        
        return base_memory

    def _estimate_host_cpu_capacity(self, host_tasks: pd.DataFrame) -> int:
        """估算主机的CPU容量"""
        # 基于历史任务的最大并发数估算
        if host_tasks.empty:
            return 8  # 默认值
        
        # 计算同时运行的任务数
        host_tasks['start_time'] = pd.to_datetime(host_tasks['start_time'])
        host_tasks['end_time'] = pd.to_datetime(host_tasks['end_time'])
        
        # 简单的估算：基于任务数量
        return min(max(len(host_tasks) // 10 + 4, 4), 16)

    def _estimate_host_memory_capacity(self, host_tasks: pd.DataFrame) -> int:
        """估算主机的内存容量"""
        if host_tasks.empty:
            return 16  # 默认值
        
        # 基于历史任务的内存需求估算
        return min(max(len(host_tasks) // 5 + 8, 8), 64)

    def _initialize_simulator(self) -> WorkflowSimulator:
        """从真实数据初始化仿真环境"""
        self.logger.info("Building simulator from real data...")
        
        # 从真实数据构建任务、资源和依赖关系
        tasks = self._build_tasks_from_data()
        resources = self._build_resources_from_data()
        dependencies = self._build_dependencies_from_data()
        
        self.logger.info(f"Built simulator with {len(tasks)} tasks, {len(resources)} resources, {len(dependencies)} dependencies")
        
        return WorkflowSimulator(tasks, resources, dependencies)

    def _create_historical_replay_simulator(self) -> WorkflowSimulator:
        """创建基于历史数据重放的仿真器"""
        self.logger.info("Creating historical replay simulator...")
        
        # 获取成功的进程实例
        process_instances = self.data['process_instance']
        successful_processes = process_instances[process_instances['state'] == 7]
        
        # 为每个成功的进程实例创建重放任务
        replay_tasks = []
        replay_resources = []
        replay_dependencies = []
        
        for _, process in successful_processes.iterrows():
            # 获取该进程的所有任务实例
            process_tasks = self.data['task_instance'][
                self.data['task_instance']['process_instance_id'] == process['id']
            ]
            
            # 按开始时间排序
            process_tasks = process_tasks.sort_values('start_time')
            
            # 创建重放任务
            for idx, task in process_tasks.iterrows():
                if task['state'] == 7:  # 只重放成功的任务
                    duration = (
                        pd.to_datetime(task['end_time']) - 
                        pd.to_datetime(task['start_time'])
                    ).total_seconds()
                    
                    replay_tasks.append({
                        "id": f"{process['id']}_{task['id']}",
                        "name": task['name'],
                        "task_type": task['task_type'],
                        "duration": duration,
                        "cpu_req": self._estimate_cpu_requirement(task),
                        "memory_req": self._estimate_memory_requirement(task),
                        "submit_time": task['submit_time'],
                        "start_time": task['start_time'],
                        "end_time": task['end_time'],
                        "host": task['host'],
                        "process_instance_id": process['id']
                    })
        
        # 创建资源（基于实际主机）
        hosts = pd.concat([self.data['task_instance']['host'], 
                          self.data['process_instance']['host']]).dropna().unique()
        
        for host in hosts:
            replay_resources.append({
                "id": host,
                "host": host,
                "cpu_capacity": self._estimate_host_cpu_capacity(
                    self.data['task_instance'][self.data['task_instance']['host'] == host]
                ),
                "memory_capacity": self._estimate_host_memory_capacity(
                    self.data['task_instance'][self.data['task_instance']['host'] == host]
                )
            })
        
        self.logger.info(f"Created replay simulator with {len(replay_tasks)} tasks, {len(replay_resources)} resources")
        
        return WorkflowSimulator(replay_tasks, replay_resources, replay_dependencies)

    def run_algorithm(self, algorithm_name: str, use_historical_replay: bool = True) -> Dict[str, Any]:
        """运行单个算法的实验"""
        self.logger.info(f"Running experiment for algorithm: {algorithm_name}")

        results = []
        for i in range(self.n_experiments):
            self.logger.info(f"  Experiment {i + 1}/{self.n_experiments}")
            
            # 选择仿真器类型
            if use_historical_replay:
                # 使用历史重放仿真器
                simulator = HistoricalReplaySimulator(
                    self.data['process_instance'],
                    self.data['task_instance'],
                    self.data['task_definition'],
                    self.data['process_task_relation']
                )
            else:
                # 使用传统仿真器
                simulator = self._initialize_simulator()
            
            # 获取算法参数
            algorithm_params = Hyperparameters.get_algorithm_params(algorithm_name)

            if algorithm_name == "FE_IDDQN":
                # 从真实数据中提取特征维度
                task_features, resource_features = simulator.get_state()
                
                # 打印数据集信息
                self.logger.info("=" * 60)
                self.logger.info("FE-IDDQN 算法数据集信息:")
                self.logger.info("=" * 60)
                self.logger.info(f"任务特征维度: {task_features.shape}")
                self.logger.info(f"资源特征维度: {resource_features.shape}")
                self.logger.info(f"可用资源数量: {simulator.num_resources}")
                
                # 打印任务特征示例
                if task_features.size > 0:
                    self.logger.info(f"\n任务特征示例 (前3个任务):")
                    for i in range(min(3, task_features.shape[1])):
                        task_feat = task_features[0, i, :]
                        self.logger.info(f"  任务 {i}: {task_feat[:10]}... (共{len(task_feat)}个特征)")
                
                # 打印资源特征示例
                if resource_features.size > 0:
                    self.logger.info(f"\n资源特征示例 (前3个资源):")
                    for i in range(min(3, resource_features.shape[1])):
                        resource_feat = resource_features[0, i, :]
                        self.logger.info(f"  资源 {i}: {resource_feat} (共{len(resource_feat)}个特征)")
                
                # 打印数据集统计信息
                self.logger.info(f"\n数据集统计信息:")
                self.logger.info(f"  总进程数: {len(self.data['process_instance'])}")
                self.logger.info(f"  总任务数: {len(self.data['task_instance'])}")
                self.logger.info(f"  任务定义数: {len(self.data['task_definition'])}")
                self.logger.info(f"  任务关系数: {len(self.data['process_task_relation'])}")
                
                # 任务类型分布
                task_types = self.data['task_instance']['task_type'].value_counts()
                self.logger.info(f"  任务类型分布:")
                for task_type, count in task_types.head(10).items():
                    self.logger.info(f"    {task_type}: {count}")
                
                self.logger.info("=" * 60)
                
                # 任务特征维度是最后一个维度（特征数量）
                task_input_dim = task_features.shape[-1] if task_features.size > 0 else 16
                resource_input_dim = resource_features.shape[-1] if resource_features.size > 0 else 7
                action_dim = simulator.num_resources

                agent = FE_IDDQN(task_input_dim, resource_input_dim, action_dim, self.device)
                
                # 实际训练过程
                episode_rewards = []
                episode_makespans = []
                
                for episode in range(algorithm_params.get('num_episodes', 100)):
                    simulator.reset()
                    episode_reward = 0
                    step_count = 0
                    
                    # 增加每个episode的最大步数，让算法能够调度更多任务
                    max_steps = algorithm_params.get('max_steps_per_episode', 1000)
                    max_steps = max(max_steps, 2000)  # 确保至少有2000步
                    
                    while not simulator.is_done() and step_count < max_steps:
                        state = simulator.get_state()
                        task_features, resource_features = state
                        
                        # 选择动作
                        action = agent.select_action(task_features, resource_features)
                        
                        # 执行动作
                        next_state, reward, done, info = simulator.step(action)
                        
                        # 添加step执行后的调试信息
                        if step_count % 50 == 0:  # 每50步检查一次
                            self.logger.info(f"        Step执行结果: reward={reward:.2f}, done={done}, info={info}")
                            
                            # 检查任务是否真的被调度了
                            if 'task_scheduled' in info and info['task_scheduled']:
                                self.logger.info(f"        ✅ 任务成功调度: {info.get('task_name', 'Unknown')} -> {info.get('host', 'Unknown')}")
                            else:
                                self.logger.warning(f"        ⚠️  任务调度失败或未调度")
                        
                        # 存储经验
                        agent.store_experience(state, action, reward, next_state, done)
                        
                        # 训练网络
                        if step_count % algorithm_params.get('train_freq', 4) == 0:
                            loss = agent.train()
                        
                        episode_reward += reward
                        step_count += 1
                        
                        # 修复：如果episode完成，立即退出循环
                        if simulator.is_done():
                            self.logger.info(f"        🎯 Episode在第{step_count}步完成，退出训练循环")
                            break
                        
                        # 添加详细的调试信息
                        if step_count % 50 == 0:  # 更频繁的日志输出
                            process_info = simulator.get_current_process_info()
                            if process_info:
                                self.logger.info(f"      Step {step_count}: Process {process_info['process_id']}, "
                                               f"Completed: {process_info['completed_tasks']}/{process_info['total_tasks']}")
                                
                                # 检查当前任务状态
                                if hasattr(simulator, 'current_task_idx') and hasattr(simulator, 'current_process_tasks'):
                                    if simulator.current_task_idx < len(simulator.current_process_tasks):
                                        current_task = simulator.current_process_tasks.iloc[simulator.current_task_idx]
                                        self.logger.info(f"        当前任务: {current_task.get('name', 'Unknown')} "
                                                       f"(ID: {current_task.get('id', 'Unknown')})")
                                        
                                        # 检查任务资源需求
                                        if hasattr(simulator, '_estimate_task_cpu_requirement'):
                                            cpu_req = simulator._estimate_task_cpu_requirement(current_task)
                                            memory_req = simulator._estimate_task_memory_requirement(current_task)
                                            self.logger.info(f"        资源需求: CPU {cpu_req:.1f}, Memory {memory_req:.1f}")
                                    
                                # 检查资源状态
                                if hasattr(simulator, 'available_resources'):
                                    self.logger.info(f"        资源状态:")
                                    for host, resource in simulator.available_resources.items():
                                        cpu_used = resource.get('cpu_used', 0)
                                        cpu_capacity = resource.get('cpu_capacity', 0)
                                        memory_used = resource.get('memory_used', 0)
                                        memory_capacity = resource.get('memory_capacity', 0)
                                        self.logger.info(f"          {host}: CPU {cpu_used:.1f}/{cpu_capacity:.1f}, "
                                                       f"Memory {memory_used:.1f}/{memory_capacity:.1f}")
                                
                                # 检查任务完成状态
                                if hasattr(simulator, 'completed_tasks'):
                                    self.logger.info(f"        已完成任务数: {len(simulator.completed_tasks)}")
                                
                                # 检查进程和任务索引
                                if hasattr(simulator, 'current_process_idx'):
                                    self.logger.info(f"        当前进程索引: {simulator.current_process_idx}")
                                if hasattr(simulator, 'current_task_idx'):
                                    self.logger.info(f"        当前任务索引: {simulator.current_task_idx}")
                                
                                # 检查是否卡住
                                if step_count > 200 and process_info['completed_tasks'] == 0:
                                    self.logger.warning(f"        ⚠️  警告: 200步后仍无任务完成，可能存在卡住问题！")
                                
                                if step_count > 500 and process_info['completed_tasks'] == 0:
                                    self.logger.error(f"        ❌ 错误: 500步后仍无任务完成，强制检查问题！")
                                    # 强制检查当前状态
                                    if hasattr(simulator, 'ready_tasks'):
                                        self.logger.info(f"          可调度任务: {simulator.ready_tasks}")
                                    if hasattr(simulator, 'is_done'):
                                        self.logger.info(f"          是否完成: {simulator.is_done()}")
                        
                        # 每100步的原有日志
                        if step_count % 100 == 0:
                            process_info = simulator.get_current_process_info()
                            if process_info:
                                self.logger.info(f"      Step {step_count}: Process {process_info['process_id']}, "
                                               f"Completed: {process_info['completed_tasks']}/{process_info['total_tasks']}")
                        
                        # 检查是否卡住，强制切换进程
                        if step_count > 1000 and process_info and process_info['completed_tasks'] == 0:
                            self.logger.error(f"        🚨 严重错误: 1000步后仍无任务完成，强制切换到下一个进程！")
                            if hasattr(simulator, 'current_process_idx') and hasattr(simulator, '_load_current_process'):
                                simulator.current_process_idx += 1
                                if simulator._load_current_process():
                                    self.logger.info(f"        强制切换到进程索引: {simulator.current_process_idx}")
                                    step_count = 0  # 重置步数
                                else:
                                    self.logger.info(f"        所有进程已完成，退出循环")
                                    break
                        
                        if done:
                            break
                    
                    # 更新目标网络
                    if episode % algorithm_params.get('target_update_frequency', 10) == 0:
                        agent.update_target_network()
                    
                    # 更新探索参数
                    agent.update_exploration_params()
                    
                    episode_rewards.append(episode_reward)
                    episode_makespans.append(simulator.get_makespan())
                    
                    if episode % 10 == 0:
                        self.logger.info(f"    Episode {episode}: Reward={episode_reward:.2f}, Makespan={simulator.get_makespan():.2f}")
                        self.logger.info(f"      Total steps: {step_count}, Tasks scheduled: {len(simulator.get_schedule_history())}")
                
                # 最终评估
                final_metrics = {
                    'makespan': simulator.get_makespan(),
                    'resource_utilization': simulator.get_resource_utilization(),
                    'average_reward': np.mean(episode_rewards),
                    'final_episode_reward': episode_rewards[-1] if episode_rewards else 0,
                    'training_losses': agent.get_training_stats().get('losses', [])
                }
                
                results.append(final_metrics)
                
                # 为FE_IDDQN设置schedule_result
                schedule_result = {
                    'algorithm': algorithm_name,
                    'metrics': final_metrics,
                    'schedule_history': simulator.get_schedule_history()
                }

            elif algorithm_name == "DQN":
                state_size = 100  # 示例状态维度
                action_size = simulator.num_resources
                agent = DQNScheduler(state_size, action_size, self.device)
                schedule_result = simulator.simulate_random_schedule(algorithm_name)

            elif algorithm_name == "DDQN":
                state_size = 100  # 示例状态维度
                action_size = simulator.num_resources
                agent = DDQNScheduler(state_size, action_size, self.device)
                schedule_result = simulator.simulate_random_schedule(algorithm_name)
            elif algorithm_name == "BF_DDQN":
                state_size = 100  # 示例状态维度
                action_size = simulator.num_resources
                agent = BF_DDQNScheduler(state_size, action_size, self.device)
                schedule_result = simulator.simulate_random_schedule(algorithm_name)
            elif algorithm_name == "FIFO":
                from baselines.traditional_schedulers import FIFOScheduler
                scheduler = FIFOScheduler()
                # 转换依赖关系格式
                dependencies = [(dep['pre_task'], dep['post_task']) 
                              for dep in simulator.dependencies 
                              if dep['pre_task'] is not None and dep['post_task'] is not None]
                schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, dependencies)
                
                # 将调度结果转换为标准格式
                if 'error' not in schedule_result:
                    # 计算makespan和资源利用率
                    if 'task_end_times' in schedule_result and schedule_result['task_end_times']:
                        makespan = max(schedule_result['task_end_times'].values())
                    else:
                        makespan = 0
                    
                    # 计算资源利用率
                    total_work = sum(task['duration'] for task in simulator.tasks)
                    total_capacity = makespan * len(simulator.resources) if makespan > 0 else 1
                    resource_utilization = total_work / total_capacity if total_capacity > 0 else 0
                    
                    schedule_result.update({
                        'makespan': makespan,
                        'resource_utilization': resource_utilization
                    })
                
            elif algorithm_name == "SJF":
                from baselines.traditional_schedulers import SJFScheduler
                scheduler = SJFScheduler()
                dependencies = [(dep['pre_task'], dep['post_task']) 
                              for dep in simulator.dependencies 
                              if dep['pre_task'] is not None and dep['post_task'] is not None]
                schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, dependencies)
                
                # 将调度结果转换为标准格式
                if 'error' not in schedule_result:
                    # 计算makespan和资源利用率
                    if 'task_end_times' in schedule_result and schedule_result['task_end_times']:
                        makespan = max(schedule_result['task_end_times'].values())
                    else:
                        makespan = 0
                    
                    # 计算资源利用率
                    total_work = sum(task['duration'] for task in simulator.tasks)
                    total_capacity = makespan * len(simulator.resources) if makespan > 0 else 1
                    resource_utilization = total_work / total_capacity if total_capacity > 0 else 0
                    
                    schedule_result.update({
                        'makespan': makespan,
                        'resource_utilization': resource_utilization
                    })
                
            elif algorithm_name == "HEFT":
                from baselines.traditional_schedulers import HEFTScheduler
                scheduler = HEFTScheduler()
                dependencies = [(dep['pre_task'], dep['post_task']) 
                              for dep in simulator.dependencies 
                              if dep['pre_task'] is not None and dep['post_task'] is not None]
                schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, dependencies)
                
                # 将调度结果转换为标准格式
                if 'error' not in schedule_result:
                    # 计算makespan和资源利用率
                    if 'task_end_times' in schedule_result and schedule_result['task_end_times']:
                        makespan = max(schedule_result['task_end_times'].values())
                    else:
                        makespan = 0
                    
                    # 计算资源利用率
                    total_work = sum(task['duration'] for task in simulator.tasks)
                    total_capacity = makespan * len(simulator.resources) if makespan > 0 else 1
                    resource_utilization = total_work / total_capacity if total_capacity > 0 else 0
                    
                    schedule_result.update({
                        'makespan': makespan,
                        'resource_utilization': resource_utilization
                    })
            elif algorithm_name == "GA":
                from baselines.meta_heuristics import GAScheduler
                scheduler = GAScheduler()
                dependencies = [(dep['pre_task'], dep['post_task']) 
                              for dep in simulator.dependencies 
                              if dep['pre_task'] is not None and dep['post_task'] is not None]
                schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, dependencies)
            elif algorithm_name == "PSO":
                from baselines.meta_heuristics import PSOScheduler
                scheduler = PSOScheduler()
                dependencies = [(dep['pre_task'], dep['post_task']) 
                              for dep in simulator.dependencies 
                              if dep['pre_task'] is not None and dep['post_task'] is not None]
                schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, dependencies)
            elif algorithm_name == "ACO":
                from baselines.meta_heuristics import ACOScheduler
                scheduler = ACOScheduler()
                dependencies = [(dep['pre_task'], dep['post_task']) 
                              for dep in simulator.dependencies 
                              if dep['pre_task'] is not None and dep['post_task'] is not None]
                schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, dependencies)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")

            # 评估结果
            metrics = self.evaluator.evaluate(schedule_result)
            results.append(metrics)

        avg_results = self._calculate_average_metrics(results)
        self.logger.info(f"  Average metrics for {algorithm_name}: {avg_results}")
        return {algorithm_name: avg_results}

    def run_comparison_experiments(self, algorithms: List[str]) -> Dict[str, Dict[str, Any]]:
        """运行多个算法的对比实验"""
        all_results = {}
        for algo in algorithms:
            all_results.update(self.run_algorithm(algo))
        return all_results

    def _calculate_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算多次实验的平均指标"""
        if not results: return {}

        # 收集所有可能的键
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        avg_metrics = {}
        for key in all_keys:
            # 收集所有结果中该键的值
            values = []
            for result in results:
                if key in result:
                    values.append(result[key])
            
            if values:
                # 如果所有值都是数值类型，计算平均值
                if all(isinstance(v, (int, float)) for v in values):
                    avg_metrics[key] = np.mean(values)
                else:
                    # 非数值型直接取第一个
                    avg_metrics[key] = values[0]
            else:
                # 如果某个键在所有结果中都不存在，设为默认值
                avg_metrics[key] = 0.0
        
        return avg_metrics

    def generate_comparison_report(self, all_results: Dict[str, Dict[str, Any]]):
        """生成对比报告，包括图表和表格"""
        self.logger.info("Generating comparison report...")

        # 将结果转换为DataFrame方便处理
        results_df = pd.DataFrame.from_dict(all_results, orient='index')

        # 保存到CSV
        table_path = Config.get_table_file_path("comparison_metrics")
        results_df.to_csv(table_path)
        self.logger.info(f"Comparison metrics saved to {table_path}")

        # 可视化 (需要实现utils.visualization)
        # from utils.visualization import plot_radar_chart, plot_bar_chart
        # plot_radar_chart(results_df, Config.get_figure_file_path("radar_chart"))
        # plot_bar_chart(results_df, Config.get_figure_file_path("bar_chart"))

        self.logger.info("Comparison report generated.")