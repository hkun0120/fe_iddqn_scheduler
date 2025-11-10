#!/usr/bin/env python3
"""
5.3 基准调度算法与对比方法
全面评估FE-IDDQN的性能，实现多种代表性的基准调度算法进行对比

5.3.1 传统启发式调度算法
- FIFO: 先来先服务算法
- SJF: 最短任务优先算法（使用随机森林回归模型预测任务执行时间）
- HEFT: 异构最早完成时间算法（通信计算比率设为0.5）

5.3.2 元启发式调度算法
- GA: 遗传算法（种群大小100，交叉概率0.8，变异概率0.1，迭代次数200）
- PSO: 粒子群优化（粒子数50，惯性权重0.7，个体学习因子1.5，群体学习因子1.5，迭代次数150）
- ACO: 蚁群优化（蚂蚁数40，信息素挥发率0.5，信息素重要性因子1.0，启发式信息重要性因子2.0，迭代次数100）
"""

import pandas as pd
import numpy as np
import torch
import logging
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import networkx as nx

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from models.fe_iddqn import FE_IDDQN
from environment.historical_replay_simulator import HistoricalReplaySimulator
from data.mysql_data_loader import MySQLDataLoader
from baselines.rl_baselines import DQNScheduler, DDQNScheduler, BF_DDQNScheduler
from baselines.meta_heuristics import GAScheduler, PSOScheduler, ACOScheduler
from baselines.traditional_schedulers import FIFOScheduler, SJFScheduler, HEFTScheduler
from config.hyperparameters import Hyperparameters

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveAlgorithmComparison:
    """全面算法对比实验"""
    
    def __init__(self, model_path: str, test_data_file: str, use_parallel=True, max_workers=None):
        self.model_path = Path(model_path)
        self.test_data_file = test_data_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 并行化设置
        self.use_parallel = use_parallel
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        
        # 模型参数
        self.task_input_dim = 16
        self.resource_input_dim = 7
        self.action_dim = 6
        
        # 数据加载器（避免重复加载）
        self.data_loader = None
        
        # 初始化调度器
        self.schedulers = self._initialize_schedulers()
        
        # 结果存储
        self.results = {}
        self.workflow_groups = {}
        
    def _initialize_schedulers(self) -> Dict[str, Any]:
        """初始化所有调度算法"""
        schedulers = {}
        
        # 传统启发式算法
        schedulers['FIFO'] = FIFOScheduler(allow_parallel=False)  # 串行FIFO
        schedulers['SJF'] = SJFScheduler(use_prediction_model=True)  # 使用随机森林预测
        schedulers['HEFT'] = HEFTScheduler(communication_computation_ratio=0.5)  # 通信计算比率0.5
        
        # 元启发式算法（启用并行化）
        schedulers['GA'] = GAScheduler(use_parallel=self.use_parallel, max_workers=self.max_workers)
        schedulers['PSO'] = PSOScheduler()  # 使用论文参数
        schedulers['ACO'] = ACOScheduler()  # 使用论文参数
        
        # 强化学习算法（需要预训练，暂时跳过）
        # schedulers['DQN'] = DQNScheduler(state_size=100, action_size=6, device=self.device)
        # schedulers['DDQN'] = DDQNScheduler(state_size=100, action_size=6, device=self.device)
        # schedulers['BF-DDQN'] = BF_DDQN   Scheduler(state_size=100, action_size=6, device=self.device)
        
        # FE-IDDQN
        schedulers['FE-IDDQN'] = self._load_fe_iddqn_model()
        
        return schedulers
    
    def _load_fe_iddqn_model(self) -> FE_IDDQN:
        """加载FE-IDDQN模型"""
        try:
            # 加载模型
            model = FE_IDDQN(
                task_input_dim=self.task_input_dim,
                resource_input_dim=self.resource_input_dim,
                action_dim=self.action_dim,
                device=self.device
            )
            
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'q_network_state_dict' in checkpoint:
                    model.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                    model.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                    model.epsilon = checkpoint.get('epsilon', 0.05)
                    model.temperature = checkpoint.get('temperature', 0.2)
                else:
                    # 如果直接保存的是模型状态
                    model.q_network.load_state_dict(checkpoint)
                model.q_network.eval()
                logger.info(f"FE-IDDQN model loaded from {self.model_path}")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                
            return model
        except Exception as e:
            logger.error(f"Failed to load FE-IDDQN model: {e}")
            return None
    
    def load_test_data(self) -> pd.DataFrame:
        """加载测试数据"""
        try:
            data = pd.read_csv(self.test_data_file)
            logger.info(f"Loaded test data: {len(data)} workflows")
            return data
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return None
    
    def group_workflows_by_size(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        """按工作流大小分组"""
        # 使用workflow_size列进行分组
        groups = {
            'Small': [],
            'Medium': [],
            'Large': []
        }
        
        for _, row in data.iterrows():
            workflow_id = row['process_id']
            workflow_size = row['workflow_size'].lower()
            
            if workflow_size == 'small':
                groups['Small'].append(workflow_id)
            elif workflow_size == 'medium':
                groups['Medium'].append(workflow_id)
            elif workflow_size == 'large':
                groups['Large'].append(workflow_id)
        
        # 记录分组结果
        for group_name, workflow_ids in groups.items():
            logger.info(f"{group_name} workflows: {len(workflow_ids)} workflows")
        
        return groups
    
    def prepare_workflow_data(self, workflow_id: int, data: pd.DataFrame) -> Tuple[List[Dict], List[Dict], List[Tuple]]:
        """准备单个工作流的数据"""
        workflow_data = data[data['process_id'] == workflow_id]
        
        if workflow_data.empty:
            logger.warning(f"No data found for workflow {workflow_id}")
            return [], [], []
        
        row = workflow_data.iloc[0]
        
        # 从数据库获取任务实例数据
        tasks = self._load_task_instances(workflow_id)
        
       
        
        # 准备资源数据（6个异构资源）
        resources = [
            {'id': 0, 'name': 'Resource_0', 'cpu_capacity': 4, 'memory_capacity': 8},
            {'id': 1, 'name': 'Resource_1', 'cpu_capacity': 8, 'memory_capacity': 16},
            {'id': 2, 'name': 'Resource_2', 'cpu_capacity': 2, 'memory_capacity': 4},
            {'id': 3, 'name': 'Resource_3', 'cpu_capacity': 16, 'memory_capacity': 32},
            {'id': 4, 'name': 'Resource_4', 'cpu_capacity': 1, 'memory_capacity': 2},
            {'id': 5, 'name': 'Resource_5', 'cpu_capacity': 32, 'memory_capacity': 64}
        ]
        
        # 准备依赖关系 - 从数据库获取真实的依赖关系
        dependencies = self._load_task_dependencies(workflow_id, row.get('process_name', ''))
        
        return tasks, resources, dependencies
    
    def _get_data_loader(self):
        """获取数据加载器（单例模式）"""
        if self.data_loader is None:
            self.data_loader = MySQLDataLoader()
        return self.data_loader
    
    def _load_task_instances(self, workflow_id: int) -> List[Dict]:
        """从数据库加载任务实例数据"""
        try:
            # 使用单例数据加载器
            data_loader = self._get_data_loader()
            task_instances = data_loader.load_task_instances_by_workflow(workflow_id)
            
            tasks = []
            for i, task in enumerate(task_instances):
                # 计算任务持续时间
                start_time = pd.to_datetime(task['start_time']) if pd.notna(task['start_time']) else None
                end_time = pd.to_datetime(task['end_time']) if pd.notna(task['end_time']) else None
                
                if start_time and end_time:
                    duration = (end_time - start_time).total_seconds()
                else:
                    duration = 1.0  # 默认持续时间
                
                # 确保任务ID有效
                task_id = task.get('task_code')
                if pd.isna(task_id) or task_id is None or task_id == '':
                    task_id = f"task_{workflow_id}_{i}"  # 生成备用ID
                
                task_data = {
                    'id': task_id,
                    'name': task.get('name', f'task_{task_id}'),
                    'duration': duration,
                    'cpu_req': task.get('cpu_usage', 1),
                    'memory_req': task.get('memory_usage', 1),
                    'task_type': hash(str(task_id)) % 100,
                    'input_size': task.get('input_size', 0),
                    'priority': task.get('task_instance_priority', 1),
                    'retry_times': task.get('retry_times', 0),
                    'complexity': task.get('complexity', 1)
                }
                tasks.append(task_data)
            
            logger.info(f"Loaded {len(tasks)} task instances for workflow {workflow_id}")
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to load task instances for workflow {workflow_id}: {e}")
            return []
    
    def _load_task_dependencies(self, workflow_id: int, process_name: str) -> List[Tuple]:
        """从数据库加载任务依赖关系"""
        try:
            # 使用单例数据加载器
            data_loader = self._get_data_loader()
            
            # 获取任务实例数据以获取process_definition_code
            task_instances = data_loader.load_task_instances_by_workflow(workflow_id)
            if not task_instances:
                logger.warning(f"No task instances found for workflow {workflow_id}")
                return []
            
            # 从第一个任务实例获取process_definition_code
            process_definition_code = task_instances[0].get('process_definition_code')
            if not process_definition_code:
                logger.warning(f"No process_definition_code found for workflow {workflow_id}")
                return []
            
            # 获取依赖关系
            dependencies_df = data_loader.process_task_relation_df
            if dependencies_df is None:
                logger.info("Loading process_task_relation data...")
                dependencies_df = pd.read_sql("SELECT * FROM t_ds_process_task_relation", data_loader.engine)
            
            # 筛选指定流程的依赖关系
            workflow_dependencies = dependencies_df[
                dependencies_df['process_definition_code'] == process_definition_code
            ]
            
            # 构建依赖列表
            dependency_list = []
            for _, dep in workflow_dependencies.iterrows():
                if not pd.isna(dep['pre_task_code']) and not pd.isna(dep['post_task_code']):
                    dependency_list.append((dep['pre_task_code'], dep['post_task_code']))
            
            logger.info(f"Loaded {len(dependency_list)} dependencies for workflow {workflow_id}")
            if dependency_list:
                logger.info(f"Dependencies: {dependency_list[:5]}...")  # 显示前5个依赖关系
            return dependency_list
            
        except Exception as e:
            logger.error(f"Failed to load dependencies for workflow {workflow_id}: {e}")
            return []
    
    def train_sjf_model(self, historical_data: pd.DataFrame):
        """训练SJF的随机森林预测模型"""
        if 'SJF' in self.schedulers:
            # 准备历史数据
            historical_tasks = []
            for _, row in historical_data.iterrows():
                # 计算工作流总持续时间
                start_time = pd.to_datetime(row['start_time']) if pd.notna(row['start_time']) else None
                end_time = pd.to_datetime(row['end_time']) if pd.notna(row['end_time']) else None
                
                if start_time and end_time:
                    total_duration = (end_time - start_time).total_seconds()
                else:
                    total_duration = 300  # 默认5分钟
                
                # 为每个任务创建训练样本
                task_count = row['task_count']
                avg_task_duration = total_duration / task_count if task_count > 0 else 60
                
                for i in range(min(task_count, 10)):  # 限制每个工作流最多10个任务样本
                    task_data = {
                        'task_type': i % 10,
                        'input_size': np.random.randint(0, 1000),
                        'cpu_req': np.random.randint(1, 4),
                        'memory_req': np.random.randint(1, 8),
                        'duration': avg_task_duration * (0.5 + np.random.random())
                    }
                    historical_tasks.append(task_data)
            
            # 训练SJF预测模型
            self.schedulers['SJF'].train_prediction_model(historical_tasks)
            logger.info("SJF prediction model trained successfully")
    
    def run_single_workflow_comparison(self, workflow_id: int, data: pd.DataFrame) -> Dict[str, Any]:
        """运行单个工作流的算法对比"""
        logger.info(f"Comparing algorithms for workflow {workflow_id}")
        
        # 准备工作流数据
        tasks, resources, dependencies = self.prepare_workflow_data(workflow_id, data)
        
        if not tasks:
            logger.warning(f"No tasks found for workflow {workflow_id}")
            return {}
        
        results = {}
        
        # 运行每个算法
        for alg_name, scheduler in self.schedulers.items():
            if scheduler is None:
                continue
                
            try:
                start_time = time.time()
                
                if alg_name == 'FE-IDDQN':
                    # FE-IDDQN需要特殊处理
                    result = self._run_fe_iddqn_scheduling(tasks, resources, dependencies)
                else:
                    # 其他算法
                    result = scheduler.schedule(tasks, resources, dependencies)
                
                execution_time = time.time() - start_time
                
                # 计算额外指标
                result['execution_time'] = execution_time
                result['task_count'] = len(tasks)
                result['workflow_id'] = workflow_id
                
                results[alg_name] = result
                
                logger.info(f"{alg_name}: makespan={result.get('makespan', 0):.2f}s, "
                          f"utilization={result.get('resource_utilization', 0):.3f}, "
                          f"execution_time={execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error running {alg_name} on workflow {workflow_id}: {e}")
                results[alg_name] = {
                    'error': str(e),
                    'workflow_id': workflow_id,
                    'task_count': len(tasks)
                }
        
        return results
    
    def _run_workflows_parallel(self, workflow_ids: List[int], data: pd.DataFrame) -> Dict[int, Dict]:
        """并行运行多个工作流的算法对比"""
        logger.info(f"Running {len(workflow_ids)} workflows in parallel with {self.max_workers} workers")
        
        # 创建工作流测试函数
        test_func = partial(self.run_single_workflow_comparison, data=data)
        
        results = {}
        completed_count = 0
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_workflow = {
                    executor.submit(test_func, workflow_id): workflow_id 
                    for workflow_id in workflow_ids
                }
                
                # 收集结果
                for future in as_completed(future_to_workflow):
                    workflow_id = future_to_workflow[future]
                    try:
                        workflow_results = future.result()
                        results[workflow_id] = workflow_results
                        completed_count += 1
                        
                        logger.info(f"Completed workflow {workflow_id} ({completed_count}/{len(workflow_ids)})")
                        
                    except Exception as e:
                        logger.error(f"Error processing workflow {workflow_id}: {e}")
                        results[workflow_id] = {"error": str(e)}
                        completed_count += 1
        
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            # 回退到串行执行
            logger.info("Falling back to serial execution")
            for workflow_id in workflow_ids:
                if workflow_id not in results:
                    results[workflow_id] = self.run_single_workflow_comparison(workflow_id, data)
        
        return results
    
    def _run_fe_iddqn_scheduling(self, tasks: List[Dict], resources: List[Dict], 
                                dependencies: List[Tuple]) -> Dict[str, Any]:
        """运行FE-IDDQN调度"""
        try:
            # 创建模拟器
            simulator = HistoricalReplaySimulator()
            
            # 准备状态
            task_features = np.array([[task['duration'], task['cpu_req'], task['memory_req'], 
                                     task['task_type'], task['priority'], task['retry_times'],
                                     task['complexity'], 0, 0, 0, 0, 0, 0, 0, 0, 0] for task in tasks])
            
            resource_features = np.array([[res['cpu_capacity'], res['memory_capacity'], 0, 0, 0, 0, 0] 
                                        for res in resources])
            
            # 转换为张量
            task_tensor = torch.FloatTensor(task_features).unsqueeze(0).to(self.device)
            resource_tensor = torch.FloatTensor(resource_features).unsqueeze(0).to(self.device)
            
            # 获取Q值并选择动作
            with torch.no_grad():
                q_values = self.schedulers['FE-IDDQN'].q_network(task_tensor, resource_tensor)
                actions = torch.argmax(q_values, dim=-1).cpu().numpy().flatten()
            
            # 模拟调度过程
            task_assignments = {}
            task_start_times = {}
            task_end_times = {}
            resource_available_time = {res['id']: 0 for res in resources}
            
            # 按依赖关系排序任务
            dag = nx.DiGraph()
            for task in tasks:
                dag.add_node(task['id'], **task)
            for pre_task, post_task in dependencies:
                dag.add_edge(pre_task, post_task)
            
            # 拓扑排序
            try:
                topo_order = list(nx.topological_sort(dag))
            except:
                topo_order = [task['id'] for task in tasks]
            
            # 调度任务
            for i, task_id in enumerate(topo_order):
                if i < len(actions):
                    resource_id = actions[i] % len(resources)
                    task = next(t for t in tasks if t['id'] == task_id)
                    
                    # 检查资源约束
                    resource = resources[resource_id]
                    if (resource['cpu_capacity'] >= task['cpu_req'] and 
                        resource['memory_capacity'] >= task['memory_req']):
                        
                        # 计算开始时间
                        dependency_ready_time = 0
                        for pre_task, post_task in dependencies:
                            if post_task == task_id and pre_task in task_end_times:
                                dependency_ready_time = max(dependency_ready_time, task_end_times[pre_task])
                        
                        start_time = max(resource_available_time[resource_id], dependency_ready_time)
                        end_time = start_time + task['duration']
                        
                        # 更新状态
                        task_assignments[task_id] = resource_id
                        task_start_times[task_id] = start_time
                        task_end_times[task_id] = end_time
                        resource_available_time[resource_id] = end_time
            
            # 计算指标
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
                'algorithm': 'FE-IDDQN'
            }
            
        except Exception as e:
            logger.error(f"FE-IDDQN scheduling error: {e}")
            return {
                'error': str(e),
                'algorithm': 'FE-IDDQN'
            }
    
    def run_comparison_experiment(self, sample_size: int = 20) -> Dict[str, Any]:
        """运行完整的对比实验"""
        logger.info("Starting comprehensive algorithm comparison experiment")
        
        # 加载测试数据
        data = self.load_test_data()
        if data is None:
            return {}
        
        # 按大小分组工作流
        self.workflow_groups = self.group_workflows_by_size(data)
        
        # 训练SJF预测模型
        self.train_sjf_model(data)
        
        # 运行对比实验
        all_results = {}
        
        for group_name, workflow_ids in self.workflow_groups.items():
            logger.info(f"Testing {group_name} workflows")
            
            # 随机采样工作流
            sample_workflows = np.random.choice(workflow_ids, 
                                             min(sample_size, len(workflow_ids)), 
                                             replace=False)
            
            group_results = {}
            
            # 并行测试工作流
            if self.use_parallel and len(sample_workflows) > 2:
                group_results = self._run_workflows_parallel(sample_workflows, data)
            else:
                # 串行测试工作流
                for workflow_id in sample_workflows:
                    workflow_results = self.run_single_workflow_comparison(workflow_id, data)
                    group_results[workflow_id] = workflow_results
            
            all_results[group_name] = group_results
        
        # 保存结果
        self.results = all_results
        self._save_results()
        
        # 生成分析报告
        self._generate_analysis_report()
        
        return all_results
    
    def _save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"comprehensive_comparison_results_{timestamp}.json"
        
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        # 转换结果数据
        converted_results = convert_numpy_types(self.results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    def _generate_analysis_report(self):
        """生成分析报告"""
        logger.info("Generating analysis report")
        
        # 计算统计指标
        stats = self._calculate_statistics()
        
        # 生成可视化图表
        self._create_visualizations(stats)
        
        # 生成文本报告
        self._create_text_report(stats)
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """计算统计指标"""
        stats = {}
        
        for group_name, group_results in self.results.items():
            group_stats = {
                'algorithms': {},
                'workflow_count': len(group_results)
            }
            
            # 收集每个算法的指标
            for alg_name in self.schedulers.keys():
                makespans = []
                utilizations = []
                execution_times = []
                
                for workflow_id, workflow_results in group_results.items():
                    if alg_name in workflow_results and 'error' not in workflow_results[alg_name]:
                        result = workflow_results[alg_name]
                        makespans.append(result.get('makespan', 0))
                        utilizations.append(result.get('resource_utilization', 0))
                        execution_times.append(result.get('execution_time', 0))
                
                if makespans:
                    group_stats['algorithms'][alg_name] = {
                        'avg_makespan': np.mean(makespans),
                        'std_makespan': np.std(makespans),
                        'avg_utilization': np.mean(utilizations),
                        'std_utilization': np.std(utilizations),
                        'avg_execution_time': np.mean(execution_times),
                        'std_execution_time': np.std(execution_times),
                        'sample_count': len(makespans)
                    }
            
            stats[group_name] = group_stats
        
        return stats
    
    def _create_visualizations(self, stats: Dict[str, Any]):
        """创建可视化图表"""
        # 创建图表目录
        viz_dir = Path("comparison_visualizations")
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Makespan对比图
        self._plot_makespan_comparison(stats, viz_dir)
        
        # 2. 资源利用率对比图
        self._plot_utilization_comparison(stats, viz_dir)
        
        # 3. 算法性能综合对比
        self._plot_comprehensive_comparison(stats, viz_dir)
    
    def _plot_makespan_comparison(self, stats: Dict[str, Any], output_dir: Path):
        """绘制Makespan对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (group_name, group_stats) in enumerate(stats.items()):
            ax = axes[i]
            
            algorithms = []
            avg_makespans = []
            std_makespans = []
            
            for alg_name, alg_stats in group_stats['algorithms'].items():
                algorithms.append(alg_name)
                avg_makespans.append(alg_stats['avg_makespan'])
                std_makespans.append(alg_stats['std_makespan'])
            
            # 绘制柱状图
            bars = ax.bar(algorithms, avg_makespans, yerr=std_makespans, 
                         capsize=5, alpha=0.7)
            
            # 突出显示FE-IDDQN
            for j, alg in enumerate(algorithms):
                if alg == 'FE-IDDQN':
                    bars[j].set_color('red')
                    bars[j].set_alpha(1.0)
            
            ax.set_title(f'{group_name} Workflows - Makespan Comparison')
            ax.set_ylabel('Makespan (seconds)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'makespan_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_utilization_comparison(self, stats: Dict[str, Any], output_dir: Path):
        """绘制资源利用率对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (group_name, group_stats) in enumerate(stats.items()):
            ax = axes[i]
            
            algorithms = []
            avg_utilizations = []
            std_utilizations = []
            
            for alg_name, alg_stats in group_stats['algorithms'].items():
                algorithms.append(alg_name)
                avg_utilizations.append(alg_stats['avg_utilization'])
                std_utilizations.append(alg_stats['std_utilization'])
            
            # 绘制柱状图
            bars = ax.bar(algorithms, avg_utilizations, yerr=std_utilizations, 
                         capsize=5, alpha=0.7)
            
            # 突出显示FE-IDDQN
            for j, alg in enumerate(algorithms):
                if alg == 'FE-IDDQN':
                    bars[j].set_color('red')
                    bars[j].set_alpha(1.0)
            
            ax.set_title(f'{group_name} Workflows - Resource Utilization Comparison')
            ax.set_ylabel('Resource Utilization')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'utilization_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_comprehensive_comparison(self, stats: Dict[str, Any], output_dir: Path):
        """绘制综合性能对比图"""
        # 收集所有数据
        all_data = []
        
        for group_name, group_stats in stats.items():
            for alg_name, alg_stats in group_stats['algorithms'].items():
                all_data.append({
                    'Group': group_name,
                    'Algorithm': alg_name,
                    'Makespan': alg_stats['avg_makespan'],
                    'Utilization': alg_stats['avg_utilization'],
                    'Execution_Time': alg_stats['avg_execution_time']
                })
        
        df = pd.DataFrame(all_data)
        
        # 创建综合对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Makespan热力图
        makespan_pivot = df.pivot(index='Algorithm', columns='Group', values='Makespan')
        sns.heatmap(makespan_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0,0])
        axes[0,0].set_title('Makespan Comparison Heatmap')
        
        # 2. 资源利用率热力图
        util_pivot = df.pivot(index='Algorithm', columns='Group', values='Utilization')
        sns.heatmap(util_pivot, annot=True, fmt='.3f', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title('Resource Utilization Comparison Heatmap')
        
        # 3. 算法性能雷达图（以FE-IDDQN为基准）
        self._plot_radar_chart(df, axes[1,0])
        
        # 4. 性能排名
        self._plot_performance_ranking(df, axes[1,1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_radar_chart(self, df: pd.DataFrame, ax):
        """绘制性能雷达图"""
        # 选择FE-IDDQN作为基准
        fe_iddqn_data = df[df['Algorithm'] == 'FE-IDDQN'].iloc[0]
        
        # 计算相对性能
        algorithms = df['Algorithm'].unique()
        metrics = ['Makespan', 'Utilization']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        
        for alg in algorithms:
            alg_data = df[df['Algorithm'] == alg]
            values = []
            for metric in metrics:
                # 对于Makespan，越小越好，所以取倒数
                if metric == 'Makespan':
                    value = 1 / alg_data[metric].mean()
                else:
                    value = alg_data[metric].mean()
                values.append(value)
            
            values += values[:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=alg)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_title('Algorithm Performance Radar Chart')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    def _plot_performance_ranking(self, df: pd.DataFrame, ax):
        """绘制性能排名图"""
        # 计算综合得分（Makespan越小越好，Utilization越大越好）
        df['Score'] = df['Utilization'] / df['Makespan']
        
        # 按组计算排名
        rankings = []
        for group in df['Group'].unique():
            group_data = df[df['Group'] == group].copy()
            group_data['Rank'] = group_data['Score'].rank(ascending=False)
            rankings.append(group_data)
        
        ranking_df = pd.concat(rankings)
        
        # 计算平均排名
        avg_rankings = ranking_df.groupby('Algorithm')['Rank'].mean().sort_values()
        
        # 绘制排名图
        bars = ax.barh(range(len(avg_rankings)), avg_rankings.values)
        
        # 突出显示FE-IDDQN
        for i, alg in enumerate(avg_rankings.index):
            if alg == 'FE-IDDQN':
                bars[i].set_color('red')
        
        ax.set_yticks(range(len(avg_rankings)))
        ax.set_yticklabels(avg_rankings.index)
        ax.set_xlabel('Average Rank (1 = Best)')
        ax.set_title('Algorithm Performance Ranking')
        ax.grid(True, alpha=0.3)
    
    def _create_text_report(self, stats: Dict[str, Any]):
        """创建文本报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"comprehensive_comparison_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 5.3 基准调度算法与对比方法 - 实验结果报告\n\n")
            f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 实验概述
            f.write("## 实验概述\n\n")
            f.write("本实验对比了FE-IDDQN与多种基准调度算法的性能，包括传统启发式算法和元启发式算法。\n\n")
            
            # 算法参数
            f.write("## 算法参数设置\n\n")
            f.write("### 5.3.1 传统启发式调度算法\n")
            f.write("- **FIFO**: 先来先服务算法（串行版本）\n")
            f.write("- **SJF**: 最短任务优先算法，使用随机森林回归模型预测任务执行时间\n")
            f.write("- **HEFT**: 异构最早完成时间算法，通信计算比率设为0.5\n\n")
            
            f.write("### 5.3.2 元启发式调度算法\n")
            f.write("- **GA**: 遗传算法（种群大小100，交叉概率0.8，变异概率0.1，迭代次数200）\n")
            f.write("- **PSO**: 粒子群优化（粒子数50，惯性权重0.7，个体学习因子1.5，群体学习因子1.5，迭代次数150）\n")
            f.write("- **ACO**: 蚁群优化（蚂蚁数40，信息素挥发率0.5，信息素重要性因子1.0，启发式信息重要性因子2.0，迭代次数100）\n\n")
            
            # 实验结果
            f.write("## 实验结果\n\n")
            
            for group_name, group_stats in stats.items():
                f.write(f"### {group_name}工作流结果\n\n")
                f.write(f"**测试工作流数量**: {group_stats['workflow_count']}\n\n")
                
                # 创建结果表格
                f.write("| 算法 | 平均Makespan(s) | 标准差 | 平均资源利用率 | 标准差 | 样本数 |\n")
                f.write("|------|----------------|--------|----------------|--------|--------|\n")
                
                for alg_name, alg_stats in group_stats['algorithms'].items():
                    f.write(f"| {alg_name} | {alg_stats['avg_makespan']:.2f} | "
                           f"{alg_stats['std_makespan']:.2f} | "
                           f"{alg_stats['avg_utilization']:.3f} | "
                           f"{alg_stats['std_utilization']:.3f} | "
                           f"{alg_stats['sample_count']} |\n")
                
                f.write("\n")
            
            # 结论
            f.write("## 结论\n\n")
            f.write("通过对比实验，可以评估FE-IDDQN相对于传统调度算法的性能优势。\n")
            f.write("详细的性能分析和可视化图表请参考生成的PNG文件。\n")
        
        logger.info(f"Report saved to {report_file}")

def main():
    """主函数"""
    # 配置参数
    model_path = "fe_iddqn_training_system/fe_iddqn_training_system/models/fe_iddqn_best.pkl"
    test_data_file = "fe_iddqn_training_system/data/test_data_20250930_122033.csv"
    
    # 检测CPU核心数
    cpu_count = mp.cpu_count()
    logger.info(f"Detected {cpu_count} CPU cores")
    
    # 创建对比实验（启用并行化）
    comparison = ComprehensiveAlgorithmComparison(
        model_path, 
        test_data_file, 
        use_parallel=True, 
        max_workers=min(cpu_count, 8)  # 限制最大线程数
    )
    
    # 运行实验
    start_time = time.time()
    results = comparison.run_comparison_experiment(sample_size=20)
    end_time = time.time()
    
    logger.info(f"Comprehensive algorithm comparison completed in {end_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()
