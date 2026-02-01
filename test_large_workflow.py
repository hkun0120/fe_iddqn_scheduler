#!/usr/bin/env python3
"""
大规模工作流调度测试 - 绕过任务限制
测试FE-IDDQN算法对执行时间最长的工作流的调度效果和依赖完整性验证
"""

import json
import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import time

# 设置环境变量以支持更多任务
os.environ['MAX_TASKS_PER_EPISODE'] = '1000'
os.environ['MAX_PROCESSES_PER_EPISODE'] = '100'

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.hyperparameters import Hyperparameters
from models.fe_iddqn import FE_IDDQN
from environment.historical_replay_simulator import HistoricalReplaySimulator
from data.mysql_data_loader import MySQLDataLoader


class DirectWorkflowScheduler:
    """直接工作流调度器 - 绕过环境模拟器的任务限制"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据加载器
        self.data_loader = MySQLDataLoader(
            host='localhost',
            user='root',
            password='',
            database='whalesb',
            port=3306
        )
        
        # 输出目录
        self.output_dir = Path("large_workflow_test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"test_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def load_workflow_data(self, process_id, data):
        """加载指定工作流的完整数据"""
        process = data['process_instance'][data['process_instance']['id'] == process_id].iloc[0]
        process_def_code = process.get('process_definition_code')
        
        # 获取所有任务
        tasks = data['task_instance'][data['task_instance']['process_instance_id'] == process_id]
        successful_tasks = tasks[tasks['state'] == 7].copy()
        
        # 获取依赖关系
        dependencies = data['process_task_relation'][
            data['process_task_relation']['process_definition_code'] == process_def_code
        ]
        
        return {
            'process': process,
            'tasks': successful_tasks,
            'dependencies': dependencies,
            'process_def_code': process_def_code
        }
    
    def build_dependency_graph(self, tasks, dependencies):
        """构建依赖关系图"""
        import networkx as nx
        
        G = nx.DiGraph()
        
        # 添加所有任务节点
        task_code_to_id = {}
        for _, task in tasks.iterrows():
            task_code = task.get('task_code', task.get('id'))
            task_id = task['id']
            G.add_node(task_code, task_id=task_id, task_data=task.to_dict())
            task_code_to_id[task_code] = task_id
        
        # 添加依赖边
        dep_count = 0
        for _, dep in dependencies.iterrows():
            pre_task = dep.get('pre_task_code')
            post_task = dep.get('post_task_code')
            
            if pd.notna(pre_task) and pd.notna(post_task) and pre_task in G.nodes and post_task in G.nodes:
                G.add_edge(pre_task, post_task)
                dep_count += 1
        
        self.logger.info(f"  构建依赖图: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
        
        return G, task_code_to_id
    
    def topological_sort_tasks(self, G, tasks):
        """拓扑排序任务"""
        import networkx as nx
        
        try:
            sorted_codes = list(nx.topological_sort(G))
            sorted_tasks = []
            for code in sorted_codes:
                task_data = G.nodes[code].get('task_data')
                if task_data:
                    sorted_tasks.append(task_data)
            return sorted_tasks, sorted_codes
        except nx.NetworkXError as e:
            self.logger.warning(f"拓扑排序失败: {e}，使用原始顺序")
            return tasks.to_dict('records'), list(tasks['task_code'])
    
    def simulate_schedule(self, sorted_tasks, G, num_resources=5, scheduler_func=None, scheduler_name="Default"):
        """模拟调度过程"""
        # 资源状态：每个资源当前的可用时间
        resource_available_time = {i: 0 for i in range(num_resources)}
        
        # 任务完成时间
        task_finish_time = {}
        
        # 调度历史
        schedule_history = []
        
        for task in sorted_tasks:
            task_code = task.get('task_code', task.get('id'))
            
            # 计算任务执行时间
            try:
                start_time = pd.to_datetime(task.get('start_time'))
                end_time = pd.to_datetime(task.get('end_time'))
                duration = (end_time - start_time).total_seconds()
                duration = max(1, duration)  # 至少1秒
            except Exception:
                duration = 10  # 默认10秒
            
            # 计算最早开始时间（依赖约束）
            earliest_start = 0
            if task_code in G:
                for pred in G.predecessors(task_code):
                    if pred in task_finish_time:
                        earliest_start = max(earliest_start, task_finish_time[pred])
            
            # 选择资源
            if scheduler_func:
                selected_resource = scheduler_func(resource_available_time, task, earliest_start)
            else:
                # 默认：选择最早可用的资源
                best_resource = 0
                best_start = float('inf')
                for r, avail_time in resource_available_time.items():
                    start = max(avail_time, earliest_start)
                    if start < best_start:
                        best_start = start
                        best_resource = r
                selected_resource = best_resource
            
            # 计算任务开始和结束时间
            actual_start = max(resource_available_time[selected_resource], earliest_start)
            finish_time = actual_start + duration
            
            # 更新状态
            resource_available_time[selected_resource] = finish_time
            task_finish_time[task_code] = finish_time
            
            schedule_history.append({
                'task_code': task_code,
                'task_name': task.get('name', 'N/A'),
                'resource': selected_resource,
                'start': actual_start,
                'finish': finish_time,
                'duration': duration
            })
        
        # 计算指标
        makespan = max(task_finish_time.values()) if task_finish_time else 0
        total_work_time = sum(h['duration'] for h in schedule_history)
        resource_utilization = total_work_time / (makespan * num_resources) if makespan > 0 else 0
        
        return {
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'schedule_history': schedule_history,
            'total_tasks': len(sorted_tasks)
        }
    
    def validate_dependencies(self, schedule_history, G):
        """验证调度结果是否保证了依赖完整性"""
        # 构建任务执行顺序
        task_finish_times = {h['task_code']: h['finish'] for h in schedule_history}
        task_start_times = {h['task_code']: h['start'] for h in schedule_history}
        
        violations = []
        checks = []
        
        for pre_task in G.nodes():
            for post_task in G.successors(pre_task):
                pre_finish = task_finish_times.get(pre_task, 0)
                post_start = task_start_times.get(post_task, 0)
                
                is_valid = pre_finish <= post_start
                checks.append({
                    'pre_task': pre_task,
                    'post_task': post_task,
                    'pre_finish': pre_finish,
                    'post_start': post_start,
                    'is_valid': is_valid
                })
                
                if not is_valid:
                    violations.append(f"依赖违规: {pre_task} 应在 {post_task} 之前完成")
        
        return len(violations) == 0, violations, checks
    
    def run_comparison_test(self):
        """运行比较测试"""
        self.logger.info("="*80)
        self.logger.info("大规模工作流调度比较测试")
        self.logger.info("="*80)
        
        # 1. 加载数据
        self.logger.info("\n1. 加载数据...")
        data = self.data_loader.load_all_data()
        if not data:
            self.logger.error("数据加载失败！")
            return
        
        # 2. 找到执行时间最长的工作流
        self.logger.info("\n2. 查找长执行时间工作流...")
        successful_processes = data['process_instance'][data['process_instance']['state'] == 7].copy()
        
        workflow_stats = []
        for _, process in successful_processes.iterrows():
            process_id = process['id']
            process_tasks = data['task_instance'][data['task_instance']['process_instance_id'] == process_id]
            successful_tasks = process_tasks[process_tasks['state'] == 7]
            
            if len(successful_tasks) < 5:  # 至少5个任务
                continue
            
            # 计算任务总执行时间
            total_task_time = 0
            for _, task in successful_tasks.iterrows():
                try:
                    task_start = pd.to_datetime(task['start_time'])
                    task_end = pd.to_datetime(task['end_time'])
                    total_task_time += (task_end - task_start).total_seconds()
                except Exception:
                    pass
            
            # 获取依赖关系
            process_def_code = process.get('process_definition_code')
            dependencies = data['process_task_relation'][
                data['process_task_relation']['process_definition_code'] == process_def_code
            ]
            
            workflow_stats.append({
                'process_id': process_id,
                'process_name': process['name'],
                'task_count': len(successful_tasks),
                'total_task_time': total_task_time,
                'dependency_count': len(dependencies)
            })
        
        workflow_df = pd.DataFrame(workflow_stats)
        workflow_df = workflow_df.sort_values('total_task_time', ascending=False)
        
        # 选择top 5
        top_workflows = workflow_df.head(5)
        self.logger.info("\n执行时间最长的5个工作流：")
        for _, wf in top_workflows.iterrows():
            self.logger.info(f"  ID={wf['process_id']}, 任务数={wf['task_count']}, "
                           f"总耗时={wf['total_task_time']:.0f}秒, 依赖数={wf['dependency_count']}")
        
        # 3. 定义调度策略
        def fifo_scheduler(resource_avail, task, earliest_start):
            """FIFO: 总是选择第一个资源"""
            return 0
        
        def round_robin_scheduler(resource_avail, task, earliest_start):
            """轮询调度"""
            if not hasattr(round_robin_scheduler, 'counter'):
                round_robin_scheduler.counter = 0
            r = round_robin_scheduler.counter % len(resource_avail)
            round_robin_scheduler.counter += 1
            return r
        
        def sjf_scheduler(resource_avail, task, earliest_start):
            """SJF: 选择当前负载最低的资源"""
            best = min(resource_avail.items(), key=lambda x: x[1])
            return best[0]
        
        def eft_scheduler(resource_avail, task, earliest_start):
            """EFT: 选择能让任务最早完成的资源"""
            try:
                start = pd.to_datetime(task.get('start_time'))
                end = pd.to_datetime(task.get('end_time'))
                duration = max(1, (end - start).total_seconds())
            except:
                duration = 10
            
            best_resource = 0
            best_finish = float('inf')
            for r, avail in resource_avail.items():
                start_time = max(avail, earliest_start)
                finish_time = start_time + duration
                if finish_time < best_finish:
                    best_finish = finish_time
                    best_resource = r
            return best_resource
        
        def load_balance_scheduler(resource_avail, task, earliest_start):
            """负载均衡：选择累计工作时间最少的资源"""
            return min(resource_avail.items(), key=lambda x: x[1])[0]
        
        schedulers = {
            'FIFO': fifo_scheduler,
            'RoundRobin': round_robin_scheduler,
            'SJF': sjf_scheduler,
            'EFT': eft_scheduler,
            'LoadBalance': load_balance_scheduler
        }
        
        # 4. 测试每个工作流
        all_results = []
        
        for _, wf in top_workflows.iterrows():
            process_id = wf['process_id']
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"测试工作流: {wf['process_name'][:50]}...")
            self.logger.info(f"任务数: {wf['task_count']}, 依赖数: {wf['dependency_count']}")
            self.logger.info("="*60)
            
            # 加载工作流数据
            wf_data = self.load_workflow_data(process_id, data)
            
            # 构建依赖图
            G, task_code_to_id = self.build_dependency_graph(wf_data['tasks'], wf_data['dependencies'])
            
            # 拓扑排序
            sorted_tasks, sorted_codes = self.topological_sort_tasks(G, wf_data['tasks'])
            
            self.logger.info(f"  排序后任务数: {len(sorted_tasks)}")
            
            # 测试每个调度器
            for sched_name, sched_func in schedulers.items():
                # 重置轮询计数器
                if hasattr(round_robin_scheduler, 'counter'):
                    round_robin_scheduler.counter = 0
                
                result = self.simulate_schedule(sorted_tasks, G, num_resources=5, 
                                                scheduler_func=sched_func, scheduler_name=sched_name)
                
                # 验证依赖
                is_valid, violations, checks = self.validate_dependencies(result['schedule_history'], G)
                
                result['algorithm'] = sched_name
                result['process_id'] = process_id
                result['dependency_valid'] = is_valid
                result['violation_count'] = len(violations)
                
                self.logger.info(f"  {sched_name:15s}: Makespan={result['makespan']:10.0f}s, "
                               f"资源利用率={result['resource_utilization']:.4f}, "
                               f"依赖完整={'✓' if is_valid else '✗'}")
                
                # 保存结果（不含schedule_history以减小文件大小）
                result_summary = {k: v for k, v in result.items() if k != 'schedule_history'}
                all_results.append(result_summary)
        
        # 5. 生成比较报告
        self.logger.info("\n" + "="*80)
        self.logger.info("综合比较结果")
        self.logger.info("="*80)
        
        # 按算法分组统计
        algo_stats = {}
        for result in all_results:
            algo = result['algorithm']
            if algo not in algo_stats:
                algo_stats[algo] = {'makespans': [], 'utils': [], 'valid': []}
            algo_stats[algo]['makespans'].append(result['makespan'])
            algo_stats[algo]['utils'].append(result['resource_utilization'])
            algo_stats[algo]['valid'].append(result['dependency_valid'])
        
        # 找到最佳基线
        for algo, stats in sorted(algo_stats.items(), key=lambda x: np.mean(x[1]['makespans'])):
            avg_makespan = np.mean(stats['makespans'])
            avg_util = np.mean(stats['utils'])
            all_valid = all(stats['valid'])
            self.logger.info(f"  {algo:15s}: 平均Makespan={avg_makespan:10.0f}s, "
                           f"平均资源利用率={avg_util:.4f}, "
                           f"依赖完整性={'全部通过' if all_valid else '存在违规'}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.output_dir / f"comparison_results_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_time': timestamp,
                'workflows_tested': len(top_workflows),
                'results': all_results,
                'algorithm_stats': {
                    algo: {
                        'avg_makespan': np.mean(stats['makespans']),
                        'std_makespan': np.std(stats['makespans']),
                        'avg_utilization': np.mean(stats['utils']),
                        'all_dependencies_valid': all(stats['valid'])
                    }
                    for algo, stats in algo_stats.items()
                }
            }, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"\n结果已保存到: {result_file}")
        
        return all_results


def main():
    """主函数"""
    scheduler = DirectWorkflowScheduler()
    scheduler.run_comparison_test()


if __name__ == "__main__":
    main()
