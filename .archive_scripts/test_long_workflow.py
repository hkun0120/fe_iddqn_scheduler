#!/usr/bin/env python3
"""
长执行时间工作流测试脚本
测试FE-IDDQN算法对长执行时间工作流的调度效果，并验证依赖完整性
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

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Config
from config.hyperparameters import Hyperparameters
from models.fe_iddqn import FE_IDDQN
from environment.historical_replay_simulator import HistoricalReplaySimulator
from data.mysql_data_loader import MySQLDataLoader


class SimpleBaselineScheduler:
    """简单的基线调度器包装器"""
    
    def __init__(self, name, action_dim):
        self.name = name
        self.action_dim = action_dim
        self.call_count = 0
    
    def select_action(self, task_features, resource_features):
        """根据调度策略选择动作"""
        if self.name == 'FIFO':
            # 先进先出：总是选择第一个资源
            return 0
        elif self.name == 'RoundRobin':
            # 轮询：循环选择资源
            action = self.call_count % self.action_dim
            self.call_count += 1
            return action
        elif self.name == 'SJF':
            # 最短作业优先：选择负载最低的资源
            if resource_features is not None and len(resource_features.shape) >= 2:
                # 获取资源特征 [batch, num_resources, features]
                rf = resource_features[0] if len(resource_features.shape) == 3 else resource_features
                # 计算每个资源的负载（已用CPU + 已用内存）
                loads = []
                for i in range(len(rf)):
                    cpu_used = rf[i][2] if len(rf[i]) > 2 else 0
                    mem_used = rf[i][3] if len(rf[i]) > 3 else 0
                    loads.append(cpu_used + mem_used)
                return int(np.argmin(loads)) if loads else 0
            return 0
        elif self.name == 'LoadBalance':
            # 负载均衡：选择剩余容量最多的资源
            if resource_features is not None and len(resource_features.shape) >= 2:
                rf = resource_features[0] if len(resource_features.shape) == 3 else resource_features
                capacities = []
                for i in range(len(rf)):
                    # 剩余CPU和内存容量在特征的第5和第6位
                    remaining_cpu = rf[i][5] if len(rf[i]) > 5 else 1
                    remaining_mem = rf[i][6] if len(rf[i]) > 6 else 1
                    capacities.append(remaining_cpu + remaining_mem)
                return int(np.argmax(capacities)) if capacities else 0
            return 0
        elif self.name == 'Random':
            # 随机选择
            return np.random.randint(0, self.action_dim)
        else:
            return 0


class LongWorkflowTester:
    """长执行时间工作流测试器"""
    
    def __init__(self, output_dir="test_long_workflow_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据加载器
        self.data_loader = MySQLDataLoader(
            host='localhost',
            user='root',
            password='',
            database='whalesb',
            port=3306
        )
        
        # 模型参数
        self.task_input_dim = 16
        self.resource_input_dim = 7
        self.action_dim = 5
    
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
    
    def find_long_execution_workflows(self, data, top_n=5):
        """找到执行时间最长的工作流实例"""
        self.logger.info("正在查找执行时间最长的工作流...")
        
        # 获取成功的进程实例
        successful_processes = data['process_instance'][data['process_instance']['state'] == 7].copy()
        
        # 计算每个工作流的执行时间和任务数量
        workflow_stats = []
        for _, process in successful_processes.iterrows():
            process_id = process['id']
            process_tasks = data['task_instance'][data['task_instance']['process_instance_id'] == process_id]
            successful_tasks = process_tasks[process_tasks['state'] == 7]
            
            if len(successful_tasks) == 0:
                continue
            
            # 计算执行时间
            try:
                start_time = pd.to_datetime(process['start_time'])
                end_time = pd.to_datetime(process['end_time'])
                execution_time = (end_time - start_time).total_seconds()
            except Exception:
                execution_time = 0
            
            # 计算任务总执行时间
            total_task_time = 0
            for _, task in successful_tasks.iterrows():
                try:
                    task_start = pd.to_datetime(task['start_time'])
                    task_end = pd.to_datetime(task['end_time'])
                    total_task_time += (task_end - task_start).total_seconds()
                except Exception:
                    pass
            
            # 获取依赖关系数量
            process_def_code = process.get('process_definition_code')
            dependencies = data['process_task_relation'][
                data['process_task_relation']['process_definition_code'] == process_def_code
            ]
            
            workflow_stats.append({
                'process_id': process_id,
                'process_name': process['name'],
                'task_count': len(successful_tasks),
                'workflow_execution_time': execution_time,
                'total_task_time': total_task_time,
                'dependency_count': len(dependencies),
                'process_definition_code': process_def_code,
                'start_time': process['start_time'],
                'end_time': process['end_time']
            })
        
        # 按执行时间排序
        workflow_df = pd.DataFrame(workflow_stats)
        workflow_df = workflow_df.sort_values('total_task_time', ascending=False)
        
        self.logger.info(f"共找到 {len(workflow_df)} 个有效工作流")
        self.logger.info("\n执行时间最长的工作流：")
        
        top_workflows = workflow_df.head(top_n)
        for _, wf in top_workflows.iterrows():
            self.logger.info(f"  ID: {wf['process_id']}, 名称: {wf['process_name'][:30] if wf['process_name'] else 'N/A'}...")
            self.logger.info(f"    任务数: {wf['task_count']}, 执行时间: {wf['workflow_execution_time']:.2f}秒")
            self.logger.info(f"    任务总耗时: {wf['total_task_time']:.2f}秒, 依赖数: {wf['dependency_count']}")
        
        return top_workflows
    
    def create_simulator_for_workflow(self, data, process_id):
        """为特定工作流创建仿真器"""
        process_instance = data['process_instance'][data['process_instance']['id'] == process_id]
        task_instances = data['task_instance'][data['task_instance']['process_instance_id'] == process_id]
        successful_tasks = task_instances[task_instances['state'] == 7]
        
        if len(successful_tasks) == 0:
            return None
        
        filtered_data = {
            'process_instance': process_instance,
            'task_instance': successful_tasks,
            'task_definition': data['task_definition'],
            'process_task_relation': data['process_task_relation']
        }
        
        return HistoricalReplaySimulator(
            filtered_data['process_instance'],
            filtered_data['task_instance'],
            filtered_data['task_definition'],
            filtered_data['process_task_relation']
        )
    
    def validate_dependency_integrity(self, simulator, schedule_history):
        """验证调度结果是否保证了依赖完整性"""
        self.logger.info("\n验证依赖完整性...")
        
        violations = []
        dependency_checks = []
        
        if not hasattr(simulator, 'current_process_dependencies') or not simulator.current_process_dependencies:
            self.logger.warning("未找到依赖关系定义，跳过依赖验证")
            return True, [], []
        
        # 构建任务调度顺序映射
        task_schedule_order = {}
        for idx, record in enumerate(schedule_history):
            task_code = record.get('task_code', record.get('task_id'))
            if task_code not in task_schedule_order:
                task_schedule_order[task_code] = idx
        
        # 检查每个依赖关系
        for dep in simulator.current_process_dependencies:
            pre_task = dep.get('pre_task_code')
            post_task = dep.get('post_task_code')
            
            if pre_task is None or post_task is None:
                continue
            
            pre_order = task_schedule_order.get(pre_task, -1)
            post_order = task_schedule_order.get(post_task, -1)
            
            check_result = {
                'pre_task': pre_task,
                'post_task': post_task,
                'pre_scheduled_at': pre_order,
                'post_scheduled_at': post_order,
                'is_valid': True
            }
            
            if pre_order != -1 and post_order != -1:
                if pre_order >= post_order:
                    check_result['is_valid'] = False
                    violations.append(f"依赖违规: 任务{pre_task}应在任务{post_task}之前调度，但实际顺序相反")
            
            dependency_checks.append(check_result)
        
        is_valid = len(violations) == 0
        
        if is_valid:
            self.logger.info(f"✓ 依赖完整性验证通过！共检查 {len(dependency_checks)} 条依赖关系")
        else:
            self.logger.error(f"✗ 发现 {len(violations)} 条依赖违规:")
            for v in violations:
                self.logger.error(f"  {v}")
        
        return is_valid, violations, dependency_checks
    
    def run_fe_iddqn_scheduling(self, data, workflow_info, agent):
        """使用FE-IDDQN运行调度"""
        process_id = workflow_info['process_id']
        self.logger.info(f"\n运行FE-IDDQN调度: 工作流ID={process_id}")
        
        simulator = self.create_simulator_for_workflow(data, process_id)
        if simulator is None:
            return None
        
        simulator.reset()
        schedule_history = []
        step_count = 0
        max_steps = 1000
        total_reward = 0
        
        start_time = time.time()
        
        while not simulator.is_done() and step_count < max_steps:
            state = simulator.get_state()
            if state is None:
                break
            
            task_features, resource_features = state
            graph_adj = simulator.get_graph_adj()
            action = agent.select_action(task_features, resource_features, graph_adj=graph_adj)
            
            # 记录调度决策
            current_task = None
            if simulator.current_task_idx < len(simulator.current_process_tasks):
                current_task = simulator.current_process_tasks.iloc[simulator.current_task_idx]
                schedule_record = {
                    'step': step_count,
                    'task_code': current_task.get('task_code', current_task.get('id')),
                    'task_name': current_task.get('name', 'N/A'),
                    'task_type': current_task.get('task_type', 'N/A'),
                    'action': action,
                    'selected_resource': list(simulator.available_resources.keys())[action] if action < len(simulator.available_resources) else 'invalid'
                }
                schedule_history.append(schedule_record)
            
            next_state, reward, done, info = simulator.step(action)
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        execution_time = time.time() - start_time
        makespan = simulator.get_makespan()
        resource_util = simulator.get_resource_utilization()
        
        # 验证依赖完整性
        is_valid, violations, dependency_checks = self.validate_dependency_integrity(simulator, schedule_history)
        
        result = {
            'algorithm': 'FE-IDDQN',
            'process_id': process_id,
            'makespan': makespan,
            'resource_utilization': resource_util,
            'total_reward': total_reward,
            'steps': step_count,
            'execution_time': execution_time,
            'dependency_valid': is_valid,
            'violation_count': len(violations),
            'schedule_history': schedule_history,
            'dependency_checks': dependency_checks
        }
        
        self.logger.info(f"  Makespan: {makespan:.2f}秒")
        self.logger.info(f"  资源利用率: {resource_util:.4f}")
        self.logger.info(f"  总奖励: {total_reward:.2f}")
        self.logger.info(f"  依赖完整: {'是' if is_valid else '否'}")
        
        return result
    
    def run_baseline_scheduling(self, data, workflow_info, scheduler, scheduler_name):
        """使用基线算法运行调度"""
        process_id = workflow_info['process_id']
        self.logger.info(f"\n运行{scheduler_name}调度: 工作流ID={process_id}")
        
        simulator = self.create_simulator_for_workflow(data, process_id)
        if simulator is None:
            return None
        
        simulator.reset()
        schedule_history = []
        step_count = 0
        max_steps = 1000
        total_reward = 0
        
        start_time = time.time()
        
        while not simulator.is_done() and step_count < max_steps:
            state = simulator.get_state()
            if state is None:
                break
            
            task_features, resource_features = state
            
            # 基线调度器选择动作
            action = scheduler.select_action(task_features, resource_features)
            
            # 确保action在有效范围内
            num_resources = len(simulator.available_resources)
            if action >= num_resources:
                action = action % num_resources
            
            # 记录调度决策
            if simulator.current_task_idx < len(simulator.current_process_tasks):
                current_task = simulator.current_process_tasks.iloc[simulator.current_task_idx]
                schedule_record = {
                    'step': step_count,
                    'task_code': current_task.get('task_code', current_task.get('id')),
                    'task_name': current_task.get('name', 'N/A'),
                    'action': action,
                    'selected_resource': list(simulator.available_resources.keys())[action] if action < num_resources else 'invalid'
                }
                schedule_history.append(schedule_record)
            
            next_state, reward, done, info = simulator.step(action)
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        execution_time = time.time() - start_time
        makespan = simulator.get_makespan()
        resource_util = simulator.get_resource_utilization()
        
        # 验证依赖完整性
        is_valid, violations, dependency_checks = self.validate_dependency_integrity(simulator, schedule_history)
        
        result = {
            'algorithm': scheduler_name,
            'process_id': process_id,
            'makespan': makespan,
            'resource_utilization': resource_util,
            'total_reward': total_reward,
            'steps': step_count,
            'execution_time': execution_time,
            'dependency_valid': is_valid,
            'violation_count': len(violations),
            'schedule_history': schedule_history
        }
        
        self.logger.info(f"  Makespan: {makespan:.2f}秒")
        self.logger.info(f"  资源利用率: {resource_util:.4f}")
        self.logger.info(f"  依赖完整: {'是' if is_valid else '否'}")
        
        return result
    
    def compare_results(self, all_results):
        """比较所有算法的结果"""
        self.logger.info("\n" + "="*80)
        self.logger.info("调度结果比较")
        self.logger.info("="*80)
        
        # 按工作流分组
        by_workflow = {}
        for result in all_results:
            wf_id = result['process_id']
            if wf_id not in by_workflow:
                by_workflow[wf_id] = []
            by_workflow[wf_id].append(result)
        
        comparison_summary = []
        
        for wf_id, results in by_workflow.items():
            self.logger.info(f"\n工作流 ID: {wf_id}")
            self.logger.info("-" * 60)
            
            # 找到FE-IDDQN的结果作为基准
            fe_iddqn_result = next((r for r in results if r['algorithm'] == 'FE-IDDQN'), None)
            
            wf_comparison = {'workflow_id': wf_id, 'algorithms': {}}
            
            for result in sorted(results, key=lambda x: x['makespan']):
                algo = result['algorithm']
                makespan = result['makespan']
                resource_util = result['resource_utilization']
                dep_valid = result['dependency_valid']
                
                # 计算相对提升
                if fe_iddqn_result and algo != 'FE-IDDQN':
                    improvement = (makespan - fe_iddqn_result['makespan']) / makespan * 100 if makespan > 0 else 0
                    improvement_str = f"(相对FE-IDDQN: {improvement:+.1f}%)"
                else:
                    improvement_str = "(基准)"
                
                self.logger.info(f"  {algo:15s}: Makespan={makespan:10.2f}s, 资源利用率={resource_util:.4f}, 依赖完整={'✓' if dep_valid else '✗'} {improvement_str}")
                
                wf_comparison['algorithms'][algo] = {
                    'makespan': makespan,
                    'resource_utilization': resource_util,
                    'dependency_valid': dep_valid
                }
            
            comparison_summary.append(wf_comparison)
        
        return comparison_summary
    
    def run_test(self):
        """运行完整测试"""
        self.logger.info("="*80)
        self.logger.info("长执行时间工作流调度测试")
        self.logger.info("="*80)
        
        # 1. 加载数据
        self.logger.info("\n1. 加载数据...")
        data = self.data_loader.load_all_data()
        if not data:
            self.logger.error("数据加载失败！")
            return
        
        self.logger.info(f"  进程实例: {len(data['process_instance'])} 条")
        self.logger.info(f"  任务实例: {len(data['task_instance'])} 条")
        self.logger.info(f"  任务关系: {len(data['process_task_relation'])} 条")
        
        # 2. 找到执行时间最长的工作流
        self.logger.info("\n2. 查找长执行时间工作流...")
        long_workflows = self.find_long_execution_workflows(data, top_n=3)
        
        if len(long_workflows) == 0:
            self.logger.error("未找到有效的工作流实例！")
            return
        
        # 3. 创建FE-IDDQN智能体
        self.logger.info("\n3. 初始化FE-IDDQN智能体...")
        agent = FE_IDDQN(
            self.task_input_dim,
            self.resource_input_dim,
            self.action_dim,
            max_tasks=5,
            max_resources=5,
            enable_graph_encoder=True
        )
        
        # 4. 创建基线调度器
        self.logger.info("\n4. 初始化基线调度器...")
        baselines = {
            'FIFO': SimpleBaselineScheduler('FIFO', self.action_dim),
            'SJF': SimpleBaselineScheduler('SJF', self.action_dim),
            'RoundRobin': SimpleBaselineScheduler('RoundRobin', self.action_dim),
            'LoadBalance': SimpleBaselineScheduler('LoadBalance', self.action_dim),
            'Random': SimpleBaselineScheduler('Random', self.action_dim)
        }
        
        # 5. 运行调度测试
        self.logger.info("\n5. 开始调度测试...")
        all_results = []
        
        for _, workflow_info in long_workflows.iterrows():
            self.logger.info(f"\n" + "="*60)
            self.logger.info(f"测试工作流: {workflow_info['process_name'][:50] if workflow_info['process_name'] else 'N/A'}...")
            self.logger.info(f"任务数: {workflow_info['task_count']}, 依赖数: {workflow_info['dependency_count']}")
            self.logger.info("="*60)
            
            # 运行FE-IDDQN
            fe_result = self.run_fe_iddqn_scheduling(data, workflow_info, agent)
            if fe_result:
                all_results.append(fe_result)
            
            # 运行基线算法
            for name, scheduler in baselines.items():
                baseline_result = self.run_baseline_scheduling(data, workflow_info, scheduler, name)
                if baseline_result:
                    all_results.append(baseline_result)
        
        # 6. 比较结果
        comparison = self.compare_results(all_results)
        
        # 7. 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_path = self.output_dir / f"test_results_{timestamp}.json"
        serializable_results = []
        for r in all_results:
            sr = {k: v for k, v in r.items() if k not in ['schedule_history', 'dependency_checks']}
            serializable_results.append(sr)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'test_time': timestamp,
                'workflows_tested': len(long_workflows),
                'results': serializable_results,
                'comparison': comparison
            }, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"\n结果已保存到: {results_path}")
        
        # 8. 输出总结
        self.logger.info("\n" + "="*80)
        self.logger.info("测试总结")
        self.logger.info("="*80)
        
        # 计算FE-IDDQN的平均提升
        fe_results = [r for r in all_results if r['algorithm'] == 'FE-IDDQN']
        baseline_results = [r for r in all_results if r['algorithm'] != 'FE-IDDQN']
        
        if fe_results and baseline_results:
            avg_fe_makespan = np.mean([r['makespan'] for r in fe_results])
            avg_fe_util = np.mean([r['resource_utilization'] for r in fe_results])
            fe_dep_valid = all([r['dependency_valid'] for r in fe_results])
            
            self.logger.info(f"\nFE-IDDQN 性能:")
            self.logger.info(f"  平均Makespan: {avg_fe_makespan:.2f}秒")
            self.logger.info(f"  平均资源利用率: {avg_fe_util:.4f}")
            self.logger.info(f"  依赖完整性: {'全部通过' if fe_dep_valid else '存在违规'}")
            
            for algo in ['FIFO', 'SJF', 'RoundRobin', 'Priority']:
                algo_results = [r for r in baseline_results if r['algorithm'] == algo]
                if algo_results:
                    avg_makespan = np.mean([r['makespan'] for r in algo_results])
                    improvement = (avg_makespan - avg_fe_makespan) / avg_makespan * 100 if avg_makespan > 0 else 0
                    self.logger.info(f"\n{algo} 对比:")
                    self.logger.info(f"  平均Makespan: {avg_makespan:.2f}秒")
                    self.logger.info(f"  FE-IDDQN 提升: {improvement:.1f}%")
        
        return all_results


def main():
    """主函数"""
    tester = LongWorkflowTester()
    tester.run_test()


if __name__ == "__main__":
    main()
