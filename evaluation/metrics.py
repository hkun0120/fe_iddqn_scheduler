import numpy as np
import logging
from typing import Dict, List, Any


class Evaluator:
    """评估器，计算各种调度性能指标"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate(self, schedule_result: Dict[str, Any]) -> Dict[str, float]:
        """评估调度结果"""
        metrics = {}

        # 基本指标
        metrics['makespan'] = schedule_result.get('makespan', 0)
        metrics['resource_utilization'] = schedule_result.get('resource_utilization', 0)

        # 计算其他指标
        if 'task_start_times' in schedule_result and 'task_end_times' in schedule_result:
            metrics.update(self._calculate_additional_metrics(schedule_result))

        return metrics

    def _calculate_additional_metrics(self, schedule_result: Dict[str, Any]) -> Dict[str, float]:
        """计算额外的性能指标"""
        task_start_times = schedule_result['task_start_times']
        task_end_times = schedule_result['task_end_times']
        task_assignments = schedule_result.get('task_assignments', {})

        metrics = {}

        # 平均完成时间
        if task_end_times:
            metrics['avg_completion_time'] = np.mean(list(task_end_times.values()))
            metrics['max_completion_time'] = np.max(list(task_end_times.values()))
            metrics['min_completion_time'] = np.min(list(task_end_times.values()))

        # 平均等待时间
        if task_start_times and task_end_times:
            wait_times = []
            for task_id in task_start_times:
                # 简化计算，假设任务提交时间为0
                wait_time = task_start_times[task_id]
                wait_times.append(wait_time)

            if wait_times:
                metrics['avg_wait_time'] = np.mean(wait_times)
                metrics['max_wait_time'] = np.max(wait_times)

        # 负载均衡度
        if task_assignments:
            resource_loads = {}
            for task_id, resource_id in task_assignments.items():
                if resource_id not in resource_loads:
                    resource_loads[resource_id] = 0
                # 这里需要任务持续时间信息，暂时简化
                resource_loads[resource_id] += 1

            if resource_loads:
                load_values = list(resource_loads.values())
                metrics['load_balance'] = 1.0 - (np.std(load_values) / (np.mean(load_values) + 1e-6))

        # 吞吐量（每单位时间完成的任务数）
        makespan = schedule_result.get('makespan', 1)
        num_tasks = len(task_end_times) if task_end_times else 0
        metrics['throughput'] = num_tasks / makespan if makespan > 0 else 0

        return metrics

    def compare_algorithms(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """比较多个算法的性能"""
        comparison = {}

        # 获取所有指标名称
        all_metrics = set()
        for algo_results in results.values():
            all_metrics.update(algo_results.keys())

        # 对每个指标进行比较
        for metric in all_metrics:
            metric_values = {}
            for algo, algo_results in results.items():
                if metric in algo_results:
                    metric_values[algo] = algo_results[metric]

            if metric_values:
                # 找出最佳算法
                if metric in ['makespan', 'avg_completion_time', 'avg_wait_time']:
                    # 越小越好的指标
                    best_algo = min(metric_values, key=metric_values.get)
                    best_value = metric_values[best_algo]
                else:
                    # 越大越好的指标
                    best_algo = max(metric_values, key=metric_values.get)
                    best_value = metric_values[best_algo]

                comparison[metric] = {
                    'best_algorithm': best_algo,
                    'best_value': best_value,
                    'all_values': metric_values
                }

        return comparison

    def calculate_improvement(self, baseline_result: Dict[str, float],
                              improved_result: Dict[str, float]) -> Dict[str, float]:
        """计算改进百分比"""
        improvements = {}

        for metric in baseline_result:
            if metric in improved_result:
                baseline_val = baseline_result[metric]
                improved_val = improved_result[metric]

                if baseline_val != 0:
                    if metric in ['makespan', 'avg_completion_time', 'avg_wait_time']:
                        # 越小越好的指标
                        improvement = (baseline_val - improved_val) / baseline_val * 100
                    else:
                        # 越大越好的指标
                        improvement = (improved_val - baseline_val) / baseline_val * 100

                    improvements[metric] = improvement

        return improvements