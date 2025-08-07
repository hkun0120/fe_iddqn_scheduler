import numpy as np
import networkx as nx
import logging
from typing import Dict, List, Tuple, Any, Optional


class WorkflowSimulator:
    """工作流调度仿真环境"""

    def __init__(self, tasks: List[Dict], resources: List[Dict], dependencies: List[Tuple[int, int]]):
        self.logger = logging.getLogger(__name__)
        self.tasks = tasks
        self.resources = resources
        self.dependencies = dependencies
        self.num_tasks = len(tasks)
        self.num_resources = len(resources)

        # 创建DAG图
        self.dag = nx.DiGraph()
        for task in tasks:
            self.dag.add_node(task['id'], **task)
        for pre_task, post_task in dependencies:
            self.dag.add_edge(pre_task, post_task)

        # 初始化状态
        self.reset()

    def reset(self):
        """重置仿真环境"""
        self.current_time = 0
        self.resource_available_time = [0] * self.num_resources
        self.task_assignments = {}
        self.task_start_times = {}
        self.task_end_times = {}
        self.completed_tasks = set()
        self.ready_tasks = self._get_ready_tasks()

    def _get_ready_tasks(self) -> List[int]:
        """获取当前可调度的任务（依赖已满足）"""
        ready = []
        for task in self.tasks:
            if task['id'] not in self.completed_tasks:
                # 检查依赖是否满足
                dependencies_satisfied = True
                for pre_task, post_task in self.dependencies:
                    if post_task == task['id'] and pre_task not in self.completed_tasks:
                        dependencies_satisfied = False
                        break

                if dependencies_satisfied:
                    ready.append(task['id'])
        return ready

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前状态（任务特征和资源特征）"""
        # 任务特征：包括任务属性和状态信息
        task_features = []
        for task in self.tasks:
            features = [
                task['duration'],
                task['cpu_req'],
                task['memory_req'],
                1 if task['id'] in self.completed_tasks else 0,
                1 if task['id'] in self.ready_tasks else 0,
                len([p for p, c in self.dependencies if c == task['id']]),  # 前驱任务数
                len([p for p, c in self.dependencies if p == task['id']]),  # 后继任务数
            ]
            task_features.append(features)

        # 资源特征：包括资源容量和当前状态
        resource_features = []
        for i, resource in enumerate(self.resources):
            features = [
                resource['cpu_capacity'],
                resource['memory_capacity'],
                self.resource_available_time[i],
                self.resource_available_time[i] / (self.current_time + 1),  # 利用率
            ]
            resource_features.append(features)

        return np.array(task_features, dtype=np.float32), np.array(resource_features, dtype=np.float32)

    def step(self, action: int) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict]:
        """执行一个调度动作"""
        if not self.ready_tasks:
            # 没有可调度的任务，跳过
            return self.get_state(), 0, self.is_done(), {}

        # 选择第一个可调度的任务
        task_id = self.ready_tasks[0]
        resource_id = action % self.num_resources  # 确保动作在有效范围内

        # 获取任务和资源信息
        task = next(t for t in self.tasks if t['id'] == task_id)
        resource = self.resources[resource_id]

        # 检查资源容量是否满足任务需求
        if (resource['cpu_capacity'] >= task['cpu_req'] and
                resource['memory_capacity'] >= task['memory_req']):

            # 计算任务开始和结束时间
            start_time = max(self.current_time, self.resource_available_time[resource_id])
            end_time = start_time + task['duration']

            # 更新状态
            self.task_assignments[task_id] = resource_id
            self.task_start_times[task_id] = start_time
            self.task_end_times[task_id] = end_time
            self.resource_available_time[resource_id] = end_time
            self.completed_tasks.add(task_id)

            # 计算奖励
            reward = self._calculate_reward(task, resource, start_time, end_time)
        else:
            # 资源不满足需求，给予负奖励
            reward = -10

        # 更新可调度任务列表
        self.ready_tasks = self._get_ready_tasks()

        # 更新当前时间
        if self.task_end_times:
            self.current_time = min(self.task_end_times.values())

        return self.get_state(), reward, self.is_done(), {}

    def _calculate_reward(self, task: Dict, resource: Dict, start_time: float, end_time: float) -> float:
        """计算奖励函数"""
        # 基础奖励：任务完成
        reward = 10

        # 时间惩罚：鼓励更早完成
        time_penalty = end_time * 0.01
        reward -= time_penalty

        # 资源利用率奖励
        cpu_utilization = task['cpu_req'] / resource['cpu_capacity']
        memory_utilization = task['memory_req'] / resource['memory_capacity']
        utilization_reward = (cpu_utilization + memory_utilization) * 5
        reward += utilization_reward

        return reward

    def is_done(self) -> bool:
        """检查是否所有任务都已完成"""
        return len(self.completed_tasks) == self.num_tasks

    def get_makespan(self) -> float:
        """获取makespan（所有任务完成的最大时间）"""
        if not self.task_end_times:
            return 0
        return max(self.task_end_times.values())

    def get_resource_utilization(self) -> float:
        """计算资源利用率"""
        if not self.task_end_times:
            return 0

        makespan = self.get_makespan()
        total_work = sum(task['duration'] for task in self.tasks)
        total_capacity = makespan * self.num_resources

        return total_work / total_capacity if total_capacity > 0 else 0

    def simulate_random_schedule(self, algorithm_name: str) -> Dict[str, Any]:
        """模拟随机调度（用于测试）"""
        self.reset()

        # 随机分配任务到资源
        for task in self.tasks:
            resource_id = np.random.randint(0, self.num_resources)
            self.task_assignments[task['id']] = resource_id

        # 计算调度结果
        resource_end_times = [0] * self.num_resources
        task_start_times = {}
        task_end_times = {}
        processed = set()

        while len(processed) < self.num_tasks:
            for task in self.tasks:
                if task['id'] in processed:
                    continue

                # 检查依赖是否满足
                dependencies_satisfied = True
                max_dependency_end = 0
                for pre_task, post_task in self.dependencies:
                    if post_task == task['id']:
                        if pre_task not in task_end_times:
                            dependencies_satisfied = False
                            break
                        max_dependency_end = max(max_dependency_end, task_end_times[pre_task])

                if dependencies_satisfied:
                    resource_id = self.task_assignments[task['id']]
                    start_time = max(resource_end_times[resource_id], max_dependency_end)
                    end_time = start_time + task['duration']

                    task_start_times[task['id']] = start_time
                    task_end_times[task['id']] = end_time
                    resource_end_times[resource_id] = end_time
                    processed.add(task['id'])

        makespan = max(resource_end_times) if resource_end_times else 0
        total_work = sum(task['duration'] for task in self.tasks)
        total_capacity = makespan * self.num_resources
        resource_utilization = total_work / total_capacity if total_capacity > 0 else 0

        return {
            'task_assignments': self.task_assignments,
            'task_start_times': task_start_times,
            'task_end_times': task_end_times,
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'algorithm': algorithm_name
        }