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

    def _initialize_simulator(self) -> WorkflowSimulator:
        """初始化仿真环境"""
        # 这里需要根据实际数据和特征来初始化仿真器
        # 暂时使用模拟数据
        tasks = [
            {
                "id": i,
                "duration": np.random.randint(10, 100),
                "cpu_req": np.random.randint(1, 4),
                "memory_req": np.random.randint(1, 8),
            }
            for i in range(50)
        ]
        resources = [
            {
                "id": i,
                "cpu_capacity": np.random.randint(4, 16),
                "memory_capacity": np.random.randint(8, 32),
            }
            for i in range(10)
        ]
        dependencies = []  # 简化为无依赖，实际应从数据中构建

        return WorkflowSimulator(tasks, resources, dependencies)

    def run_algorithm(self, algorithm_name: str) -> Dict[str, Any]:
        """运行单个算法的实验"""
        self.logger.info(f"Running experiment for algorithm: {algorithm_name}")

        results = []
        for i in range(self.n_experiments):
            self.logger.info(f"  Experiment {i + 1}/{self.n_experiments}")
            simulator = self._initialize_simulator()

            if algorithm_name == "FE_IDDQN":
                # 假设任务特征和资源特征的维度
                task_input_dim = 10  # 示例维度
                resource_input_dim = 5  # 示例维度
                action_dim = simulator.num_resources

                agent = FE_IDDQN(task_input_dim, resource_input_dim, action_dim, self.device)
                # 实际训练过程会在这里进行，目前简化为模拟调度
                # state = simulator.get_initial_state()
                # for step in range(Hyperparameters.FE_IDDQN["max_steps_per_episode"]):
                #     task_features, resource_features = state # 假设state包含这两部分
                #     action = agent.select_action(task_features, resource_features)
                #     next_state, reward, done, _ = simulator.step(action)
                #     agent.store_experience(state, action, reward, next_state, done)
                #     agent.train()
                #     state = next_state
                #     if done: break

                # 模拟调度结果
                schedule_result = simulator.simulate_random_schedule(algorithm_name)

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
                scheduler = FIFOScheduler()
                schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, simulator.dependencies)
            elif algorithm_name == "SJF":
                scheduler = SJFScheduler()
                schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, simulator.dependencies)
            elif algorithm_name == "HEFT":
                scheduler = HEFTScheduler()
                schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, simulator.dependencies)
            elif algorithm_name == "GA":
                scheduler = GAScheduler()
                schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, simulator.dependencies)
            elif algorithm_name == "PSO":
                scheduler = PSOScheduler()
                schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, simulator.dependencies)
            elif algorithm_name == "ACO":
                scheduler = ACOScheduler()
                schedule_result = scheduler.schedule(simulator.tasks, simulator.resources, simulator.dependencies)
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

        avg_metrics = {}
        for key in results[0].keys():
            if isinstance(results[0][key], (int, float)):
                avg_metrics[key] = np.mean([r[key] for r in results])
            else:
                avg_metrics[key] = results[0][key]  # 非数值型直接取第一个
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