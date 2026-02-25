"""
对比实验脚本：原始FE-IDDQN vs 增强版FE-IDDQN

这个脚本运行对比实验，评估增强算法相对于原始算法的性能提升：
- 时间跨度（Makespan）
- 资源利用率
- 负载均衡
- 并行度
- 训练效率
"""

import torch
import numpy as np
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入原始和增强版算法
try:
    from models import FE_IDDQN  # 原始版本
except ImportError:
    logger.warning("未找到原始FE_IDDQN，跳过对比")
    FE_IDDQN = None

from models import EnhancedFE_IDDQN, EnhancedFE_IDDQN_Config
from environment import EnhancedWorkflowSimulator
from config import Hyperparameters


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, env: EnhancedWorkflowSimulator, num_runs: int = 3, device: str = 'cuda'):
        self.env = env
        self.num_runs = num_runs
        self.device = device
        self.results = {
            'original': [],
            'enhanced': []
        }
    
    def run_baseline(self, num_episodes: int = 50) -> Dict[str, float]:
        """运行原始算法基准"""
        if FE_IDDQN is None:
            logger.warning("原始FE_IDDQN不可用，使用随机策略作为基准")
            return self._run_random_baseline(num_episodes)
        
        logger.info("="*60)
        logger.info(f"运行原始FE-IDDQN (共{self.num_runs}次运行)")
        logger.info("="*60)
        
        run_results = []
        
        for run in range(self.num_runs):
            logger.info(f"运行 {run + 1}/{self.num_runs}")
            
            agent = FE_IDDQN(
                task_input_dim=64,
                resource_input_dim=32,
                hidden_dim=128,
                num_actions=6,
                learning_rate=1e-4
            ).to(self.device)
            
            metrics = self._train_agent(agent, num_episodes, is_enhanced=False)
            run_results.append(metrics)
        
        # 汇总结果
        summary = self._summarize_results(run_results)
        logger.info(f"原始FE-IDDQN结果: {summary}")
        
        return summary
    
    def run_enhanced(self, num_episodes: int = 50) -> Dict[str, float]:
        """运行增强版算法"""
        logger.info("="*60)
        logger.info(f"运行增强版FE-IDDQN (共{self.num_runs}次运行)")
        logger.info("="*60)
        
        run_results = []
        hyper = Hyperparameters.ENHANCED_FE_IDDQN
        
        for run in range(self.num_runs):
            logger.info(f"运行 {run + 1}/{self.num_runs}")
            
            config = EnhancedFE_IDDQN_Config(
                task_input_dim=64,
                resource_input_dim=32,
                hidden_dim=hyper.get('hidden_dim', 256),
                num_actions=6,
                use_gnn=True,
                use_transformer=True,
                use_n_step=True,
                n_step=hyper.get('n_step', 3),
                use_per=True,
                learning_rate=hyper.get('learning_rate', 3e-4),
                batch_size=hyper.get('batch_size', 64),
                gamma=hyper.get('gamma', 0.99)
            )
            
            agent = EnhancedFE_IDDQN(config).to(self.device)
            metrics = self._train_agent(agent, num_episodes, is_enhanced=True)
            run_results.append(metrics)
        
        # 汇总结果
        summary = self._summarize_results(run_results)
        logger.info(f"增强版FE-IDDQN结果: {summary}")
        
        return summary
    
    def _run_random_baseline(self, num_episodes: int) -> Dict[str, float]:
        """运行随机策略基准"""
        logger.info(f"运行随机策略基准 (共{self.num_runs}次运行)")
        
        makespan_list = []
        utilization_list = []
        load_balance_list = []
        
        for run in range(self.num_runs):
            episode_makespans = []
            episode_utilizations = []
            episode_loads = []
            
            for episode in range(num_episodes):
                state = self.env.reset()
                done = False
                
                while not done:
                    # 随机选择动作
                    action = np.random.randint(0, 6)
                    next_state, reward, done, info = self.env.step(action)
                    state = next_state
                
                episode_makespans.append(info.get('makespan', 0))
                episode_utilizations.append(info.get('utilization', 0))
                episode_loads.append(info.get('load_balance', 0))
            
            makespan_list.append(np.mean(episode_makespans))
            utilization_list.append(np.mean(episode_utilizations))
            load_balance_list.append(np.mean(episode_loads))
        
        return {
            'avg_makespan': np.mean(makespan_list),
            'std_makespan': np.std(makespan_list),
            'avg_utilization': np.mean(utilization_list),
            'std_utilization': np.std(utilization_list),
            'avg_load_balance': np.mean(load_balance_list),
            'std_load_balance': np.std(load_balance_list),
            'training_time': 0
        }
    
    def _train_agent(self, agent, num_episodes: int, is_enhanced: bool = False) -> Dict[str, float]:
        """训练Agent并评估"""
        agent.train()
        
        makespan_list = []
        utilization_list = []
        load_balance_list = []
        loss_list = []
        training_start = time.time()
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_loss = 0
            episode_reward = 0
            step_count = 0
            
            while True:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = self.env.step(action)
                
                if is_enhanced:
                    agent.remember(state, action, reward, next_state, done)
                    if len(agent.memory) > agent.config.batch_size:
                        loss = agent.train_step()
                        episode_loss += loss
                
                episode_reward += reward
                step_count += 1
                state = next_state
                
                if done or step_count >= 500:
                    break
            
            makespan_list.append(info.get('makespan', 0))
            utilization_list.append(info.get('utilization', 0))
            load_balance_list.append(info.get('load_balance', 0))
            loss_list.append(episode_loss / max(step_count, 1))
        
        training_time = time.time() - training_start
        
        return {
            'avg_makespan': np.mean(makespan_list),
            'std_makespan': np.std(makespan_list),
            'avg_utilization': np.mean(utilization_list),
            'std_utilization': np.std(utilization_list),
            'avg_load_balance': np.mean(load_balance_list),
            'std_load_balance': np.std(load_balance_list),
            'avg_loss': np.mean(loss_list),
            'training_time': training_time
        }
    
    def _summarize_results(self, run_results: List[Dict]) -> Dict[str, float]:
        """汇总多次运行的结果"""
        metrics = {}
        
        for key in run_results[0].keys():
            values = [r[key] for r in run_results]
            metrics[f'avg_{key}'] = np.mean(values)
            metrics[f'std_{key}'] = np.std(values)
        
        return metrics
    
    def calculate_improvements(self, baseline: Dict, enhanced: Dict) -> Dict[str, float]:
        """计算改进百分比"""
        improvements = {}
        
        # 时间和负载应该越小越好
        for metric in ['makespan', 'load_balance']:
            baseline_key = f'avg_{metric}'
            if baseline_key in baseline and baseline_key in enhanced:
                improvement = (baseline[baseline_key] - enhanced[baseline_key]) / baseline[baseline_key] * 100
                improvements[f'{metric}_improvement'] = improvement
        
        # 利用率应该越大越好
        for metric in ['utilization']:
            baseline_key = f'avg_{metric}'
            if baseline_key in baseline and baseline_key in enhanced:
                improvement = (enhanced[baseline_key] - baseline[baseline_key]) / baseline[baseline_key] * 100
                improvements[f'{metric}_improvement'] = improvement
        
        return improvements
    
    def save_results(self, baseline: Dict, enhanced: Dict, improvements: Dict, path: str = 'experiment_results.json'):
        """保存实验结果"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline,
            'enhanced': enhanced,
            'improvements': improvements
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"实验结果已保存到 {path}")
    
    def plot_comparison(self, baseline: Dict, enhanced: Dict, output_path: str = 'comparison.png'):
        """绘制对比图表"""
        metrics = ['makespan', 'utilization', 'load_balance']
        baseline_values = [baseline.get(f'avg_{m}', 0) for m in metrics]
        enhanced_values = [enhanced.get(f'avg_{m}', 0) for m in metrics]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (ax, metric) in enumerate(zip(axes, metrics)):
            x = np.arange(2)
            values = [baseline_values[idx], enhanced_values[idx]]
            colors = ['#FF6B6B', '#4ECDC4']
            
            bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('值')
            ax.set_title(f'{metric.title()}对比')
            ax.set_xticks(x)
            ax.set_xticklabels(['原始FE-IDDQN', '增强FE-IDDQN'])
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        logger.info(f"对比图已保存到 {output_path}")


def create_sample_environment():
    """创建示例工作流环境"""
    tasks = [
        {'id': 0, 'name': 'task_0', 'duration': 10, 'resource_req': 2},
        {'id': 1, 'name': 'task_1', 'duration': 15, 'resource_req': 3},
        {'id': 2, 'name': 'task_2', 'duration': 20, 'resource_req': 4},
        {'id': 3, 'name': 'task_3', 'duration': 30, 'resource_req': 6},
        {'id': 4, 'name': 'task_4', 'duration': 10, 'resource_req': 2},
        {'id': 5, 'name': 'task_5', 'duration': 12, 'resource_req': 3},
        {'id': 6, 'name': 'task_6', 'duration': 8, 'resource_req': 2},
        {'id': 7, 'name': 'task_7', 'duration': 5, 'resource_req': 1},
    ]
    
    dependencies = [
        (0, 1), (0, 5),
        (1, 2), (5, 6),
        (2, 3), (6, 3),
        (3, 4), (4, 7)
    ]
    
    resources = [
        {'id': 0, 'name': 'worker_0', 'capacity': 4, 'speed': 1.0},
        {'id': 1, 'name': 'worker_1', 'capacity': 6, 'speed': 1.2},
        {'id': 2, 'name': 'worker_2', 'capacity': 8, 'speed': 0.9},
        {'id': 3, 'name': 'worker_3', 'capacity': 4, 'speed': 1.1},
        {'id': 4, 'name': 'worker_4', 'capacity': 10, 'speed': 2.0},
        {'id': 5, 'name': 'worker_5', 'capacity': 12, 'speed': 0.8},
    ]
    
    return EnhancedWorkflowSimulator(tasks, resources, dependencies)


def main():
    parser = argparse.ArgumentParser(description='FE-IDDQN对比实验')
    parser.add_argument('--episodes', type=int, default=20, help='每次运行的episodes数')
    parser.add_argument('--runs', type=int, default=3, help='运行次数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='使用的设备')
    parser.add_argument('--output-dir', type=str, default='experiment_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("FE-IDDQN对比实验: 原始版本 vs 增强版本")
    logger.info("="*80)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建环境
    logger.info("初始化工作流环境...")
    env = create_sample_environment()
    
    # 创建实验运行器
    runner = ExperimentRunner(env, num_runs=args.runs, device=args.device)
    
    # 运行基准
    baseline_results = runner.run_baseline(args.episodes)
    
    # 运行增强版
    enhanced_results = runner.run_enhanced(args.episodes)
    
    # 计算改进
    improvements = runner.calculate_improvements(baseline_results, enhanced_results)
    
    # 保存结果
    results_file = Path(args.output_dir) / 'results.json'
    runner.save_results(baseline_results, enhanced_results, improvements, str(results_file))
    
    # 绘制对比图
    plot_file = Path(args.output_dir) / 'comparison.png'
    runner.plot_comparison(baseline_results, enhanced_results, str(plot_file))
    
    # 打印对比总结
    logger.info("\n" + "="*80)
    logger.info("对比实验结果")
    logger.info("="*80)
    logger.info("\n原始FE-IDDQN:")
    for key, value in baseline_results.items():
        if not key.startswith('std_'):
            logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\n增强版FE-IDDQN:")
    for key, value in enhanced_results.items():
        if not key.startswith('std_'):
            logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\n性能改进:")
    for key, value in improvements.items():
        direction = "↓" if "makespan" in key or "load_balance" in key else "↑"
        logger.info(f"  {key}: {value:+.2f}% {direction}")


if __name__ == "__main__":
    main()
