"""
增强版FE-IDDQN完整训练脚本

这个脚本演示如何：
1. 加载实际的工作流数据
2. 初始化增强版FE-IDDQN Agent
3. 运行完整的训练循环
4. 评估性能
5. 保存和加载模型
"""

import torch
import numpy as np
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入项目模块
from config import Hyperparameters
from models import (
    EnhancedFE_IDDQN,
    EnhancedFE_IDDQN_Config,
    CombinedReplayBuffer
)
from data import (
    DataLoader,
    DataPreprocessor,
    EnhancedStateEncoder,
    EnhancedStateConfig
)
from environment import EnhancedWorkflowSimulator
from evaluation import MetricCalculator


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建带时间戳的日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.json"
        
        self.history = {
            'episodes': [],
            'rewards': [],
            'makespan': [],
            'utilization': [],
            'load_balance': [],
            'loss': [],
            'epsilon': []
        }
    
    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """记录episode的指标"""
        self.history['episodes'].append(episode)
        self.history['rewards'].append(metrics.get('reward', 0))
        self.history['makespan'].append(metrics.get('makespan', 0))
        self.history['utilization'].append(metrics.get('utilization', 0))
        self.history['load_balance'].append(metrics.get('load_balance', 0))
        self.history['loss'].append(metrics.get('loss', 0))
        self.history['epsilon'].append(metrics.get('epsilon', 0))
        
        if episode % 50 == 0:
            self.save()
    
    def save(self):
        """保存日志到文件"""
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_summary(self) -> Dict[str, float]:
        """获取训练摘要"""
        if not self.history['rewards']:
            return {}
        
        return {
            'avg_reward_last_100': np.mean(self.history['rewards'][-100:]),
            'avg_makespan_last_100': np.mean(self.history['makespan'][-100:]),
            'avg_utilization_last_100': np.mean(self.history['utilization'][-100:]),
            'max_reward': np.max(self.history['rewards']),
            'min_makespan': np.min(self.history['makespan'])
        }


class EnhancedTrainer:
    """增强版FE-IDDQN训练器"""
    
    def __init__(self, config: EnhancedFE_IDDQN_Config, 
                 task_input_dim: int = 64,
                 resource_input_dim: int = 32,
                 action_dim: int = 6,
                 device: str = 'cuda'):
        self.config = config
        self.device = device
        self.task_input_dim = task_input_dim
        self.resource_input_dim = resource_input_dim
        self.action_dim = action_dim
        
        # 初始化Agent
        self.agent = EnhancedFE_IDDQN(
            task_input_dim=task_input_dim,
            resource_input_dim=resource_input_dim,
            action_dim=action_dim,
            config=config
        )
        logger.info(f"Agent初始化完成，设备: {device}")
        
        # 初始化日志记录器
        self.logger = TrainingLogger()
        
        # 初始化指标计算器
        self.metric_calculator = MetricCalculator()
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            self.agent.q_network.parameters(),
            lr=config.learning_rate
        )
        
        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=config.learning_rate * 0.1
        )
    
    def prepare_state_batch(self, states: List[Dict]) -> Dict[str, torch.Tensor]:
        """准备状态批次"""
        batch = {
            'task_features': [],
            'resource_features': [],
            'adjacency_matrix': [],
            'critical_path_mask': []
        }
        
        for state in states:
            batch['task_features'].append(torch.tensor(state['task_features'], dtype=torch.float32))
            batch['resource_features'].append(torch.tensor(state['resource_features'], dtype=torch.float32))
            batch['adjacency_matrix'].append(torch.tensor(state['adjacency_matrix'], dtype=torch.float32))
            batch['critical_path_mask'].append(torch.tensor(state['critical_path_mask'], dtype=torch.float32))
        
        return {
            k: torch.stack(v).to(self.device) if v else None
            for k, v in batch.items()
        }
    
    def train_episode(self, env: EnhancedWorkflowSimulator, episode: int) -> Dict[str, float]:
        """训练一个episode"""
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        
        while True:
            # 提取状态特征
            task_features = state.get('task_features', np.zeros((8, self.task_input_dim)))
            resource_features = state.get('resource_features', np.zeros((self.action_dim, self.resource_input_dim)))
            adj_matrix = state.get('adjacency_matrix', None)
            critical_path_mask = state.get('critical_path_mask', None)
            
            # 选择动作
            action = self.agent.select_action(
                task_features=task_features,
                resource_features=resource_features,
                adj_matrix=adj_matrix,
                critical_path_mask=critical_path_mask,
                training=True
            )
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            self.agent.store_experience(state, action, reward, next_state, done, info)
            
            # 训练
            if len(self.agent.replay_buffer) > self.config.batch_size:
                train_result = self.agent.train_step()
                if train_result is not None:
                    episode_loss += train_result.get('total_loss', 0.0)
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            if done or step_count >= 500:
                break
        
        # 计算平均损失（只在有损失时除以步数）
        if episode_loss > 0:
            avg_loss = episode_loss / max(step_count, 1)
        else:
            avg_loss = 0.0
        
        # 获取epsilon值
        epsilon = getattr(self.agent.exploration, 'epsilon', 0.0)
        
        return {
            'reward': episode_reward,
            'loss': avg_loss,
            'epsilon': epsilon,
            'makespan': info.get('makespan', 0),
            'utilization': info.get('utilization', 0),
            'load_balance': info.get('load_balance', 0)
        }
    
    def evaluate(self, env: EnhancedWorkflowSimulator, num_episodes: int = 10) -> Dict[str, float]:
        """评估Agent性能"""
        # 切换网络到评估模式
        self.agent.q_network.eval()
        
        total_reward = 0
        total_makespan = 0
        total_utilization = 0
        total_load_balance = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            step_count = 0
            
            while True:
                # 提取状态特征
                task_features = state.get('task_features', np.zeros((8, self.task_input_dim)))
                resource_features = state.get('resource_features', np.zeros((self.action_dim, self.resource_input_dim)))
                adj_matrix = state.get('adj_matrix', None)
                critical_path_mask = state.get('critical_path_mask', None)
                
                action = self.agent.select_action(
                    task_features=task_features,
                    resource_features=resource_features,
                    adj_matrix=adj_matrix,
                    critical_path_mask=critical_path_mask,
                    training=False
                )
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
                step_count += 1
                
                if done or step_count >= 500:
                    break
            
            total_reward += episode_reward
            total_makespan += info.get('makespan', 0)
            total_utilization += info.get('utilization', 0)
            total_load_balance += info.get('load_balance', 0)
        
        # 切换回训练模式
        self.agent.q_network.train()
        
        return {
            'avg_reward': total_reward / num_episodes,
            'avg_makespan': total_makespan / num_episodes,
            'avg_utilization': total_utilization / num_episodes,
            'avg_load_balance': total_load_balance / num_episodes
        }
    
    def train(self, env: EnhancedWorkflowSimulator, num_episodes: int = 1000, eval_freq: int = 100):
        """完整训练循环"""
        logger.info(f"开始训练，共{num_episodes}个episodes")
        
        for episode in range(num_episodes):
            # 训练一个episode
            metrics = self.train_episode(env, episode)
            self.logger.log_episode(episode, metrics)
            
            # 定期评估
            if (episode + 1) % eval_freq == 0:
                eval_metrics = self.evaluate(env, num_episodes=5)
                logger.info(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Reward: {metrics['reward']:.2f} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"ε: {metrics['epsilon']:.4f} | "
                    f"Eval Reward: {eval_metrics['avg_reward']:.2f} | "
                    f"Makespan: {eval_metrics['avg_makespan']:.2f}"
                )
            elif (episode + 1) % 10 == 0:
                logger.info(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Reward: {metrics['reward']:.2f} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"ε: {metrics['epsilon']:.4f}"
                )
        
        logger.info("训练完成！")
        summary = self.logger.get_summary()
        logger.info(f"训练摘要: {summary}")
        
        return self.logger.history
    
    def save_model(self, path: str = "checkpoints/enhanced_fe_iddqn.pt"):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'q_network_state': self.agent.q_network.state_dict(),
            'target_network_state': self.agent.target_network.state_dict(),
            'config': vars(self.config) if hasattr(self.config, '__dict__') else {},
            'optimizer_state': self.optimizer.state_dict(),
            'step_count': getattr(self.agent, 'step_count', 0)
        }, path)
        logger.info(f"模型已保存到 {path}")
    
    def load_model(self, path: str = "checkpoints/enhanced_fe_iddqn.pt"):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.q_network.load_state_dict(checkpoint['q_network_state'])
        self.agent.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'step_count' in checkpoint:
            self.agent.step_count = checkpoint['step_count']
        logger.info(f"模型已从 {path} 加载")


def create_sample_environment():
    """创建示例工作流环境"""
    # 简单的8个任务的DAG
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
    
    env = EnhancedWorkflowSimulator(tasks, resources, dependencies)
    return env


def main():
    parser = argparse.ArgumentParser(description='增强版FE-IDDQN训练脚本')
    parser.add_argument('--episodes', type=int, default=100, help='训练episodes数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='使用的设备')
    parser.add_argument('--eval-freq', type=int, default=10, help='评估频率')
    parser.add_argument('--save-path', type=str, default='checkpoints/enhanced_fe_iddqn.pt',
                       help='模型保存路径')
    parser.add_argument('--load-path', type=str, default=None, help='模型加载路径')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("增强版FE-IDDQN训练")
    logger.info("="*60)
    logger.info(f"参数配置: episodes={args.episodes}, batch_size={args.batch_size}, lr={args.lr}")
    
    # 创建环境
    logger.info("初始化工作流环境...")
    env = create_sample_environment()
    
    # 创建Agent配置
    hyper = Hyperparameters.ENHANCED_FE_IDDQN
    config = EnhancedFE_IDDQN_Config()
    config.hidden_dim = hyper.get('hidden_dim', 256)
    config.use_gnn = hyper.get('use_gnn', True)
    config.use_n_step = hyper.get('use_n_step', True)
    config.n_step = hyper.get('n_step', 3)
    config.use_per = hyper.get('use_per', True)
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.gamma = hyper.get('gamma', 0.99)
    config.tau = hyper.get('tau', 0.005)
    config.device = args.device
    
    # 创建训练器
    # 实际的特征维度由环境决定
    # task_features: 19维 (7基础 + 8DAG + 4时序)
    # resource_features: 11维 (4基础 + 4负载 + 3历史)
    logger.info(f"初始化训练器，使用设备: {args.device}")
    trainer = EnhancedTrainer(
        config=config,
        task_input_dim=19,
        resource_input_dim=11,
        action_dim=6,
        device=args.device
    )
    
    # 加载已有模型（如果指定）
    if args.load_path and os.path.exists(args.load_path):
        logger.info(f"从 {args.load_path} 加载模型")
        trainer.load_model(args.load_path)
    
    # 运行训练
    logger.info(f"开始训练 ({args.episodes} episodes)...")
    history = trainer.train(env, num_episodes=args.episodes, eval_freq=args.eval_freq)
    
    # 保存模型
    trainer.save_model(args.save_path)
    
    # 输出最终结果
    summary = trainer.logger.get_summary()
    logger.info("\n" + "="*60)
    logger.info("训练完成！最终结果:")
    logger.info("="*60)
    for key, value in summary.items():
        logger.info(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
