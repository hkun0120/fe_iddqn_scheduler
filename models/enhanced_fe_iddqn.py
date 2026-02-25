# -*- coding: utf-8 -*-
"""
增强版FE-IDDQN算法
整合GNN、Transformer、多目标奖励、高级探索策略、N-step回报等改进
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field

from .enhanced_network import EnhancedDualStreamNetwork
from .enhanced_replay_buffer import CombinedReplayBuffer, Experience
from .exploration_strategies import (
    AdaptiveEpsilonGreedy, BoltzmannExploration, 
    HeuristicGuidedExploration, IntrinsicCuriosityModule,
    NoisyNetwork
)
from .reward_functions import (
    EnhancedRewardCalculator, RewardConfig,
    AdaptiveRewardShaper, CurriculumRewardScheduler
)


@dataclass
class EnhancedFE_IDDQN_Config:
    """增强版FE-IDDQN配置"""
    # 网络架构
    hidden_dim: int = 256
    fusion_dim: int = 256
    num_transformer_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    use_gnn: bool = True
    use_noisy_net: bool = False
    
    # 训练参数
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005  # 软更新参数
    
    # N-step
    n_step: int = 3
    use_n_step: bool = True
    
    # 经验回放
    replay_buffer_size: int = 100000
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    
    # 探索策略
    exploration_strategy: str = 'adaptive_epsilon'  # 'adaptive_epsilon', 'boltzmann', 'noisy', 'icm'
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995  # 更慢的衰减，约1400步降到0.5
    
    # 训练控制
    max_episodes: int = 1000
    max_steps_per_episode: int = 500
    warmup_steps: int = 500  # 减少预热步数，更快开始学习
    train_freq: int = 4
    target_update_freq: int = 100
    gradient_clip: float = 1.0
    
    # 课程学习
    use_curriculum: bool = True
    curriculum_stages: int = 3
    
    # 多任务学习
    use_multi_task: bool = True
    auxiliary_loss_weight: float = 0.1
    
    # 奖励塑形
    use_reward_shaping: bool = True
    
    # 设备
    device: str = 'auto'


class EnhancedFE_IDDQN:
    """增强版FE-IDDQN算法"""
    
    def __init__(self, task_input_dim: int, resource_input_dim: int,
                 action_dim: int, config: Optional[EnhancedFE_IDDQN_Config] = None):
        """
        初始化增强版FE-IDDQN
        
        Args:
            task_input_dim: 任务特征维度
            resource_input_dim: 资源特征维度
            action_dim: 动作空间维度
            config: 配置对象
        """
        self.config = config or EnhancedFE_IDDQN_Config()
        self.logger = logging.getLogger(__name__)
        
        # 设置设备
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # 保存维度信息
        self.task_input_dim = task_input_dim
        self.resource_input_dim = resource_input_dim
        self.action_dim = action_dim
        
        # 创建网络
        self._build_networks()
        
        # 创建优化器
        self._build_optimizers()
        
        # 创建经验回放
        self._build_replay_buffer()
        
        # 创建探索策略
        self._build_exploration_strategy()
        
        # 创建奖励计算器
        self._build_reward_components()
        
        # 训练统计
        self.step_count = 0
        self.episode_count = 0
        self.training_losses = []
        self.episode_rewards = []
        
    def _build_networks(self):
        """构建网络"""
        # 主Q网络
        self.q_network = EnhancedDualStreamNetwork(
            task_input_dim=self.task_input_dim,
            resource_input_dim=self.resource_input_dim,
            hidden_dim=self.config.hidden_dim,
            fusion_dim=self.config.fusion_dim,
            output_dim=self.action_dim,
            num_transformer_layers=self.config.num_transformer_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            use_gnn=self.config.use_gnn
        ).to(self.device)
        
        # 目标Q网络
        self.target_network = EnhancedDualStreamNetwork(
            task_input_dim=self.task_input_dim,
            resource_input_dim=self.resource_input_dim,
            hidden_dim=self.config.hidden_dim,
            fusion_dim=self.config.fusion_dim,
            output_dim=self.action_dim,
            num_transformer_layers=self.config.num_transformer_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            use_gnn=self.config.use_gnn
        ).to(self.device)
        
        # 初始化目标网络
        self.hard_update_target()
        
        # 可选：ICM模块
        if self.config.exploration_strategy == 'icm':
            state_dim = self.task_input_dim * 10 + self.resource_input_dim * 6  # 估计的状态维度
            self.icm = IntrinsicCuriosityModule(
                state_dim=state_dim,
                action_dim=self.action_dim,
                feature_dim=64,
                hidden_dim=128
            ).to(self.device)
    
    def _build_optimizers(self):
        """构建优化器"""
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,
            gamma=0.95
        )
        
        # ICM优化器
        if hasattr(self, 'icm'):
            self.icm_optimizer = optim.Adam(
                self.icm.parameters(),
                lr=self.config.learning_rate
            )
    
    def _build_replay_buffer(self):
        """构建经验回放缓冲区"""
        self.replay_buffer = CombinedReplayBuffer(
            capacity=self.config.replay_buffer_size,
            n_step=self.config.n_step,
            gamma=self.config.gamma,
            use_per=self.config.use_per,
            use_n_step=self.config.use_n_step,
            alpha=self.config.per_alpha,
            beta_start=self.config.per_beta_start
        )
    
    def _build_exploration_strategy(self):
        """构建探索策略"""
        if self.config.exploration_strategy == 'adaptive_epsilon':
            self.exploration = AdaptiveEpsilonGreedy(
                epsilon_start=self.config.epsilon_start,
                epsilon_end=self.config.epsilon_end,
                epsilon_decay=self.config.epsilon_decay,
                adaptive_mode='performance'
            )
        elif self.config.exploration_strategy == 'boltzmann':
            self.exploration = BoltzmannExploration(
                temperature_start=1.0,
                temperature_end=0.1,
                temperature_decay=0.995
            )
        else:
            self.exploration = AdaptiveEpsilonGreedy()
    
    def _build_reward_components(self):
        """构建奖励组件"""
        # 增强版奖励计算器
        self.reward_calculator = EnhancedRewardCalculator(RewardConfig())
        
        # 奖励塑形器
        if self.config.use_reward_shaping:
            self.reward_shaper = AdaptiveRewardShaper()
        
        # 课程学习调度器
        if self.config.use_curriculum:
            self.curriculum_scheduler = CurriculumRewardScheduler(
                total_episodes=self.config.max_episodes
            )
    
    def select_action(self, task_features: np.ndarray, 
                     resource_features: np.ndarray,
                     adj_matrix: Optional[np.ndarray] = None,
                     node_depths: Optional[np.ndarray] = None,
                     critical_path_mask: Optional[np.ndarray] = None,
                     training: bool = True) -> int:
        """
        选择动作
        
        Args:
            task_features: 任务特征 [num_tasks, task_input_dim]
            resource_features: 资源特征 [num_resources, resource_input_dim]
            adj_matrix: 邻接矩阵 [num_tasks, num_tasks]
            node_depths: 节点深度 [num_tasks]
            critical_path_mask: 关键路径掩码 [num_tasks]
            training: 是否在训练模式
            
        Returns:
            选择的动作
        """
        # 转换为张量
        task_tensor = torch.FloatTensor(task_features).unsqueeze(0).to(self.device)
        resource_tensor = torch.FloatTensor(resource_features).unsqueeze(0).to(self.device)
        
        adj_tensor = None
        if adj_matrix is not None:
            adj_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0).to(self.device)
        
        depth_tensor = None
        if node_depths is not None:
            depth_tensor = torch.LongTensor(node_depths).unsqueeze(0).to(self.device)
        
        mask_tensor = None
        if critical_path_mask is not None:
            mask_tensor = torch.FloatTensor(critical_path_mask).unsqueeze(0).to(self.device)
        
        # 获取Q值
        with torch.no_grad():
            q_values = self.q_network(
                task_tensor, resource_tensor,
                adj_tensor, depth_tensor, mask_tensor
            ).squeeze(0)
        
        # 探索/利用
        if training:
            action = self.exploration.select_action(q_values)
        else:
            action = torch.argmax(q_values).item()
        
        return action
    
    def store_experience(self, state: Dict[str, np.ndarray], action: int,
                        reward: float, next_state: Dict[str, np.ndarray],
                        done: bool, info: Optional[Dict] = None):
        """存储经验"""
        # 展平状态为单一数组
        state_flat = self._flatten_state(state)
        next_state_flat = self._flatten_state(next_state)
        
        self.replay_buffer.add(state_flat, action, reward, next_state_flat, done, info)
    
    def _flatten_state(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        """展平状态字典为单一数组"""
        task_features = state.get('task_features', np.array([]))
        resource_features = state.get('resource_features', np.array([]))
        global_features = state.get('global_features', np.array([]))
        
        return np.concatenate([
            task_features.flatten(),
            resource_features.flatten(),
            global_features.flatten() if global_features.size > 0 else np.array([])
        ])
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """执行一步训练"""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        if len(self.replay_buffer) < self.config.warmup_steps:
            return None
        
        # 采样经验
        batch_data = self.replay_buffer.sample(self.config.batch_size)
        
        # 计算损失
        losses = self._compute_losses(batch_data)
        
        # 反向传播
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # 梯度裁剪
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(),
                self.config.gradient_clip
            )
        
        self.optimizer.step()
        
        # 更新统计
        self.step_count += 1
        self.training_losses.append(losses['total_loss'].item())
        
        # 软更新目标网络
        if self.step_count % self.config.target_update_freq == 0:
            self.soft_update_target()
        
        # 更新探索参数
        self.exploration.update()
        
        return {k: v.item() if torch.is_tensor(v) else v 
                for k, v in losses.items()}
    
    def _compute_losses(self, batch_data: Dict) -> Dict[str, torch.Tensor]:
        """计算损失"""
        losses = {}
        
        # 从PER缓冲区获取数据
        if 'per' in batch_data:
            per_data = batch_data['per']
            experiences = per_data['experiences']
            indices = per_data['indices']
            weights = torch.FloatTensor(per_data['weights']).to(self.device)
            
            # 提取批次数据
            states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
            actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
            next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
            dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
            
            # 重构状态
            task_features, resource_features = self._reconstruct_features(states)
            next_task_features, next_resource_features = self._reconstruct_features(next_states)
            
            # 计算当前Q值
            current_q_all = self.q_network(task_features, resource_features)
            current_q = current_q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Double DQN: 用主网络选择动作，目标网络评估
            with torch.no_grad():
                next_q_main = self.q_network(next_task_features, next_resource_features)
                next_actions = next_q_main.argmax(dim=1)
                
                next_q_target = self.target_network(next_task_features, next_resource_features)
                next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                
                target_q = rewards + self.config.gamma * next_q * (~dones)
            
            # TD误差
            td_errors = torch.abs(current_q - target_q)
            
            # 加权MSE损失
            td_loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
            losses['td_loss'] = td_loss
            
            # 更新优先级
            self.replay_buffer.update_priorities(
                indices, td_errors.detach().cpu().numpy()
            )
        
        # N-step损失
        if 'n_step' in batch_data and self.config.use_n_step:
            n_step_data = batch_data['n_step']
            states, actions, n_step_rewards, nth_states, dones, gamma_ns = n_step_data
            
            states = states.to(self.device)
            actions = actions.to(self.device)
            n_step_rewards = n_step_rewards.to(self.device)
            nth_states = nth_states.to(self.device)
            dones = dones.to(self.device)
            gamma_ns = gamma_ns.to(self.device)
            
            # 重构特征
            task_features, resource_features = self._reconstruct_features(states)
            nth_task_features, nth_resource_features = self._reconstruct_features(nth_states)
            
            # N-step Q值
            current_q_all = self.q_network(task_features, resource_features)
            current_q = current_q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                nth_q = self.target_network(nth_task_features, nth_resource_features)
                nth_q_max = nth_q.max(dim=1)[0]
                target_q = n_step_rewards + gamma_ns * nth_q_max * (~dones)
            
            n_step_loss = F.mse_loss(current_q, target_q)
            losses['n_step_loss'] = n_step_loss
        
        # 总损失
        total_loss = losses.get('td_loss', torch.tensor(0.0))
        if 'n_step_loss' in losses:
            total_loss = total_loss + 0.5 * losses['n_step_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _reconstruct_features(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从展平的状态重构任务和资源特征"""
        batch_size = states.shape[0]
        total_features = states.shape[1]
        
        # 实际的任务和资源数量
        num_tasks = 8
        num_resources = 6
        
        task_features_size = num_tasks * self.task_input_dim
        resource_features_size = num_resources * self.resource_input_dim
        
        # 确保不会超出边界
        task_end = min(task_features_size, total_features)
        resource_end = min(task_end + resource_features_size, total_features)
        
        # 提取任务特征
        task_flat = states[:, :task_end]
        if task_flat.shape[1] < task_features_size:
            # 填充不足的部分
            padding = torch.zeros(batch_size, task_features_size - task_flat.shape[1], device=states.device)
            task_flat = torch.cat([task_flat, padding], dim=1)
        task_features = task_flat.view(batch_size, num_tasks, self.task_input_dim)
        
        # 提取资源特征
        resource_flat = states[:, task_end:resource_end]
        if resource_flat.shape[1] < resource_features_size:
            # 填充不足的部分
            padding = torch.zeros(batch_size, resource_features_size - resource_flat.shape[1], device=states.device)
            resource_flat = torch.cat([resource_flat, padding], dim=1)
        resource_features = resource_flat.view(batch_size, num_resources, self.resource_input_dim)
        
        return task_features, resource_features
    
    def soft_update_target(self):
        """软更新目标网络"""
        for target_param, main_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * main_param.data +
                (1.0 - self.config.tau) * target_param.data
            )
    
    def hard_update_target(self):
        """硬更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def on_episode_end(self, episode_reward: float, episode_stats: Dict):
        """Episode结束时的回调"""
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        
        # 更新课程学习调度器
        if self.config.use_curriculum:
            self.curriculum_scheduler.step()
        
        # 更新学习率
        self.scheduler.step()
        
        # 日志
        if self.episode_count % 10 == 0:
            avg_reward = np.mean(self.episode_rewards[-10:])
            avg_loss = np.mean(self.training_losses[-100:]) if self.training_losses else 0
            
            self.logger.info(
                f"Episode {self.episode_count}: "
                f"Reward={episode_reward:.2f}, "
                f"Avg Reward={avg_reward:.2f}, "
                f"Avg Loss={avg_loss:.4f}, "
                f"Epsilon={self.exploration.get_stats().get('epsilon', 0):.3f}"
            )
    
    def save(self, filepath: str):
        """保存模型"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'training_losses': self.training_losses,
            'episode_rewards': self.episode_rewards,
            'exploration_stats': self.exploration.get_stats(),
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.training_losses = checkpoint['training_losses']
        self.episode_rewards = checkpoint['episode_rewards']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计"""
        return {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0,
            'avg_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'exploration_stats': self.exploration.get_stats(),
            'replay_buffer_size': len(self.replay_buffer),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }


class DAGAwareActionMasker:
    """DAG感知动作掩码器 - 只允许选择满足依赖的任务"""
    
    def __init__(self):
        pass
    
    def get_valid_actions(self, ready_tasks: List[int], 
                         num_resources: int) -> np.ndarray:
        """
        获取有效动作掩码
        
        Args:
            ready_tasks: 当前可调度的任务列表
            num_resources: 资源数量
            
        Returns:
            有效动作掩码 [action_dim]
        """
        action_dim = len(ready_tasks) * num_resources
        mask = np.zeros(action_dim, dtype=np.float32)
        
        for i, task_id in enumerate(ready_tasks):
            for r in range(num_resources):
                action_idx = i * num_resources + r
                if action_idx < action_dim:
                    mask[action_idx] = 1.0
        
        return mask
    
    def apply_mask(self, q_values: torch.Tensor, 
                   mask: np.ndarray) -> torch.Tensor:
        """应用动作掩码"""
        mask_tensor = torch.FloatTensor(mask).to(q_values.device)
        
        # 将无效动作的Q值设为很小的负数
        masked_q = q_values.clone()
        masked_q[mask_tensor == 0] = -1e9
        
        return masked_q


class LookaheadPlanner:
    """前瞻规划器 - 使用蒙特卡洛树搜索进行前瞻"""
    
    def __init__(self, lookahead_depth: int = 3, 
                 num_simulations: int = 10):
        self.lookahead_depth = lookahead_depth
        self.num_simulations = num_simulations
    
    def plan(self, q_network, state: Dict, 
            available_actions: List[int]) -> int:
        """
        使用前瞻规划选择动作
        
        简化版本：使用贪婪前瞻
        """
        if len(available_actions) == 0:
            return 0
        
        if len(available_actions) == 1:
            return available_actions[0]
        
        # 对每个可用动作评估前瞻价值
        action_values = []
        
        for action in available_actions:
            # 简化：直接使用Q值作为评估
            # 完整实现应该模拟执行动作并评估后续状态
            value = self._evaluate_action(q_network, state, action)
            action_values.append((action, value))
        
        # 选择最高价值的动作
        best_action = max(action_values, key=lambda x: x[1])[0]
        
        return best_action
    
    def _evaluate_action(self, q_network, state: Dict, action: int) -> float:
        """评估动作价值"""
        # 简化实现：返回Q值
        # 完整实现应该进行蒙特卡洛模拟
        return 0.0
