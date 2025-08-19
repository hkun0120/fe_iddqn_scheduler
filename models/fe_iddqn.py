import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, List, Optional, Dict
from .dual_stream_network import DualStreamNetwork
from .replay_buffer import PrioritizedReplayBuffer
from config.hyperparameters import Hyperparameters

class FE_IDDQN:
    """基于特征工程的改进双重深度Q网络"""
    
    def __init__(self, task_input_dim: int, resource_input_dim: int, 
                 action_dim: int, device: torch.device = None):
        """
        初始化FE-IDDQN算法
        
        Args:
            task_input_dim: 任务特征维度
            resource_input_dim: 资源特征维度
            action_dim: 动作空间维度
            device: 计算设备
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # 超参数
        self.params = Hyperparameters.get_algorithm_params('FE_IDDQN')
        
        # 网络参数
        self.task_input_dim = task_input_dim
        self.resource_input_dim = resource_input_dim
        self.action_dim = action_dim
        
        # 创建主网络和目标网络
        self.q_network = DualStreamNetwork(
            task_input_dim=task_input_dim,
            resource_input_dim=resource_input_dim,
            task_hidden_dims=self.params['task_stream_hidden_dims'],
            resource_hidden_dims=self.params['resource_stream_hidden_dims'],
            fusion_dim=self.params['fusion_dim'],
            output_dim=action_dim,
            attention_dim=self.params['attention_dim'],
            num_heads=self.params['attention_heads'],
            dropout_rate=self.params['dropout_rate']
        ).to(self.device)
        
        self.target_network = DualStreamNetwork(
            task_input_dim=task_input_dim,
            resource_input_dim=resource_input_dim,
            task_hidden_dims=self.params['task_stream_hidden_dims'],
            resource_hidden_dims=self.params['resource_stream_hidden_dims'],
            fusion_dim=self.params['fusion_dim'],
            output_dim=action_dim,
            attention_dim=self.params['attention_dim'],
            num_heads=self.params['attention_heads'],
            dropout_rate=self.params['dropout_rate']
        ).to(self.device)
        
        # 初始化目标网络
        self.update_target_network()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                   lr=self.params['learning_rate'])
        
        # 经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.params['replay_buffer_size'],
            alpha=self.params['priority_alpha'],
            beta=self.params['priority_beta'],
            beta_increment=self.params['priority_beta_increment']
        )
        
        # 探索策略参数
        self.epsilon = self.params['epsilon_start']
        self.temperature = self.params['temperature_start']
        
        # 训练统计
        self.step_count = 0
        self.episode_count = 0
        self.training_losses = []
        
    def select_action(self, task_features: np.ndarray, resource_features: np.ndarray, 
                     exploration_strategy: str = 'epsilon_greedy') -> int:
        """
        选择动作
        
        Args:
            task_features: 任务特征 [num_tasks, task_input_dim]
            resource_features: 资源特征 [num_resources, resource_input_dim]
            exploration_strategy: 探索策略 ('epsilon_greedy', 'boltzmann', 'ucb')
            
        Returns:
            选择的动作
        """
        # 转换为张量，如果已经是3D则不需要添加批次维度
        task_tensor = torch.FloatTensor(task_features).to(self.device)
        resource_tensor = torch.FloatTensor(resource_features).to(self.device)
        
        # 确保是3D张量 [batch_size, num_items, features]
        if len(task_tensor.shape) == 2:
            task_tensor = task_tensor.unsqueeze(0)
        if len(resource_tensor.shape) == 2:
            resource_tensor = resource_tensor.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(task_tensor, resource_tensor)
            q_values = q_values.squeeze(0)  # 移除批次维度
        
        if exploration_strategy == 'epsilon_greedy':
            return self._epsilon_greedy_action(q_values)
        elif exploration_strategy == 'boltzmann':
            return self._boltzmann_action(q_values)
        elif exploration_strategy == 'ucb':
            return self._ucb_action(q_values)
        else:
            return torch.argmax(q_values).item()
    
    def _epsilon_greedy_action(self, q_values: torch.Tensor) -> int:
        """ε-贪婪策略"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            return torch.argmax(q_values).item()
    
    def _boltzmann_action(self, q_values: torch.Tensor) -> int:
        """Boltzmann探索策略"""
        probabilities = F.softmax(q_values / self.temperature, dim=0)
        action = torch.multinomial(probabilities, 1).item()
        return action
    
    def _ucb_action(self, q_values: torch.Tensor) -> int:
        """上置信界探索策略（简化版本）"""
        # 这里简化实现，实际UCB需要维护动作选择计数
        exploration_bonus = np.sqrt(2 * np.log(self.step_count + 1) / (self.step_count + 1))
        ucb_values = q_values + exploration_bonus
        return torch.argmax(ucb_values).item()
    
    def store_experience(self, state: Tuple[np.ndarray, np.ndarray], action: int, 
                        reward: float, next_state: Tuple[np.ndarray, np.ndarray], 
                        done: bool):
        """存储经验到回放缓冲区"""
        # 将状态元组转换为单一数组
        state_array = np.concatenate([state[0].flatten(), state[1].flatten()])
        next_state_array = np.concatenate([next_state[0].flatten(), next_state[1].flatten()])
        
        self.replay_buffer.add(state_array, action, reward, next_state_array, done)
    
    def train(self) -> Optional[float]:
        """训练网络"""
        if len(self.replay_buffer) < self.params['batch_size']:
            return None
        
        # 采样经验
        states, actions, rewards, next_states, dones, is_weights, idxs = \
            self.replay_buffer.sample(self.params['batch_size'])
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        is_weights = is_weights.to(self.device)
        
        # 重构状态为任务和资源特征
        task_features, resource_features = self._reconstruct_features(states)
        next_task_features, next_resource_features = self._reconstruct_features(next_states)
        
        # 计算当前Q值
        current_q_values = self.q_network(task_features, resource_features)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            # 使用主网络选择动作
            next_q_values_main = self.q_network(next_task_features, next_resource_features)
            next_actions = torch.argmax(next_q_values_main, dim=1)
            
            # 使用目标网络评估动作
            next_q_values_target = self.target_network(next_task_features, next_resource_features)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q_values = rewards + (self.params['gamma'] * next_q_values * ~dones)
        
        # 计算TD误差
        td_errors = torch.abs(current_q_values - target_q_values)
        
        # 计算加权损失
        loss = (is_weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新优先级
        priorities = td_errors.detach().cpu().numpy()
        self.replay_buffer.update_priorities(idxs, priorities)
        
        # 更新beta
        self.replay_buffer.update_beta()
        
        # 记录损失
        self.training_losses.append(loss.item())
        
        return loss.item()
    
    def _reconstruct_features(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """重构状态为任务和资源特征"""
        batch_size = states.shape[0]
        total_features = states.shape[1]
        
        # 动态计算任务和资源特征的分割点
        # 假设任务特征和资源特征的总维度是固定的
        # 我们需要根据实际的状态维度来推断任务和资源的数量
        
        # 计算每个样本的总特征数
        features_per_sample = total_features // batch_size if batch_size > 0 else total_features
        
        # 假设任务特征占2/3，资源特征占1/3
        task_features_count = int(features_per_sample * 2 / 3)
        resource_features_count = features_per_sample - task_features_count
        
        # 确保任务特征数量是task_input_dim的倍数
        num_tasks = max(1, task_features_count // self.task_input_dim)
        task_feature_size = num_tasks * self.task_input_dim
        
        # 确保资源特征数量是resource_input_dim的倍数
        num_resources = max(1, resource_features_count // self.resource_input_dim)
        resource_feature_size = num_resources * self.resource_input_dim
        
        # 如果总特征数不够，调整分割
        if task_feature_size + resource_feature_size > total_features:
            # 按比例调整
            ratio = total_features / (task_feature_size + resource_feature_size)
            task_feature_size = int(task_feature_size * ratio)
            resource_feature_size = total_features - task_feature_size
        
        # 重塑为3D张量
        task_features = states[:, :task_feature_size].view(batch_size, num_tasks, self.task_input_dim)
        resource_features = states[:, task_feature_size:task_feature_size + resource_feature_size].view(
            batch_size, num_resources, self.resource_input_dim)
        
        return task_features, resource_features
    
    def update_target_network(self):
        """软更新目标网络"""
        for target_param, main_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(
                self.params['tau'] * main_param.data + 
                (1.0 - self.params['tau']) * target_param.data
            )
    
    def update_exploration_params(self):
        """更新探索参数"""
        # 更新epsilon
        self.epsilon = max(
            self.params['epsilon_end'],
            self.epsilon * self.params['epsilon_decay']
        )
        
        # 更新temperature
        self.temperature = max(
            self.params['temperature_end'],
            self.temperature * self.params['temperature_decay']
        )
        
        self.step_count += 1
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'temperature': self.temperature,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'training_losses': self.training_losses
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.temperature = checkpoint['temperature']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.training_losses = checkpoint['training_losses']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'temperature': self.temperature,
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0,
            'replay_buffer_size': len(self.replay_buffer)
        }

