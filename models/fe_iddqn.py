import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, List, Optional, Dict
from .dual_stream_network import DualStreamNetwork
from .replay_buffer import PrioritizedReplayBuffer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.hyperparameters import Hyperparameters

class FE_IDDQN:
    """基于特征工程的改进双重深度Q网络"""
    
    def __init__(self, task_input_dim: int, resource_input_dim: int, 
                 action_dim: int, device: torch.device = None,
                 max_tasks: Optional[int] = None, max_resources: Optional[int] = None,
                 enable_graph_encoder: bool = True,
                 graph_encoder_layers: int = 2):
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

        # 状态结构（用于从flatten状态恢复回(任务,资源)）
        self.max_tasks = max_tasks
        self.max_resources = max_resources
        self.enable_graph_encoder = enable_graph_encoder
        
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
            dropout_rate=self.params['dropout_rate'],
            enable_graph_encoder=enable_graph_encoder,
            graph_encoder_layers=graph_encoder_layers,
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
            dropout_rate=self.params['dropout_rate'],
            enable_graph_encoder=enable_graph_encoder,
            graph_encoder_layers=graph_encoder_layers,
        ).to(self.device)
        
        # 初始化目标网络
        self.update_target_network()
        
        # 优化器（添加L2正则化）
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                   lr=self.params['learning_rate'],
                                   weight_decay=1e-5)  # L2正则化
        
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
        
    def select_action(
        self,
        task_features: np.ndarray,
        resource_features: np.ndarray,
        graph_adj: Optional[np.ndarray] = None,
        exploration_strategy: str = 'epsilon_greedy',
    ) -> int:
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

        graph_tensor = None
        if graph_adj is not None:
            graph_tensor = torch.FloatTensor(graph_adj).to(self.device)
            if len(graph_tensor.shape) == 2:
                graph_tensor = graph_tensor.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(task_tensor, resource_tensor, graph_adj=graph_tensor)
            q_values = q_values.squeeze(0)  # 移除批次维度

        # 动作mask：从状态特征推断资源可行性（提升样本效率与稳定性）
        action_mask = self._infer_action_mask(task_tensor, resource_tensor, q_values.shape[-1])
        if action_mask is not None and action_mask.numel() == q_values.numel():
            # invalid动作设置为很小的值，避免被选中
            q_values = q_values.masked_fill(~action_mask, float('-inf'))
        
        if exploration_strategy == 'epsilon_greedy':
            return self._epsilon_greedy_action(q_values, action_mask=action_mask)
        elif exploration_strategy == 'boltzmann':
            return self._boltzmann_action(q_values, action_mask=action_mask)
        elif exploration_strategy == 'ucb':
            return self._ucb_action(q_values, action_mask=action_mask)
        elif exploration_strategy == 'greedy':
            return self._greedy_action(q_values, action_mask=action_mask)
        else:
            return self._greedy_action(q_values, action_mask=action_mask)

    def _infer_action_mask(
        self,
        task_tensor: torch.Tensor,
        resource_tensor: torch.Tensor,
        action_dim: int,
        cpu_overload: float = 1.2,
        mem_overload: float = 1.1,
    ) -> Optional[torch.Tensor]:
        """从(任务/资源)特征中推断动作可行性mask。"""
        try:
            # task_tensor: [1, num_tasks, 16]
            # resource_tensor: [1, num_resources, 7]
            if task_tensor.ndim != 3 or resource_tensor.ndim != 3:
                return None

            # 当前要调度的任务默认是batch里的第0个
            current_task = task_tensor[0, 0]
            # 约定：第7/8维是cpu/mem需求（见HistoricalReplaySimulator._extract_task_features）
            cpu_req = float(current_task[7].item())
            mem_req = float(current_task[8].item())

            # 约定：资源特征第0/1维是cpu/mem容量，第2/3维是cpu/mem已用
            cpu_cap = resource_tensor[0, :action_dim, 0]
            mem_cap = resource_tensor[0, :action_dim, 1]
            cpu_used = resource_tensor[0, :action_dim, 2]
            mem_used = resource_tensor[0, :action_dim, 3]

            valid = (cpu_used + cpu_req <= cpu_cap * cpu_overload) & (mem_used + mem_req <= mem_cap * mem_overload)

            # 若全部不可行，返回None以便fallback到原策略（避免全-inf导致异常）
            if valid.sum().item() == 0:
                return None
            return valid
        except Exception:
            return None

    def _greedy_action(self, q_values: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> int:
        if action_mask is None:
            return torch.argmax(q_values).item()
        # 若mask导致全-inf，fallback
        if torch.isinf(q_values).all():
            return np.random.randint(0, self.action_dim)
        return torch.argmax(q_values).item()
    
    def _epsilon_greedy_action(self, q_values: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> int:
        """ε-贪婪策略"""
        if np.random.random() < self.epsilon:
            if action_mask is None:
                return np.random.randint(0, self.action_dim)
            valid_idxs = torch.where(action_mask)[0]
            if valid_idxs.numel() == 0:
                return np.random.randint(0, self.action_dim)
            return valid_idxs[torch.randint(0, valid_idxs.numel(), (1,))].item()
        else:
            return self._greedy_action(q_values, action_mask=action_mask)
    
    def _boltzmann_action(self, q_values: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> int:
        """Boltzmann探索策略"""
        if torch.isinf(q_values).all():
            return np.random.randint(0, self.action_dim)
        probabilities = F.softmax(q_values / max(self.temperature, 1e-6), dim=0)
        action = torch.multinomial(probabilities, 1).item()
        return action
    
    def _ucb_action(self, q_values: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> int:
        """上置信界探索策略（简化版本）"""
        # 这里简化实现，实际UCB需要维护动作选择计数
        exploration_bonus = np.sqrt(2 * np.log(self.step_count + 1) / (self.step_count + 1))
        ucb_values = q_values + exploration_bonus
        if torch.isinf(ucb_values).all():
            return np.random.randint(0, self.action_dim)
        return torch.argmax(ucb_values).item()
    
    def store_experience(
        self,
        state: Tuple[np.ndarray, np.ndarray],
        action: int,
        reward: float,
        next_state: Tuple[np.ndarray, np.ndarray],
        done: bool,
        graph_adj: Optional[np.ndarray] = None,
        next_graph_adj: Optional[np.ndarray] = None,
    ):
        """存储经验到回放缓冲区"""
        # 标准化状态维度，确保所有样本维度一致
        max_tasks = self.max_tasks if self.max_tasks else 5
        max_resources = self.max_resources if self.max_resources else 5
        expected_task_size = max_tasks * self.task_input_dim
        expected_resource_size = max_resources * self.resource_input_dim
        expected_graph_size = max_tasks * max_tasks
        expected_total = expected_task_size + expected_resource_size + expected_graph_size
        
        def normalize_state(s, g_adj):
            """标准化单个状态到固定维度"""
            task_flat = np.asarray(s[0], dtype=np.float32).flatten()
            resource_flat = np.asarray(s[1], dtype=np.float32).flatten()
            
            # Pad or truncate task features
            if len(task_flat) < expected_task_size:
                task_flat = np.pad(task_flat, (0, expected_task_size - len(task_flat)), mode='constant')
            else:
                task_flat = task_flat[:expected_task_size]
            
            # Pad or truncate resource features  
            if len(resource_flat) < expected_resource_size:
                resource_flat = np.pad(resource_flat, (0, expected_resource_size - len(resource_flat)), mode='constant')
            else:
                resource_flat = resource_flat[:expected_resource_size]
            
            # Handle graph adjacency
            if g_adj is not None:
                graph_flat = np.asarray(g_adj, dtype=np.float32).flatten()
                if len(graph_flat) < expected_graph_size:
                    graph_flat = np.pad(graph_flat, (0, expected_graph_size - len(graph_flat)), mode='constant')
                else:
                    graph_flat = graph_flat[:expected_graph_size]
            else:
                graph_flat = np.zeros(expected_graph_size, dtype=np.float32)
            
            result = np.concatenate([task_flat, resource_flat, graph_flat]).astype(np.float32)
            return result
        
        state_array = normalize_state(state, graph_adj)
        next_state_array = normalize_state(next_state, next_graph_adj)
        
        # 验证维度
        assert len(state_array) == expected_total, f"State size {len(state_array)} != expected {expected_total}"
        assert len(next_state_array) == expected_total, f"Next state size {len(next_state_array)} != expected {expected_total}"
        
        self.replay_buffer.add(state_array, action, reward, next_state_array, done)
    
    def train(self) -> Optional[float]:
        """训练网络"""
        if len(self.replay_buffer) < self.params['batch_size']:
            return None
        
        # 采样经验
        states, actions, rewards, next_states, dones, is_weights, idxs = \
            self.replay_buffer.sample(self.params['batch_size'])
        
        # 调试：检查状态维度
        expected_size = self.max_tasks * self.task_input_dim + self.max_resources * self.resource_input_dim + self.max_tasks * self.max_tasks
        if states.shape[1] != expected_size:
            self.logger.warning(f"State size mismatch: got {states.shape[1]}, expected {expected_size}. Skipping this batch.")
            return None
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        is_weights = is_weights.to(self.device)
        
        # 重构状态为任务、资源、图结构
        task_features, resource_features, graph_adj = self._reconstruct_features(states)
        next_task_features, next_resource_features, next_graph_adj = self._reconstruct_features(next_states)
        
        # 计算当前Q值
        current_q_values = self.q_network(task_features, resource_features, graph_adj=graph_adj)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            # 使用主网络选择动作
            next_q_values_main = self.q_network(next_task_features, next_resource_features, graph_adj=next_graph_adj)
            next_actions = torch.argmax(next_q_values_main, dim=1)
            
            # 使用目标网络评估动作
            next_q_values_target = self.target_network(next_task_features, next_resource_features, graph_adj=next_graph_adj)
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
    
    def _reconstruct_features(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """重构状态为任务、资源特征以及可选图结构"""
        batch_size = states.shape[0]
        total_features = states.shape[1]

        graph_adj = None

        # 方式1：如果已知(任务数、资源数)，直接按固定结构拆分
        if self.max_tasks is not None and self.max_resources is not None:
            task_feature_size = self.max_tasks * self.task_input_dim
            resource_feature_size = self.max_resources * self.resource_input_dim
            base_expected = task_feature_size + resource_feature_size

            # 允许拼接邻接矩阵（max_tasks^2）
            graph_expected = self.max_tasks * self.max_tasks
            if total_features == base_expected + graph_expected:
                # 提取各部分特征
                task_features = states[:, :task_feature_size].view(batch_size, self.max_tasks, self.task_input_dim)
                resource_features = states[:, task_feature_size:task_feature_size + resource_feature_size].view(batch_size, self.max_resources, self.resource_input_dim)
                graph_flat = states[:, base_expected:base_expected + graph_expected]
                graph_adj = graph_flat.view(batch_size, self.max_tasks, self.max_tasks)
                return task_features, resource_features, graph_adj

            if base_expected == total_features:
                task_features = states[:, :task_feature_size].view(batch_size, self.max_tasks, self.task_input_dim)
                resource_features = states[:, task_feature_size:task_feature_size + resource_feature_size].view(batch_size, self.max_resources, self.resource_input_dim)
                return task_features, resource_features, graph_adj

        # 方式2：从总长度反推(任务数、资源数)的整数解
        candidates: List[Tuple[int, int]] = []
        max_search_tasks = min(64, max(1, total_features // max(1, self.task_input_dim)))
        for num_tasks in range(1, max_search_tasks + 1):
            remaining = total_features - num_tasks * self.task_input_dim
            if remaining <= 0:
                continue
            if remaining % self.resource_input_dim != 0:
                continue
            num_resources = remaining // self.resource_input_dim
            if num_resources <= 0:
                continue
            candidates.append((num_tasks, num_resources))

        if not candidates:
            raise ValueError(
                f"Cannot reconstruct features: total_features={total_features}, "
                f"task_dim={self.task_input_dim}, resource_dim={self.resource_input_dim}."
            )

        # 若没显式提供max_tasks/max_resources，则偏好“规模适中”的拆分
        def score(pair: Tuple[int, int]) -> Tuple[int, int]:
            t, r = pair
            # 经验偏好：任务与资源都不极端偏大；其次优先较大的任务批（更贴近批处理设定）
            return (abs(t - r), -t)

        num_tasks, num_resources = sorted(candidates, key=score)[0]
        task_feature_size = num_tasks * self.task_input_dim

        task_features = states[:, :task_feature_size].view(batch_size, num_tasks, self.task_input_dim)
        resource_features = states[:, task_feature_size:].view(batch_size, num_resources, self.resource_input_dim)
        return task_features, resource_features, graph_adj
    
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
            'training_losses': self.training_losses,
            'replay_buffer_size': len(self.replay_buffer)
        }

