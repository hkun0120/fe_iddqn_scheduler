# -*- coding: utf-8 -*-
"""
On-Policy轨迹缓冲区（Rollout Buffer）
用于PPO算法的轨迹数据存储与GAE计算
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field


@dataclass
class RolloutSample:
    """单步轨迹样本"""
    state: np.ndarray
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool
    # 可选的结构化状态
    task_features: Optional[np.ndarray] = None
    resource_features: Optional[np.ndarray] = None
    adj_matrix: Optional[np.ndarray] = None
    action_mask: Optional[np.ndarray] = None


class RolloutBuffer:
    """
    PPO轨迹缓冲区
    
    支持GAE (Generalized Advantage Estimation) 计算
    支持Mini-batch迭代采样
    """
    
    def __init__(self, buffer_size: int, gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        """
        Args:
            buffer_size: 缓冲区容量（每次rollout收集的步数）
            gamma: 折扣因子
            gae_lambda: GAE的λ参数，平衡偏差与方差
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 存储轨迹数据
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        
        # 结构化状态（用于双流网络）
        self.task_features_list: List[Optional[np.ndarray]] = []
        self.resource_features_list: List[Optional[np.ndarray]] = []
        self.adj_matrices: List[Optional[np.ndarray]] = []
        self.action_masks: List[Optional[np.ndarray]] = []
        
        # GAE计算后的结果
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None
        
        self.pos = 0
        self.full = False
    
    def add(self, state: np.ndarray, action: int, reward: float,
            value: float, log_prob: float, done: bool,
            task_features: Optional[np.ndarray] = None,
            resource_features: Optional[np.ndarray] = None,
            adj_matrix: Optional[np.ndarray] = None,
            action_mask: Optional[np.ndarray] = None):
        """添加一步经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.task_features_list.append(task_features)
        self.resource_features_list.append(resource_features)
        self.adj_matrices.append(adj_matrix)
        self.action_masks.append(action_mask)
        
        self.pos += 1
    
    def compute_gae(self, last_value: float, last_done: bool):
        """
        计算GAE (Generalized Advantage Estimation)
        
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        A_t = δ_t + (γ * λ) * A_{t+1}
        
        Args:
            last_value: 最后一个状态的价值估计
            last_done: 最后一个状态是否为终止状态
        """
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)
        
        last_gae = 0.0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t + 1])
                next_value = self.values[t + 1]
            
            # TD误差: δ_t = r_t + γ * V(s_{t+1}) * (1-done) - V(s_t)
            delta = (self.rewards[t] 
                     + self.gamma * next_value * next_non_terminal
                     - self.values[t])
            
            # GAE: A_t = δ_t + (γ * λ) * (1-done) * A_{t+1}
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        # Returns = Advantages + Values
        self.returns = self.advantages + np.array(self.values, dtype=np.float32)
    
    def get_batches(self, batch_size: int, device: torch.device
                    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        生成Mini-batch数据用于PPO策略更新
        
        Args:
            batch_size: Mini-batch大小
            device: 计算设备
            
        Yields:
            包含batch数据的字典
        """
        n = len(self.states)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            batch = {
                'states': torch.FloatTensor(
                    np.array([self.states[i] for i in batch_indices])
                ).to(device),
                'actions': torch.LongTensor(
                    [self.actions[i] for i in batch_indices]
                ).to(device),
                'old_log_probs': torch.FloatTensor(
                    [self.log_probs[i] for i in batch_indices]
                ).to(device),
                'advantages': torch.FloatTensor(
                    self.advantages[batch_indices]
                ).to(device),
                'returns': torch.FloatTensor(
                    self.returns[batch_indices]
                ).to(device),
                'old_values': torch.FloatTensor(
                    [self.values[i] for i in batch_indices]
                ).to(device),
            }
            
            # 结构化状态（如果可用）
            if self.task_features_list[0] is not None:
                batch['task_features'] = torch.FloatTensor(
                    np.array([self.task_features_list[i] for i in batch_indices])
                ).to(device)
            
            if self.resource_features_list[0] is not None:
                batch['resource_features'] = torch.FloatTensor(
                    np.array([self.resource_features_list[i] for i in batch_indices])
                ).to(device)
            
            if self.adj_matrices[0] is not None:
                batch['adj_matrix'] = torch.FloatTensor(
                    np.array([self.adj_matrices[i] for i in batch_indices])
                ).to(device)
            
            if self.action_masks[0] is not None:
                batch['action_masks'] = torch.FloatTensor(
                    np.array([self.action_masks[i] for i in batch_indices])
                ).to(device)
            
            yield batch
    
    def reset(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.task_features_list.clear()
        self.resource_features_list.clear()
        self.adj_matrices.clear()
        self.action_masks.clear()
        self.advantages = None
        self.returns = None
        self.pos = 0
        self.full = False
    
    def __len__(self) -> int:
        return len(self.states)
