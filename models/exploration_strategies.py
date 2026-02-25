# -*- coding: utf-8 -*-
"""
增强版探索策略模块
包含Noisy Networks、动态ε调整、启发式引导探索等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, List, Callable
from abc import ABC, abstractmethod


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer - 参数化噪声实现自适应探索
    参考: Fortunato et al. "Noisy Networks for Exploration" (2018)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 sigma_init: float = 0.5, factorized: bool = True):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.factorized = factorized
        
        # 可学习参数
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # 噪声缓冲区
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        """初始化参数"""
        if self.factorized:
            mu_range = 1 / math.sqrt(self.in_features)
            sigma_init = self.sigma_init / math.sqrt(self.in_features)
        else:
            mu_range = math.sqrt(3 / self.in_features)
            sigma_init = 0.017
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init)
        
    def _scale_noise(self, size: int) -> torch.Tensor:
        """生成缩放噪声"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """重置噪声"""
        if self.factorized:
            # Factorized Gaussian Noise
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)
        else:
            # Independent Gaussian Noise
            self.weight_epsilon.copy_(torch.randn_like(self.weight_epsilon))
            self.bias_epsilon.copy_(torch.randn_like(self.bias_epsilon))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class NoisyNetwork(nn.Module):
    """使用Noisy Layers的网络"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 sigma_init: float = 0.5):
        super(NoisyNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(NoisyLinear(prev_dim, hidden_dim, sigma_init))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(NoisyLinear(prev_dim, output_dim, sigma_init))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def reset_noise(self):
        """重置所有Noisy层的噪声"""
        for module in self.network.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class ExplorationStrategy(ABC):
    """探索策略抽象基类"""
    
    @abstractmethod
    def select_action(self, q_values: torch.Tensor, **kwargs) -> int:
        pass
    
    @abstractmethod
    def update(self, **kwargs):
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        pass


class AdaptiveEpsilonGreedy(ExplorationStrategy):
    """自适应ε-贪婪策略"""
    
    def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, 
                 adaptive_mode: str = 'performance'):
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.adaptive_mode = adaptive_mode
        
        # 性能追踪
        self.recent_rewards = []
        self.window_size = 100
        self.improvement_threshold = 0.05
        
        # 统计
        self.total_actions = 0
        self.random_actions = 0
        
    def select_action(self, q_values: torch.Tensor, **kwargs) -> int:
        self.total_actions += 1
        
        if np.random.random() < self.epsilon:
            self.random_actions += 1
            return np.random.randint(0, q_values.shape[-1])
        else:
            return torch.argmax(q_values).item()
    
    def update(self, reward: Optional[float] = None, **kwargs):
        """更新探索参数"""
        if self.adaptive_mode == 'performance' and reward is not None:
            self._adaptive_update(reward)
        else:
            self._standard_update()
    
    def _standard_update(self):
        """标准衰减更新"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _adaptive_update(self, reward: float):
        """自适应更新：根据性能调整"""
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
        
        if len(self.recent_rewards) >= self.window_size:
            # 计算最近性能趋势
            first_half = np.mean(self.recent_rewards[:self.window_size//2])
            second_half = np.mean(self.recent_rewards[self.window_size//2:])
            
            improvement = (second_half - first_half) / (abs(first_half) + 1e-8)
            
            if improvement > self.improvement_threshold:
                # 性能改善，加快衰减
                self.epsilon = max(self.epsilon_end, self.epsilon * 0.99)
            elif improvement < -self.improvement_threshold:
                # 性能下降，增加探索
                self.epsilon = min(self.epsilon_start, self.epsilon * 1.02)
            else:
                # 正常衰减
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_stats(self) -> Dict:
        return {
            'epsilon': self.epsilon,
            'random_ratio': self.random_actions / max(1, self.total_actions),
            'total_actions': self.total_actions
        }


class BoltzmannExploration(ExplorationStrategy):
    """Boltzmann (Softmax) 探索策略"""
    
    def __init__(self, temperature_start: float = 1.0, temperature_end: float = 0.1,
                 temperature_decay: float = 0.995):
        self.temperature = temperature_start
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
        
    def select_action(self, q_values: torch.Tensor, **kwargs) -> int:
        # 数值稳定的softmax
        q_values = q_values.detach()
        q_max = q_values.max()
        exp_q = torch.exp((q_values - q_max) / self.temperature)
        probs = exp_q / exp_q.sum()
        
        action = torch.multinomial(probs, 1).item()
        return action
    
    def update(self, **kwargs):
        self.temperature = max(
            self.temperature_end, 
            self.temperature * self.temperature_decay
        )
    
    def get_stats(self) -> Dict:
        return {'temperature': self.temperature}


class UCBExploration(ExplorationStrategy):
    """Upper Confidence Bound探索策略"""
    
    def __init__(self, c: float = 2.0, action_dim: int = 6):
        self.c = c
        self.action_dim = action_dim
        self.action_counts = np.zeros(action_dim)
        self.total_steps = 0
        
    def select_action(self, q_values: torch.Tensor, **kwargs) -> int:
        self.total_steps += 1
        
        # 计算UCB值
        ucb_values = q_values.detach().cpu().numpy()
        
        for a in range(self.action_dim):
            if self.action_counts[a] == 0:
                # 未尝试的动作优先
                self.action_counts[a] += 1
                return a
            
            # UCB bonus
            ucb_bonus = self.c * np.sqrt(
                np.log(self.total_steps) / self.action_counts[a]
            )
            ucb_values[a] += ucb_bonus
        
        action = np.argmax(ucb_values)
        self.action_counts[action] += 1
        
        return action
    
    def update(self, **kwargs):
        pass  # UCB在select_action中更新
    
    def get_stats(self) -> Dict:
        return {
            'total_steps': self.total_steps,
            'action_counts': self.action_counts.tolist()
        }


class HeuristicGuidedExploration(ExplorationStrategy):
    """启发式引导的探索策略"""
    
    def __init__(self, heuristic_fn: Callable, 
                 guidance_weight_start: float = 0.8,
                 guidance_weight_end: float = 0.1,
                 guidance_decay: float = 0.995,
                 epsilon: float = 0.1):
        self.heuristic_fn = heuristic_fn
        self.guidance_weight = guidance_weight_start
        self.guidance_weight_start = guidance_weight_start
        self.guidance_weight_end = guidance_weight_end
        self.guidance_decay = guidance_decay
        self.epsilon = epsilon
        
    def select_action(self, q_values: torch.Tensor, 
                     state: Optional[Dict] = None, **kwargs) -> int:
        # 随机探索
        if np.random.random() < self.epsilon:
            return np.random.randint(0, q_values.shape[-1])
        
        # 获取启发式建议
        heuristic_scores = self.heuristic_fn(state) if state else q_values.detach()
        
        if isinstance(heuristic_scores, np.ndarray):
            heuristic_scores = torch.FloatTensor(heuristic_scores)
        
        # 归一化
        q_normalized = F.softmax(q_values.detach(), dim=-1)
        h_normalized = F.softmax(heuristic_scores, dim=-1)
        
        # 混合Q值和启发式
        combined = (1 - self.guidance_weight) * q_normalized + \
                   self.guidance_weight * h_normalized
        
        action = torch.argmax(combined).item()
        return action
    
    def update(self, **kwargs):
        self.guidance_weight = max(
            self.guidance_weight_end,
            self.guidance_weight * self.guidance_decay
        )
    
    def get_stats(self) -> Dict:
        return {'guidance_weight': self.guidance_weight}


class IntrinsicCuriosityModule(nn.Module):
    """
    内在好奇心模块 (ICM)
    参考: Pathak et al. "Curiosity-driven Exploration" (2017)
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 feature_dim: int = 64, hidden_dim: int = 128,
                 eta: float = 0.01, beta: float = 0.2):
        super(IntrinsicCuriosityModule, self).__init__()
        
        self.eta = eta  # 内在奖励缩放
        self.beta = beta  # 前向/逆向损失权重
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 逆向模型：预测动作
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 前向模型：预测下一状态特征
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor,
                next_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            intrinsic_reward: 内在奖励
            inverse_loss: 逆向模型损失
            forward_loss: 前向模型损失
        """
        # 编码状态
        phi_s = self.feature_encoder(state)
        phi_s_next = self.feature_encoder(next_state)
        
        # 逆向模型
        phi_concat = torch.cat([phi_s, phi_s_next], dim=-1)
        action_pred = self.inverse_model(phi_concat)
        inverse_loss = F.cross_entropy(action_pred, action.long())
        
        # 前向模型
        action_onehot = F.one_hot(action.long(), num_classes=action_pred.shape[-1]).float()
        forward_input = torch.cat([phi_s, action_onehot], dim=-1)
        phi_s_next_pred = self.forward_model(forward_input)
        forward_loss = F.mse_loss(phi_s_next_pred, phi_s_next.detach())
        
        # 内在奖励：前向预测误差
        intrinsic_reward = self.eta * forward_loss.detach()
        
        return intrinsic_reward, inverse_loss, forward_loss
    
    def get_intrinsic_reward(self, state: torch.Tensor, action: torch.Tensor,
                            next_state: torch.Tensor) -> torch.Tensor:
        """计算内在奖励"""
        phi_s = self.feature_encoder(state)
        phi_s_next = self.feature_encoder(next_state)
        
        action_onehot = F.one_hot(action.long(), num_classes=6).float()  # 假设6个动作
        forward_input = torch.cat([phi_s, action_onehot], dim=-1)
        phi_s_next_pred = self.forward_model(forward_input)
        
        intrinsic_reward = self.eta * F.mse_loss(
            phi_s_next_pred, phi_s_next.detach(), reduction='none'
        ).mean(dim=-1)
        
        return intrinsic_reward


class CombinedExplorationStrategy(ExplorationStrategy):
    """组合探索策略"""
    
    def __init__(self, strategies: Dict[str, Tuple[ExplorationStrategy, float]]):
        """
        Args:
            strategies: {name: (strategy, initial_weight)}
        """
        self.strategies = {name: s for name, (s, _) in strategies.items()}
        self.weights = {name: w for name, (_, w) in strategies.items()}
        self.active_strategy = None
        
    def select_action(self, q_values: torch.Tensor, **kwargs) -> int:
        # 根据权重选择策略
        strategy_names = list(self.strategies.keys())
        weights = [self.weights[name] for name in strategy_names]
        weights = np.array(weights) / sum(weights)
        
        self.active_strategy = np.random.choice(strategy_names, p=weights)
        
        return self.strategies[self.active_strategy].select_action(q_values, **kwargs)
    
    def update(self, **kwargs):
        for strategy in self.strategies.values():
            strategy.update(**kwargs)
    
    def get_stats(self) -> Dict:
        stats = {'active_strategy': self.active_strategy}
        for name, strategy in self.strategies.items():
            stats[name] = strategy.get_stats()
        return stats
