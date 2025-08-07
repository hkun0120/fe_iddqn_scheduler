import numpy as np
import torch
import random
from typing import Tuple, List, Optional
from collections import namedtuple, deque

# 定义经验元组
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class SumTree:
    """求和树数据结构，用于优先级采样"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """检索叶子节点索引"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """返回总优先级"""
        return self.tree[0]
    
    def add(self, priority: float, data: Experience):
        """添加新的经验"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """更新优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Experience]:
        """根据优先级采样"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001, epsilon: float = 1e-6):
        """
        初始化优先级回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数，控制优先级的重要性
            beta: 重要性采样指数
            beta_increment: beta的增长率
            epsilon: 防止优先级为0的小常数
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """添加经验到缓冲区"""
        experience = Experience(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                              torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """采样一批经验"""
        batch = []
        idxs = []
        priorities = []
        
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)
        
        # 计算重要性采样权重
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        
        # 转换为张量
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        is_weights = torch.FloatTensor(is_weights)
        
        return states, actions, rewards, next_states, dones, is_weights, idxs
    
    def update_priorities(self, idxs: List[int], priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(idxs, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def update_beta(self):
        """更新beta值"""
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def __len__(self) -> int:
        """返回缓冲区中的经验数量"""
        return self.tree.n_entries

class ReplayBuffer:
    """标准经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """添加经验到缓冲区"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                              torch.Tensor, torch.Tensor]:
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """返回缓冲区中的经验数量"""
        return len(self.buffer)

