#!/usr/bin/env python3
"""
Prioritized Experience Replay Buffer
基于TD-error的优先级采样
"""

import numpy as np
import random
from collections import namedtuple


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class SumTree:
    """
    Sum Tree数据结构，用于高效的优先级采样
    
    叶节点存储优先级，父节点存储子节点之和
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """检索优先级对应的叶节点"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """返回所有优先级之和"""
        return self.tree[0]
    
    def add(self, priority, data):
        """添加数据"""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, priority):
        """更新优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """根据优先级采样"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    使用TD-error作为优先级，优先采样重要的经验
    
    Args:
        capacity: buffer容量
        alpha: 优先级指数 (0=uniform, 1=full prioritization)
        beta_start: 重要性采样起始值
        beta_frames: beta增长到1.0的帧数
    """
    
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 0.01  # 小常数，避免零优先级
        self.abs_err_upper = 1.0  # TD-error上限
    
    def _get_beta(self):
        """Beta线性增长"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def _get_priority(self, error):
        """计算优先级"""
        return (np.abs(error) + self.epsilon) ** self.alpha
    
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        # 新经验使用最大优先级
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        
        experience = Experience(state, action, reward, next_state, done)
        self.tree.add(max_priority, experience)
    
    def sample(self, batch_size):
        """
        采样一个batch
        
        Returns:
            states, actions, rewards, next_states, dones, is_weights, idxs
        """
        batch = []
        idxs = []
        priorities = []
        
        # 分段采样
        segment = self.tree.total() / batch_size
        
        beta = self._get_beta()
        self.frame += 1
        
        # 计算最小概率（用于重要性采样权重）
        min_prob = np.min(self.tree.tree[-self.tree.capacity:self.tree.capacity + self.tree.n_entries]) / self.tree.total()
        if min_prob == 0:
            min_prob = 1e-10
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            if data is not None:
                priorities.append(priority)
                batch.append(data)
                idxs.append(idx)
        
        # 计算重要性采样权重
        sampling_probs = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probs, -beta)
        is_weights /= is_weights.max()
        
        # 解包经验
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        return states, actions, rewards, next_states, dones, is_weights, idxs
    
    def update_priorities(self, idxs, errors):
        """更新优先级（基于TD-error）"""
        for idx, error in zip(idxs, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)
    
    def __len__(self):
        """返回当前buffer大小"""
        return self.tree.n_entries

