# -*- coding: utf-8 -*-
"""
增强版经验回放模块
包含N-step Returns、优先级经验回放增强、分层缓冲、HER等
"""

import torch
import numpy as np
import random
from typing import Tuple, List, Optional, Dict, Any, NamedTuple
from collections import deque, namedtuple
from dataclasses import dataclass
import heapq


# 经验元组定义
class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Optional[Dict] = None


class NStepExperience(NamedTuple):
    """N-step经验"""
    state: np.ndarray
    action: int
    n_step_reward: float  # 累积N步奖励
    nth_state: np.ndarray  # N步后的状态
    done: bool
    gamma_n: float  # gamma^n
    info: Optional[Dict] = None


class SegmentTree:
    """线段树，支持高效的区间查询和更新"""
    
    def __init__(self, capacity: int, operation, neutral_element):
        self.capacity = capacity
        self.operation = operation
        self.neutral_element = neutral_element
        
        # 完整二叉树需要2n-1个节点
        self.tree = [neutral_element] * (2 * capacity)
        
    def _reduce(self, start: int, end: int, node: int, 
                node_start: int, node_end: int):
        """递归区间查询"""
        if start == node_start and end == node_end:
            return self.tree[node]
        
        mid = (node_start + node_end) // 2
        
        if end <= mid:
            return self._reduce(start, end, 2 * node, node_start, mid)
        elif start > mid:
            return self._reduce(start, end, 2 * node + 1, mid + 1, node_end)
        else:
            left = self._reduce(start, mid, 2 * node, node_start, mid)
            right = self._reduce(mid + 1, end, 2 * node + 1, mid + 1, node_end)
            return self.operation(left, right)
    
    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """区间聚合操作"""
        if end is None:
            end = self.capacity - 1
        return self._reduce(start, end, 1, 0, self.capacity - 1)
    
    def __setitem__(self, idx: int, val: float):
        """更新单个值"""
        idx += self.capacity
        self.tree[idx] = val
        
        # 向上更新
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(
                self.tree[2 * idx], 
                self.tree[2 * idx + 1]
            )
            idx //= 2
    
    def __getitem__(self, idx: int) -> float:
        return self.tree[idx + self.capacity]


class SumSegmentTree(SegmentTree):
    """求和线段树"""
    
    def __init__(self, capacity: int):
        super().__init__(capacity, lambda a, b: a + b, 0.0)
    
    def sum(self, start: int = 0, end: Optional[int] = None) -> float:
        return self.reduce(start, end)
    
    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """找到满足前缀和条件的最小索引"""
        idx = 1
        while idx < self.capacity:
            if self.tree[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self.tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """最小值线段树"""
    
    def __init__(self, capacity: int):
        super().__init__(capacity, min, float('inf'))
    
    def min(self, start: int = 0, end: Optional[int] = None) -> float:
        return self.reduce(start, end)


class EnhancedPrioritizedReplayBuffer:
    """增强版优先级经验回放缓冲区"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, 
                 beta_start: float = 0.4, beta_frames: int = 100000,
                 epsilon: float = 1e-6):
        # 确保容量是2的幂
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
        
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0
        
        # 线段树
        self.sum_tree = SumSegmentTree(self.capacity)
        self.min_tree = MinSegmentTree(self.capacity)
        
        # 数据存储
        self.data = [None] * self.capacity
        self.write_idx = 0
        self.size = 0
        self.max_priority = 1.0
        
    @property
    def beta(self) -> float:
        """动态计算beta值"""
        return min(1.0, self.beta_start + 
                  (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def add(self, experience: Experience, priority: Optional[float] = None):
        """添加经验"""
        if priority is None:
            priority = self.max_priority
        
        # 存储数据
        self.data[self.write_idx] = experience
        
        # 更新树
        priority_alpha = (priority + self.epsilon) ** self.alpha
        self.sum_tree[self.write_idx] = priority_alpha
        self.min_tree[self.write_idx] = priority_alpha
        
        # 更新索引
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.frame += 1
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """采样一批经验"""
        indices = []
        priorities = []
        
        # 分段采样
        total_priority = self.sum_tree.sum(0, self.size - 1)
        segment = total_priority / batch_size
        
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            sample = random.uniform(low, high)
            
            idx = self.sum_tree.find_prefixsum_idx(sample)
            idx = min(idx, self.size - 1)
            
            indices.append(idx)
            priorities.append(self.sum_tree[idx])
        
        # 计算重要性采样权重
        min_priority = self.min_tree.min(0, self.size - 1)
        max_weight = (min_priority * self.size) ** (-self.beta)
        
        sampling_probs = np.array(priorities) / total_priority
        weights = (sampling_probs * self.size) ** (-self.beta)
        weights = weights / max_weight  # 归一化
        
        # 获取经验
        experiences = [self.data[idx] for idx in indices]
        
        return experiences, np.array(indices), weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            priority_alpha = (priority + self.epsilon) ** self.alpha
            self.sum_tree[idx] = priority_alpha
            self.min_tree[idx] = priority_alpha
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.size


class NStepReplayBuffer:
    """N-step经验回放缓冲区"""
    
    def __init__(self, capacity: int, n_step: int = 3, gamma: float = 0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        
        # 主缓冲区
        self.buffer = deque(maxlen=capacity)
        
        # N-step临时缓冲区
        self.n_step_buffer = deque(maxlen=n_step)
        
    def _get_n_step_info(self) -> Tuple[float, np.ndarray, bool]:
        """计算N-step回报和最终状态"""
        n_step_reward = 0
        gamma_n = 1.0
        
        for idx, exp in enumerate(self.n_step_buffer):
            n_step_reward += gamma_n * exp.reward
            gamma_n *= self.gamma
            
            if exp.done:
                break
        
        # 最终状态是N-step缓冲区中最后一个经验的next_state
        last_exp = self.n_step_buffer[-1]
        
        return n_step_reward, last_exp.next_state, last_exp.done, gamma_n
    
    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool, info: Optional[Dict] = None):
        """添加经验"""
        exp = Experience(state, action, reward, next_state, done, info)
        self.n_step_buffer.append(exp)
        
        # 只有当N-step缓冲区满时才添加到主缓冲区
        if len(self.n_step_buffer) == self.n_step:
            n_step_reward, nth_state, nth_done, gamma_n = self._get_n_step_info()
            
            # 获取初始状态和动作
            first_exp = self.n_step_buffer[0]
            
            n_step_exp = NStepExperience(
                state=first_exp.state,
                action=first_exp.action,
                n_step_reward=n_step_reward,
                nth_state=nth_state,
                done=nth_done,
                gamma_n=gamma_n,
                info=first_exp.info
            )
            
            self.buffer.append(n_step_exp)
        
        # 如果episode结束，清空N-step缓冲区并添加剩余经验
        if done:
            while len(self.n_step_buffer) > 0:
                n_step_reward, nth_state, nth_done, gamma_n = self._get_n_step_info()
                first_exp = self.n_step_buffer[0]
                
                n_step_exp = NStepExperience(
                    state=first_exp.state,
                    action=first_exp.action,
                    n_step_reward=n_step_reward,
                    nth_state=nth_state,
                    done=nth_done,
                    gamma_n=gamma_n,
                    info=first_exp.info
                )
                
                self.buffer.append(n_step_exp)
                self.n_step_buffer.popleft()
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """采样一批N-step经验"""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        n_step_rewards = torch.FloatTensor([e.n_step_reward for e in batch])
        nth_states = torch.FloatTensor([e.nth_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        gamma_ns = torch.FloatTensor([e.gamma_n for e in batch])
        
        return states, actions, n_step_rewards, nth_states, dones, gamma_ns
    
    def __len__(self) -> int:
        return len(self.buffer)


class HierarchicalReplayBuffer:
    """分层经验回放缓冲区 - 按工作流类型分层存储"""
    
    def __init__(self, capacity: int, num_levels: int = 3):
        self.capacity = capacity
        self.num_levels = num_levels
        
        # 每个层级的缓冲区
        self.level_buffers = [
            deque(maxlen=capacity // num_levels) 
            for _ in range(num_levels)
        ]
        
        # 层级采样权重
        self.level_weights = np.ones(num_levels) / num_levels
        
        # 统计信息
        self.level_stats = {i: {'count': 0, 'avg_reward': 0} 
                          for i in range(num_levels)}
    
    def _get_level(self, experience: Experience) -> int:
        """根据经验特征确定层级"""
        info = experience.info or {}
        
        # 可以根据工作流复杂度、任务数量等确定层级
        complexity = info.get('workflow_complexity', 0.5)
        
        level = int(complexity * (self.num_levels - 1))
        return min(level, self.num_levels - 1)
    
    def add(self, experience: Experience):
        """添加经验到对应层级"""
        level = self._get_level(experience)
        self.level_buffers[level].append(experience)
        
        # 更新统计
        self.level_stats[level]['count'] += 1
        alpha = 0.01
        self.level_stats[level]['avg_reward'] = (
            (1 - alpha) * self.level_stats[level]['avg_reward'] + 
            alpha * experience.reward
        )
    
    def sample(self, batch_size: int, 
               level_probs: Optional[np.ndarray] = None) -> List[Experience]:
        """从各层级采样"""
        if level_probs is None:
            level_probs = self.level_weights
        
        # 确定每个层级采样数量
        level_counts = np.random.multinomial(batch_size, level_probs)
        
        batch = []
        for level, count in enumerate(level_counts):
            if count > 0 and len(self.level_buffers[level]) > 0:
                samples = random.sample(
                    list(self.level_buffers[level]), 
                    min(count, len(self.level_buffers[level]))
                )
                batch.extend(samples)
        
        # 如果采样不足，从其他层级补充
        while len(batch) < batch_size:
            level = random.randint(0, self.num_levels - 1)
            if len(self.level_buffers[level]) > 0:
                batch.append(random.choice(self.level_buffers[level]))
        
        return batch[:batch_size]
    
    def update_weights(self, td_errors_by_level: Dict[int, float]):
        """根据TD误差更新层级权重"""
        for level, td_error in td_errors_by_level.items():
            self.level_weights[level] = abs(td_error) + 0.01
        
        self.level_weights /= self.level_weights.sum()
    
    def __len__(self) -> int:
        return sum(len(buf) for buf in self.level_buffers)


class HindsightExperienceReplay:
    """
    Hindsight Experience Replay (HER)
    用于从失败的调度中学习
    """
    
    def __init__(self, capacity: int, k: int = 4, 
                 strategy: str = 'future'):
        """
        Args:
            capacity: 缓冲区容量
            k: 每个episode生成的hindsight经验数量
            strategy: 'future', 'final', 'episode'
        """
        self.capacity = capacity
        self.k = k
        self.strategy = strategy
        
        # Episode缓冲区
        self.episode_buffer = []
        
        # 主缓冲区
        self.buffer = deque(maxlen=capacity)
    
    def add_transition(self, state: np.ndarray, action: int, reward: float,
                       next_state: np.ndarray, done: bool, 
                       achieved_goal: Dict, desired_goal: Dict):
        """添加一个transition到当前episode"""
        self.episode_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        })
        
        if done:
            self._store_episode()
            self.episode_buffer = []
    
    def _store_episode(self):
        """存储episode并生成hindsight经验"""
        episode = self.episode_buffer
        
        for t, transition in enumerate(episode):
            # 存储原始经验
            exp = Experience(
                state=transition['state'],
                action=transition['action'],
                reward=transition['reward'],
                next_state=transition['next_state'],
                done=transition['done']
            )
            self.buffer.append(exp)
            
            # 生成hindsight经验
            if self.strategy == 'future':
                # 从未来的achieved goals中采样
                future_indices = list(range(t + 1, len(episode)))
                if len(future_indices) > 0:
                    selected = random.sample(
                        future_indices, 
                        min(self.k, len(future_indices))
                    )
                    for idx in selected:
                        hindsight_goal = episode[idx]['achieved_goal']
                        hindsight_exp = self._create_hindsight_experience(
                            transition, hindsight_goal
                        )
                        self.buffer.append(hindsight_exp)
            
            elif self.strategy == 'final':
                # 使用最终achieved goal
                final_goal = episode[-1]['achieved_goal']
                hindsight_exp = self._create_hindsight_experience(
                    transition, final_goal
                )
                self.buffer.append(hindsight_exp)
    
    def _create_hindsight_experience(self, transition: Dict, 
                                     new_goal: Dict) -> Experience:
        """创建hindsight经验"""
        # 根据新目标重新计算奖励
        new_reward = self._compute_reward(
            transition['achieved_goal'], 
            new_goal
        )
        
        return Experience(
            state=transition['state'],
            action=transition['action'],
            reward=new_reward,
            next_state=transition['next_state'],
            done=transition['done']
        )
    
    def _compute_reward(self, achieved_goal: Dict, desired_goal: Dict) -> float:
        """计算达成目标的奖励"""
        # 这里可以根据具体任务定义奖励函数
        makespan_achieved = achieved_goal.get('makespan', float('inf'))
        makespan_desired = desired_goal.get('makespan', float('inf'))
        
        if makespan_achieved <= makespan_desired:
            return 10.0  # 达成目标
        else:
            return -1.0 + (makespan_desired / makespan_achieved)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """采样"""
        return random.sample(list(self.buffer), 
                           min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class CombinedReplayBuffer:
    """组合经验回放缓冲区 - 整合多种回放策略"""
    
    def __init__(self, capacity: int, n_step: int = 3, gamma: float = 0.99,
                 use_per: bool = True, use_n_step: bool = True,
                 alpha: float = 0.6, beta_start: float = 0.4):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.use_per = use_per
        self.use_n_step = use_n_step
        
        # N-step缓冲区
        if use_n_step:
            self.n_step_buffer = NStepReplayBuffer(capacity, n_step, gamma)
        
        # 优先级缓冲区
        if use_per:
            self.per_buffer = EnhancedPrioritizedReplayBuffer(
                capacity, alpha, beta_start
            )
        else:
            self.simple_buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool, info: Optional[Dict] = None):
        """添加经验"""
        exp = Experience(state, action, reward, next_state, done, info)
        
        if self.use_n_step:
            self.n_step_buffer.add(state, action, reward, next_state, done, info)
        
        if self.use_per:
            self.per_buffer.add(exp)
        else:
            self.simple_buffer.append(exp)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """采样"""
        result = {}
        
        # 从优先级缓冲区采样
        if self.use_per and len(self.per_buffer) >= batch_size:
            experiences, indices, weights = self.per_buffer.sample(batch_size)
            result['per'] = {
                'experiences': experiences,
                'indices': indices,
                'weights': weights
            }
        elif not self.use_per and len(self.simple_buffer) >= batch_size:
            import random
            experiences = random.sample(list(self.simple_buffer), batch_size)
            indices = np.zeros(batch_size, dtype=np.int32)
            weights = np.ones(batch_size, dtype=np.float32)
            result['per'] = {
                'experiences': experiences,
                'indices': indices,
                'weights': weights
            }
        
        # 从N-step缓冲区采样
        if self.use_n_step and len(self.n_step_buffer) >= batch_size:
            n_step_data = self.n_step_buffer.sample(batch_size)
            result['n_step'] = n_step_data
        
        return result
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        if self.use_per:
            self.per_buffer.update_priorities(indices, priorities)
    
    def __len__(self) -> int:
        if self.use_per:
            return len(self.per_buffer)
        return len(self.simple_buffer)
