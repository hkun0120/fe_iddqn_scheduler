#!/usr/bin/env python3
"""
依赖关系感知的DDQN调度器
确保任务分配不违反依赖关系约束
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import logging
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional, Set
from baselines.traditional_schedulers import BaseScheduler
from config.hyperparameters import Hyperparameters
import networkx as nx

# 定义经验元组
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """标准经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        e = Experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        experiences = random.sample(self.buffer, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e.state is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e.action is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e.reward is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e.next_state is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e.done is not None]).astype(np.uint8)).bool()

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

class DualStreamQNetwork(nn.Module):
    """双流Q网络：任务特征流 + 资源特征流 + 依赖关系融合"""

    def __init__(self, task_input_dim: int, resource_input_dim: int, action_size: int,
                 task_hidden_dims: List[int], resource_hidden_dims: List[int],
                 fusion_dim: int, dependency_dim: int = 16):
        super(DualStreamQNetwork, self).__init__()

        # 任务特征流
        self.task_layers = nn.ModuleList()
        input_dim = task_input_dim
        for h_dim in task_hidden_dims:
            self.task_layers.append(nn.Linear(input_dim, h_dim))
            input_dim = h_dim

        # 资源特征流
        self.resource_layers = nn.ModuleList()
        input_dim = resource_input_dim
        for h_dim in resource_hidden_dims:
            self.resource_layers.append(nn.Linear(input_dim, h_dim))
            input_dim = h_dim

        # 依赖关系编码器
        self.dependency_encoder = nn.Linear(dependency_dim, fusion_dim // 2)

        # 融合层
        self.fusion_input_dim = task_hidden_dims[-1] + resource_hidden_dims[-1] + fusion_dim // 2
        self.fusion_layers = nn.ModuleList()
        input_dim = self.fusion_input_dim
        for h_dim in [fusion_dim, fusion_dim // 2]:
            self.fusion_layers.append(nn.Linear(input_dim, h_dim))
            input_dim = h_dim

        # 输出层
        self.output_layer = nn.Linear(fusion_dim // 2, action_size)

    def forward(self, task_features, resource_features, dependency_features=None):
        # 处理任务特征 - 输入可能是 [batch_size, num_tasks, task_feature_dim]
        if len(task_features.shape) == 3:
            # 展平为 [batch_size * num_tasks, task_feature_dim]
            batch_size, num_tasks, task_feature_dim = task_features.shape
            x_task = task_features.view(-1, task_feature_dim)
        else:
            x_task = task_features

        # 任务特征流前向传播
        for layer in self.task_layers:
            x_task = F.relu(layer(x_task))

        # 如果输入是3D的，需要聚合回批次维度
        if len(task_features.shape) == 3:
            # 从 [batch_size * num_tasks, hidden_dim] 聚合为 [batch_size, hidden_dim]
            x_task = x_task.view(batch_size, num_tasks, -1).mean(dim=1)

        # 处理资源特征 - 输入可能是 [batch_size, num_resources, resource_feature_dim]
        if len(resource_features.shape) == 3:
            # 展平为 [batch_size * num_resources, resource_feature_dim]
            batch_size, num_resources, resource_feature_dim = resource_features.shape
            x_resource = resource_features.view(-1, resource_feature_dim)
        else:
            x_resource = resource_features

        # 资源特征流前向传播
        for layer in self.resource_layers:
            x_resource = F.relu(layer(x_resource))

        # 如果输入是3D的，需要聚合回批次维度
        if len(resource_features.shape) == 3:
            # 从 [batch_size * num_resources, hidden_dim] 聚合为 [batch_size, hidden_dim]
            x_resource = x_resource.view(batch_size, num_resources, -1).mean(dim=1)

        # 依赖关系特征（如果提供）
        if dependency_features is not None:
            x_dependency = F.relu(self.dependency_encoder(dependency_features))
        else:
            # 创建零向量作为默认值
            batch_size = x_task.shape[0]
            x_dependency = torch.zeros(batch_size, self.dependency_encoder.out_features).to(x_task.device)

        # 特征融合
        x_fusion = torch.cat([x_task, x_resource, x_dependency], dim=1)

        # 融合层前向传播
        for layer in self.fusion_layers:
            x_fusion = F.relu(layer(x_fusion))

        # 输出层
        action_values = self.output_layer(x_fusion)
        return action_values

class DependencyAwareDDQNScheduler(BaseScheduler):
    """依赖关系感知的DDQN调度器"""

    def __init__(self, task_input_dim: int, resource_input_dim: int, action_size: int, device: torch.device = None):
        super().__init__("Dependency-Aware DDQN")
        self.task_input_dim = task_input_dim
        self.resource_input_dim = resource_input_dim
        self.action_size = action_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 依赖关系图（需要在调用时设置）
        self.dependency_graph = None
        self.task_id_to_index = {}
        self.task_dependencies = {}

        # DDQN参数
        self.params = {
            'task_stream_hidden_dims': [64, 32],
            'resource_stream_hidden_dims': [64, 32],
            'fusion_dim': 64,
            'dependency_dim': 16,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'gamma': 0.99,
            'replay_buffer_size': 10000,
            'target_update_freq': 100,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'dependency_violation_penalty': -10.0,  # 依赖关系违反惩罚
        }

        # 创建Q网络
        self.qnetwork_local = DualStreamQNetwork(
            task_input_dim=task_input_dim,
            resource_input_dim=resource_input_dim,
            action_size=action_size,
            task_hidden_dims=self.params["task_stream_hidden_dims"],
            resource_hidden_dims=self.params["resource_stream_hidden_dims"],
            fusion_dim=self.params["fusion_dim"],
            dependency_dim=self.params["dependency_dim"]
        ).to(self.device)

        self.qnetwork_target = DualStreamQNetwork(
            task_input_dim=task_input_dim,
            resource_input_dim=resource_input_dim,
            action_size=action_size,
            task_hidden_dims=self.params["task_stream_hidden_dims"],
            resource_hidden_dims=self.params["resource_stream_hidden_dims"],
            fusion_dim=self.params["fusion_dim"],
            dependency_dim=self.params["dependency_dim"]
        ).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.params["learning_rate"])
        self.memory = ReplayBuffer(self.params["replay_buffer_size"])

        self.epsilon = self.params["epsilon_start"]
        self.t_step = 0

    def set_dependency_graph(self, dependency_graph: nx.DiGraph, task_id_to_index: Dict, task_dependencies: Dict):
        """设置依赖关系图"""
        self.dependency_graph = dependency_graph
        self.task_id_to_index = task_id_to_index
        self.task_dependencies = task_dependencies
        self.logger.info(f"Dependency graph set with {len(dependency_graph.nodes)} nodes and {len(dependency_graph.edges)} edges")

    def get_dependency_features(self, current_task_id: str, host_assignments: Dict[str, int]) -> torch.Tensor:
        """获取当前任务的依赖关系特征"""

        if not self.dependency_graph or current_task_id not in self.task_dependencies:
            # 返回零向量
            return torch.zeros(self.params['dependency_dim']).to(self.device)

        features = []

        # 1. 前置任务完成情况
        predecessors = list(self.dependency_graph.predecessors(current_task_id))
        completed_predecessors = sum(1 for pred in predecessors if pred in host_assignments)
        features.append(completed_predecessors / max(1, len(predecessors)))  # 完成比例

        # 2. 后置任务依赖情况
        successors = list(self.dependency_graph.successors(current_task_id))
        features.append(len(successors))  # 后置任务数量

        # 3. 关键路径位置
        try:
            centrality = nx.betweenness_centrality(self.dependency_graph)[current_task_id]
            features.append(centrality)
        except:
            features.append(0.0)

        # 4. 依赖链长度
        try:
            longest_path = nx.dag_longest_path_length(self.dependency_graph)
            features.append(longest_path)
        except:
            features.append(0.0)

        # 5. 并行度
        try:
            parallelism = len(list(nx.antichains(self.dependency_graph)))
            features.append(parallelism)
        except:
            features.append(1.0)

        # 填充到指定维度
        while len(features) < self.params['dependency_dim']:
            features.append(0.0)

        return torch.tensor(features[:self.params['dependency_dim']], dtype=torch.float32).to(self.device)

    def check_dependency_violation(self, task_id: str, selected_host: int, host_assignments: Dict[str, int]) -> bool:
        """检查任务分配是否违反依赖关系"""

        if not self.dependency_graph or task_id not in self.task_dependencies:
            return False

        # 检查前置任务是否在不同主机上
        predecessors = list(self.dependency_graph.predecessors(task_id))
        for pred in predecessors:
            if pred in host_assignments and host_assignments[pred] != selected_host:
                # 前置任务在不同主机，检查是否已经完成（简化检查）
                # 在实际系统中，需要更复杂的检查
                continue

        # 检查后置任务是否已经在其他主机上分配
        successors = list(self.dependency_graph.successors(task_id))
        for succ in successors:
            if succ in host_assignments and host_assignments[succ] != selected_host:
                # 后置任务已经在不同主机上，这可能违反依赖关系
                return True

        return False

    def get_valid_actions(self, current_task_id: str, host_assignments: Dict[str, int]) -> List[int]:
        """获取当前任务的有效动作（不违反依赖关系的分配）"""

        valid_actions = []

        for action in range(self.action_size):
            if not self.check_dependency_violation(current_task_id, action, host_assignments):
                valid_actions.append(action)

        # 如果没有有效动作，返回所有动作（避免死锁）
        return valid_actions if valid_actions else list(range(self.action_size))

    def schedule(self, tasks: List[Dict], resources: List[Dict],
                dependencies: List[Tuple[int, int]]) -> Dict:
        """
        依赖关系感知的调度方法

        注意：这个方法主要用于兼容基类接口，
        实际的强化学习调度在仿真环境中进行
        """
        self.logger.warning("schedule() 方法主要用于兼容性，实际调度请使用仿真环境")

        # 返回基本的调度结果格式
        return {
            'task_assignments': {},
            'makespan': 0,
            'resource_utilization': {},
            'algorithm': 'Dependency-Aware DDQN',
            'note': '实际调度在仿真环境中进行'
        }

    def act(self, task_features, resource_features, current_task_id: str = None, host_assignments: Dict[str, int] = None):
        """选择动作（依赖关系感知版本）"""

        task_features = torch.from_numpy(task_features).float().to(self.device)
        resource_features = torch.from_numpy(resource_features).float().to(self.device)

        # 获取依赖关系特征
        dependency_features = None
        if current_task_id:
            dependency_features = self.get_dependency_features(current_task_id, host_assignments or {})
            dependency_features = dependency_features.unsqueeze(0)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(task_features, resource_features, dependency_features)
        self.qnetwork_local.train()

        # 获取有效动作
        valid_actions = self.get_valid_actions(current_task_id, host_assignments or {}) if current_task_id else list(range(self.action_size))

        if random.random() > self.epsilon:
            # 选择有效动作中的最优动作
            valid_action_values = action_values[0][valid_actions]
            best_valid_idx = torch.argmax(valid_action_values)
            return valid_actions[best_valid_idx]
        else:
            # 随机选择有效动作
            return random.choice(valid_actions)

    def step(self, task_features, resource_features, action: int, reward: float,
             next_task_features, next_resource_features, done: bool,
             current_task_id: str = None, next_task_id: str = None,
             host_assignments: Dict[str, int] = None):
        """执行一步学习"""

        # 存储经验
        state = np.concatenate([task_features, resource_features])
        next_state = np.concatenate([next_task_features, next_resource_features]) if next_task_features is not None else None

        self.memory.add(state, action, reward, next_state, done)

        # 学习
        self.learn()

        # 更新epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

    def learn(self):
        """依赖关系感知的学习过程"""

        if len(self.memory) < self.params["batch_size"]:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.params["batch_size"])

        # 将所有张量移动到正确的设备
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 重新拆分状态为任务和资源特征
        batch_size = states.shape[0]
        task_features = states[:, :self.task_input_dim]
        resource_features = states[:, self.task_input_dim:self.task_input_dim + self.resource_input_dim]

        next_task_features = next_states[:, :self.task_input_dim]
        next_resource_features = next_states[:, self.task_input_dim:self.task_input_dim + self.resource_input_dim]

        # Get max predicted Q values (for next states) from local model
        # 注意：这里简化了依赖关系特征，实际应用中应该传递正确的依赖特征
        next_dependency_features = torch.zeros(batch_size, self.params['dependency_dim']).to(self.device)
        Q_best_action = self.qnetwork_local(next_task_features, next_resource_features, next_dependency_features).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_task_features, next_resource_features, next_dependency_features).detach().gather(1, Q_best_action)

        # Compute Q targets for current states
        Q_targets = rewards + (self.params["gamma"] * Q_targets_next * (~dones))

        # Get expected Q values from local model
        current_dependency_features = torch.zeros(batch_size, self.params['dependency_dim']).to(self.device)
        Q_expected = self.qnetwork_local(task_features, resource_features, current_dependency_features).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.t_step = (self.t_step + 1) % self.params["target_update_freq"]
        if self.t_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def update_target_network(self):
        """更新目标网络"""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
