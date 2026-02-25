import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import logging
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional
from baselines.traditional_schedulers import BaseScheduler
from config.hyperparameters import Hyperparameters

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

class QNetwork(nn.Module):
    """Q网络"""
    
    def __init__(self, state_size: int, action_size: int, hidden_dims: List[int]):
        super(QNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        input_dim = state_size
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, h_dim))
            input_dim = h_dim
        self.layers.append(nn.Linear(input_dim, action_size))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x

class DualStreamQNetwork(nn.Module):
    """双流Q网络（用于DQN和DDQN，与FE-IDDQN兼容）"""

    def __init__(self, task_input_dim: int, resource_input_dim: int, action_size: int,
                 task_hidden_dims: List[int], resource_hidden_dims: List[int], fusion_dim: int):
        super(DualStreamQNetwork, self).__init__()

        # 任务流
        self.task_layers = nn.ModuleList()
        task_input = task_input_dim
        for h_dim in task_hidden_dims:
            self.task_layers.append(nn.Linear(task_input, h_dim))
            task_input = h_dim

        # 资源流
        self.resource_layers = nn.ModuleList()
        resource_input = resource_input_dim
        for h_dim in resource_hidden_dims:
            self.resource_layers.append(nn.Linear(resource_input, h_dim))
            resource_input = h_dim

        # 融合层
        self.fusion = nn.Linear(task_input + resource_input, fusion_dim)
        self.output = nn.Linear(fusion_dim, action_size)

    def forward(self, task_features, resource_features):
        # 任务流前向传播
        x_task = task_features
        for layer in self.task_layers:
            x_task = F.relu(layer(x_task))

        # 资源流前向传播
        x_resource = resource_features
        for layer in self.resource_layers:
            x_resource = F.relu(layer(x_resource))

        # 融合
        x_combined = torch.cat([x_task, x_resource], dim=-1)
        x_fusion = F.relu(self.fusion(x_combined))

        return self.output(x_fusion)

class DQNScheduler(BaseScheduler):
    """DQN调度器（双流网络版本）"""

    def __init__(self, task_input_dim: int, resource_input_dim: int, action_size: int, device: torch.device = None):
        super().__init__("DQN")
        self.task_input_dim = task_input_dim
        self.resource_input_dim = resource_input_dim
        self.action_size = action_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用双流网络参数（基于FE-IDDQN的参数但简化）
        self.params = {
            'task_stream_hidden_dims': [64, 32],
            'resource_stream_hidden_dims': [64, 32],
            'fusion_dim': 64,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'gamma': 0.99,
            'replay_buffer_size': 10000,
            'target_update_freq': 100,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995
        }

        self.qnetwork_local = DualStreamQNetwork(
            task_input_dim=task_input_dim,
            resource_input_dim=resource_input_dim,
            action_size=action_size,
            task_hidden_dims=self.params["task_stream_hidden_dims"],
            resource_hidden_dims=self.params["resource_stream_hidden_dims"],
            fusion_dim=self.params["fusion_dim"]
        ).to(self.device)

        self.qnetwork_target = DualStreamQNetwork(
            task_input_dim=task_input_dim,
            resource_input_dim=resource_input_dim,
            action_size=action_size,
            task_hidden_dims=self.params["task_stream_hidden_dims"],
            resource_hidden_dims=self.params["resource_stream_hidden_dims"],
            fusion_dim=self.params["fusion_dim"]
        ).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.params["learning_rate"])
        self.memory = ReplayBuffer(self.params["replay_buffer_size"])

        self.epsilon = self.params["epsilon_start"]
        self.t_step = 0

    def step(self, task_features, resource_features, action: int, reward: float,
             next_task_features, next_resource_features, done: bool):
        """存储经验（双流版本）"""
        # 将双流特征展平为单一流用于存储（简化处理）
        state = np.concatenate([task_features.flatten(), resource_features.flatten()])
        next_state = np.concatenate([next_task_features.flatten(), next_resource_features.flatten()])

        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.params["target_update_freq"]
        if self.t_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, task_features, resource_features):
        """选择动作（双流特征版本）"""
        task_features = torch.from_numpy(task_features).float().unsqueeze(0).to(self.device)
        resource_features = torch.from_numpy(resource_features).float().unsqueeze(0).to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(task_features, resource_features)
        self.qnetwork_local.train()

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self):
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

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_task_features, next_resource_features).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.params["gamma"] * Q_targets_next * (~dones))

        # Get expected Q values from local model
        # 确保actions不超过动作空间大小
        valid_actions = torch.clamp(actions, 0, self.action_size - 1)
        Q_expected = self.qnetwork_local(task_features, resource_features).gather(1, valid_actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.params["epsilon_end"], self.epsilon * self.params["epsilon_decay"])
        
    def schedule(self, tasks: List[Dict], resources: List[Dict], 
                dependencies: List[Tuple[int, int]]) -> Dict:
        # DQN的调度需要一个环境进行交互，这里只是一个占位符
        # 实际调度逻辑将在仿真环境中实现
        self.logger.warning("DQN scheduling requires an environment. This is a placeholder.")
        return {"algorithm": self.name, "makespan": 0, "resource_utilization": 0}

class DDQNScheduler(BaseScheduler):
    """DDQN调度器（双流网络版本）"""

    def __init__(self, task_input_dim: int, resource_input_dim: int, action_size: int, device: torch.device = None):
        super().__init__("DDQN")
        self.task_input_dim = task_input_dim
        self.resource_input_dim = resource_input_dim
        self.action_size = action_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用双流网络参数（基于FE-IDDQN的参数但简化）
        self.params = {
            'task_stream_hidden_dims': [64, 32],
            'resource_stream_hidden_dims': [64, 32],
            'fusion_dim': 64,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'gamma': 0.99,
            'replay_buffer_size': 10000,
            'target_update_freq': 100,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995
        }

        self.qnetwork_local = DualStreamQNetwork(
            task_input_dim=task_input_dim,
            resource_input_dim=resource_input_dim,
            action_size=action_size,
            task_hidden_dims=self.params["task_stream_hidden_dims"],
            resource_hidden_dims=self.params["resource_stream_hidden_dims"],
            fusion_dim=self.params["fusion_dim"]
        ).to(self.device)

        self.qnetwork_target = DualStreamQNetwork(
            task_input_dim=task_input_dim,
            resource_input_dim=resource_input_dim,
            action_size=action_size,
            task_hidden_dims=self.params["task_stream_hidden_dims"],
            resource_hidden_dims=self.params["resource_stream_hidden_dims"],
            fusion_dim=self.params["fusion_dim"]
        ).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.params["learning_rate"])
        self.memory = ReplayBuffer(self.params["replay_buffer_size"])

        self.epsilon = self.params["epsilon_start"]
        self.t_step = 0
        
    def step(self, task_features, resource_features, action: int, reward: float,
             next_task_features, next_resource_features, done: bool):
        """存储经验（双流版本）"""
        # 将双流特征展平为单一流用于存储（简化处理）
        state = np.concatenate([task_features.flatten(), resource_features.flatten()])
        next_state = np.concatenate([next_task_features.flatten(), next_resource_features.flatten()])

        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.params["target_update_freq"]
        if self.t_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, task_features, resource_features):
        """选择动作（双流特征版本）"""
        task_features = torch.from_numpy(task_features).float().unsqueeze(0).to(self.device)
        resource_features = torch.from_numpy(resource_features).float().unsqueeze(0).to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(task_features, resource_features)
        self.qnetwork_local.train()

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
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
        Q_best_action = self.qnetwork_local(next_task_features, next_resource_features).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_task_features, next_resource_features).detach().gather(1, Q_best_action)

        # Compute Q targets for current states
        Q_targets = rewards + (self.params["gamma"] * Q_targets_next * (~dones))

        # Get expected Q values from local model
        # 确保actions不超过动作空间大小
        valid_actions = torch.clamp(actions, 0, self.action_size - 1)
        Q_expected = self.qnetwork_local(task_features, resource_features).gather(1, valid_actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.params["epsilon_end"], self.epsilon * self.params["epsilon_decay"])

    def schedule(self, tasks: List[Dict], resources: List[Dict],
                dependencies: List[Tuple[int, int]]) -> Dict:
        self.logger.warning("DDQN scheduling requires an environment. This is a placeholder.")
        return {"algorithm": self.name, "makespan": 0, "resource_utilization": 0}

class BF_DDQNScheduler(BaseScheduler):
    """BF-DDQN调度器 (Batch-First DDQN)"""
    
    def __init__(self, state_size: int, action_size: int, device: torch.device = None):
        super().__init__("BF-DDQN")
        self.state_size = state_size
        self.action_size = action_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.params = Hyperparameters.get_algorithm_params("BF_DDQN")
        
        self.qnetwork_local = QNetwork(state_size, action_size, self.params["hidden_dims"]).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.params["hidden_dims"]).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.params["learning_rate"])
        self.memory = ReplayBuffer(self.params["replay_buffer_size"])
        
        self.epsilon = self.params["epsilon_start"]
        self.t_step = 0

    def step(self, task_features, resource_features, action: int, reward: float,
             next_task_features, next_resource_features, done: bool):
        """存储经验（双流版本）"""
        # 将双流特征展平为单一流用于存储（简化处理）
        state = np.concatenate([task_features.flatten(), resource_features.flatten()])
        next_state = np.concatenate([next_task_features.flatten(), next_resource_features.flatten()])

        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.params["target_update_freq"]
        if self.t_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, task_features, resource_features):
        """选择动作（双流特征版本）"""
        task_features = torch.from_numpy(task_features).float().unsqueeze(0).to(self.device)
        resource_features = torch.from_numpy(resource_features).float().unsqueeze(0).to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(task_features, resource_features)
        self.qnetwork_local.train()

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
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
        Q_best_action = self.qnetwork_local(next_task_features, next_resource_features).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_task_features, next_resource_features).detach().gather(1, Q_best_action)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.params["gamma"] * Q_targets_next * (~dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.params["epsilon_end"], self.epsilon * self.params["epsilon_decay"])
        
    def schedule(self, tasks: List[Dict], resources: List[Dict], 
                dependencies: List[Tuple[int, int]]) -> Dict:
        self.logger.warning("BF-DDQN scheduling requires an environment. This is a placeholder.")
        return {"algorithm": self.name, "makespan": 0, "resource_utilization": 0}


