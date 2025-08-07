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

class DQNScheduler(BaseScheduler):
    """DQN调度器"""
    
    def __init__(self, state_size: int, action_size: int, device: torch.device = None):
        super().__init__("DQN")
        self.state_size = state_size
        self.action_size = action_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.params = Hyperparameters.get_algorithm_params("DQN")
        
        self.qnetwork_local = QNetwork(state_size, action_size, self.params["hidden_dims"]).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.params["hidden_dims"]).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.params["learning_rate"])
        self.memory = ReplayBuffer(self.params["replay_buffer_size"])
        
        self.epsilon = self.params["epsilon_start"]
        self.t_step = 0
        
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.params["target_update_freq"]
        if self.t_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            
    def act(self, state: np.ndarray) -> int:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self):
        if len(self.memory) < self.params["batch_size"]:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.params["batch_size"])
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
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
        # DQN的调度需要一个环境进行交互，这里只是一个占位符
        # 实际调度逻辑将在仿真环境中实现
        self.logger.warning("DQN scheduling requires an environment. This is a placeholder.")
        return {"algorithm": self.name, "makespan": 0, "resource_utilization": 0}

class DDQNScheduler(BaseScheduler):
    """DDQN调度器"""
    
    def __init__(self, state_size: int, action_size: int, device: torch.device = None):
        super().__init__("DDQN")
        self.state_size = state_size
        self.action_size = action_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.params = Hyperparameters.get_algorithm_params("DDQN")
        
        self.qnetwork_local = QNetwork(state_size, action_size, self.params["hidden_dims"]).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.params["hidden_dims"]).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.params["learning_rate"])
        self.memory = ReplayBuffer(self.params["replay_buffer_size"])
        
        self.epsilon = self.params["epsilon_start"]
        self.t_step = 0
        
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.params["target_update_freq"]
        if self.t_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            
    def act(self, state: np.ndarray) -> int:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self):
        if len(self.memory) < self.params["batch_size"]:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.params["batch_size"])
        
        # Get max predicted Q values (for next states) from local model
        Q_best_action = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_best_action)
        
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
        
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.params["target_update_freq"]
        if self.t_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            
    def act(self, state: np.ndarray) -> int:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self):
        if len(self.memory) < self.params["batch_size"]:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.params["batch_size"])
        
        # Get max predicted Q values (for next states) from local model
        Q_best_action = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_best_action)
        
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


