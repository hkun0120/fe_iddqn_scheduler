# -*- coding: utf-8 -*-
"""
GDS-PPO: Graph-enhanced Dual-Stream PPO for DAG-Aware Workflow Scheduling

核心算法模块 — 基于双流Actor-Critic架构的PPO, 融合:
  1. 双流网络 (任务流+资源流) + 交叉注意力特征融合
  2. Graph Transformer / GAT 编码工作流DAG结构
  3. DAG感知动作掩码 (合法动作约束)
  4. GAE (Generalized Advantage Estimation)
  5. PPO-Clip 策略更新

参考: GA-HPO PPO (Zhou et al., Sensors 2025) 的混合优化框架，
      本模块为其中 PPO Actor-Critic 核心部分。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import copy
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .rollout_buffer import RolloutBuffer


# ─────────────────── 配置 ─────────────────── #

@dataclass
class GDS_PPO_Config:
    """GDS-PPO 超参数配置"""
    # ---- 网络架构 ----
    hidden_dim: int = 256
    fusion_dim: int = 256
    num_transformer_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    use_gnn: bool = True
    
    # ---- PPO 核心参数 ----
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2            # 策略clip范围
    value_loss_coef: float = 0.5     # 价值损失权重 c1
    entropy_coef: float = 0.01       # 熵正则权重 c2
    max_grad_norm: float = 0.5       # 梯度裁剪
    
    # ---- 训练控制 ----
    rollout_steps: int = 2048        # 每次rollout收集的步数
    batch_size: int = 64             # Mini-batch大小
    k_epochs: int = 10               # 每次更新的epoch数
    max_episodes: int = 1000
    
    # ---- 学习率调度 ----
    lr_schedule: str = 'linear'      # 'linear' | 'cosine' | 'constant'
    warmup_steps: int = 0
    
    # ---- 高级选项 ----
    value_clip: bool = True          # 是否clip价值损失
    normalize_advantages: bool = True
    use_orthogonal_init: bool = True # 正交初始化
    
    # ---- 设备 ----
    device: str = 'auto'


# ─────────── Actor-Critic 双流网络 ─────────── #

class ResidualBlock(nn.Module):
    """残差块 — 改善梯度流, 参考 GA-HPO PPO 架构"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(self.net(x)))


class DualStreamActorCritic(nn.Module):
    """
    双流 Actor-Critic 网络
    
    结构:
      任务流: LinearProj → PositionalEncoding → Transformer × L → (可选GNN)
      资源流: LinearProj → PositionalEncoding → Transformer × L
      融合:   双向Cross-Attention → GlobalPool → ResidualBlocks → Actor头 / Critic头
    """
    
    def __init__(self, task_input_dim: int, resource_input_dim: int,
                 action_dim: int, hidden_dim: int = 256,
                 fusion_dim: int = 256,
                 num_transformer_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1,
                 use_gnn: bool = True,
                 use_orthogonal_init: bool = True):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_gnn = use_gnn
        
        # ---- 任务流 ----
        self.task_proj = nn.Linear(task_input_dim, hidden_dim)
        self.task_transformers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True
            )
            for _ in range(num_transformer_layers)
        ])
        
        # 可选 GNN (Graph Transformer)
        if use_gnn:
            from .graph_transformer import GraphTransformerEncoder
            self.graph_encoder = GraphTransformerEncoder(
                dim=hidden_dim, num_layers=2,
                num_heads=num_heads, dropout=dropout
            )
        
        # ---- 资源流 ----
        self.resource_proj = nn.Linear(resource_input_dim, hidden_dim)
        self.resource_transformers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True
            )
            for _ in range(num_transformer_layers)
        ])
        
        # ---- 双向交叉注意力 ----
        self.task_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.resource_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm_t = nn.LayerNorm(hidden_dim)
        self.cross_norm_r = nn.LayerNorm(hidden_dim)
        
        # ---- 融合层 ----
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(fusion_dim, dropout),
            ResidualBlock(fusion_dim, dropout),
        )
        
        # ---- Multi-Head Attention (GA-HPO PPO 风格) ----
        self.mha_layer = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_norm = nn.LayerNorm(fusion_dim)
        
        # ---- Actor 头 (策略) ----
        self.actor_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(fusion_dim // 2, action_dim),
        )
        
        # ---- Critic 头 (价值) ----
        self.critic_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(fusion_dim // 2, 1),
        )
        
        # 初始化
        if use_orthogonal_init:
            self._orthogonal_init()
    
    def _orthogonal_init(self):
        """正交初始化 — PPO最佳实践"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.zeros_(param)
        # Actor和Critic最后一层用较小的gain
        for module in [self.actor_head[-1], self.critic_head[-1]]:
            if hasattr(module, 'weight'):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
    
    def _encode_tasks(self, task_features: torch.Tensor,
                      adj_matrix: Optional[torch.Tensor] = None
                      ) -> torch.Tensor:
        """任务流编码"""
        x = self.task_proj(task_features)
        for layer in self.task_transformers:
            x = layer(x)
        # GNN增强
        if self.use_gnn and adj_matrix is not None:
            x = x + self.graph_encoder(x, adj_matrix)
        return x
    
    def _encode_resources(self, resource_features: torch.Tensor
                          ) -> torch.Tensor:
        """资源流编码"""
        x = self.resource_proj(resource_features)
        for layer in self.resource_transformers:
            x = layer(x)
        return x
    
    def _fuse(self, task_enc: torch.Tensor,
              resource_enc: torch.Tensor) -> torch.Tensor:
        """双向交叉注意力 + 全局池化 + 融合"""
        # 任务关注资源
        t_cross, _ = self.task_cross_attn(task_enc, resource_enc, resource_enc)
        t_cross = self.cross_norm_t(task_enc + t_cross)
        # 资源关注任务
        r_cross, _ = self.resource_cross_attn(resource_enc, task_enc, task_enc)
        r_cross = self.cross_norm_r(resource_enc + r_cross)
        
        # 全局池化 (mean + max)
        t_mean = t_cross.mean(dim=1)
        t_max = t_cross.max(dim=1)[0]
        r_mean = r_cross.mean(dim=1)
        r_max = r_cross.max(dim=1)[0]
        
        global_feat = torch.cat([t_mean, t_max, r_mean, r_max], dim=-1)
        fused = self.fusion(global_feat)  # [B, fusion_dim]
        
        # Multi-Head Attention 进一步精炼 (GA-HPO PPO 风格)
        fused_seq = fused.unsqueeze(1)  # [B, 1, fusion_dim]
        attn_out, _ = self.mha_layer(fused_seq, fused_seq, fused_seq)
        fused = self.mha_norm(fused_seq + attn_out).squeeze(1)
        
        return fused
    
    def forward(self, task_features: torch.Tensor,
                resource_features: torch.Tensor,
                adj_matrix: Optional[torch.Tensor] = None,
                action_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            action_logits: [B, action_dim] — 未归一化的动作得分
            state_value:   [B, 1]          — 状态价值估计
        """
        task_enc = self._encode_tasks(task_features, adj_matrix)
        resource_enc = self._encode_resources(resource_features)
        fused = self._fuse(task_enc, resource_enc)
        
        action_logits = self.actor_head(fused)
        state_value = self.critic_head(fused)
        
        # 动作掩码: 将非法动作logits设为 -inf
        if action_mask is not None:
            action_logits = action_logits.masked_fill(
                action_mask == 0, float('-inf'))
        
        return action_logits, state_value


# ─────────────── PPO 算法 ─────────────── #

class GDS_PPO:
    """
    GDS-PPO: Graph-enhanced Dual-Stream PPO
    
    特点:
      - 双流Actor-Critic (复用已有DualStream架构思想)
      - DAG感知 Graph Transformer 编码
      - PPO-Clip + GAE + 熵正则
      - DAG感知动作掩码
      - 支持GA架构搜索 & Optuna HPO (外部调用)
    """
    
    def __init__(self, task_input_dim: int, resource_input_dim: int,
                 action_dim: int, config: Optional[GDS_PPO_Config] = None,
                 network_structure: Optional[Dict] = None):
        """
        Args:
            task_input_dim:  任务特征维度
            resource_input_dim: 资源特征维度
            action_dim:  动作空间维度 (资源数量)
            config:      超参数配置
            network_structure: GA优化的网络结构 (可选, 用于架构搜索)
        """
        self.config = config or GDS_PPO_Config()
        self.logger = logging.getLogger(__name__)
        
        # 设备
        if self.config.device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
        
        self.task_input_dim = task_input_dim
        self.resource_input_dim = resource_input_dim
        self.action_dim = action_dim
        
        # 如果GA提供了网络结构，覆盖默认值
        if network_structure:
            self.config.hidden_dim = network_structure.get(
                'hidden_dim', self.config.hidden_dim)
            self.config.num_transformer_layers = network_structure.get(
                'num_transformer_layers', self.config.num_transformer_layers)
            self.config.num_heads = network_structure.get(
                'num_heads', self.config.num_heads)
        
        # 构建网络
        self.policy = DualStreamActorCritic(
            task_input_dim=task_input_dim,
            resource_input_dim=resource_input_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
            fusion_dim=self.config.fusion_dim,
            num_transformer_layers=self.config.num_transformer_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            use_gnn=self.config.use_gnn,
            use_orthogonal_init=self.config.use_orthogonal_init,
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate, eps=1e-5)
        
        # 轨迹缓冲区
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.config.rollout_steps,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        
        # 训练统计
        self.step_count = 0
        self.episode_count = 0
        self.update_count = 0
        self.training_stats: List[Dict[str, float]] = []
    
    # ──────── 动作选择 ──────── #
    
    def select_action(self, task_features: np.ndarray,
                      resource_features: np.ndarray,
                      adj_matrix: Optional[np.ndarray] = None,
                      action_mask: Optional[np.ndarray] = None,
                      deterministic: bool = False
                      ) -> Tuple[int, float, float]:
        """
        根据当前策略选择动作
        
        Returns:
            action:   选择的动作
            log_prob: 动作的对数概率
            value:    状态价值估计
        """
        task_t = torch.FloatTensor(task_features).unsqueeze(0).to(self.device)
        res_t = torch.FloatTensor(resource_features).unsqueeze(0).to(self.device)
        
        adj_t = None
        if adj_matrix is not None:
            adj_t = torch.FloatTensor(adj_matrix).unsqueeze(0).to(self.device)
        
        mask_t = None
        if action_mask is not None:
            mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.policy(task_t, res_t, adj_t, mask_t)
            
            # NaN防护
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            logits = torch.clamp(logits, -20.0, 20.0)
            
            # 构建策略分布
            dist = torch.distributions.Categorical(logits=logits)
            
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
        
        return (action.item(), log_prob.item(), value.squeeze().item())
    
    # ──────── 策略评估 ──────── #
    
    def evaluate_actions(self, task_features: torch.Tensor,
                         resource_features: torch.Tensor,
                         actions: torch.Tensor,
                         adj_matrix: Optional[torch.Tensor] = None,
                         action_mask: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估一批动作 — 用于PPO策略更新
        
        Returns:
            log_probs: 动作对数概率
            values:    状态价值
            entropy:   策略熵
        """
        logits, values = self.policy(
            task_features, resource_features, adj_matrix, action_mask)
        
        # NaN防护: 替换NaN并钳位防止极端值
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = torch.clamp(logits, -20.0, 20.0)
        
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy
    
    # ──────── PPO-Clip 策略更新 ──────── #
    
    def update(self) -> Dict[str, float]:
        """
        执行PPO-Clip策略更新
        
        L = L_CLIP(θ) - c1 * L_VF(θ) + c2 * H[π_θ]
        
        Returns:
            训练统计字典
        """
        cfg = self.config
        
        # 统计量累加
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        n_updates = 0
        clip_fractions = []
        approx_kl_divs = []
        
        for epoch in range(cfg.k_epochs):
            for batch in self.rollout_buffer.get_batches(
                    cfg.batch_size, self.device):
                
                # 解包
                states = batch['states']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                old_values = batch['old_values']
                
                task_features = batch.get('task_features')
                resource_features = batch.get('resource_features')
                adj_matrix = batch.get('adj_matrix')
                action_masks = batch.get('action_masks')
                
                # 如果没有结构化状态，从flat state重构
                if task_features is None:
                    task_features, resource_features = \
                        self._reconstruct_features(states)
                
                # 归一化优势
                if cfg.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8)
                
                # 评估动作
                new_log_probs, new_values, entropy = self.evaluate_actions(
                    task_features, resource_features, actions,
                    adj_matrix, action_masks)
                
                # ---- 策略损失 (PPO-Clip) ----
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # ---- 价值损失 ----
                if cfg.value_clip:
                    # Clipped value loss (PPO变体)
                    value_pred_clipped = old_values + torch.clamp(
                        new_values - old_values,
                        -cfg.eps_clip, cfg.eps_clip)
                    vf_loss1 = F.mse_loss(new_values, returns)
                    vf_loss2 = F.mse_loss(value_pred_clipped, returns)
                    value_loss = torch.max(vf_loss1, vf_loss2)
                else:
                    value_loss = F.mse_loss(new_values, returns)
                
                # ---- 熵正则 ----
                entropy_loss = -entropy.mean()
                
                # ---- 总损失 ----
                # L = L_policy + c1 * L_value + c2 * (-H)
                loss = (policy_loss
                        + cfg.value_loss_coef * value_loss
                        + cfg.entropy_coef * entropy_loss)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), cfg.max_grad_norm)
                
                self.optimizer.step()
                
                # 统计
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss.item())
                total_loss += loss.item()
                n_updates += 1
                
                # clip分数 (多少ratio被clip了)
                clip_fraction = (
                    (ratio - 1.0).abs() > cfg.eps_clip
                ).float().mean().item()
                clip_fractions.append(clip_fraction)
                
                # 近似KL散度 (用于早停判断)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                    approx_kl_divs.append(approx_kl)
        
        self.update_count += 1
        
        stats = {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'total_loss': total_loss / max(n_updates, 1),
            'clip_fraction': np.mean(clip_fractions) if clip_fractions else 0,
            'approx_kl': np.mean(approx_kl_divs) if approx_kl_divs else 0,
            'n_updates': n_updates,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
        self.training_stats.append(stats)
        
        # 清空缓冲区 (on-policy)
        self.rollout_buffer.reset()
        
        return stats
    
    # ──────── 辅助方法 ──────── #
    
    def _reconstruct_features(self, states: torch.Tensor,
                               num_tasks: int = 8,
                               num_resources: int = 6
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从展平状态重构任务/资源特征"""
        B = states.shape[0]
        total = states.shape[1]
        
        task_size = num_tasks * self.task_input_dim
        res_size = num_resources * self.resource_input_dim
        
        task_end = min(task_size, total)
        res_end = min(task_end + res_size, total)
        
        task_flat = states[:, :task_end]
        if task_flat.shape[1] < task_size:
            pad = torch.zeros(B, task_size - task_flat.shape[1],
                              device=states.device)
            task_flat = torch.cat([task_flat, pad], dim=1)
        
        res_flat = states[:, task_end:res_end]
        if res_flat.shape[1] < res_size:
            pad = torch.zeros(B, res_size - res_flat.shape[1],
                              device=states.device)
            res_flat = torch.cat([res_flat, pad], dim=1)
        
        task_features = task_flat.view(B, num_tasks, self.task_input_dim)
        resource_features = res_flat.view(B, num_resources,
                                          self.resource_input_dim)
        return task_features, resource_features
    
    def store_transition(self, state: np.ndarray, action: int,
                         reward: float, value: float, log_prob: float,
                         done: bool, **kwargs):
        """存储一步转移到缓冲区"""
        self.rollout_buffer.add(
            state=state, action=action, reward=reward,
            value=value, log_prob=log_prob, done=done,
            task_features=kwargs.get('task_features'),
            resource_features=kwargs.get('resource_features'),
            adj_matrix=kwargs.get('adj_matrix'),
            action_mask=kwargs.get('action_mask'),
        )
        self.step_count += 1
    
    def compute_gae(self, last_value: float, last_done: bool):
        """计算GAE优势估计"""
        self.rollout_buffer.compute_gae(last_value, last_done)
    
    def save(self, filepath: str):
        """保存模型"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'training_stats': self.training_stats,
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"GDS-PPO model saved to {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.update_count = checkpoint.get('update_count', 0)
        self.training_stats = checkpoint.get('training_stats', [])
        self.logger.info(f"GDS-PPO model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        recent = self.training_stats[-10:] if self.training_stats else []
        return {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'avg_policy_loss': np.mean([s['policy_loss'] for s in recent]) if recent else 0,
            'avg_value_loss': np.mean([s['value_loss'] for s in recent]) if recent else 0,
            'avg_entropy': np.mean([s['entropy'] for s in recent]) if recent else 0,
            'avg_clip_fraction': np.mean([s['clip_fraction'] for s in recent]) if recent else 0,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
    
    def get_policy_params(self) -> Dict[str, torch.Tensor]:
        """获取策略参数 (用于GA交叉/变异)"""
        return {k: v.clone() for k, v in self.policy.state_dict().items()}
    
    def set_policy_params(self, params: Dict[str, torch.Tensor]):
        """设置策略参数 (用于GA进化)"""
        self.policy.load_state_dict(params)
    
    def get_network_structure(self) -> Dict[str, int]:
        """获取网络结构描述 (用于GA搜索)"""
        return {
            'hidden_dim': self.config.hidden_dim,
            'fusion_dim': self.config.fusion_dim,
            'num_transformer_layers': self.config.num_transformer_layers,
            'num_heads': self.config.num_heads,
        }
