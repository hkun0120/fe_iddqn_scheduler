# -*- coding: utf-8 -*-
"""
增强版双流网络架构 - 集成GNN、Transformer和Cross-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Optional, Dict

from .gnn_module import DAGAwareModule


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加位置编码"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # 自注意力 + 残差连接
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络 + 残差连接
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class CrossAttentionBlock(nn.Module):
    """交叉注意力块 - 用于任务流和资源流之间的早期特征交互"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(CrossAttentionBlock, self).__init__()
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: 查询序列 [batch_size, seq_len_q, d_model]
            key_value: 键值序列 [batch_size, seq_len_kv, d_model]
            
        Returns:
            交叉注意力输出 [batch_size, seq_len_q, d_model]
        """
        # 交叉注意力
        attn_out, attn_weights = self.cross_attn(query, key_value, key_value)
        
        # 门控融合
        gate_input = torch.cat([query, attn_out], dim=-1)
        gate = self.gate(gate_input)
        
        # 残差连接和归一化
        output = self.norm(query + self.dropout(gate * attn_out))
        
        return output


class EnhancedTaskStream(nn.Module):
    """增强版任务流网络 - 集成Transformer和GNN"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_transformer_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1, use_gnn: bool = True):
        super(EnhancedTaskStream, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gnn = use_gnn
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                d_ff=hidden_dim * 4,
                dropout=dropout
            )
            for _ in range(num_transformer_layers)
        ])
        
        # GNN模块（可选）
        if use_gnn:
            self.dag_module = DAGAwareModule(
                node_feature_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_gnn_layers=2,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor, 
                adj_matrix: Optional[torch.Tensor] = None,
                node_depths: Optional[torch.Tensor] = None,
                critical_path_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 任务特征 [batch_size, num_tasks, input_dim]
            adj_matrix: DAG邻接矩阵 [batch_size, num_tasks, num_tasks]
            node_depths: 节点拓扑深度 [batch_size, num_tasks]
            critical_path_mask: 关键路径掩码 [batch_size, num_tasks]
            
        Returns:
            包含任务表示的字典
        """
        batch_size, num_tasks, _ = x.shape
        
        # 输入投影和位置编码
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # Transformer编码
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        # GNN编码（如果启用且有邻接矩阵）
        dag_features = None
        if self.use_gnn and adj_matrix is not None:
            dag_features = self.dag_module(x, adj_matrix, node_depths, critical_path_mask)
            # 融合Transformer和GNN特征
            x = x + dag_features['node_embeddings']
        
        # 输出投影
        task_embeddings = self.layer_norm(self.output_projection(x))
        
        return {
            'task_embeddings': task_embeddings,
            'dag_features': dag_features
        }


class EnhancedResourceStream(nn.Module):
    """增强版资源流网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_transformer_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1):
        super(EnhancedResourceStream, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                d_ff=hidden_dim * 4,
                dropout=dropout
            )
            for _ in range(num_transformer_layers)
        ])
        
        # 资源状态编码器（处理动态负载信息）
        self.load_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 资源特征 [batch_size, num_resources, input_dim]
            
        Returns:
            资源表示 [batch_size, num_resources, output_dim]
        """
        # 输入投影和位置编码
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # 负载编码
        load_features = self.load_encoder(x)
        x = x + load_features
        
        # Transformer编码
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        # 输出投影
        resource_embeddings = self.layer_norm(self.output_projection(x))
        
        return resource_embeddings


class EnhancedFeatureFusion(nn.Module):
    """增强版特征融合模块 - 使用双向交叉注意力"""
    
    def __init__(self, task_dim: int, resource_dim: int, fusion_dim: int,
                 output_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(EnhancedFeatureFusion, self).__init__()
        
        self.task_dim = task_dim
        self.resource_dim = resource_dim
        self.fusion_dim = fusion_dim
        
        # 特征投影
        self.task_projection = nn.Linear(task_dim, fusion_dim)
        self.resource_projection = nn.Linear(resource_dim, fusion_dim)
        
        # 双向交叉注意力
        self.task_cross_attn = CrossAttentionBlock(fusion_dim, num_heads, dropout)
        self.resource_cross_attn = CrossAttentionBlock(fusion_dim, num_heads, dropout)
        
        # 全局上下文编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        
        # Dueling网络分支
        # 价值分支
        self.value_stream = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1)
        )
        
        # 优势分支
        self.advantage_stream = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, output_dim)
        )
        
    def forward(self, task_features: torch.Tensor, 
                resource_features: torch.Tensor,
                dag_representation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            task_features: 任务特征 [batch_size, num_tasks, task_dim]
            resource_features: 资源特征 [batch_size, num_resources, resource_dim]
            dag_representation: DAG表示 [batch_size, dag_dim]（可选）
            
        Returns:
            Q值 [batch_size, output_dim]
        """
        batch_size = task_features.shape[0]
        
        # 特征投影
        task_proj = self.task_projection(task_features)
        resource_proj = self.resource_projection(resource_features)
        
        # 双向交叉注意力
        task_attended = self.task_cross_attn(task_proj, resource_proj)
        resource_attended = self.resource_cross_attn(resource_proj, task_proj)
        
        # 全局池化
        task_mean = torch.mean(task_attended, dim=1)
        task_max = torch.max(task_attended, dim=1)[0]
        resource_mean = torch.mean(resource_attended, dim=1)
        resource_max = torch.max(resource_attended, dim=1)[0]
        
        # 拼接全局特征
        global_features = torch.cat([task_mean, task_max, resource_mean, resource_max], dim=-1)
        
        # 全局编码
        fused = self.global_encoder(global_features)
        
        # 如果有DAG表示，融合进来
        if dag_representation is not None:
            fused = fused + dag_representation
        
        # Dueling DQN: Q = V + (A - mean(A))
        value = self.value_stream(fused)
        advantage = self.advantage_stream(fused)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values


class EnhancedDualStreamNetwork(nn.Module):
    """增强版双流网络 - 集成GNN、Transformer、Cross-Attention和Dueling架构"""
    
    def __init__(self, task_input_dim: int, resource_input_dim: int,
                 hidden_dim: int = 256, fusion_dim: int = 256, output_dim: int = 6,
                 num_transformer_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1, use_gnn: bool = True):
        super(EnhancedDualStreamNetwork, self).__init__()
        
        self.task_input_dim = task_input_dim
        self.resource_input_dim = resource_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gnn = use_gnn
        
        # 增强版任务流
        self.task_stream = EnhancedTaskStream(
            input_dim=task_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_gnn=use_gnn
        )
        
        # 增强版资源流
        self.resource_stream = EnhancedResourceStream(
            input_dim=resource_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 增强版特征融合
        self.feature_fusion = EnhancedFeatureFusion(
            task_dim=hidden_dim,
            resource_dim=hidden_dim,
            fusion_dim=fusion_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(self, task_features: torch.Tensor, resource_features: torch.Tensor,
                adj_matrix: Optional[torch.Tensor] = None,
                node_depths: Optional[torch.Tensor] = None,
                critical_path_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            task_features: 任务特征 [batch_size, num_tasks, task_input_dim]
            resource_features: 资源特征 [batch_size, num_resources, resource_input_dim]
            adj_matrix: DAG邻接矩阵 [batch_size, num_tasks, num_tasks]
            node_depths: 节点拓扑深度 [batch_size, num_tasks]
            critical_path_mask: 关键路径掩码 [batch_size, num_tasks]
            
        Returns:
            Q值 [batch_size, output_dim]
        """
        # 任务流处理
        task_output = self.task_stream(
            task_features, adj_matrix, node_depths, critical_path_mask
        )
        task_embeddings = task_output['task_embeddings']
        dag_features = task_output['dag_features']
        
        # 资源流处理
        resource_embeddings = self.resource_stream(resource_features)
        
        # 获取DAG表示（如果可用）
        dag_representation = None
        if dag_features is not None:
            dag_representation = dag_features['dag_representation']
        
        # 特征融合
        q_values = self.feature_fusion(
            task_embeddings, resource_embeddings, dag_representation
        )
        
        return q_values
    
    def get_attention_weights(self, task_features: torch.Tensor, 
                             resource_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取注意力权重用于可解释性分析"""
        # 这里可以添加获取各层注意力权重的逻辑
        pass
