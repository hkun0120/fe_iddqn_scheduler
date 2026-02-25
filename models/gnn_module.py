# -*- coding: utf-8 -*-
"""
图神经网络模块 - 用于处理DAG结构的工作流调度
实现Graph Attention Network (GAT) 和 Graph Convolutional Network (GCN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict


class GraphAttentionLayer(nn.Module):
    """图注意力层 (GAT Layer)"""
    
    def __init__(self, in_features: int, out_features: int, 
                 dropout: float = 0.1, alpha: float = 0.2, concat: bool = True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # 线性变换
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # 注意力参数
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU激活
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h: 节点特征 [batch_size, num_nodes, in_features]
            adj: 邻接矩阵 [batch_size, num_nodes, num_nodes]
            
        Returns:
            更新后的节点特征 [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = h.shape
        
        # 线性变换
        Wh = self.W(h)  # [batch_size, num_nodes, out_features]
        
        # 计算注意力系数
        # 扩展Wh以计算所有节点对之间的注意力
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)  # [batch_size, num_nodes, num_nodes, out_features]
        Wh2 = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)  # [batch_size, num_nodes, num_nodes, out_features]
        
        # 拼接并计算注意力分数
        a_input = torch.cat([Wh1, Wh2], dim=-1)  # [batch_size, num_nodes, num_nodes, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [batch_size, num_nodes, num_nodes]
        
        # 使用邻接矩阵进行掩码
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 聚合邻居特征
        h_prime = torch.bmm(attention, Wh)  # [batch_size, num_nodes, out_features]
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class MultiHeadGAT(nn.Module):
    """多头图注意力网络"""
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 num_heads: int = 4, dropout: float = 0.1, alpha: float = 0.2):
        super(MultiHeadGAT, self).__init__()
        
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 多头注意力层
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, hidden_features, dropout, alpha, concat=True)
            for _ in range(num_heads)
        ])
        
        # 输出层
        self.out_layer = GraphAttentionLayer(
            hidden_features * num_heads, out_features, dropout, alpha, concat=False
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_features * num_heads)
        self.layer_norm2 = nn.LayerNorm(out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [batch_size, num_nodes, in_features]
            adj: 邻接矩阵 [batch_size, num_nodes, num_nodes]
            
        Returns:
            更新后的节点特征 [batch_size, num_nodes, out_features]
        """
        # 多头注意力
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([attn(x, adj) for attn in self.attention_heads], dim=-1)
        x = self.layer_norm1(x)
        
        # 输出层
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_layer(x, adj)
        x = self.layer_norm2(x)
        
        return x


class DAGEncoder(nn.Module):
    """DAG编码器 - 使用GNN编码工作流DAG结构"""
    
    def __init__(self, node_feature_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super(DAGEncoder, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 输入投影
        self.input_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        # 位置编码（基于拓扑深度）
        self.position_embedding = nn.Embedding(100, hidden_dim)  # 最多100层深度
        
        # GNN层
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.gnn_layers.append(
                MultiHeadGAT(in_dim, hidden_dim // num_heads, out_dim, 
                           num_heads=num_heads, dropout=dropout)
            )
        
        # 图级别读出
        self.graph_readout = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor,
                node_depths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_features: 节点特征 [batch_size, num_nodes, node_feature_dim]
            adj_matrix: 邻接矩阵 [batch_size, num_nodes, num_nodes]
            node_depths: 节点拓扑深度 [batch_size, num_nodes]
            
        Returns:
            node_embeddings: 节点嵌入 [batch_size, num_nodes, output_dim]
            graph_embedding: 图嵌入 [batch_size, output_dim]
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # 输入投影
        x = self.input_projection(node_features)
        
        # 添加位置编码
        if node_depths is not None:
            node_depths = node_depths.clamp(0, 99).long()
            pos_encoding = self.position_embedding(node_depths)
            x = x + pos_encoding
        
        # GNN传播
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, adj_matrix)
        
        # 节点嵌入
        node_embeddings = x
        
        # 图级别池化（均值 + 最大值）
        mean_pool = torch.mean(node_embeddings, dim=1)
        max_pool = torch.max(node_embeddings, dim=1)[0]
        graph_embedding = self.graph_readout(mean_pool + max_pool)
        
        return node_embeddings, graph_embedding


class CriticalPathEncoder(nn.Module):
    """关键路径编码器 - 专门编码DAG中的关键路径信息"""
    
    def __init__(self, node_dim: int, hidden_dim: int, output_dim: int):
        super(CriticalPathEncoder, self).__init__()
        
        # 关键路径LSTM
        self.path_lstm = nn.LSTM(
            input_size=node_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim * 2, output_dim)
        
        # 关键性评分网络
        self.criticality_scorer = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features: torch.Tensor, 
                critical_path_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_features: 节点特征 [batch_size, num_nodes, node_dim]
            critical_path_mask: 关键路径掩码 [batch_size, num_nodes]
            
        Returns:
            path_encoding: 关键路径编码 [batch_size, output_dim]
            criticality_scores: 每个节点的关键性评分 [batch_size, num_nodes]
        """
        batch_size, num_nodes, node_dim = node_features.shape
        
        # 计算关键性评分
        criticality_scores = self.criticality_scorer(node_features).squeeze(-1)
        
        # 提取关键路径节点（使用掩码加权）
        weighted_features = node_features * critical_path_mask.unsqueeze(-1)
        
        # LSTM编码
        lstm_out, (h_n, c_n) = self.path_lstm(weighted_features)
        
        # 使用最后的隐藏状态
        path_encoding = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        path_encoding = self.output_projection(path_encoding)
        
        return path_encoding, criticality_scores


class DAGAwareModule(nn.Module):
    """DAG感知模块 - 整合DAG编码和关键路径分析"""
    
    def __init__(self, node_feature_dim: int, hidden_dim: int, output_dim: int,
                 num_gnn_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super(DAGAwareModule, self).__init__()
        
        # DAG编码器
        self.dag_encoder = DAGEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 关键路径编码器
        self.critical_path_encoder = CriticalPathEncoder(
            node_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )
        
    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor,
                node_depths: Optional[torch.Tensor] = None,
                critical_path_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_features: 节点特征 [batch_size, num_nodes, node_feature_dim]
            adj_matrix: 邻接矩阵 [batch_size, num_nodes, num_nodes]
            node_depths: 节点拓扑深度 [batch_size, num_nodes]
            critical_path_mask: 关键路径掩码 [batch_size, num_nodes]
            
        Returns:
            包含各种DAG感知特征的字典
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # DAG编码
        node_embeddings, graph_embedding = self.dag_encoder(
            node_features, adj_matrix, node_depths
        )
        
        # 关键路径编码
        if critical_path_mask is None:
            critical_path_mask = torch.ones(batch_size, num_nodes, device=node_features.device)
        
        path_encoding, criticality_scores = self.critical_path_encoder(
            node_features, critical_path_mask
        )
        
        # 融合
        combined = torch.cat([graph_embedding, path_encoding, 
                             torch.mean(node_embeddings, dim=1)], dim=-1)
        dag_representation = self.fusion(combined)
        
        return {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
            'path_encoding': path_encoding,
            'criticality_scores': criticality_scores,
            'dag_representation': dag_representation
        }
