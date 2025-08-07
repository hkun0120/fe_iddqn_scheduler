import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
from config.hyperparameters import Hyperparameters

class AttentionModule(nn.Module):
    """注意力机制模块"""
    
    def __init__(self, input_dim: int, attention_dim: int, num_heads: int = 4):
        super(AttentionModule, self).__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attention_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            输出张量 [batch_size, seq_len, input_dim]
        """
        # 多头注意力
        attn_output, _ = self.multihead_attn(x, x, x)
        
        # 残差连接和层归一化
        x = self.layer_norm(x + attn_output)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        
        # 残差连接和层归一化
        output = self.layer_norm(x + ff_output)
        
        return output

class TaskStream(nn.Module):
    """任务流网络"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 attention_dim: int, num_heads: int = 4, dropout_rate: float = 0.1):
        super(TaskStream, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, hidden_dims[0])
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.BatchNorm1d(hidden_dims[i + 1])
                )
            )
        
        # 注意力机制
        self.attention = AttentionModule(
            input_dim=hidden_dims[-1],
            attention_dim=attention_dim,
            num_heads=num_heads
        )
        
        # 输出层
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 任务特征张量 [batch_size, num_tasks, input_dim]
            
        Returns:
            任务流输出 [batch_size, num_tasks, output_dim]
        """
        batch_size, num_tasks, _ = x.shape
        
        # 重塑为二维张量进行线性变换
        x = x.view(-1, self.input_dim)
        
        # 输入嵌入
        x = F.relu(self.input_embedding(x))
        
        # 隐藏层
        for layer in self.hidden_layers:
            x = layer(x)
        
        # 重塑回三维张量
        x = x.view(batch_size, num_tasks, -1)
        
        # 注意力机制
        x = self.attention(x)
        
        return x

class ResourceStream(nn.Module):
    """资源流网络"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 attention_dim: int, num_heads: int = 4, dropout_rate: float = 0.1):
        super(ResourceStream, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, hidden_dims[0])
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.BatchNorm1d(hidden_dims[i + 1])
                )
            )
        
        # 注意力机制
        self.attention = AttentionModule(
            input_dim=hidden_dims[-1],
            attention_dim=attention_dim,
            num_heads=num_heads
        )
        
        # 输出层
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 资源特征张量 [batch_size, num_resources, input_dim]
            
        Returns:
            资源流输出 [batch_size, num_resources, output_dim]
        """
        batch_size, num_resources, _ = x.shape
        
        # 重塑为二维张量进行线性变换
        x = x.view(-1, self.input_dim)
        
        # 输入嵌入
        x = F.relu(self.input_embedding(x))
        
        # 隐藏层
        for layer in self.hidden_layers:
            x = layer(x)
        
        # 重塑回三维张量
        x = x.view(batch_size, num_resources, -1)
        
        # 注意力机制
        x = self.attention(x)
        
        return x

class FeatureFusion(nn.Module):
    """特征融合模块"""
    
    def __init__(self, task_dim: int, resource_dim: int, fusion_dim: int, 
                 output_dim: int, dropout_rate: float = 0.1):
        super(FeatureFusion, self).__init__()
        
        self.task_dim = task_dim
        self.resource_dim = resource_dim
        self.fusion_dim = fusion_dim
        
        # 任务特征投影
        self.task_projection = nn.Linear(task_dim, fusion_dim)
        
        # 资源特征投影
        self.resource_projection = nn.Linear(resource_dim, fusion_dim)
        
        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, output_dim)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
    def forward(self, task_features: torch.Tensor, 
                resource_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            task_features: 任务特征 [batch_size, num_tasks, task_dim]
            resource_features: 资源特征 [batch_size, num_resources, resource_dim]
            
        Returns:
            融合后的特征 [batch_size, output_dim]
        """
        batch_size = task_features.shape[0]
        
        # 特征投影
        task_proj = self.task_projection(task_features)  # [batch_size, num_tasks, fusion_dim]
        resource_proj = self.resource_projection(resource_features)  # [batch_size, num_resources, fusion_dim]
        
        # 交叉注意力：任务关注资源
        task_attended, _ = self.cross_attention(task_proj, resource_proj, resource_proj)
        task_attended = self.layer_norm(task_proj + task_attended)
        
        # 交叉注意力：资源关注任务
        resource_attended, _ = self.cross_attention(resource_proj, task_proj, task_proj)
        resource_attended = self.layer_norm(resource_proj + resource_attended)
        
        # 全局池化
        task_global = torch.mean(task_attended, dim=1)  # [batch_size, fusion_dim]
        resource_global = torch.mean(resource_attended, dim=1)  # [batch_size, fusion_dim]
        
        # 特征拼接
        fused_features = torch.cat([task_global, resource_global], dim=1)  # [batch_size, fusion_dim * 2]
        
        # 融合网络
        output = self.fusion_network(fused_features)
        
        return output

class DualStreamNetwork(nn.Module):
    """双流网络架构"""
    
    def __init__(self, task_input_dim: int, resource_input_dim: int, 
                 task_hidden_dims: List[int], resource_hidden_dims: List[int],
                 fusion_dim: int, output_dim: int, 
                 attention_dim: int = 64, num_heads: int = 4, 
                 dropout_rate: float = 0.1):
        super(DualStreamNetwork, self).__init__()
        
        # 任务流
        self.task_stream = TaskStream(
            input_dim=task_input_dim,
            hidden_dims=task_hidden_dims,
            attention_dim=attention_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
        # 资源流
        self.resource_stream = ResourceStream(
            input_dim=resource_input_dim,
            hidden_dims=resource_hidden_dims,
            attention_dim=attention_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
        # 特征融合
        self.feature_fusion = FeatureFusion(
            task_dim=task_hidden_dims[-1],
            resource_dim=resource_hidden_dims[-1],
            fusion_dim=fusion_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        )
        
    def forward(self, task_features: torch.Tensor, 
                resource_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            task_features: 任务特征 [batch_size, num_tasks, task_input_dim]
            resource_features: 资源特征 [batch_size, num_resources, resource_input_dim]
            
        Returns:
            网络输出 [batch_size, output_dim]
        """
        # 任务流处理
        task_output = self.task_stream(task_features)
        
        # 资源流处理
        resource_output = self.resource_stream(resource_features)
        
        # 特征融合
        fused_output = self.feature_fusion(task_output, resource_output)
        
        return fused_output
    
    def get_feature_representations(self, task_features: torch.Tensor, 
                                   resource_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取中间特征表示
        
        Args:
            task_features: 任务特征
            resource_features: 资源特征
            
        Returns:
            任务流输出和资源流输出
        """
        task_output = self.task_stream(task_features)
        resource_output = self.resource_stream(resource_features)
        
        return task_output, resource_output

