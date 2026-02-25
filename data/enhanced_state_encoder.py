# -*- coding: utf-8 -*-
"""
增强版状态表示模块
包含图嵌入、关键路径特征、全局进度特征等
"""

import numpy as np
import networkx as nx
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class EnhancedStateConfig:
    """增强版状态配置"""
    # 任务特征维度
    task_base_features: int = 7  # 基础任务特征数
    task_dag_features: int = 8   # DAG相关特征数
    task_temporal_features: int = 4  # 时序特征数
    
    # 资源特征维度
    resource_base_features: int = 4  # 基础资源特征数
    resource_load_features: int = 4  # 负载特征数
    resource_history_features: int = 3  # 历史性能特征数
    
    # 全局特征维度
    global_progress_features: int = 6  # 进度特征数
    global_dag_features: int = 5  # DAG全局特征数
    
    # 归一化
    normalize: bool = True
    clip_range: Tuple[float, float] = (-10.0, 10.0)


class CriticalPathAnalyzer:
    """关键路径分析器"""
    
    def __init__(self):
        self.critical_path_cache = {}
    
    def find_critical_path(self, dag: nx.DiGraph, 
                          task_durations: Dict[int, float]) -> Tuple[List[int], float]:
        """
        找到DAG中的关键路径
        
        Args:
            dag: 任务DAG
            task_durations: 任务ID到持续时间的映射
            
        Returns:
            critical_path: 关键路径上的任务ID列表
            critical_length: 关键路径长度
        """
        if len(dag.nodes()) == 0:
            return [], 0.0
        
        # 拓扑排序
        try:
            topo_order = list(nx.topological_sort(dag))
        except nx.NetworkXError:
            return [], 0.0
        
        # 计算每个节点的最早完成时间 (EST + duration)
        earliest_finish = {}
        predecessor = {}
        
        for node in topo_order:
            duration = task_durations.get(node, 1.0)
            
            # 找到所有前驱节点的最大完成时间
            predecessors = list(dag.predecessors(node))
            if not predecessors:
                earliest_finish[node] = duration
                predecessor[node] = None
            else:
                max_pred_finish = 0
                max_pred = None
                for pred in predecessors:
                    if earliest_finish[pred] > max_pred_finish:
                        max_pred_finish = earliest_finish[pred]
                        max_pred = pred
                earliest_finish[node] = max_pred_finish + duration
                predecessor[node] = max_pred
        
        # 找到最大完成时间的节点（终点）
        end_node = max(earliest_finish, key=earliest_finish.get)
        critical_length = earliest_finish[end_node]
        
        # 回溯找到关键路径
        critical_path = []
        current = end_node
        while current is not None:
            critical_path.append(current)
            current = predecessor[current]
        
        critical_path.reverse()
        
        return critical_path, critical_length
    
    def calculate_criticality_scores(self, dag: nx.DiGraph,
                                    task_durations: Dict[int, float]) -> Dict[int, float]:
        """
        计算每个任务的关键性评分
        评分基于任务对整体makespan的影响程度
        """
        if len(dag.nodes()) == 0:
            return {}
        
        critical_path, critical_length = self.find_critical_path(dag, task_durations)
        critical_set = set(critical_path)
        
        scores = {}
        
        for node in dag.nodes():
            if node in critical_set:
                # 关键路径上的任务评分最高
                scores[node] = 1.0
            else:
                # 计算任务的松弛时间 (slack)
                # 松弛时间越小，任务越关键
                slack = self._calculate_slack(dag, node, task_durations, critical_length)
                # 归一化到[0, 1]
                scores[node] = max(0, 1 - slack / (critical_length + 1e-6))
        
        return scores
    
    def _calculate_slack(self, dag: nx.DiGraph, node: int,
                        task_durations: Dict[int, float], 
                        total_length: float) -> float:
        """计算任务的松弛时间"""
        duration = task_durations.get(node, 1.0)
        
        # 计算最早开始时间 (EST)
        predecessors = list(dag.predecessors(node))
        if not predecessors:
            est = 0
        else:
            est = max(self._get_earliest_finish(dag, p, task_durations) 
                     for p in predecessors)
        
        # 计算最晚开始时间 (LST)
        successors = list(dag.successors(node))
        if not successors:
            lst = total_length - duration
        else:
            lst = min(self._get_latest_start(dag, s, task_durations, total_length) 
                     for s in successors) - duration
        
        return max(0, lst - est)
    
    def _get_earliest_finish(self, dag: nx.DiGraph, node: int,
                            task_durations: Dict[int, float]) -> float:
        """获取节点的最早完成时间"""
        duration = task_durations.get(node, 1.0)
        predecessors = list(dag.predecessors(node))
        
        if not predecessors:
            return duration
        
        return max(self._get_earliest_finish(dag, p, task_durations) 
                  for p in predecessors) + duration
    
    def _get_latest_start(self, dag: nx.DiGraph, node: int,
                         task_durations: Dict[int, float],
                         total_length: float) -> float:
        """获取节点的最晚开始时间"""
        duration = task_durations.get(node, 1.0)
        successors = list(dag.successors(node))
        
        if not successors:
            return total_length - duration
        
        return min(self._get_latest_start(dag, s, task_durations, total_length) 
                  for s in successors) - duration


class EnhancedStateEncoder:
    """增强版状态编码器"""
    
    def __init__(self, config: Optional[EnhancedStateConfig] = None):
        self.config = config or EnhancedStateConfig()
        self.critical_path_analyzer = CriticalPathAnalyzer()
        
        # 归一化统计
        self.feature_stats = defaultdict(lambda: {'mean': 0, 'std': 1, 'count': 0})
    
    def encode_state(self, tasks: List[Dict], resources: List[Dict],
                    dag: nx.DiGraph, scheduler_state: Dict) -> Dict[str, np.ndarray]:
        """
        编码完整状态表示
        
        Args:
            tasks: 任务列表
            resources: 资源列表
            dag: 任务DAG
            scheduler_state: 调度器状态
            
        Returns:
            包含各种特征的字典
        """
        # 1. 任务特征
        task_features = self._encode_task_features(tasks, dag, scheduler_state)
        
        # 2. 资源特征
        resource_features = self._encode_resource_features(resources, scheduler_state)
        
        # 3. DAG结构特征
        dag_features = self._encode_dag_features(dag, tasks, scheduler_state)
        
        # 4. 全局进度特征
        global_features = self._encode_global_features(tasks, resources, dag, scheduler_state)
        
        # 5. 邻接矩阵
        adj_matrix = self._build_adjacency_matrix(dag, tasks)
        
        # 6. 关键路径掩码
        critical_path_mask = self._build_critical_path_mask(dag, tasks, scheduler_state)
        
        # 7. 节点深度
        node_depths = self._calculate_node_depths(dag, tasks)
        
        return {
            'task_features': task_features,
            'resource_features': resource_features,
            'dag_features': dag_features,
            'global_features': global_features,
            'adj_matrix': adj_matrix,
            'critical_path_mask': critical_path_mask,
            'node_depths': node_depths
        }
    
    def _encode_task_features(self, tasks: List[Dict], dag: nx.DiGraph,
                             scheduler_state: Dict) -> np.ndarray:
        """编码任务特征"""
        completed_tasks = scheduler_state.get('completed_tasks', set())
        ready_tasks = scheduler_state.get('ready_tasks', [])
        task_end_times = scheduler_state.get('task_end_times', {})
        
        # 获取任务持续时间
        task_durations = {t['id']: t.get('duration', 1.0) for t in tasks}
        
        # 计算关键性评分
        criticality_scores = self.critical_path_analyzer.calculate_criticality_scores(
            dag, task_durations
        )
        
        features_list = []
        for task in tasks:
            task_id = task['id']
            
            # 基础特征
            base_features = [
                task.get('duration', 1.0),
                task.get('cpu_req', 1.0),
                task.get('memory_req', 1.0),
                1.0 if task_id in completed_tasks else 0.0,
                1.0 if task_id in ready_tasks else 0.0,
                task.get('priority', 0) / 10.0,  # 归一化优先级
                task.get('retry_times', 0) / 3.0  # 归一化重试次数
            ]
            
            # DAG相关特征
            in_degree = dag.in_degree(task_id) if task_id in dag else 0
            out_degree = dag.out_degree(task_id) if task_id in dag else 0
            
            # 已完成的前驱数量
            predecessors = list(dag.predecessors(task_id)) if task_id in dag else []
            completed_preds = sum(1 for p in predecessors if p in completed_tasks)
            
            # 等待的后继数量
            successors = list(dag.successors(task_id)) if task_id in dag else []
            waiting_succs = sum(1 for s in successors if s not in completed_tasks)
            
            dag_features = [
                in_degree / 10.0,
                out_degree / 10.0,
                completed_preds / max(1, len(predecessors)),
                waiting_succs / max(1, len(successors)),
                criticality_scores.get(task_id, 0.5),
                1.0 if len(predecessors) == 0 else 0.0,  # 是否是根节点
                1.0 if len(successors) == 0 else 0.0,    # 是否是叶节点
                len(successors) / 10.0  # 后续任务数量
            ]
            
            # 时序特征
            earliest_start = self._calculate_earliest_start(
                task_id, dag, task_end_times
            )
            
            temporal_features = [
                earliest_start / 1000.0,  # 归一化最早开始时间
                task.get('duration', 1.0) / 100.0,  # 归一化持续时间
                0.0,  # 预留：预估等待时间
                0.0   # 预留：历史执行时间方差
            ]
            
            features = base_features + dag_features + temporal_features
            features_list.append(features)
        
        features_array = np.array(features_list, dtype=np.float32)
        
        if self.config.normalize:
            features_array = self._normalize_features(features_array, 'task')
        
        return features_array
    
    def _encode_resource_features(self, resources: List[Dict],
                                  scheduler_state: Dict) -> np.ndarray:
        """编码资源特征"""
        resource_available_time = scheduler_state.get('resource_available_time', {})
        current_time = scheduler_state.get('current_time', 0)
        
        features_list = []
        for resource in resources:
            resource_id = resource['id']
            
            # 基础特征
            base_features = [
                resource.get('cpu_capacity', 4.0) / 16.0,
                resource.get('memory_capacity', 8.0) / 64.0,
                resource.get('disk_capacity', 100.0) / 1000.0,
                resource.get('network_bandwidth', 100.0) / 1000.0
            ]
            
            # 负载特征
            available_time = resource_available_time.get(resource_id, 0)
            wait_time = max(0, available_time - current_time)
            
            load_features = [
                available_time / 1000.0,
                wait_time / 100.0,
                1.0 if wait_time == 0 else 0.0,  # 是否空闲
                wait_time / max(1, current_time + 1)  # 相对等待时间
            ]
            
            # 历史性能特征
            history = scheduler_state.get('resource_history', {}).get(resource_id, {})
            history_features = [
                history.get('avg_task_duration', 10.0) / 100.0,
                history.get('task_count', 0) / 100.0,
                history.get('failure_rate', 0.0)
            ]
            
            features = base_features + load_features + history_features
            features_list.append(features)
        
        features_array = np.array(features_list, dtype=np.float32)
        
        if self.config.normalize:
            features_array = self._normalize_features(features_array, 'resource')
        
        return features_array
    
    def _encode_dag_features(self, dag: nx.DiGraph, tasks: List[Dict],
                            scheduler_state: Dict) -> Dict[str, Any]:
        """编码DAG结构特征"""
        if len(dag.nodes()) == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'max_depth': 0,
                'max_width': 0,
                'density': 0.0
            }
        
        # 计算DAG统计特征
        num_nodes = len(dag.nodes())
        num_edges = len(dag.edges())
        density = nx.density(dag)
        
        # 计算深度和宽度
        levels = defaultdict(list)
        for node in dag.nodes():
            depth = self._get_node_depth(dag, node)
            levels[depth].append(node)
        
        max_depth = max(levels.keys()) if levels else 0
        max_width = max(len(nodes) for nodes in levels.values()) if levels else 0
        
        # 计算关键路径
        task_durations = {t['id']: t.get('duration', 1.0) for t in tasks}
        critical_path, critical_length = self.critical_path_analyzer.find_critical_path(
            dag, task_durations
        )
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'max_depth': max_depth,
            'max_width': max_width,
            'density': density,
            'critical_path_length': critical_length,
            'critical_path_nodes': len(critical_path),
            'parallelism': max_width / max(1, max_depth)
        }
    
    def _encode_global_features(self, tasks: List[Dict], resources: List[Dict],
                               dag: nx.DiGraph, scheduler_state: Dict) -> np.ndarray:
        """编码全局进度特征"""
        completed_tasks = scheduler_state.get('completed_tasks', set())
        ready_tasks = scheduler_state.get('ready_tasks', [])
        current_time = scheduler_state.get('current_time', 0)
        current_makespan = scheduler_state.get('current_makespan', 0)
        
        total_tasks = len(tasks)
        total_resources = len(resources)
        
        # 进度特征
        progress_features = [
            len(completed_tasks) / max(1, total_tasks),  # 完成进度
            len(ready_tasks) / max(1, total_tasks),      # 就绪任务比例
            current_time / 1000.0,                        # 当前时间
            current_makespan / 1000.0,                    # 当前makespan
            (total_tasks - len(completed_tasks)) / max(1, total_tasks),  # 剩余任务比例
            len(ready_tasks) / max(1, total_resources)    # 就绪任务/资源比
        ]
        
        # DAG全局特征
        if len(dag.nodes()) > 0:
            dag_stats = self._encode_dag_features(dag, tasks, scheduler_state)
            dag_features = [
                dag_stats['max_depth'] / 20.0,
                dag_stats['max_width'] / 10.0,
                dag_stats['density'],
                dag_stats.get('parallelism', 1.0),
                dag_stats.get('critical_path_length', 0) / 1000.0
            ]
        else:
            dag_features = [0.0] * 5
        
        features = progress_features + dag_features
        return np.array(features, dtype=np.float32)
    
    def _build_adjacency_matrix(self, dag: nx.DiGraph, 
                               tasks: List[Dict]) -> np.ndarray:
        """构建邻接矩阵"""
        task_ids = [t['id'] for t in tasks]
        id_to_idx = {tid: idx for idx, tid in enumerate(task_ids)}
        
        n = len(tasks)
        adj = np.zeros((n, n), dtype=np.float32)
        
        for u, v in dag.edges():
            if u in id_to_idx and v in id_to_idx:
                adj[id_to_idx[u], id_to_idx[v]] = 1.0
        
        # 添加自环
        np.fill_diagonal(adj, 1.0)
        
        return adj
    
    def _build_critical_path_mask(self, dag: nx.DiGraph, tasks: List[Dict],
                                  scheduler_state: Dict) -> np.ndarray:
        """构建关键路径掩码"""
        task_durations = {t['id']: t.get('duration', 1.0) for t in tasks}
        critical_path, _ = self.critical_path_analyzer.find_critical_path(
            dag, task_durations
        )
        critical_set = set(critical_path)
        
        mask = np.array([
            1.0 if t['id'] in critical_set else 0.0
            for t in tasks
        ], dtype=np.float32)
        
        return mask
    
    def _calculate_node_depths(self, dag: nx.DiGraph, 
                              tasks: List[Dict]) -> np.ndarray:
        """计算节点深度"""
        depths = []
        for task in tasks:
            task_id = task['id']
            depth = self._get_node_depth(dag, task_id)
            depths.append(depth)
        
        return np.array(depths, dtype=np.int64)
    
    def _get_node_depth(self, dag: nx.DiGraph, node: int) -> int:
        """获取节点深度"""
        if node not in dag:
            return 0
        
        predecessors = list(dag.predecessors(node))
        if not predecessors:
            return 0
        
        return 1 + max(self._get_node_depth(dag, p) for p in predecessors)
    
    def _calculate_earliest_start(self, task_id: int, dag: nx.DiGraph,
                                  task_end_times: Dict[int, float]) -> float:
        """计算任务最早开始时间"""
        if task_id not in dag:
            return 0.0
        
        predecessors = list(dag.predecessors(task_id))
        if not predecessors:
            return 0.0
        
        # 最早开始时间 = 所有前驱的最大完成时间
        max_end = 0.0
        for pred in predecessors:
            if pred in task_end_times:
                max_end = max(max_end, task_end_times[pred])
        
        return max_end
    
    def _normalize_features(self, features: np.ndarray, 
                           feature_type: str) -> np.ndarray:
        """归一化特征"""
        # 使用在线更新的均值和标准差
        stats = self.feature_stats[feature_type]
        
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        
        # 在线更新统计量
        alpha = 0.01
        if stats['count'] == 0:
            stats['mean'] = mean
            stats['std'] = std
        else:
            stats['mean'] = (1 - alpha) * stats['mean'] + alpha * mean
            stats['std'] = (1 - alpha) * stats['std'] + alpha * std
        stats['count'] += 1
        
        # 归一化
        normalized = (features - stats['mean']) / stats['std']
        
        # 裁剪
        normalized = np.clip(normalized, 
                           self.config.clip_range[0], 
                           self.config.clip_range[1])
        
        return normalized


class StateToTensor:
    """状态到张量的转换器"""
    
    @staticmethod
    def convert(state: Dict[str, np.ndarray], 
               device: torch.device = None) -> Dict[str, torch.Tensor]:
        """将numpy状态转换为PyTorch张量"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        tensors = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                tensors[key] = torch.from_numpy(value).to(device)
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                tensors[key] = StateToTensor.convert(value, device)
            else:
                tensors[key] = value
        
        return tensors
    
    @staticmethod
    def add_batch_dim(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """添加批次维度"""
        batched = {}
        for key, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor):
                batched[key] = tensor.unsqueeze(0)
            else:
                batched[key] = tensor
        return batched
