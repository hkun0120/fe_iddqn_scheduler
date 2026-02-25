# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
# 移除不需要的导入


class FeatureEngineer:
    """特征工程类 - 同时考虑工作流图特征和任务特征"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.feature_selector = None
        
    def create_workflow_graphs(self, process_definitions: pd.DataFrame, 
                             process_task_relations: pd.DataFrame) -> Dict[int, nx.DiGraph]:
        """为每个工作流定义创建有向图"""
        self.logger.info("Creating workflow graphs...")
        
        graphs = {}
        for _, process in process_definitions.iterrows():
            process_code = process['code']
            
            # 获取该工作流的所有任务关系
            relations = process_task_relations[
                process_task_relations['process_definition_code'] == process_code
            ]
            
            # 创建有向图
            G = nx.DiGraph()
            
            # 添加节点（任务）
            for _, relation in relations.iterrows():
                pre_task = relation['pre_task_code']
                post_task = relation['post_task_code']
                
                if pd.notna(pre_task):
                    G.add_node(pre_task)
                if pd.notna(post_task):
                    G.add_node(post_task)
                
                # 添加边（依赖关系）
                if pd.notna(pre_task) and pd.notna(post_task):
                    G.add_edge(pre_task, post_task)
            
            graphs[process_code] = G
        
        self.logger.info(f"Created {len(graphs)} workflow graphs")
        return graphs

    def extract_workflow_graph_features(self, graphs: Dict[int, nx.DiGraph]) -> pd.DataFrame:
        """提取工作流图特征"""
        self.logger.info("Extracting workflow graph features...")
        
        graph_features = []
        
        for process_code, G in graphs.items():
            if len(G.nodes()) == 0:
                continue
                
            # 基本图特征
            features = {
                'process_definition_code': process_code,
                'graph_num_nodes': len(G.nodes()),
                'graph_num_edges': len(G.edges()),
                'graph_is_dag': nx.is_directed_acyclic_graph(G),
                'graph_max_depth': self._calculate_max_depth(G),
                'graph_avg_in_degree': np.mean([G.in_degree(node) for node in G.nodes()]) if G.nodes() else 0,
                'graph_avg_out_degree': np.mean([G.out_degree(node) for node in G.nodes()]) if G.nodes() else 0,
                'graph_density': nx.density(G),
                'graph_diameter': self._safe_diameter(G),
                'graph_parallelism': self._calculate_parallelism(G),
                'graph_critical_path_length': self._calculate_critical_path_length(G),
                'graph_avg_path_length': self._calculate_avg_path_length(G),
                'graph_max_degree': max([G.degree(node) for node in G.nodes()]) if G.nodes() else 0,
                'graph_min_degree': min([G.degree(node) for node in G.nodes()]) if G.nodes() else 0,
                'graph_leaf_nodes': len([node for node in G.nodes() if G.out_degree(node) == 0]),
                'graph_root_nodes': len([node for node in G.nodes() if G.in_degree(node) == 0])
            }
            
            graph_features.append(features)
        
        return pd.DataFrame(graph_features)

    def _calculate_max_depth(self, G: nx.DiGraph) -> int:
        """计算图的最大深度"""
        if len(G.nodes()) == 0:
            return 0
        
        # 找到所有入度为0的节点（根节点）
        roots = [node for node in G.nodes() if G.in_degree(node) == 0]
        
        if not roots:
            return 0
        
        max_depth = 0
        for root in roots:
            depths = nx.single_source_shortest_path_length(G, root)
            max_depth = max(max_depth, max(depths.values()) if depths else 0)
        
        return max_depth

    def _safe_diameter(self, G: nx.DiGraph) -> int:
        """安全计算图的直径，处理不连通的情况"""
        if len(G.nodes()) == 0:
            return -1
        
        try:
            # 转换为无向图
            G_undirected = G.to_undirected()
            
            # 检查是否连通
            if not nx.is_connected(G_undirected):
                return -1
            
            # 计算直径
            return nx.diameter(G_undirected)
        except (nx.NetworkXError, nx.NetworkXNoPath):
            return -1

    def _calculate_parallelism(self, G: nx.DiGraph) -> float:
        """计算工作流的并行度"""
        if len(G.nodes()) == 0:
            return 0
        
        # 计算每一层的节点数，取最大值作为并行度
        levels = {}
        for node in G.nodes():
            depth = self._calculate_node_depth(G, node)
            if depth not in levels:
                levels[depth] = 0
            levels[depth] += 1
        
        return max(levels.values()) if levels else 0

    def _calculate_node_depth(self, G: nx.DiGraph, node) -> int:
        """计算节点的深度"""
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        if not roots:
            return 0
        
        max_depth = 0
        for root in roots:
            try:
                depth = nx.shortest_path_length(G, root, node)
                max_depth = max(max_depth, depth)
            except nx.NetworkXNoPath:
                continue
        
        return max_depth

    def _calculate_critical_path_length(self, G: nx.DiGraph) -> int:
        """计算关键路径长度"""
        if len(G.nodes()) == 0:
            return 0
        
        # 找到所有根节点
        roots = [node for node in G.nodes() if G.in_degree(node) == 0]
        
        if not roots:
            return 0
        
        # 计算从每个根节点到所有叶节点的最长路径
        max_length = 0
        for root in roots:
            for node in G.nodes():
                if G.out_degree(node) == 0:  # 叶节点
                    try:
                        length = nx.shortest_path_length(G, root, node)
                        max_length = max(max_length, length)
                    except nx.NetworkXNoPath:
                        continue
        
        return max_length

    def _calculate_avg_path_length(self, G: nx.DiGraph) -> float:
        """计算平均路径长度"""
        if len(G.nodes()) == 0:
            return 0
        
        try:
            return nx.average_shortest_path_length(G)
        except (nx.NetworkXError, nx.NetworkXNoPath):
            return 0

    def extract_task_features(self, task_instances: pd.DataFrame, 
                            task_definitions: pd.DataFrame,
                            process_instances: pd.DataFrame,
                            workflow_graphs: pd.DataFrame) -> pd.DataFrame:
        """提取任务级别的特征，包含工作流图特征"""
        self.logger.info("Extracting task-level features with workflow graph features...")
        
        # 合并任务实例和任务定义
        merged_tasks = pd.merge(
            task_instances,
            task_definitions,
            left_on=['task_code', 'task_definition_version'],
            right_on=['code', 'version'],
            how='left',
            suffixes=('_instance', '_definition')
        )
        
        # 合并进程实例信息
        merged_tasks = pd.merge(
            merged_tasks,
            process_instances[['id', 'process_definition_code', 'process_instance_priority', 'global_params', 'command_param']],
            left_on='process_instance_id',
            right_on='id',
            how='left',
            suffixes=('', '_process')
        )
        
        # 合并工作流图特征
        merged_tasks = pd.merge(
            merged_tasks,
            workflow_graphs,
            left_on='process_definition_code',
            right_on='process_definition_code',
            how='left'
        )
        
        task_features = []
        
        for _, task in merged_tasks.iterrows():
            # 基本任务特征
            features = {
                'task_id': task['id_instance'],
                'task_name': task['name_instance'],
                'task_type': task['task_type_instance'],
                'process_instance_id': task['process_instance_id'],
                'process_definition_code': task['process_definition_code'],
                'state': task['state'],
                'priority': task.get('task_instance_priority', 0),
                'retry_times': task.get('retry_times', 0),
                'max_retry_times': task.get('max_retry_times', 0),
                'worker_group': task.get('worker_group', 'default'),
                'executor_id': task.get('executor_id', 0),
                'flag': task.get('flag', 0),
                'alert_flag': task.get('alert_flag', 0),
                'delay_time': task.get('delay_time', 0),
                'task_group_id': task.get('task_group_id', 0),
                'dry_run': task.get('dry_run', 0)
            }
            
            # 任务类型编码
            task_type_encoding = {
                'SQL': [1, 0, 0, 0, 0, 0, 0],
                'SHELL': [0, 1, 0, 0, 0, 0, 0],
                'PYTHON': [0, 0, 1, 0, 0, 0, 0],
                'JAVA': [0, 0, 0, 1, 0, 0, 0],
                'SPARK': [0, 0, 0, 0, 1, 0, 0],
                'FLINK': [0, 0, 0, 0, 0, 1, 0],
                'HTTP': [0, 0, 0, 0, 0, 0, 1]
            }
            
            task_type = task.get('task_type', 'SHELL')
            type_encoding = task_type_encoding.get(task_type, [0, 0, 0, 0, 0, 0, 0])
            features.update({
                'is_sql': type_encoding[0],
                'is_shell': type_encoding[1],
                'is_python': type_encoding[2],
                'is_java': type_encoding[3],
                'is_spark': type_encoding[4],
                'is_flink': type_encoding[5],
                'is_http': type_encoding[6]
            })
            
            # 资源需求特征
            features.update({
                'cpu_requirement': self._estimate_task_cpu_requirement(task),
                'memory_requirement': self._estimate_task_memory_requirement(task),
                'timeout_requirement': task.get('timeout', 0),
                'fail_retry_times': task.get('fail_retry_times', 0),
                'fail_retry_interval': task.get('fail_retry_interval', 0),
                'timeout_flag': task.get('timeout_flag', 0),
                'timeout_notify_strategy': task.get('timeout_notify_strategy', 0)
            })
            
            # 任务定义特征
            features.update({
                'task_priority': task.get('task_priority', 0),
                'environment_code': task.get('environment_code', ''),
                'has_resource_ids': 1 if pd.notna(task.get('resource_ids')) else 0,
                'has_task_params': 1 if pd.notna(task.get('task_params')) else 0,
                'task_group_priority': task.get('task_group_priority', 0)
            })
            
            # 时间特征
            if pd.notna(task.get('submit_time')):
                submit_time = pd.to_datetime(task['submit_time'])
                features.update({
                    'submit_hour': submit_time.hour,
                    'submit_day_of_week': submit_time.dayofweek,
                    'submit_month': submit_time.month,
                    'submit_is_weekend': 1 if submit_time.dayofweek >= 5 else 0
                })
            
            if pd.notna(task.get('start_time')):
                start_time = pd.to_datetime(task['start_time'])
                features.update({
                    'start_hour': start_time.hour,
                    'start_day_of_week': start_time.dayofweek,
                    'start_month': start_time.month,
                    'start_is_weekend': 1 if start_time.dayofweek >= 5 else 0
                })
            
            # 执行时间特征
            if pd.notna(task.get('start_time')) and pd.notna(task.get('end_time')):
                start_time = pd.to_datetime(task['start_time'])
                end_time = pd.to_datetime(task['end_time'])
                duration = (end_time - start_time).total_seconds()
                features.update({
                    'execution_duration': duration,
                    'execution_duration_minutes': duration / 60,
                    'execution_duration_hours': duration / 3600
                })
            else:
                features.update({
                    'execution_duration': 0,
                    'execution_duration_minutes': 0,
                    'execution_duration_hours': 0
                })
            
            # 进程实例特征
            features.update({
                'process_priority': task.get('process_instance_priority', 0),
                'has_global_params': 1 if pd.notna(task.get('global_params')) else 0,
                'has_command_param': 1 if pd.notna(task.get('command_param')) else 0
            })
            
            # 状态特征
            state_features = {
                'state_commit_succeeded': 1 if task['state'] == 0 else 0,
                'state_running': 1 if task['state'] == 1 else 0,
                'state_prepare_to_pause': 1 if task['state'] == 2 else 0,
                'state_pause': 1 if task['state'] == 3 else 0,
                'state_prepare_to_stop': 1 if task['state'] == 4 else 0,
                'state_stop': 1 if task['state'] == 5 else 0,
                'state_fail': 1 if task['state'] == 6 else 0,
                'state_succeed': 1 if task['state'] == 7 else 0,
                'state_need_fault_tolerance': 1 if task['state'] == 8 else 0,
                'state_kill': 1 if task['state'] == 9 else 0,
                'state_wait_for_thread': 1 if task['state'] == 10 else 0,
                'state_wait_for_dependency': 1 if task['state'] == 11 else 0
            }
            features.update(state_features)
            
            # 工作流图特征
            workflow_graph_features = [
                'graph_num_nodes', 'graph_num_edges', 'graph_is_dag', 'graph_max_depth',
                'graph_avg_in_degree', 'graph_avg_out_degree', 'graph_density', 'graph_diameter',
                'graph_parallelism', 'graph_critical_path_length', 'graph_avg_path_length',
                'graph_max_degree', 'graph_min_degree', 'graph_leaf_nodes', 'graph_root_nodes'
            ]
            
            for feature in workflow_graph_features:
                features[feature] = task.get(feature, 0)
            
            task_features.append(features)
        
        return pd.DataFrame(task_features)

    def _estimate_task_cpu_requirement(self, task: pd.Series) -> float:
        """估算任务的CPU需求"""
        task_type = task.get('task_type', 'SHELL')
        
        # 根据任务类型设置基础CPU需求
        base_cpu = {
            'SQL': 2.0,
            'SHELL': 1.0,
            'PYTHON': 2.0,
            'JAVA': 3.0,
            'SPARK': 4.0,
            'FLINK': 4.0,
            'HTTP': 1.0
        }.get(task_type, 1.0)
        
        # 根据优先级调整
        priority = task.get('task_priority', 0)
        if priority > 0:
            base_cpu *= 1.2
        
        # 根据执行时间调整
        if pd.notna(task.get('start_time')) and pd.notna(task.get('end_time')):
            start_time = pd.to_datetime(task['start_time'])
            end_time = pd.to_datetime(task['end_time'])
            duration = (end_time - start_time).total_seconds()
            if duration > 300:  # 超过5分钟的任务
                base_cpu *= 1.1
        
        return min(base_cpu, 8.0)  # 限制最大CPU需求

    def _estimate_task_memory_requirement(self, task: pd.Series) -> float:
        """估算任务的内存需求"""
        task_type = task.get('task_type', 'SHELL')
        
        # 根据任务类型设置基础内存需求（GB）
        base_memory = {
            'SQL': 1.0,
            'SHELL': 0.5,
            'PYTHON': 2.0,
            'JAVA': 4.0,
            'SPARK': 8.0,
            'FLINK': 8.0,
            'HTTP': 1.0
        }.get(task_type, 1.0)
        
        # 根据优先级调整
        priority = task.get('task_priority', 0)
        if priority > 0:
            base_memory *= 1.2
        
        # 根据执行时间调整
        if pd.notna(task.get('start_time')) and pd.notna(task.get('end_time')):
            start_time = pd.to_datetime(task['start_time'])
            end_time = pd.to_datetime(task['end_time'])
            duration = (end_time - start_time).total_seconds()
            if duration > 300:  # 超过5分钟的任务
                base_memory *= 1.1
        
        return min(base_memory, 16.0)  # 限制最大内存需求

    def extract_resource_features(self, task_instances: pd.DataFrame) -> pd.DataFrame:
        """提取资源级别的特征"""
        self.logger.info("Extracting resource-level features...")
        
        # 获取所有主机
        hosts = task_instances['host'].dropna().unique()
        
        resource_features = []
        
        for host in hosts:
            host_tasks = task_instances[task_instances['host'] == host]
            
            # 基本资源特征
            features = {
                'host': host,
                'total_tasks_processed': len(host_tasks),
                'successful_tasks': len(host_tasks[host_tasks['state'] == 7]),
                'failed_tasks': len(host_tasks[host_tasks['state'] == 6]),
                'running_tasks': len(host_tasks[host_tasks['state'] == 1]),
                'avg_task_duration': host_tasks['execution_duration'].mean() if 'execution_duration' in host_tasks.columns else 0,
                'max_task_duration': host_tasks['execution_duration'].max() if 'execution_duration' in host_tasks.columns else 0,
                'min_task_duration': host_tasks['execution_duration'].min() if 'execution_duration' in host_tasks.columns else 0
            }
            
            # 任务类型分布
            task_types = host_tasks['task_type'].value_counts()
            features.update({
                'num_sql_tasks': task_types.get('SQL', 0),
                'num_shell_tasks': task_types.get('SHELL', 0),
                'num_python_tasks': task_types.get('PYTHON', 0),
                'num_java_tasks': task_types.get('JAVA', 0),
                'num_spark_tasks': task_types.get('SPARK', 0),
                'num_flink_tasks': task_types.get('FLINK', 0),
                'num_http_tasks': task_types.get('HTTP', 0),
                'unique_task_types': len(task_types)
            })
            
            # 估算资源容量
            features.update({
                'estimated_cpu_capacity': self._estimate_host_cpu_capacity(host_tasks),
                'estimated_memory_capacity': self._estimate_host_memory_capacity(host_tasks)
            })
            
            resource_features.append(features)
        
        return pd.DataFrame(resource_features)

    def _estimate_host_cpu_capacity(self, host_tasks: pd.DataFrame) -> int:
        """估算主机的CPU容量"""
        if host_tasks.empty:
            return 8
        
        # 基于任务类型和数量估算
        task_types = host_tasks['task_type'].value_counts()
        estimated_cpu = 4  # 基础CPU
        
        # 根据任务类型调整
        if 'SPARK' in task_types or 'FLINK' in task_types:
            estimated_cpu += 4
        if 'JAVA' in task_types:
            estimated_cpu += 2
        if 'PYTHON' in task_types:
            estimated_cpu += 1
        
        return min(estimated_cpu, 16)

    def _estimate_host_memory_capacity(self, host_tasks: pd.DataFrame) -> int:
        """估算主机的内存容量"""
        if host_tasks.empty:
            return 16
        
        # 基于任务类型估算
        task_types = host_tasks['task_type'].value_counts()
        estimated_memory = 8  # 基础内存
        
        # 根据任务类型调整
        if 'SPARK' in task_types or 'FLINK' in task_types:
            estimated_memory += 8
        if 'JAVA' in task_types:
            estimated_memory += 4
        if 'PYTHON' in task_types:
            estimated_memory += 2
        
        return min(estimated_memory, 64)

    def normalize_features(self, df: pd.DataFrame, 
                         columns_to_normalize: Optional[List[str]] = None) -> pd.DataFrame:
        """对数值特征进行标准化"""
        self.logger.info("Normalizing features...")
        
        df_normalized = df.copy()
        if columns_to_normalize is None:
            columns_to_normalize = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
            # 排除ID列
            exclude_cols = ['task_id', 'process_instance_id', 'process_definition_code']
            columns_to_normalize = [col for col in columns_to_normalize if col not in exclude_cols]
        
        for col in columns_to_normalize:
            if col in df_normalized.columns:
                scaler = StandardScaler()
                df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
                self.scalers[col] = scaler
        
        self.logger.info("Features normalized.")
        return df_normalized

    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """特征选择"""
        self.logger.info(f"Selecting top {k} features...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        # 排除ID列
        exclude_cols = ['task_id', 'process_instance_id', 'process_definition_code']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X_numeric = X[numeric_cols]

        self.feature_selector = SelectKBest(score_func=f_classif, 
                                          k=min(k, X_numeric.shape[1]))
        self.feature_selector.fit(X_numeric, y)
        
        selected_features_mask = self.feature_selector.get_support()
        selected_feature_names = X_numeric.columns[selected_features_mask].tolist()
        
        self.logger.info(f"Selected features: {selected_feature_names}")
        return X[selected_feature_names], selected_feature_names

    def run_feature_engineering_pipeline(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """运行完整的特征工程管道"""
        self.logger.info("Running feature engineering pipeline...")
        
        # 1. 创建工作流图
        workflow_graphs = self.create_workflow_graphs(
            data["process_definition"], data["process_task_relation"]
        )
        
        # 2. 提取工作流图特征
        workflow_graph_features_df = self.extract_workflow_graph_features(workflow_graphs)
        
        # 3. 提取任务级别特征（包含工作流图特征）
        task_features_df = self.extract_task_features(
            data["task_instance"], data["task_definition"], 
            data["process_instance"], workflow_graph_features_df
        )
        
        # 4. 标准化特征
        if Config.NORMALIZE_FEATURES:
            cols_to_normalize = [col for col in task_features_df.columns 
                               if task_features_df[col].dtype in [np.float64, np.int64] 
                               and col not in ["task_id", "process_instance_id", "process_definition_code"]]
            task_features_df = self.normalize_features(task_features_df, cols_to_normalize)
        
        self.logger.info("Feature engineering pipeline completed.")
        return task_features_df

    def apply_feature_engineering_pipeline(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """对新数据应用特征工程管道"""
        self.logger.info("Applying feature engineering pipeline to new data...")
        
        # 1. 创建工作流图
        workflow_graphs = self.create_workflow_graphs(
            data["process_definition"], data["process_task_relation"]
        )
        
        # 2. 提取工作流图特征
        workflow_graph_features_df = self.extract_workflow_graph_features(workflow_graphs)
        
        # 3. 提取任务级别特征（包含工作流图特征）
        task_features_df = self.extract_task_features(
            data["task_instance"], data["task_definition"], 
            data["process_instance"], workflow_graph_features_df
        )
        
        # 4. 标准化特征
        if Config.NORMALIZE_FEATURES:
            cols_to_normalize = [col for col in task_features_df.columns 
                               if task_features_df[col].dtype in [np.float64, np.int64] 
                               and col not in ["task_id", "process_instance_id", "process_definition_code"]]
            for col in cols_to_normalize:
                if col in task_features_df.columns and col in self.scalers:
                    task_features_df[col] = self.scalers[col].transform(task_features_df[[col]])
                else:
                    self.logger.warning(f"Scaler not found for column {col}. Skipping normalization.")
        
        # 5. 特征选择
        if self.feature_selector:
            numeric_cols = task_features_df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['task_id', 'process_instance_id', 'process_definition_code']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            X_numeric = task_features_df[numeric_cols]
            
            selected_features_mask = self.feature_selector.get_support()
            selected_feature_names = X_numeric.columns[selected_features_mask].tolist()
            
            return task_features_df[selected_feature_names]
        
        self.logger.info("Feature engineering application completed.")
        return task_features_df
