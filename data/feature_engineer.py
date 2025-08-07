import pandas as pd
import numpy as np
import logging
import networkx as nx
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from config.config import Config

class FeatureEngineer:
    """DolphinScheduler数据特征工程师"""
    
    def __init__(self):
        """初始化特征工程师"""
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selector = None
        
    def create_process_graph(self, process_definitions: pd.DataFrame, 
                             process_task_relations: pd.DataFrame) -> Dict[int, nx.DiGraph]:
        """为每个流程定义创建DAG图"""
        self.logger.info("Creating process graphs...")
        graphs = {}
        
        for _, process_def in process_definitions.iterrows():
            process_code = process_def["code"]
            process_name = process_def["name"]
            
            # 过滤出当前流程的关系
            relations = process_task_relations[
                process_task_relations["process_definition_code"] == process_code
            ]
            
            graph = nx.DiGraph()
            
            # 添加节点
            # 假设所有在关系中出现的task_code都是节点
            all_task_codes = set(relations["pre_task_code"]).union(set(relations["post_task_code"])) - {0}
            for task_code in all_task_codes:
                graph.add_node(task_code)
            
            # 添加边
            for _, row in relations.iterrows():
                pre_task = row["pre_task_code"]
                post_task = row["post_task_code"]
                
                if pre_task != 0 and post_task != 0:
                    graph.add_edge(pre_task, post_task)
            
            # 检查图中是否存在环
            if not nx.is_directed_acyclic_graph(graph):
                self.logger.warning(f"Process {process_name} (code: {process_code}) contains a cycle. Skipping graph creation.")
                continue
                
            graphs[process_code] = graph
            
        self.logger.info(f"Created {len(graphs)} process graphs.")
        return graphs

    def extract_graph_features(self, graphs: Dict[int, nx.DiGraph], task_definitions: pd.DataFrame) -> pd.DataFrame:
        """从DAG图中提取特征"""
        self.logger.info("Extracting graph features...")
        
        graph_features = []
        for process_code, graph in graphs.items():
            features = {
                "process_definition_code": process_code,
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0,
                "is_dag": nx.is_directed_acyclic_graph(graph),
                "num_connected_components": nx.number_weakly_connected_components(graph)
            }
            
            # 拓扑特征
            if graph.number_of_nodes() > 0:
                # 入度/出度统计
                in_degrees = [graph.in_degree(node) for node in graph.nodes()]
                out_degrees = [graph.out_degree(node) for node in graph.nodes()]
                features["avg_in_degree"] = np.mean(in_degrees) if in_degrees else 0
                features["max_in_degree"] = np.max(in_degrees) if in_degrees else 0
                features["avg_out_degree"] = np.mean(out_degrees) if out_degrees else 0
                features["max_out_degree"] = np.max(out_degrees) if out_degrees else 0
                
                # 最长路径
                try:
                    features["longest_path_length"] = nx.dag_longest_path_length(graph)
                except nx.NetworkXError:
                    features["longest_path_length"] = 0 # 对于没有路径的图
                
                # 关键路径任务数
                # 这是一个简化的近似，实际关键路径需要任务执行时间
                features["critical_path_tasks_ratio"] = features["longest_path_length"] / features["num_nodes"] if features["num_nodes"] > 0 else 0

            else:
                features["avg_in_degree"] = 0
                features["max_in_degree"] = 0
                features["avg_out_degree"] = 0
                features["max_out_degree"] = 0
                features["longest_path_length"] = 0
                features["critical_path_tasks_ratio"] = 0

            graph_features.append(features)
            
        self.logger.info("Graph features extracted.")
        return pd.DataFrame(graph_features)

    def aggregate_task_features(self, task_definitions: pd.DataFrame, 
                                process_task_relations: pd.DataFrame) -> pd.DataFrame:
        """聚合任务定义特征到流程级别"""
        self.logger.info("Aggregating task features to process level...")
        
        # 将任务定义与关系合并，以便按流程聚合
        merged_df = pd.merge(process_task_relations, task_definitions,
                             left_on=\'post_task_code\', right_on=\'code\', how=\'left\', suffixes=('_relation', '_definition'))
        
        # 填充缺失的任务定义信息（例如，对于没有后置任务的流程）
        merged_df["task_type_encoded"] = merged_df["task_type_encoded"].fillna(-1) # -1表示未知任务类型
        merged_df["cpu_quota"] = merged_df["cpu_quota"].fillna(0)
        merged_df["memory_max"] = merged_df["memory_max"].fillna(0)
        merged_df["timeout"] = merged_df["timeout"].fillna(0)
        merged_df["task_retry_times"] = merged_df["task_retry_times"].fillna(0)

        # 聚合特征
        agg_features = merged_df.groupby("process_definition_code").agg(
            total_tasks=("code_definition", "count"),
            unique_task_types=("task_type_encoded", lambda x: x.nunique()),
            avg_cpu_quota=("cpu_quota", "mean"),
            max_cpu_quota=("cpu_quota", "max"),
            avg_memory_max=("memory_max", "mean"),
            max_memory_max=("memory_max", "max"),
            avg_timeout=("timeout", "mean"),
            max_timeout=("timeout", "max"),
            avg_task_retry_times=("task_retry_times", "mean"),
            max_task_retry_times=("task_retry_times", "max"),
            # 可以添加更多聚合特征，例如不同任务类型的计数
        ).reset_index()
        
        self.logger.info("Task features aggregated.")
        return agg_features

    def combine_features(self, *dataframes: pd.DataFrame) -> pd.DataFrame:
        """合并所有特征DataFrame"""
        self.logger.info("Combining all features...")
        
        combined_df = dataframes[0]
        for i in range(1, len(dataframes)):
            # 假设所有DataFrame都有一个共同的合并键，例如 'process_definition_code'
            # 需要根据实际情况调整合并键和合并方式
            combined_df = pd.merge(combined_df, dataframes[i], on=\'process_definition_code\', how=\'left\')
            
        self.logger.info(f"Combined features shape: {combined_df.shape}")
        return combined_df

    def normalize_features(self, df: pd.DataFrame, columns_to_normalize: Optional[List[str]] = None) -> pd.DataFrame:
        """对数值特征进行标准化"""
        self.logger.info("Normalizing features...")
        
        df_normalized = df.copy()
        if columns_to_normalize is None:
            # 默认对所有数值型特征进行标准化，排除ID列
            columns_to_normalize = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
            if 'process_definition_code' in columns_to_normalize:
                columns_to_normalize.remove('process_definition_code')
        
        for col in columns_to_normalize:
            if col in df_normalized.columns:
                scaler = StandardScaler()
                df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
                self.scalers[col] = scaler # 保存scaler以便后续对新数据进行转换
        
        self.logger.info("Features normalized.")
        return df_normalized

    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = Config.FEATURE_SELECTION_K) -> Tuple[pd.DataFrame, List[str]]:
        """特征选择 (SelectKBest)"""
        self.logger.info(f"Selecting top {k} features...")
        
        # 确保X中不包含非数值列，或者只选择数值列进行特征选择
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols]

        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X_numeric.shape[1]))
        self.feature_selector.fit(X_numeric, y)
        
        selected_features_mask = self.feature_selector.get_support()
        selected_feature_names = X_numeric.columns[selected_features_mask].tolist()
        
        self.logger.info(f"Selected features: {selected_feature_names}")
        return X[selected_feature_names], selected_feature_names

    def run_feature_engineering_pipeline(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """运行完整的特征工程管道"""
        self.logger.info("Running feature engineering pipeline...")
        
        # 1. 创建流程图并提取图特征
        graphs = self.create_process_graph(
            data["process_definition"], data["process_task_relation"]
        )
        graph_features_df = self.extract_graph_features(graphs, data["task_definition"])
        
        # 2. 聚合任务定义特征
        aggregated_task_features_df = self.aggregate_task_features(
            data["task_definition"], data["process_task_relation"]
        )
        
        # 3. 合并所有特征
        # 假设我们主要关注流程定义级别的特征，所以以process_definition为基础合并
        # 需要确保process_definition_code是唯一的
        process_def_features = data["process_definition"].copy()
        
        # 提取process_definition中的一些基本特征
        process_def_features = process_def_features[[
            "code", "name", "version", "release_state", "create_time_hour",
            "create_time_day_of_week", "create_time_month", "has_global_params"
        ]].rename(columns={\n            "code": "process_definition_code",
            "name": "process_definition_name"
        })
        
        # 合并图特征和聚合任务特征
        combined_features = self.combine_features(
            process_def_features,
            graph_features_df,
            aggregated_task_features_df
        )
        
        # 4. 标准化特征
        if Config.NORMALIZE_FEATURES:
            # 排除非数值列和ID列
            cols_to_normalize = [col for col in combined_features.columns 
                                 if combined_features[col].dtype in [np.float64, np.int64] 
                                 and col not in ["process_definition_code", "process_definition_name", "is_dag"]]
            combined_features = self.normalize_features(combined_features, cols_to_normalize)
        
        self.logger.info("Feature engineering pipeline completed.")
        return combined_features

    def apply_feature_engineering_pipeline(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """对新数据应用特征工程管道（使用之前fit的scaler和selector）"""
        self.logger.info("Applying feature engineering pipeline to new data...")
        
        # 1. 创建流程图并提取图特征
        graphs = self.create_process_graph(
            data["process_definition"], data["process_task_relation"]
        )
        graph_features_df = self.extract_graph_features(graphs, data["task_definition"])
        
        # 2. 聚合任务定义特征
        aggregated_task_features_df = self.aggregate_task_features(
            data["task_definition"], data["process_task_relation"]
        )
        
        # 3. 合并所有特征
        process_def_features = data["process_definition"].copy()
        process_def_features = process_def_features[[
            "code", "name", "version", "release_state", "create_time_hour",
            "create_time_day_of_week", "create_time_month", "has_global_params"
        ]].rename(columns={\n            "code": "process_definition_code",
            "name": "process_definition_name"
        })
        
        combined_features = self.combine_features(
            process_def_features,
            graph_features_df,
            aggregated_task_features_df
        )
        
        # 4. 标准化特征
        if Config.NORMALIZE_FEATURES:
            cols_to_normalize = [col for col in combined_features.columns 
                                 if combined_features[col].dtype in [np.float64, np.int64] 
                                 and col not in ["process_definition_code", "process_definition_name", "is_dag"]]
            for col in cols_to_normalize:
                if col in combined_features.columns and col in self.scalers:
                    combined_features[col] = self.scalers[col].transform(combined_features[[col]])
                else:
                    self.logger.warning(f"Scaler not found for column {col}. Skipping normalization.")
        
        # 5. 特征选择
        if self.feature_selector:
            # 确保X中不包含非数值列，或者只选择数值列进行特征选择
            numeric_cols = combined_features.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = combined_features[numeric_cols]
            
            selected_features_mask = self.feature_selector.get_support()
            selected_feature_names = X_numeric.columns[selected_features_mask].tolist()
            
            return combined_features[selected_feature_names]
        
        self.logger.info("Feature engineering application completed.")
        return combined_features

