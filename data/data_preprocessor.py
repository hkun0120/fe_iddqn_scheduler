import pandas as pd
import numpy as np
from datetime import datetime
import logging

class DataPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def preprocess_all_data(self, dataframes):
        """
        对所有数据表进行预处理
        
        Args:
            dataframes: dict, 包含所有DataFrame的字典
            
        Returns:
            dict: 预处理后的DataFrame字典
        """
        preprocessed_data = {}
        
        for name, df in dataframes.items():
            self.logger.info(f"Preprocessing {name} dataframe...")
            preprocessed_data[name] = self._preprocess_dataframe(df, name)
            
        return preprocessed_data
    
    def _preprocess_dataframe(self, df, table_name):
        """
        对单个DataFrame进行预处理
        
        Args:
            df: pandas.DataFrame
            table_name: str, 表名
            
        Returns:
            pandas.DataFrame: 预处理后的DataFrame
        """
        df_processed = df.copy()
        
        # 1. 处理缺失值
        df_processed = self._handle_missing_values(df_processed, table_name)
        
        # 2. 处理时间字段
        df_processed = self._handle_datetime_fields(df_processed, table_name)
        
        # 3. 处理异常值
        df_processed = self._handle_outliers(df_processed, table_name)
        
        # 4. 数据类型转换
        df_processed = self._convert_data_types(df_processed, table_name)
        
        # 5. 特定表的预处理
        df_processed = self._table_specific_preprocessing(df_processed, table_name)
        
        return df_processed
    
    def _handle_missing_values(self, df, table_name):
        """处理缺失值"""
        # 记录缺失值情况
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            self.logger.info(f"Missing values in {table_name}:")
            for col, count in missing_info[missing_info > 0].items():
                self.logger.info(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
        
        # 根据不同表采用不同的缺失值处理策略
        if table_name == 'process_instance':
            # 对于工作流实例，关键字段不能为空
            df = df.dropna(subset=['id', 'process_definition_code', 'state'])
            # 对于可选字段，用默认值填充
            if 'host' in df.columns:
                df['host'] = df['host'].fillna('unknown')
            if 'executor_id' in df.columns:
                df['executor_id'] = df['executor_id'].fillna(-1)
                
        elif table_name == 'task_instance':
            # 对于任务实例，关键字段不能为空
            df = df.dropna(subset=['id', 'task_code', 'process_instance_id', 'state'])
            # 填充可选字段
            if 'host' in df.columns:
                df['host'] = df['host'].fillna('unknown')
            if 'executor_id' in df.columns:
                df['executor_id'] = df['executor_id'].fillna(-1)
            if 'pid' in df.columns:
                df['pid'] = df['pid'].fillna(-1)
                
        elif table_name == 'process_definition':
            # 工作流定义的关键字段不能为空
            df = df.dropna(subset=['code', 'name', 'version'])
            
        elif table_name == 'task_definition':
            # 任务定义的关键字段不能为空
            df = df.dropna(subset=['code', 'name', 'version', 'task_type'])
            
        elif table_name == 'process_task_relation':
            # 任务关系的关键字段不能为空
            df = df.dropna(subset=['process_definition_code', 'process_definition_version'])
        
        return df
    
    def _handle_datetime_fields(self, df, table_name):
        """处理时间字段"""
        datetime_columns = []
        
        # 识别时间字段
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['time', 'date', 'create', 'update', 'start', 'end']):
                datetime_columns.append(col)
        
        # 转换时间字段
        for col in datetime_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    self.logger.info(f"Converted {col} to datetime in {table_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to convert {col} to datetime in {table_name}: {e}")
        
        return df
    
    def _handle_outliers(self, df, table_name):
        """处理异常值"""
        # 对于数值型字段，检测和处理异常值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['id', 'code', 'version', 'process_instance_id', 'task_instance_id']:
                # 跳过ID类字段
                continue
                
            # 使用IQR方法检测异常值
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                self.logger.info(f"Found {len(outliers)} outliers in {col} of {table_name}")
                
                # 对于执行时间等关键指标，使用截断而不是删除
                if 'time' in col.lower() or 'duration' in col.lower():
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _convert_data_types(self, df, table_name):
        """数据类型转换"""
        # 将ID类字段转换为整数
        id_columns = [col for col in df.columns if 'id' in col.lower() and col != 'task_code']
        for col in id_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                except Exception as e:
                    self.logger.warning(f"Failed to convert {col} to integer in {table_name}: {e}")
        
        # 将状态字段转换为分类变量
        if 'state' in df.columns:
            df['state'] = df['state'].astype('category')
        
        # 将任务类型转换为分类变量
        if 'task_type' in df.columns:
            df['task_type'] = df['task_type'].astype('category')
        
        return df
    
    def _table_specific_preprocessing(self, df, table_name):
        """特定表的预处理"""
        if table_name == 'task_instance':
            # 计算任务执行时间
            if 'start_time' in df.columns and 'end_time' in df.columns:
                df['execution_duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
                # 处理负数或异常的执行时间
                df['execution_duration'] = df['execution_duration'].clip(lower=0)
                
            # 添加任务状态编码
            if 'state' in df.columns:
                state_mapping = {
                    'SUCCESS': 1,
                    'FAILURE': 0,
                    'RUNNING': 2,
                    'WAITING': 3,
                    'PAUSE': 4,
                    'KILL': 5,
                    'STOP': 6
                }
                df['state_code'] = df['state'].map(state_mapping).fillna(-1)
                
        elif table_name == 'process_instance':
            # 计算工作流执行时间
            if 'start_time' in df.columns and 'end_time' in df.columns:
                df['workflow_duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
                df['workflow_duration'] = df['workflow_duration'].clip(lower=0)
                
            # 添加工作流状态编码
            if 'state' in df.columns:
                state_mapping = {
                    'SUCCESS': 1,
                    'FAILURE': 0,
                    'RUNNING': 2,
                    'WAITING': 3,
                    'PAUSE': 4,
                    'KILL': 5,
                    'STOP': 6
                }
                df['state_code'] = df['state'].map(state_mapping).fillna(-1)
        
        return df
    
    def get_data_summary(self, dataframes):
        """获取数据摘要信息"""
        summary = {}
        
        for name, df in dataframes.items():
            summary[name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
            # 添加数值型字段的统计信息
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary[name]['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        return summary

# 示例用法
# if __name__ == '__main__':
#     # 这里可以添加测试代码
#     pass

