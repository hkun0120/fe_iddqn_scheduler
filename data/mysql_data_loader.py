import pandas as pd
import pymysql
from sqlalchemy import create_engine
import logging

class MySQLDataLoader:
    """从MySQL数据库加载数据"""
    
    def __init__(self, host='localhost', user='root', password='', database='whalesb', port=3306):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # 创建数据库连接
        self.connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        self.engine = create_engine(self.connection_string)
        
        # 数据缓存
        self.process_definition_df = None
        self.process_instance_df = None
        self.task_definition_df = None
        self.task_instance_df = None
        self.process_task_relation_df = None
        self.project_df = None
        self.relation_process_instance_df = None
    
    def load_all_data(self):
        """从MySQL数据库加载所有数据"""
        self.logger.info(f"正在从MySQL数据库 {self.database} 加载数据...")
        
        try:
            # 加载各个表的数据
            self.logger.info("加载 t_ds_process_definition...")
            self.process_definition_df = pd.read_sql("SELECT * FROM t_ds_process_definition", self.engine)
            
            self.logger.info("加载 t_ds_process_instance...")
            self.process_instance_df = pd.read_sql("SELECT * FROM t_ds_process_instance", self.engine)
            
            self.logger.info("加载 t_ds_task_definition...")
            self.task_definition_df = pd.read_sql("SELECT * FROM t_ds_task_definition", self.engine)
            
            self.logger.info("加载 t_ds_task_instance...")
            self.task_instance_df = pd.read_sql("SELECT * FROM t_ds_task_instance", self.engine)
            
            self.logger.info("加载 t_ds_process_task_relation...")
            self.process_task_relation_df = pd.read_sql("SELECT * FROM t_ds_process_task_relation", self.engine)
            
            self.logger.info("加载 t_ds_project...")
            self.project_df = pd.read_sql("SELECT * FROM t_ds_project", self.engine)
            
            self.logger.info("加载 t_ds_relation_process_instance...")
            self.relation_process_instance_df = pd.read_sql("SELECT * FROM t_ds_relation_process_instance", self.engine)
            
            self.logger.info("所有数据加载成功！")
            
            # 打印数据统计信息
            self._print_data_statistics()
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
        
        return {
            'process_definition': self.process_definition_df,
            'process_instance': self.process_instance_df,
            'task_definition': self.task_definition_df,
            'task_instance': self.task_instance_df,
            'process_task_relation': self.process_task_relation_df,
            'project': self.project_df,
            'relation_process_instance': self.relation_process_instance_df
        }
    
    def _print_data_statistics(self):
        """打印数据统计信息"""
        self.logger.info("=" * 60)
        self.logger.info("数据统计信息:")
        self.logger.info("=" * 60)
        
        tables = [
            ('process_definition', self.process_definition_df),
            ('process_instance', self.process_instance_df),
            ('task_definition', self.task_definition_df),
            ('task_instance', self.task_instance_df),
            ('process_task_relation', self.process_task_relation_df),
            ('project', self.project_df),
            ('relation_process_instance', self.relation_process_instance_df)
        ]
        
        for table_name, df in tables:
            if df is not None:
                self.logger.info(f"  {table_name}: {len(df)} 条记录, {len(df.columns)} 个字段")
        
        # 任务类型分布
        if self.task_instance_df is not None:
            self.logger.info("\n任务类型分布:")
            task_type_counts = self.task_instance_df['task_type'].value_counts()
            for task_type, count in task_type_counts.head(10).items():
                self.logger.info(f"  {task_type}: {count} 个任务")
        
        # 进程状态分布
        if self.process_instance_df is not None:
            self.logger.info("\n进程状态分布:")
            state_counts = self.process_instance_df['state'].value_counts()
            for state, count in state_counts.items():
                self.logger.info(f"  状态 {state}: {count} 个进程")
        
        self.logger.info("=" * 60)
    
    def get_dataframes(self):
        """返回所有已加载的DataFrame"""
        return {
            'process_definition': self.process_definition_df,
            'process_instance': self.process_instance_df,
            'task_definition': self.task_definition_df,
            'task_instance': self.task_instance_df,
            'process_task_relation': self.process_task_relation_df,
            'project': self.project_df,
            'relation_process_instance': self.relation_process_instance_df
        }
    
    def get_successful_processes(self):
        """获取成功的进程实例"""
        if self.process_instance_df is None:
            return pd.DataFrame()
        
        # 状态7表示成功
        successful_processes = self.process_instance_df[
            self.process_instance_df['state'] == 7
        ].copy()
        
        self.logger.info(f"成功进程数量: {len(successful_processes)}")
        return successful_processes
    
    def get_tasks_for_process(self, process_id):
        """获取指定进程的所有任务"""
        if self.task_instance_df is None:
            return pd.DataFrame()
        
        tasks = self.task_instance_df[
            self.task_instance_df['process_instance_id'] == process_id
        ].copy()
        
        return tasks
    
    def load_task_instances_by_workflow(self, workflow_id):
        """根据工作流ID加载任务实例数据"""
        try:
            if self.task_instance_df is None:
                self.logger.info("任务实例数据未加载，正在加载...")
                self.task_instance_df = pd.read_sql("SELECT * FROM t_ds_task_instance", self.engine)
                self.logger.info(f"任务实例数据加载完成，共 {len(self.task_instance_df)} 条记录")
            
            # 筛选指定工作流的任务实例
            tasks = self.task_instance_df[
                self.task_instance_df['process_instance_id'] == workflow_id
            ].copy()
            
            if tasks.empty:
                self.logger.warning(f"未找到工作流 {workflow_id} 的任务实例")
                return []
            
            # 转换为字典列表
            task_list = []
            for _, task in tasks.iterrows():
                task_dict = task.to_dict()
                task_list.append(task_dict)
            
            self.logger.info(f"成功加载工作流 {workflow_id} 的 {len(task_list)} 个任务实例")
            return task_list
            
        except Exception as e:
            self.logger.error(f"加载工作流 {workflow_id} 的任务实例失败: {e}")
            return []
    
    def get_process_dependencies(self, process_definition_id):
        """获取指定流程的依赖关系"""
        if self.process_task_relation_df is None:
            return pd.DataFrame()
        
        dependencies = self.process_task_relation_df[
            self.process_task_relation_df['process_definition_code'] == process_definition_id
        ].copy()
        
        return dependencies
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
