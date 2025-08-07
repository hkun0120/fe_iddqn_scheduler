import pandas as pd
import os

class DataLoader:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.process_definition_df = None
        self.process_instance_df = None
        self.task_definition_df = None
        self.task_instance_df = None
        self.process_task_relation_df = None

    def load_all_data(self):
        """加载所有CSV文件"""
        print(f"Loading data from: {self.raw_data_path}")
        try:
            self.process_definition_df = pd.read_csv(os.path.join(self.raw_data_path, 'oceanbase_t_ds_process_definition.csv'))
            self.process_instance_df = pd.read_csv(os.path.join(self.raw_data_path, 'gaussdb_t_ds_process_instance_a.csv'))
            self.task_definition_df = pd.read_csv(os.path.join(self.raw_data_path, 'oceanbase_t_ds_task_definition.csv'))
            self.task_instance_df = pd.read_csv(os.path.join(self.raw_data_path, 'gaussdb_t_ds_task_instance_a.csv'))
            self.process_task_relation_df = pd.read_csv(os.path.join(self.raw_data_path, 'oceanbase_t_ds_process_task_relation.csv'))
            print("All data loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading file: {e}. Please ensure all CSV files are in the specified raw_data_path.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}")
            raise
        
        return {
            'process_definition': self.process_definition_df,
            'process_instance': self.process_instance_df,
            'task_definition': self.task_definition_df,
            'task_instance': self.task_instance_df,
            'process_task_relation': self.process_task_relation_df
        }

    def get_dataframes(self):
        """返回所有已加载的DataFrame"""
        return {
            'process_definition': self.process_definition_df,
            'process_instance': self.process_instance_df,
            'task_definition': self.task_definition_df,
            'task_instance': self.task_instance_df,
            'process_task_relation': self.process_task_relation_df
        }

    def merge_data(self):
        """合并相关数据表，建立关联关系"""
        print("Merging dataframes...")
        
        # 1. 合并任务定义和任务实例
        # 假设 task_instance.task_code 对应 task_definition.code
        # 假设 task_instance.task_definition_version 对应 task_definition.version
        # 还需要考虑 task_instance.process_instance_id 关联 process_instance.id
        # 以及 process_instance.process_definition_code 关联 process_definition.code

        # 任务实例与任务定义关联
        # 注意：这里需要根据实际的DolphinScheduler表结构来确定正确的JOIN键
        # 常见的关联方式是 task_instance.task_code = task_definition.code
        # 但如果存在版本，可能还需要 task_instance.task_definition_version = task_definition.version
        # 鉴于用户提供了dolphinscheduler_mysql.sql，我将基于此进行更精确的关联

        # 假设：
        # t_ds_process_instance (工作流实例) -> id, process_definition_code, process_definition_version
        # t_ds_process_definition (工作流定义) -> code, version
        # t_ds_task_instance (任务实例) -> id, task_code, task_definition_version, process_instance_id
        # t_ds_task_definition (任务定义) -> code, version
        # t_ds_process_task_relation (工作流任务关系) -> process_definition_code, process_definition_version, pre_task_code, post_task_code

        # 1. 合并 process_instance 和 process_definition
        merged_df = pd.merge(
            self.process_instance_df,
            self.process_definition_df,
            left_on=['process_definition_code', 'process_definition_version'],
            right_on=['code', 'version'],
            how='left',
            suffixes=('_instance', '_definition')
        )
        # 重命名冲突的列，例如 'name_definition' -> 'process_definition_name'
        merged_df.rename(columns={'name_definition': 'process_definition_name'}, inplace=True)
        merged_df.drop(columns=['code', 'version'], inplace=True) # 移除重复的code和version列

        # 2. 合并 task_instance 和 task_definition
        task_merged_df = pd.merge(
            self.task_instance_df,
            self.task_definition_df,
            left_on=['task_code', 'task_definition_version'],
            right_on=['code', 'version'],
            how='left',
            suffixes=('_instance', '_definition')
        )
        task_merged_df.rename(columns={'name_definition': 'task_definition_name'}, inplace=True)
        task_merged_df.drop(columns=['code', 'version'], inplace=True) # 移除重复的code和version列

        # 3. 合并主 merged_df (process_instance + process_definition) 和 task_merged_df (task_instance + task_definition)
        # 通过 process_instance_id 关联
        final_merged_df = pd.merge(
            merged_df,
            task_merged_df,
            left_on='id_instance',
            right_on='process_instance_id',
            how='left',
            suffixes=('_process', '_task')
        )
        # 重命名冲突的列，例如 'id_instance' -> 'process_instance_id', 'id_task' -> 'task_instance_id'
        final_merged_df.rename(columns={'id_process': 'process_instance_id', 'id_task': 'task_instance_id'}, inplace=True)

        # 4. 合并 process_task_relation
        # 这一步比较复杂，因为 relation 表描述的是 DAG 结构，需要特殊处理来构建图
        # 这里先不直接merge到主表中，而是作为单独的DAG构建数据源
        print("Data merging complete. DAG relation data is kept separate for graph construction.")
        
        return final_merged_df

# 示例用法 (在实际运行中，这些将由 run_experiments.py 调用)
# if __name__ == '__main__':
#     # 假设 raw_data 目录在当前脚本的同级目录
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     raw_data_path = os.path.join(current_dir, 'raw_data')

#     loader = DataLoader(raw_data_path)
#     loaded_data = loader.load_all_data()
    
#     # 打印每个DataFrame的前几行和信息，以验证加载
#     for name, df in loaded_data.items():
#         print(f"\n--- {name} Dataframe ---")
#         print(df.head())
#         print(df.info())

#     # 尝试合并数据
#     try:
#         merged_df = loader.merge_data()
#         print("\n--- Merged Dataframe ---")
#         print(merged_df.head())
#         print(merged_df.info())
#     except Exception as e:
#         print(f"Error during merging: {e}")



