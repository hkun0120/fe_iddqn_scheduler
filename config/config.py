import os
from pathlib import Path

class Config:
    """项目配置类"""
    
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 数据路径
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw_data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # 结果输出路径
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = RESULTS_DIR / "logs"
    MODELS_DIR = RESULTS_DIR / "models"
    FIGURES_DIR = RESULTS_DIR / "figures"
    TABLES_DIR = RESULTS_DIR / "tables"
    
    # 实验配置路径
    EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
    EXPERIMENT_CONFIGS_DIR = EXPERIMENTS_DIR / "experiment_configs"
    
    # 数据文件名
    DATA_FILES = {
        'process_definition': 'oceanbase_t_ds_process_definition.csv',
        'process_instance': 'gaussdb_t_ds_process_instance_a.csv',
        'task_definition': 'oceanbase_t_ds_task_definition.csv',
        'task_instance': 'gaussdb_t_ds_task_instance_a.csv',
        'process_task_relation': 'oceanbase_t_ds_process_task_relation.csv'
    }
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # 数据集划分比例
    TRAIN_RATIO = 0.7
    VALIDATION_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 数据处理配置
    MAX_PROCESSES_PER_EPISODE = int(os.getenv('MAX_PROCESSES_PER_EPISODE', 1000))  # 每个episode最大进程数
    MAX_TASKS_PER_EPISODE = int(os.getenv('MAX_TASKS_PER_EPISODE', 10000))        # 每个episode最大任务数
    DATA_SAMPLING_RATIO = float(os.getenv('DATA_SAMPLING_RATIO', 1.0))            # 数据采样比例 (0.0-1.0)
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 并行处理
    N_JOBS = -1  # 使用所有可用CPU核心
    
    # 内存限制（GB）
    MEMORY_LIMIT = 8
    
    # 特征工程配置
    FEATURE_SELECTION_K = 50  # 选择前K个最重要的特征
    NORMALIZE_FEATURES = True
    
    # 实验配置
    N_EXPERIMENTS = 10  # 每个算法运行的次数
    EXPERIMENT_TIMEOUT = 3600  # 单个实验超时时间（秒）
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.RESULTS_DIR,
            cls.LOGS_DIR,
            cls.MODELS_DIR,
            cls.FIGURES_DIR,
            cls.TABLES_DIR,
            cls.EXPERIMENTS_DIR,
            cls.EXPERIMENT_CONFIGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_data_file_path(cls, file_key):
        """获取数据文件的完整路径"""
        if file_key not in cls.DATA_FILES:
            raise ValueError(f"Unknown data file key: {file_key}")
        
        return cls.RAW_DATA_DIR / cls.DATA_FILES[file_key]
    
    @classmethod
    def get_log_file_path(cls, log_name):
        """获取日志文件路径"""
        return cls.LOGS_DIR / f"{log_name}.log"
    
    @classmethod
    def get_model_file_path(cls, model_name):
        """获取模型文件路径"""
        return cls.MODELS_DIR / f"{model_name}.pkl"
    
    @classmethod
    def get_figure_file_path(cls, figure_name):
        """获取图表文件路径"""
        return cls.FIGURES_DIR / f"{figure_name}.png"
    
    @classmethod
    def get_table_file_path(cls, table_name):
        """获取表格文件路径"""
        return cls.TABLES_DIR / f"{table_name}.csv"

