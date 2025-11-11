#!/usr/bin/env python3
"""
FE-IDDQN训练系统配置文件
"""

import os
from pathlib import Path

class TrainingConfig:
    """训练配置类"""
    
    # 基础配置
    OUTPUT_DIR = "fe_iddqn_training_system"
    LOGS_DIR = "logs"
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    DATA_DIR = "data"
    
    # 数据集配置
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    RANDOM_SEED = 42
    
    # 工作流大小分类
    WORKFLOW_SIZE_BINS = [0, 10, 30, float('inf')]
    WORKFLOW_SIZE_LABELS = ['small', 'medium', 'large']
    
    # 训练配置
    N_EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    
    # 早停配置
    PATIENCE = 20
    MIN_DELTA = 0.001
    
    # 评估配置
    MAX_STEPS_PER_EPISODE = 1000
    EVAL_EPISODES_PER_SIZE = 10
    
    # 模型配置
    TASK_INPUT_DIM = 16
    RESOURCE_INPUT_DIM = 7
    ACTION_DIM = 6
    
    # 数据库配置
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'whalesb',
        'port': 3306
    }
    
    # 奖励函数权重
    REWARD_WEIGHTS = {
        'base_reward': 1.0,
        'makespan_weight': 10.0,
        'balance_weight': 2.0,
        'load_balance_weight': 1.0,
        'priority_weight': 0.1
    }
    
    @classmethod
    def get_output_path(cls, subdir=""):
        """获取输出路径"""
        base_path = Path(cls.OUTPUT_DIR)
        if subdir:
            return base_path / subdir
        return base_path
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.get_output_path(),
            cls.get_output_path(cls.LOGS_DIR),
            cls.get_output_path(cls.MODELS_DIR),
            cls.get_output_path(cls.RESULTS_DIR),
            cls.get_output_path(cls.DATA_DIR)
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name, is_best=False):
        """获取模型保存路径"""
        models_dir = cls.get_output_path(cls.MODELS_DIR)
        if is_best:
            return models_dir / f"fe_iddqn_best_{model_name}.pkl"
        else:
            return models_dir / f"fe_iddqn_{model_name}.pkl"
    
    @classmethod
    def get_log_path(cls, log_name):
        """获取日志文件路径"""
        logs_dir = cls.get_output_path(cls.LOGS_DIR)
        return logs_dir / f"{log_name}.log"
    
    @classmethod
    def get_result_path(cls, result_name):
        """获取结果文件路径"""
        results_dir = cls.get_output_path(cls.RESULTS_DIR)
        return results_dir / f"{result_name}.json"
