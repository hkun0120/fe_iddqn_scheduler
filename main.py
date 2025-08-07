#!/usr/bin/env python3
"""
FE-IDDQN云工作流调度算法主程序

使用方法:
    python main.py                              # 运行完整实验
    python main.py --algorithm FE_IDDQN        # 运行特定算法
    python main.py --baseline_comparison       # 运行基线对比
    python main.py --config config.json        # 使用自定义配置
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Config
from config.hyperparameters import Hyperparameters
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer
from experiments.experiment_runner import ExperimentRunner
from utils.logger import setup_logger

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FE-IDDQN云工作流调度算法")
    
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["FE_IDDQN", "DQN", "DDQN", "BF_DDQN", "FIFO", "SJF", "HEFT", "GA", "PSO", "ACO"],
        help="指定要运行的算法"
    )
    
    parser.add_argument(
        "--baseline_comparison",
        action="store_true",
        help="运行所有基线算法对比实验"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="自定义配置文件路径"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Config.RAW_DATA_DIR),
        help="数据目录路径"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Config.RESULTS_DIR),
        help="结果输出目录路径"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="指定GPU设备ID"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=Config.RANDOM_SEED,
        help="随机种子"
    )
    
    parser.add_argument(
        "--n_experiments",
        type=int,
        default=Config.N_EXPERIMENTS,
        help="每个算法运行的实验次数"
    )
    
    return parser.parse_args()

def setup_environment(args):
    """设置运行环境"""
    # 创建必要的目录
    Config.create_directories()
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else getattr(logging, args.log_level)
    setup_logger(log_level, Config.get_log_file_path("main"))
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("FE-IDDQN云工作流调度算法启动")
    logger.info("=" * 80)
    
    # 设置随机种子
    import random
    import numpy as np
    import torch
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info(f"随机种子设置为: {args.seed}")
    
    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logger.info(f"使用GPU设备: {args.gpu}")
    else:
        logger.info("使用CPU设备")
    
    return logger

def load_and_preprocess_data(data_dir, logger):
    """加载和预处理数据"""
    logger.info("开始数据加载和预处理...")
    
    # 数据加载
    data_loader = DataLoader(data_dir)
    raw_data = data_loader.load_all_data()
    
    # 验证数据完整性
    if not data_loader.validate_data_integrity(raw_data):
        logger.error("数据完整性验证失败")
        sys.exit(1)
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_all_data(raw_data)
    
    # 特征工程
    feature_engineer = FeatureEngineer()
    features = feature_engineer.run_feature_engineering_pipeline(processed_data)
    
    logger.info("数据加载和预处理完成")
    logger.info(f"特征矩阵形状: {features.shape}")
    
    return processed_data, features

def run_single_algorithm(algorithm_name, data, features, args, logger):
    """运行单个算法"""
    logger.info(f"运行算法: {algorithm_name}")
    
    experiment_runner = ExperimentRunner(
        data=data,
        features=features,
        output_dir=args.output_dir,
        n_experiments=args.n_experiments
    )
    
    results = experiment_runner.run_algorithm(algorithm_name)
    
    logger.info(f"算法 {algorithm_name} 运行完成")
    return results

def run_baseline_comparison(data, features, args, logger):
    """运行基线对比实验"""
    logger.info("开始基线对比实验...")
    
    algorithms = ["FE_IDDQN", "DDQN", "DQN", "FIFO", "SJF", "HEFT", "GA", "PSO", "ACO"]
    
    experiment_runner = ExperimentRunner(
        data=data,
        features=features,
        output_dir=args.output_dir,
        n_experiments=args.n_experiments
    )
    
    all_results = experiment_runner.run_comparison_experiments(algorithms)
    
    # 生成对比报告
    experiment_runner.generate_comparison_report(all_results)
    
    logger.info("基线对比实验完成")
    return all_results

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置环境
    logger = setup_environment(args)
    
    try:
        # 加载和预处理数据
        data, features = load_and_preprocess_data(args.data_dir, logger)
        
        # 根据参数运行相应的实验
        if args.baseline_comparison:
            # 运行基线对比实验
            results = run_baseline_comparison(data, features, args, logger)
            
        elif args.algorithm:
            # 运行指定算法
            results = run_single_algorithm(args.algorithm, data, features, args, logger)
            
        else:
            # 默认运行FE-IDDQN算法
            logger.info("未指定算法，默认运行FE-IDDQN")
            results = run_single_algorithm("FE_IDDQN", data, features, args, logger)
        
        logger.info("=" * 80)
        logger.info("实验运行完成")
        logger.info(f"结果保存在: {args.output_dir}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("用户中断实验")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"实验运行出错: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

