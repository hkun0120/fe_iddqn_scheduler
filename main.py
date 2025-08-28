#!/usr/bin/env python3
"""
FE-IDDQN 调度算法主运行脚本
运行FE-IDDQN算法并与基线算法进行比较
"""

import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Config
from data.data_loader import DataLoader
from experiments.experiment_runner import ExperimentRunner
from experiments.fair_comparison_runner import FairComparisonRunner
from utils.logger import setup_logger


def main():
    """主函数"""
    # 设置日志
    setup_logger()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("FE-IDDQN 工作流调度算法实验")
    logger.info("=" * 80)
    
    try:
        # 1. 加载数据
        logger.info("正在加载数据...")
        data_loader = DataLoader("data/raw_data")
        data = data_loader.load_all_data()
        
        if not data:
            logger.error("数据加载失败！")
            return
        
        logger.info(f"数据加载成功！")
        logger.info(f"  进程实例: {len(data['process_instance'])} 条")
        logger.info(f"  任务实例: {len(data['task_instance'])} 条")
        logger.info(f"  任务定义: {len(data['task_definition'])} 条")
        logger.info(f"  任务关系: {len(data['process_task_relation'])} 条")
        
        # 2. 创建实验运行器
        logger.info("\n正在初始化实验运行器...")
        output_dir = "results/experiments"
        os.makedirs(output_dir, exist_ok=True)
        
        experiment_runner = ExperimentRunner(
            data=data,
            features=None,  # 不使用预计算特征
            output_dir=output_dir,
            n_experiments=1  # 先运行1次实验
        )
        
        # 3. 运行FE-IDDQN算法
        logger.info("\n" + "="*60)
        logger.info("运行 FE-IDDQN 算法")
        logger.info("="*60)
        
        fe_iddqn_results = experiment_runner.run_algorithm(
            algorithm_name="FE_IDDQN",
            use_historical_replay=True
        )
        
        logger.info(f"FE-IDDQN 算法完成！")
        logger.info(f"  最终makespan: {fe_iddqn_results.get('makespan', 'N/A')}")
        
        resource_util = fe_iddqn_results.get('resource_utilization', 'N/A')
        if isinstance(resource_util, (int, float)):
            logger.info(f"  资源利用率: {resource_util:.2f}")
        else:
            logger.info(f"  资源利用率: {resource_util}")
            
        avg_reward = fe_iddqn_results.get('average_reward', 'N/A')
        if isinstance(avg_reward, (int, float)):
            logger.info(f"  平均奖励: {avg_reward:.2f}")
        else:
            logger.info(f"  平均奖励: {avg_reward}")
        
        # 4. 运行公平比较实验
        logger.info("\n" + "="*60)
        logger.info("运行公平比较实验")
        logger.info("="*60)
        
        # 创建公平比较运行器
        fair_comparison_runner = FairComparisonRunner(
            data=data,
            output_dir=output_dir
        )
        
        # 定义要比较的算法
        algorithms_to_compare = [
            "FE_IDDQN",  # 我们的算法
            "FIFO",      # 先进先出
            "SJF",       # 最短作业优先
            "HEFT",      # 异构最早完成时间
            "DQN",       # 深度Q网络
            "DDQN",      # 双重深度Q网络
            "GA",        # 遗传算法
            "PSO"        # 粒子群优化
        ]
        
        # 运行公平比较实验
        try:
            logger.info("开始公平比较实验...")
            logger.info("注意：所有算法将使用相同的工作流实例集，确保公平比较")
            
            comparison_results = fair_comparison_runner.run_algorithm_comparison(
                algorithms=algorithms_to_compare,
                n_experiments=3  # 每个算法运行3次
            )
            
            logger.info("公平比较实验完成！")
            
        except Exception as e:
            logger.error(f"公平比较实验失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果公平比较失败，回退到原来的比较方式
            logger.info("回退到原始比较方式...")
            baseline_algorithms = [
                "FIFO", "SJF", "HEFT", "DQN", "DDQN", "GA", "PSO"
            ]
            
            comparison_results = {}
            for algorithm in baseline_algorithms:
                try:
                    logger.info(f"\n正在运行 {algorithm} 算法...")
                    result = experiment_runner.run_algorithm(
                        algorithm_name=algorithm,
                        use_historical_replay=True
                    )
                    comparison_results[algorithm] = result
                    
                    logger.info(f"{algorithm} 完成:")
                    logger.info(f"  Makespan: {result.get('makespan', 'N/A')}")
                    logger.info(f"  资源利用率: {result.get('resource_utilization', 'N/A'):.2f}")
                    
                except Exception as e:
                    logger.error(f"{algorithm} 算法运行失败: {e}")
                    comparison_results[algorithm] = {"error": str(e)}
            
            # 添加FE-IDDQN结果到比较中
            comparison_results["FE_IDDQN"] = fe_iddqn_results
        
        # 5. 结果比较
        logger.info("\n" + "="*60)
        logger.info("算法性能比较")
        logger.info("="*60)
        
        # 按makespan排序（越小越好）
        valid_results = {k: v for k, v in comparison_results.items() 
                        if isinstance(v, dict) and 'makespan' in v and not isinstance(v.get('makespan'), str)}
        
        if valid_results:
            sorted_algorithms = sorted(valid_results.keys(), 
                                     key=lambda x: valid_results[x]['makespan'])
            
            logger.info("算法性能排名 (按makespan排序，越小越好):")
            for i, algorithm in enumerate(sorted_algorithms, 1):
                result = valid_results[algorithm]
                logger.info(f"  {i}. {algorithm}: Makespan={result['makespan']:.2f}, "
                           f"资源利用率={result.get('resource_utilization', 0):.2f}")
        else:
            logger.warning("没有有效的比较结果")
        
        # 6. 保存结果
        logger.info(f"\n结果已保存到: {output_dir}")
        
        logger.info("\n" + "="*80)
        logger.info("实验完成！")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

