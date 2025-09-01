#!/usr/bin/env python3
"""
调试任务进度和进程切换逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.historical_replay_simulator import HistoricalReplaySimulator
from data.data_loader import DataLoader
from config.config import Config
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_task_progression():
    """调试任务进度"""
    logger.info("开始调试任务进度...")
    
    # 加载数据
    data_loader = DataLoader(raw_data_path="data/raw_data")
    data_dict = data_loader.load_all_data()
    task_definitions = data_dict['task_definition']
    process_task_relations = data_dict['process_task_relation']
    task_instances = data_dict['task_instance']
    process_instances = data_dict['process_instance']
    
    # 创建模拟器
    simulator = HistoricalReplaySimulator(
        task_definitions=task_definitions,
        process_task_relations=process_task_relations,
        task_instances=task_instances,
        process_instances=process_instances
    )
    
    logger.info(f"初始化完成，进程数量: {len(simulator.successful_processes)}")
    logger.info(f"当前进程索引: {simulator.current_process_idx}")
    logger.info(f"当前任务索引: {simulator.current_task_idx}")
    
    # 检查第一个进程
    if len(simulator.successful_processes) > 0:
        first_process = simulator.successful_processes.iloc[0]
        logger.info(f"第一个进程: {first_process['id']}")
        
        # 获取该进程的任务
        process_tasks = simulator.task_instances[
            simulator.task_instances['process_instance_id'] == first_process['id']
        ]
        logger.info(f"第一个进程的任务数量: {len(process_tasks)}")
        
        # 显示前几个任务
        for i, task in process_tasks.head(5).iterrows():
            logger.info(f"  任务 {i}: {task['name']} (ID: {task['id']})")
    
    # 模拟几个step
    logger.info("\n开始模拟step...")
    for step in range(5):
        logger.info(f"\n--- Step {step} ---")
        
        # 检查当前状态
        logger.info(f"当前进程索引: {simulator.current_process_idx}")
        logger.info(f"当前任务索引: {simulator.current_task_idx}")
        
        if simulator.current_process_idx < len(simulator.successful_processes):
            current_process = simulator.successful_processes.iloc[simulator.current_process_idx]
            logger.info(f"当前进程: {current_process['id']}")
            
            if hasattr(simulator, 'current_process_tasks') and simulator.current_process_tasks is not None:
                logger.info(f"当前进程任务数量: {len(simulator.current_process_tasks)}")
                
                if simulator.current_task_idx < len(simulator.current_process_tasks):
                    current_task = simulator.current_process_tasks.iloc[simulator.current_task_idx]
                    logger.info(f"当前任务: {current_task['name']} (ID: {current_task['id']})")
                else:
                    logger.info("当前任务索引超出范围")
            else:
                logger.info("当前进程任务未加载")
        else:
            logger.info("所有进程已完成")
            break
        
        # 执行step
        try:
            state, reward, done, info = simulator.step(0)  # 使用action 0
            logger.info(f"Step结果: reward={reward}, done={done}, info={info}")
            
            if done:
                logger.info("Episode完成")
                break
                
        except Exception as e:
            logger.error(f"Step执行失败: {e}")
            break
    
    # 检查最终状态
    logger.info(f"\n最终状态:")
    logger.info(f"当前进程索引: {simulator.current_process_idx}")
    logger.info(f"当前任务索引: {simulator.current_task_idx}")
    logger.info(f"已完成任务数量: {len(simulator.completed_tasks)}")
    logger.info(f"调度历史数量: {len(simulator.task_schedule_history)}")

if __name__ == "__main__":
    debug_task_progression()
