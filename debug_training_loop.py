#!/usr/bin/env python3
"""
调试训练循环和episode重置问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.historical_replay_simulator import HistoricalReplaySimulator
from data.data_loader import DataLoader
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_training_loop():
    """调试训练循环"""
    logger.info("开始调试训练循环...")
    
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
    
    # 模拟训练循环
    for episode in range(3):  # 模拟3个episode
        logger.info(f"\n{'='*50}")
        logger.info(f"Episode {episode + 1} 开始")
        logger.info(f"{'='*50}")
        
        # 重置环境
        simulator.reset()
        logger.info(f"环境重置完成")
        logger.info(f"重置后进程数量: {len(simulator.successful_processes)}")
        logger.info(f"重置后进程索引: {simulator.current_process_idx}")
        logger.info(f"重置后任务索引: {simulator.current_task_idx}")
        
        # 检查前几个进程ID是否相同
        if episode > 0:
            logger.info(f"前5个进程ID:")
            for i in range(min(5, len(simulator.successful_processes))):
                process = simulator.successful_processes.iloc[i]
                logger.info(f"  进程 {i}: {process['id']}")
        
        # 模拟episode执行
        step_count = 0
        max_steps = 50  # 限制步数观察
        
        while not simulator.is_done() and step_count < max_steps:
            step_count += 1
            
            # 检查当前状态
            if simulator.current_process_idx < len(simulator.successful_processes):
                current_process = simulator.successful_processes.iloc[simulator.current_process_idx]
                
                if step_count % 10 == 0:  # 每10步输出一次
                    logger.info(f"  Step {step_count}: 进程 {simulator.current_process_idx} ({current_process['id']})")
                
                if hasattr(simulator, 'current_process_tasks') and simulator.current_process_tasks is not None:
                    if simulator.current_task_idx < len(simulator.current_process_tasks):
                        current_task = simulator.current_process_tasks.iloc[simulator.current_task_idx]
                        if step_count % 10 == 0:
                            logger.info(f"    任务: {current_task['name']}")
                    else:
                        if step_count % 10 == 0:
                            logger.info(f"    任务索引超出范围")
            else:
                logger.info(f"  Step {step_count}: 所有进程已完成")
                break
            
            # 执行step
            try:
                state, reward, done, info = simulator.step(0)  # 使用action 0
                
                if done:
                    logger.info(f"  🎯 Episode在第{step_count}步完成")
                    break
                    
            except Exception as e:
                logger.error(f"  Step {step_count} 执行失败: {e}")
                break
        
        # Episode结束统计
        logger.info(f"\nEpisode {episode + 1} 结束统计:")
        logger.info(f"  总步数: {step_count}")
        logger.info(f"  已完成任务数: {len(simulator.completed_tasks)}")
        logger.info(f"  最终进程索引: {simulator.current_process_idx}")
        logger.info(f"  是否完成: {simulator.is_done()}")
        
        # 检查数据是否被重新采样
        if episode > 0:
            logger.info(f"  数据采样检查:")
            logger.info(f"    进程数量: {len(simulator.successful_processes)}")
            logger.info(f"    总任务数: {len(simulator.task_instances[simulator.task_instances['process_instance_id'].isin(simulator.successful_processes['id'])])}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"训练循环调试完成")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    debug_training_loop()
