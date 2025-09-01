#!/usr/bin/env python3
"""
调试episode循环和任务完成情况
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

def debug_episode_cycle():
    """调试episode循环"""
    logger.info("开始调试episode循环...")
    
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
    logger.info(f"总任务数: {len(simulator.task_instances[simulator.task_instances['process_instance_id'].isin(simulator.successful_processes['id'])])}")
    
    # 模拟更多step，观察episode何时完成
    logger.info(f"\n开始模拟step，观察episode循环...")
    step_count = 0
    max_steps = 200  # 增加步数观察
    
    while step_count < max_steps:
        step_count += 1
        
        # 检查当前状态
        if simulator.current_process_idx < len(simulator.successful_processes):
            current_process = simulator.successful_processes.iloc[simulator.current_process_idx]
            logger.info(f"\n--- Step {step_count} ---")
            logger.info(f"当前进程索引: {simulator.current_process_idx}/{len(simulator.successful_processes)}")
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
            logger.info(f"\n--- Step {step_count} ---")
            logger.info("所有进程已完成，episode结束")
            break
        
        # 执行step
        try:
            state, reward, done, info = simulator.step(0)  # 使用action 0
            
            # 每10步输出一次详细信息
            if step_count % 10 == 0:
                logger.info(f"Step {step_count} 结果: reward={reward:.2f}, done={done}")
                logger.info(f"任务调度信息: {info}")
                logger.info(f"已完成任务数量: {len(simulator.completed_tasks)}")
                logger.info(f"当前进程索引: {simulator.current_process_idx}")
                
                # 检查是否所有进程都完成了
                if simulator.current_process_idx >= len(simulator.successful_processes):
                    logger.info("🎯 所有进程已完成！")
                    break
            
            if done:
                logger.info(f"🎯 Episode在第{step_count}步完成")
                break
                
        except Exception as e:
            logger.error(f"Step {step_count} 执行失败: {e}")
            break
    
    # 最终统计
    logger.info(f"\n=== 最终统计 ===")
    logger.info(f"总步数: {step_count}")
    logger.info(f"已完成任务总数: {len(simulator.completed_tasks)}")
    logger.info(f"调度历史总数: {len(simulator.task_schedule_history)}")
    logger.info(f"最终进程索引: {simulator.current_process_idx}")
    logger.info(f"总进程数: {len(simulator.successful_processes)}")
    
    # 分析episode长度
    if step_count < len(simulator.successful_processes):
        logger.info(f"⚠️  Episode在第{step_count}步就结束了，少于进程数量{len(simulator.successful_processes)}")
        logger.info(f"   这可能意味着某些进程没有任务，或者任务索引问题")
    else:
        logger.info(f"✅ Episode正常完成，步数({step_count}) >= 进程数({len(simulator.successful_processes)})")
    
    # 检查每个进程的任务数量
    logger.info(f"\n各进程任务数量分布:")
    for i, process in simulator.successful_processes.iterrows():
        process_id = process['id']
        process_tasks = simulator.task_instances[
            simulator.task_instances['process_instance_id'] == process_id
        ]
        logger.info(f"  进程 {i}: {process_id} -> {len(process_tasks)} 个任务")

if __name__ == "__main__":
    debug_episode_cycle()
