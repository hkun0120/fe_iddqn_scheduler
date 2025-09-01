#!/usr/bin/env python3
"""
调试已完成的任务和任务名称分布
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

def debug_completed_tasks():
    """调试已完成的任务"""
    logger.info("开始调试已完成的任务...")
    
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
    
    # 检查前几个进程的任务名称分布
    logger.info(f"\n前10个进程的任务名称分布:")
    for i in range(min(10, len(simulator.successful_processes))):
        process = simulator.successful_processes.iloc[i]
        process_id = process['id']
        
        # 获取该进程的所有任务
        process_tasks = simulator.task_instances[
            simulator.task_instances['process_instance_id'] == process_id
        ]
        
        logger.info(f"\n进程 {process_id}:")
        for j, task in process_tasks.iterrows():
            logger.info(f"  任务 {j}: {task['name']} (ID: {task['id']}, 类型: {task.get('task_type', 'N/A')})")
    
    # 模拟几个step，观察任务完成情况
    logger.info(f"\n开始模拟step，观察任务完成情况...")
    for step in range(10):
        logger.info(f"\n--- Step {step} ---")
        
        # 检查当前状态
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
            logger.info(f"Step结果: reward={reward}, done={done}")
            logger.info(f"任务调度信息: {info}")
            
            # 检查已完成的任务
            logger.info(f"已完成任务数量: {len(simulator.completed_tasks)}")
            if simulator.completed_tasks:
                logger.info(f"已完成任务ID: {list(simulator.completed_tasks)[-3:]}")  # 显示最后3个
            
            # 检查调度历史
            if simulator.task_schedule_history:
                last_record = simulator.task_schedule_history[-1]
                logger.info(f"最后调度的任务: {last_record['task_name']} -> {last_record['host']}")
            
            if done:
                logger.info("Episode完成")
                break
                
        except Exception as e:
            logger.error(f"Step执行失败: {e}")
            break
    
    # 最终统计
    logger.info(f"\n=== 最终统计 ===")
    logger.info(f"已完成任务总数: {len(simulator.completed_tasks)}")
    logger.info(f"调度历史总数: {len(simulator.task_schedule_history)}")
    
    # 分析已完成任务的名称分布
    if simulator.task_schedule_history:
        logger.info(f"\n已完成任务的名称分布:")
        task_names = [record['task_name'] for record in simulator.task_schedule_history]
        from collections import Counter
        name_counts = Counter(task_names)
        
        for name, count in name_counts.most_common(10):
            logger.info(f"  {name}: {count} 次")
    
    # 检查是否有重复的任务名称
    if simulator.task_schedule_history:
        unique_names = set(record['task_name'] for record in simulator.task_schedule_history)
        total_tasks = len(simulator.task_schedule_history)
        logger.info(f"\n任务名称统计:")
        logger.info(f"  总任务数: {total_tasks}")
        logger.info(f"  唯一任务名称数: {len(unique_names)}")
        logger.info(f"  重复率: {(1 - len(unique_names) / total_tasks) * 100:.1f}%")

if __name__ == "__main__":
    debug_completed_tasks()
