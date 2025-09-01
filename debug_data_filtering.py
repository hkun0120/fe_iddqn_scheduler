#!/usr/bin/env python3
"""
调试数据筛选逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import DataLoader
import pandas as pd
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_data_filtering():
    """调试数据筛选逻辑"""
    logger.info("开始调试数据筛选逻辑...")
    
    # 加载数据
    data_loader = DataLoader(raw_data_path="data/raw_data")
    data_dict = data_loader.load_all_data()
    task_definitions = data_dict['task_definition']
    process_task_relations = data_dict['process_task_relation']
    task_instances = data_dict['task_instance']
    process_instances = data_dict['process_instance']
    
    logger.info(f"原始数据统计:")
    logger.info(f"  进程实例总数: {len(process_instances)}")
    logger.info(f"  任务实例总数: {len(task_instances)}")
    logger.info(f"  任务定义总数: {len(task_definitions)}")
    logger.info(f"  进程任务关系总数: {len(process_task_relations)}")
    
    # 检查进程状态分布
    logger.info(f"\n进程状态分布:")
    state_counts = process_instances['state'].value_counts()
    for state, count in state_counts.items():
        logger.info(f"  状态 {state}: {count}")
    
    # 检查成功状态的进程
    successful_processes = process_instances[process_instances['state'] == 7]
    logger.info(f"\n成功状态的进程数量: {len(successful_processes)}")
    
    # 检查有任务的进程
    processes_with_tasks = task_instances['process_instance_id'].unique()
    logger.info(f"有任务的进程ID数量: {len(processes_with_tasks)}")
    
    # 检查成功且有任务的进程
    successful_with_tasks = process_instances[
        (process_instances['state'] == 7) & 
        (process_instances['id'].isin(processes_with_tasks))
    ]
    logger.info(f"成功且有任务的进程数量: {len(successful_with_tasks)}")
    
    # 检查前几个进程的任务分布
    logger.info(f"\n前5个进程的任务分布:")
    for i, process in successful_with_tasks.head(5).iterrows():
        process_id = process['id']
        process_tasks = task_instances[task_instances['process_instance_id'] == process_id]
        logger.info(f"  进程 {process_id}: {len(process_tasks)} 个任务")
        
        # 显示任务详情
        for j, task in process_tasks.head(3).iterrows():
            logger.info(f"    任务 {j}: {task['name']} (ID: {task['id']}, 类型: {task.get('task_type', 'N/A')})")
        
        if len(process_tasks) > 3:
            logger.info(f"    ... 还有 {len(process_tasks) - 3} 个任务")
    
    # 检查任务类型分布
    logger.info(f"\n任务类型分布:")
    task_type_counts = task_instances['task_type'].value_counts()
    for task_type, count in task_type_counts.head(10).items():
        logger.info(f"  {task_type}: {count}")
    
    # 检查主机分布
    logger.info(f"\n主机分布:")
    host_counts = task_instances['host'].value_counts()
    for host, count in host_counts.head(10).items():
        logger.info(f"  {host}: {count}")

if __name__ == "__main__":
    debug_data_filtering()
