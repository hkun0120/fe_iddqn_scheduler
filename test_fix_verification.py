#!/usr/bin/env python3
"""
测试随机种子和训练循环的修复
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

def test_fix_verification():
    """测试修复是否有效"""
    logger.info("开始测试修复效果...")
    
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
    
    # 测试多个episode，检查数据是否不同
    episode_data = []
    
    for episode in range(3):
        logger.info(f"\n{'='*50}")
        logger.info(f"测试 Episode {episode + 1}")
        logger.info(f"{'='*50}")
        
        # 重置环境
        simulator.reset()
        logger.info(f"重置后进程数量: {len(simulator.successful_processes)}")
        
        # 记录前5个进程ID
        first_five_ids = []
        for i in range(min(5, len(simulator.successful_processes))):
            process = simulator.successful_processes.iloc[i]
            first_five_ids.append(process['id'])
            logger.info(f"  进程 {i}: {process['id']}")
        
        episode_data.append({
            'episode': episode + 1,
            'process_count': len(simulator.successful_processes),
            'first_five_ids': first_five_ids
        })
        
        # 快速测试episode执行
        step_count = 0
        while not simulator.is_done() and step_count < 30:
            step_count += 1
            try:
                state, reward, done, info = simulator.step(0)
                if done:
                    logger.info(f"  Episode在第{step_count}步完成")
                    break
            except Exception as e:
                logger.error(f"  Step {step_count} 执行失败: {e}")
                break
    
    # 分析结果
    logger.info(f"\n{'='*50}")
    logger.info(f"修复效果分析")
    logger.info(f"{'='*50}")
    
    # 检查进程数量是否一致
    process_counts = [data['process_count'] for data in episode_data]
    if len(set(process_counts)) == 1:
        logger.info(f"✅ 所有Episode的进程数量一致: {process_counts[0]}")
    else:
        logger.warning(f"⚠️  Episode进程数量不一致: {process_counts}")
    
    # 检查进程ID是否不同
    all_ids = []
    for data in episode_data:
        all_ids.extend(data['first_five_ids'])
    
    unique_ids = set(all_ids)
    total_ids = len(all_ids)
    
    logger.info(f"总进程ID数量: {total_ids}")
    logger.info(f"唯一进程ID数量: {len(unique_ids)}")
    logger.info(f"重复率: {(1 - len(unique_ids) / total_ids) * 100:.1f}%")
    
    if len(unique_ids) == total_ids:
        logger.info(f"✅ 所有Episode的进程ID都不同，修复成功！")
    elif len(unique_ids) > total_ids * 0.8:
        logger.info(f"✅ 大部分Episode的进程ID不同，修复基本成功！")
    else:
        logger.warning(f"⚠️  很多Episode的进程ID重复，修复可能不完整")
    
    # 检查episode计数器
    if hasattr(simulator, 'episode_count'):
        logger.info(f"✅ Episode计数器正常工作，当前值: {simulator.episode_count}")
    else:
        logger.error(f"❌ Episode计数器未找到")
    
    logger.info(f"\n测试完成！")

if __name__ == "__main__":
    test_fix_verification()
