#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查找低串行度的工作流实例（修复版本）
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def find_low_parallelism_workflows():
    """查找串行度很低（并行度很高）的工作流"""
    
    # 读取数据
    print("加载工作流实例数据...")
    process_instance_df = pd.read_csv('./data/Commercial_B_t_ds_process_instance.csv', dtype={'id': 'int64'})
    
    print("加载任务实例数据...")
    task_instance_df = pd.read_csv(
        './data/Commercial_B_t_ds_task_instance.csv',
        dtype={'process_instance_id': 'int64', 'id': 'int64'},
        usecols=['id', 'name', 'process_instance_id', 'start_time', 'end_time', 'state'],
        low_memory=False
    )
    
    print(f"工作流实例总数: {len(process_instance_df)}")
    print(f"任务实例总数: {len(task_instance_df)}")
    
    # 分析每个工作流实例
    results = []
    
    for idx, instance in process_instance_df.iterrows():
        instance_id = instance['id']
        
        # 获取该实例的所有任务
        tasks = task_instance_df[task_instance_df['process_instance_id'] == instance_id]
        
        if len(tasks) < 2:  # 至少2个任务
            continue
        
        # 计算任务执行时间
        task_times = []
        valid_tasks = 0
        
        for t_idx, task in tasks.iterrows():
            try:
                t_start = pd.to_datetime(task['start_time'], errors='coerce')
                t_end = pd.to_datetime(task['end_time'], errors='coerce')
                
                if pd.isna(t_start) or pd.isna(t_end):
                    continue
                
                duration = (t_end - t_start).total_seconds()
                if duration >= 0:
                    task_times.append(duration)
                    valid_tasks += 1
            except Exception as e:
                pass
        
        if len(task_times) < 2:  # 至少2个有效任务
            continue
        
        # 计算整个工作流的执行时间
        try:
            start_time = pd.to_datetime(instance['start_time'], errors='coerce')
            end_time = pd.to_datetime(instance['end_time'], errors='coerce')
            
            if pd.isna(start_time) or pd.isna(end_time):
                continue
            
            total_time = (end_time - start_time).total_seconds()
            
            if total_time <= 0:
                continue
        except Exception as e:
            continue
        
        total_work = sum(task_times)
        num_tasks = len(task_times)
        
        # 跳过总工作量为0的
        if total_work <= 0:
            continue
        
        # 串行度 = 总执行时间 / 总工作量（越小越好，说明并行度越高）
        # 理论最小值 = 1（完全并行）
        serialism_degree = total_time / total_work if total_work > 0 else 0
        
        # 并行度 = 总工作量 / 总执行时间（越大越好）
        parallelism = total_work / total_time if total_time > 0 else 0
        
        results.append({
            'instance_id': instance_id,
            'name': str(instance['name'])[:40],  # 截断名字
            'num_tasks': num_tasks,
            'total_time': total_time,
            'total_work': total_work,
            'serialism_degree': serialism_degree,
            'parallelism': parallelism,
            'avg_task_time': total_work / num_tasks if num_tasks > 0 else 0
        })
    
    if len(results) == 0:
        print("未找到有效的工作流实例！")
        return None
    
    # 排序：找串行度最低的（并行度最高）
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('serialism_degree')
    
    print(f"\n{'='*110}")
    print(f"工作流实例分析 - 按串行度排序（最低 = 最高并行度）")
    print(f"{'='*110}\n")
    
    print(f"{'实例ID':<12} {'任务数':<8} {'执行时间(s)':<12} {'总工作量(s)':<12} {'串行度':<10} {'并行度':<10} {'名称':<30}")
    print("-" * 110)
    
    for idx, row in results_df.head(20).iterrows():
        print(f"{int(row['instance_id']):<12} {int(row['num_tasks']):<8} {row['total_time']:<12.2f} {row['total_work']:<12.2f} {row['serialism_degree']:<10.2f} {row['parallelism']:<10.2f} {row['name']:<30}")
    
    print(f"\n{'='*110}")
    print(f"找到低串行度实例TOP5详细信息")
    print(f"{'='*110}\n")
    
    for i, (idx, row) in enumerate(results_df.head(5).iterrows(), 1):
        print(f"{i}. 实例ID: {int(row['instance_id'])}")
        print(f"   名称: {row['name']}")
        print(f"   任务数: {int(row['num_tasks'])}")
        print(f"   执行时间: {row['total_time']:.2f}秒")
        print(f"   总工作量: {row['total_work']:.2f}秒")
        print(f"   串行度: {row['serialism_degree']:.2f} (越低越高并行)")
        print(f"   并行度: {row['parallelism']:.2f} (越高越好)")
        print(f"   平均任务时长: {row['avg_task_time']:.2f}秒\n")
    
    # 返回最低串行度的实例ID
    if len(results_df) > 0:
        return int(results_df.iloc[0]['instance_id'])
    return None

if __name__ == '__main__':
    best_instance_id = find_low_parallelism_workflows()
    if best_instance_id:
        print(f"最低串行度实例ID: {best_instance_id}")
