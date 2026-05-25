#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查找任务数多且并行度高的工作流实例
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 加载数据
print("加载数据...")
pi_df = pd.read_csv('./data/Commercial_B_t_ds_process_instance.csv')
ti_df = pd.read_csv('./data/Commercial_B_t_ds_task_instance.csv', 
                    usecols=['id', 'name', 'process_instance_id', 'start_time', 'end_time'])

print(f"工作流实例范围: {pi_df['id'].min()} - {pi_df['id'].max()}")
print(f"292677是否存在: {292677 in pi_df['id'].values}")

# 找任务数多的工作流
task_counts = ti_df.groupby('process_instance_id').size().reset_index(name='task_count')
task_counts = task_counts.sort_values('task_count', ascending=False).head(30)

print('\n任务数最多的工作流实例:')
print(f"{'实例ID':<12} {'任务数':<8} {'名称'}")
print('-' * 90)

for _, row in task_counts.iterrows():
    inst = pi_df[pi_df['id'] == row['process_instance_id']]
    if len(inst) > 0:
        name = str(inst.iloc[0]['name'])[:60]
        print(f"{row['process_instance_id']:<12} {row['task_count']:<8} {name}")

# 分析具体工作流的并行度
print('\n' + '='*90)
print('分析并行度')
print('='*90)

results = []
for _, row in task_counts.iterrows():
    inst_id = row['process_instance_id']
    inst = pi_df[pi_df['id'] == inst_id]
    if len(inst) == 0:
        continue
    
    inst = inst.iloc[0]
    tasks = ti_df[ti_df['process_instance_id'] == inst_id]
    
    # 计算时间
    try:
        wf_start = pd.to_datetime(inst['start_time'])
        wf_end = pd.to_datetime(inst['end_time'])
        total_time = (wf_end - wf_start).total_seconds()
        
        if total_time <= 0:
            continue
        
        # 任务总工作量
        total_work = 0
        for _, task in tasks.iterrows():
            t_start = pd.to_datetime(task['start_time'], errors='coerce')
            t_end = pd.to_datetime(task['end_time'], errors='coerce')
            if pd.notna(t_start) and pd.notna(t_end):
                duration = (t_end - t_start).total_seconds()
                if duration >= 0:
                    total_work += duration
        
        if total_work > 0:
            parallelism = total_work / total_time
            serialism = total_time / total_work
            results.append({
                'id': inst_id,
                'name': str(inst['name'])[:40],
                'tasks': row['task_count'],
                'total_time': total_time,
                'total_work': total_work,
                'parallelism': parallelism,
                'serialism': serialism
            })
    except:
        pass

# 按并行度排序
results = sorted(results, key=lambda x: x['parallelism'], reverse=True)

print(f"\n{'实例ID':<12} {'任务数':<8} {'执行时间':<12} {'总工作量':<12} {'并行度':<10} {'串行度':<10}")
print('-' * 80)
for r in results[:10]:
    print(f"{r['id']:<12} {r['tasks']:<8} {r['total_time']:<12.1f} {r['total_work']:<12.1f} {r['parallelism']:<10.2f} {r['serialism']:<10.4f}")

if results:
    best = results[0]
    print(f"\n推荐测试实例: {best['id']} (并行度: {best['parallelism']:.2f})")
