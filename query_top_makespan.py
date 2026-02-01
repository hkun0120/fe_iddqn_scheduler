#!/usr/bin/env python3
"""查询前10个Makespan最大的工作流"""

import pandas as pd
import sys
sys.path.insert(0, '.')

from data.mysql_data_loader import MySQLDataLoader

# 加载数据
loader = MySQLDataLoader(host='localhost', user='root', password='', database='whalesb', port=3306)
data = loader.load_all_data()

# 获取成功的进程实例
successful = data['process_instance'][data['process_instance']['state'] == 7].copy()

# 计算每个工作流的执行时间
results = []
for _, proc in successful.iterrows():
    try:
        start = pd.to_datetime(proc['start_time'])
        end = pd.to_datetime(proc['end_time'])
        duration = (end - start).total_seconds()
        
        # 获取任务数
        tasks = data['task_instance'][data['task_instance']['process_instance_id'] == proc['id']]
        task_count = len(tasks[tasks['state'] == 7])
        
        results.append({
            'process_id': proc['id'],
            'name': proc['name'][:50] if proc['name'] else 'N/A',
            'makespan_seconds': duration,
            'task_count': task_count
        })
    except:
        pass

# 排序并取前10
df = pd.DataFrame(results)
df = df.sort_values('makespan_seconds', ascending=False).head(10)

print('\n前10个Makespan最大的工作流：')
print('=' * 100)
print(f'{"ID":>10} | {"Makespan(秒)":>12} | {"时分秒":>12} | {"任务数":>6} | 名称')
print('-' * 100)

for _, row in df.iterrows():
    secs = int(row['makespan_seconds'])
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    time_str = f"{h:02d}:{m:02d}:{s:02d}"
    print(f'{row["process_id"]:>10} | {secs:>12} | {time_str:>12} | {row["task_count"]:>6} | {row["name"]}')

print('=' * 100)
