#!/usr/bin/env python3
"""检查 task_instance 表结构"""
import sys
sys.path.insert(0, '/Users/hong/Documents/GitHub/fe_iddqn_scheduler')
from data.mysql_data_loader import MySQLDataLoader

loader = MySQLDataLoader(host='localhost', user='root', password='', database='whalesb', port=3306)
data = loader.load_all_data()

# 293718 的 process_definition_code
p = data['process_instance'][data['process_instance']['id'] == 293718].iloc[0]
print('293718 的 process_definition_code:', p['process_definition_code'])

# 查看 t_ds_task_instance 表结构
tasks = data['task_instance'][data['task_instance']['process_instance_id'] == 293718]
print()
print('task_instance 表的列:')
print(tasks.columns.tolist())

# 检查是否有 task_definition_code 列
if 'task_definition_code' in tasks.columns:
    print()
    print('task_definition_code 列存在！')
    print('前10个任务的对比:')
    for _, t in tasks[tasks['state']==7].head(10).iterrows():
        name = t['name'][:30]
        tc = t['task_code']
        tdc = t.get('task_definition_code', 'N/A')
        print(f'  {name:30} task_code={tc:15} task_definition_code={tdc}')
