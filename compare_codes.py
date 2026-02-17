#!/usr/bin/env python3
"""比较任务实例和任务定义的code"""
import sys
sys.path.insert(0, '/Users/hong/Documents/GitHub/fe_iddqn_scheduler')
from data.mysql_data_loader import MySQLDataLoader

loader = MySQLDataLoader(host='localhost', user='root', password='', database='whalesb', port=3306)
data = loader.load_all_data()

# 任务实例
tasks = data['task_instance'][data['task_instance']['process_instance_id'] == 293718]
successful = tasks[tasks['state'] == 7]

# 任务定义
task_defs = data.get('task_definition')

print('比较任务实例的 task_code 与任务定义的 code:')
print('=' * 80)

instance_to_def = {}
for _, t in successful.iterrows():
    name = t['name']
    inst_code = t['task_code']
    match = task_defs[task_defs['name'] == name]
    if len(match) > 0:
        def_code = match.iloc[0]['code']
        same = '相同' if inst_code == def_code else '不同'
        print(f'{name[:40]:40} inst={inst_code} def={def_code} [{same}]')
        instance_to_def[inst_code] = def_code
    else:
        print(f'{name[:40]:40} inst={inst_code} def=未找到')

# 统计
same_count = sum(1 for ic, dc in instance_to_def.items() if ic == dc)
diff_count = len(instance_to_def) - same_count

print()
print(f'相同的: {same_count}')
print(f'不同的: {diff_count}')
