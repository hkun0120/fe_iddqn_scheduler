#!/usr/bin/env python3
"""检查任务实例和依赖关系的code是否匹配"""

import pandas as pd
import sys
sys.path.insert(0, '/Users/hong/Documents/GitHub/fe_iddqn_scheduler')
from data.mysql_data_loader import MySQLDataLoader

loader = MySQLDataLoader(host='localhost', user='root', password='', database='whalesb', port=3306)
data = loader.load_all_data()

# 293718 的任务实例
tasks = data['task_instance'][data['task_instance']['process_instance_id'] == 293718]
successful = tasks[tasks['state'] == 7]

# 获取所有成功任务的 task_code
instance_codes = set(successful['task_code'].dropna().astype(int))
print(f'任务实例中的 task_code 数量: {len(instance_codes)}')

# 获取 process_definition_code
process = data['process_instance'][data['process_instance']['id'] == 293718].iloc[0]
pdc = process['process_definition_code']

# 依赖关系中的 code
deps = data['process_task_relation'][data['process_task_relation']['process_definition_code'] == pdc]
dep_pre_codes = set(deps['pre_task_code'].dropna().astype(int)) - {0}
dep_post_codes = set(deps['post_task_code'].dropna().astype(int))
all_dep_codes = dep_pre_codes | dep_post_codes
print(f'依赖关系中的 code 数量: {len(all_dep_codes)}')

# 检查重叠
overlap = instance_codes & all_dep_codes
print(f'重叠数量: {len(overlap)}')

# 只在实例中的
only_in_instance = instance_codes - all_dep_codes
print(f'只在实例中的: {len(only_in_instance)}')

# 只在依赖中的
only_in_deps = all_dep_codes - instance_codes
print(f'只在依赖中的: {len(only_in_deps)}')

if only_in_deps:
    # 查看这些 code 对应的任务名称
    task_defs = data.get('task_definition')
    if task_defs is not None:
        code_to_name = dict(zip(task_defs['code'], task_defs['name']))
        code_to_type = dict(zip(task_defs['code'], task_defs['task_type']))
        print('\n只在依赖中但不在实例中的任务:')
        for code in list(only_in_deps)[:20]:
            print(f'  {code}: [{code_to_type.get(code, "N/A")}] {code_to_name.get(code, "N/A")}')
