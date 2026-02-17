#!/usr/bin/env python3
"""检查 task_code 与 relation 中 code 的关系"""
import sys
sys.path.insert(0, '/Users/hong/Documents/GitHub/fe_iddqn_scheduler')
from data.mysql_data_loader import MySQLDataLoader

loader = MySQLDataLoader(host='localhost', user='root', password='', database='whalesb', port=3306)
data = loader.load_all_data()

# 获取 process_task_relation (使用 293718 对应的 process_definition_code = 43714)
deps = data['process_task_relation'][data['process_task_relation']['process_definition_code'] == 43714]
print('process_task_relation 中的依赖边数:', len(deps))

# 获取 task_instance
tasks = data['task_instance'][data['task_instance']['process_instance_id'] == 293718]
successful = tasks[tasks['state'] == 7]

# 获取 task_instance 中的 task_code
inst_codes = set(successful['task_code'].dropna())
inst_codes_list = sorted([int(x) for x in inst_codes if x != 0])
print(f'task_instance 中的 task_code (共{len(inst_codes_list)}个):')
for c in inst_codes_list[:5]:
    print(f'  {c}')

# 获取依赖关系中的 codes  
dep_codes = set(deps['pre_task_code'].dropna()) | set(deps['post_task_code'].dropna())
dep_codes = dep_codes - {0}
dep_codes_list = sorted([int(x) for x in dep_codes])
print(f'process_task_relation 中的 codes (共{len(dep_codes_list)}个):')
for c in dep_codes_list[:5]:
    print(f'  {c}')

# 检查是否有交集
overlap = set(inst_codes_list) & set(dep_codes_list)
print(f'\n交集数量: {len(overlap)}')

if len(overlap) > 0:
    print('交集中的 code:')
    for c in list(overlap)[:5]:
        print(f'  {c}')

# 检查 task_instance.task_code 是否等于 task_definition.code
print('\n' + '=' * 80)
print('关键检查：task_instance.task_code 与 process_task_relation 中的 code 是否匹配')
print('=' * 80)

task_defs = data.get('task_definition')
if task_defs is not None:
    # 通过名称找到对应的 task_definition.code
    for _, t in successful.iterrows():
        name = t['name']
        inst_code = int(t['task_code']) if t['task_code'] != 0 else 0
        
        # 在 task_definition 中找同名任务
        match = task_defs[task_defs['name'] == name]
        if len(match) > 0:
            def_code = int(match.iloc[0]['code'])
            
            # 检查这个 def_code 是否在依赖关系中
            in_deps = def_code in dep_codes_list
            
            if inst_code != 0:
                print(f'{name[:35]:35} inst_code={inst_code:15} def_code={def_code:15} in_deps={in_deps}')
