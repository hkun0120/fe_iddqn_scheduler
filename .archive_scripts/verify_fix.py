#!/usr/bin/env python3
"""
验证 build_dag 修复后的效果
使用已缓存的数据
"""

import sys
sys.path.insert(0, '/Users/hong/Documents/GitHub/fe_iddqn_scheduler')

import pandas as pd
from sqlalchemy import create_engine

# 直接连接数据库
engine = create_engine('mysql+pymysql://root:@localhost:3306/whalesb')

print("加载数据...")

# 加载必要的表
process_instance = pd.read_sql("SELECT * FROM t_ds_process_instance WHERE id = 293718", engine)
task_instance = pd.read_sql("SELECT * FROM t_ds_task_instance WHERE process_instance_id = 293718", engine)
process_def_code = process_instance.iloc[0]['process_definition_code']
print(f"process_definition_code: {process_def_code}")

process_task_relation = pd.read_sql(f"SELECT * FROM t_ds_process_task_relation WHERE process_definition_code = {process_def_code}", engine)
task_definition = pd.read_sql(f"SELECT code, name, task_type FROM t_ds_task_definition WHERE process_definition_code = {process_def_code}", engine)

print(f"任务实例数: {len(task_instance)}")
print(f"任务定义数: {len(task_definition)}")
print(f"依赖边数: {len(process_task_relation)}")

# 成功任务
successful = task_instance[task_instance['state'] == 7]
print(f"成功任务数: {len(successful)}")

# 建立映射
def_code_to_name = dict(zip(task_definition['code'], task_definition['name']))
name_to_inst_code = dict(zip(successful['name'], successful['task_code']))

# 建立 task_definition.code -> task_instance.task_code 的映射
def_code_to_inst_code = {}
for def_code, name in def_code_to_name.items():
    if name in name_to_inst_code:
        def_code_to_inst_code[def_code] = name_to_inst_code[name]

print(f"\n成功建立的映射数: {len(def_code_to_inst_code)}")

# 模拟构建 DAG
print("\n" + "=" * 80)
print("模拟构建 DAG:")
print("=" * 80)

# 添加节点
nodes = set()
for _, task in successful.iterrows():
    nodes.add(task['task_code'])
print(f"节点数: {len(nodes)}")

# 添加边
edges_added = 0
edges_failed = 0
for _, dep in process_task_relation.iterrows():
    pre_def = dep['pre_task_code']
    post_def = dep['post_task_code']
    
    if pd.notna(pre_def) and pd.notna(post_def):
        pre_inst = def_code_to_inst_code.get(pre_def)
        post_inst = def_code_to_inst_code.get(post_def)
        
        if pre_inst is not None and post_inst is not None:
            if pre_inst in nodes and post_inst in nodes:
                edges_added += 1
            else:
                edges_failed += 1
        else:
            edges_failed += 1

print(f"成功添加的边数: {edges_added}")
print(f"未能添加的边数: {edges_failed}")

# 对比：旧方法
print("\n" + "=" * 80)
print("对比：旧方法 (直接使用 task_code):")
print("=" * 80)

old_edges_added = 0
for _, dep in process_task_relation.iterrows():
    pre = dep['pre_task_code']
    post = dep['post_task_code']
    if pd.notna(pre) and pd.notna(post) and pre in nodes and post in nodes:
        old_edges_added += 1

print(f"旧方法添加的边数: {old_edges_added}")

print("\n" + "=" * 80)
print("结论:")
print("=" * 80)
print(f"  修复后: {edges_added} 条边")
print(f"  修复前: {old_edges_added} 条边")
print(f"  改进: +{edges_added - old_edges_added} 条边")
