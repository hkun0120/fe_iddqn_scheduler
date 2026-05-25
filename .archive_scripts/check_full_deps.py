#!/usr/bin/env python3
"""
重新分析 293718 的完整依赖关系
包括 CONDITIONS 和 SQL 任务
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.mysql_data_loader import MySQLDataLoader

def main():
    loader = MySQLDataLoader(host='localhost', user='root', password='', database='whalesb', port=3306)
    data = loader.load_all_data()
    
    # 找到 293718 的 process_definition_code
    process = data['process_instance'][data['process_instance']['id'] == 293718].iloc[0]
    pdc = process['process_definition_code']
    print(f"293718 的 process_definition_code: {pdc}")
    
    # 查询依赖关系
    deps = data['process_task_relation'][data['process_task_relation']['process_definition_code'] == pdc]
    print(f"依赖关系数: {len(deps)}")
    
    # 获取任务定义
    task_defs = data.get('task_definition')
    if task_defs is not None:
        code_to_name = dict(zip(task_defs['code'], task_defs['name']))
        code_to_type = dict(zip(task_defs['code'], task_defs['task_type']))
        
        print("\n" + "=" * 120)
        print("完整依赖关系详情:")
        print("=" * 120)
        
        # 按类型统计
        type_stats = {}
        
        for _, row in deps.iterrows():
            pre = row['pre_task_code']
            post = row['post_task_code']
            pre_name = code_to_name.get(pre, 'START')
            post_name = code_to_name.get(post, 'N/A')
            pre_type = code_to_type.get(pre, 'START')
            post_type = code_to_type.get(post, 'N/A')
            
            pair = f"{pre_type} -> {post_type}"
            type_stats[pair] = type_stats.get(pair, 0) + 1
            
            print(f"  [{pre_type:12}] {pre_name[:35]:35} -> [{post_type:12}] {post_name[:35]}")
        
        print("\n" + "=" * 120)
        print("依赖类型统计:")
        print("=" * 120)
        for pair, count in sorted(type_stats.items(), key=lambda x: -x[1]):
            print(f"  {pair}: {count}条")
        
        # 特别检查 CONDITIONS 相关的依赖
        print("\n" + "=" * 120)
        print("CONDITIONS 相关依赖:")
        print("=" * 120)
        
        for _, row in deps.iterrows():
            pre = row['pre_task_code']
            post = row['post_task_code']
            pre_type = code_to_type.get(pre, 'START')
            post_type = code_to_type.get(post, 'N/A')
            
            if pre_type == 'CONDITIONS' or post_type == 'CONDITIONS':
                pre_name = code_to_name.get(pre, 'START')
                post_name = code_to_name.get(post, 'N/A')
                print(f"  [{pre_type:12}] {pre_name[:35]:35} -> [{post_type:12}] {post_name[:35]}")

if __name__ == '__main__':
    main()
