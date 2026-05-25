#!/usr/bin/env python3
"""
分析工作流293718的依赖关系
检查调度是否正确遵守依赖约束
"""

import pandas as pd
import sys
from pathlib import Path
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from data.mysql_data_loader import MySQLDataLoader

def analyze_workflow(process_id=293718):
    print(f"\n{'='*80}")
    print(f"分析工作流 {process_id} 的依赖关系")
    print(f"{'='*80}\n")
    
    # 加载数据
    loader = MySQLDataLoader(host='localhost', user='root', password='', database='whalesb', port=3306)
    data = loader.load_all_data()
    
    # 获取工作流信息
    process = data['process_instance'][data['process_instance']['id'] == process_id].iloc[0]
    print(f"工作流名称: {process['name']}")
    print(f"原始开始时间: {process['start_time']}")
    print(f"原始结束时间: {process['end_time']}")
    
    # 获取任务
    tasks = data['task_instance'][data['task_instance']['process_instance_id'] == process_id]
    successful_tasks = tasks[tasks['state'] == 7].copy()
    
    print(f"\n成功任务数: {len(successful_tasks)}")
    
    # 获取依赖关系
    process_def_code = process.get('process_definition_code')
    dependencies = data['process_task_relation'][
        data['process_task_relation']['process_definition_code'] == process_def_code
    ]
    
    print(f"依赖关系数: {len(dependencies)}")
    
    # 构建任务代码到名称的映射
    task_code_to_name = {}
    task_code_to_type = {}
    task_code_to_times = {}
    
    for _, task in successful_tasks.iterrows():
        code = task.get('task_code')
        task_code_to_name[code] = task.get('name', 'N/A')
        task_code_to_type[code] = task.get('task_type', 'N/A')
        try:
            start = pd.to_datetime(task.get('start_time'))
            end = pd.to_datetime(task.get('end_time'))
            task_code_to_times[code] = {
                'start': start,
                'end': end,
                'duration': (end - start).total_seconds()
            }
        except:
            pass
    
    # 打印所有任务
    print(f"\n{'='*80}")
    print("所有任务列表:")
    print(f"{'='*80}")
    print(f"{'序号':<4} | {'任务类型':<15} | {'开始时间':<25} | {'结束时间':<25} | {'时长(秒)':<10} | 任务名称")
    print("-" * 120)
    
    # 按开始时间排序
    sorted_tasks = successful_tasks.sort_values('start_time')
    for idx, (_, task) in enumerate(sorted_tasks.iterrows(), 1):
        code = task.get('task_code')
        name = task.get('name', 'N/A')
        task_type = task.get('task_type', 'N/A')
        start = task.get('start_time')
        end = task.get('end_time')
        try:
            duration = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
        except:
            duration = 0
        print(f"{idx:<4} | {task_type:<15} | {str(start):<25} | {str(end):<25} | {duration:<10.0f} | {name}")
    
    # 打印依赖关系
    print(f"\n{'='*80}")
    print("依赖关系 (pre_task -> post_task):")
    print(f"{'='*80}")
    
    valid_deps = []
    for _, dep in dependencies.iterrows():
        pre = dep.get('pre_task_code')
        post = dep.get('post_task_code')
        
        # 只显示在成功任务中的依赖
        pre_name = task_code_to_name.get(pre, None)
        post_name = task_code_to_name.get(post, None)
        
        if pre_name and post_name:
            pre_type = task_code_to_type.get(pre, 'N/A')
            post_type = task_code_to_type.get(post, 'N/A')
            print(f"  [{pre_type}] {pre_name[:40]:<40} -> [{post_type}] {post_name[:40]}")
            valid_deps.append((pre, post, pre_name, post_name))
    
    # 检查原始执行是否遵守依赖
    print(f"\n{'='*80}")
    print("检查原始执行是否遵守依赖约束:")
    print(f"{'='*80}")
    
    violations = []
    for pre, post, pre_name, post_name in valid_deps:
        if pre in task_code_to_times and post in task_code_to_times:
            pre_end = task_code_to_times[pre]['end']
            post_start = task_code_to_times[post]['start']
            
            if post_start < pre_end:
                violations.append({
                    'pre': pre_name,
                    'post': post_name,
                    'pre_end': pre_end,
                    'post_start': post_start,
                    'gap': (post_start - pre_end).total_seconds()
                })
                print(f"  ❌ 违反依赖: {pre_name[:30]} (结束于 {pre_end}) -> {post_name[:30]} (开始于 {post_start})")
            else:
                gap = (post_start - pre_end).total_seconds()
                print(f"  ✓ 遵守依赖: {pre_name[:30]} -> {post_name[:30]} (等待 {gap:.0f}秒)")
    
    if not violations:
        print("\n  所有依赖约束都被正确遵守!")
    else:
        print(f"\n  发现 {len(violations)} 个依赖违反!")
    
    # 分析 CONDITIONS 和 SUB_PROCESS 的关系
    print(f"\n{'='*80}")
    print("分析 SUB_PROCESS 和 CONDITIONS 的关系:")
    print(f"{'='*80}")
    
    for _, task in successful_tasks.iterrows():
        task_type = task.get('task_type', '')
        if task_type == 'SUB_PROCESS':
            code = task.get('task_code')
            name = task.get('name')
            print(f"\nSUB_PROCESS: {name}")
            
            # 找到依赖于这个任务的后续任务
            for _, dep in dependencies.iterrows():
                if dep.get('pre_task_code') == code:
                    post_code = dep.get('post_task_code')
                    post_name = task_code_to_name.get(post_code, 'N/A')
                    post_type = task_code_to_type.get(post_code, 'N/A')
                    print(f"  -> 后续任务: [{post_type}] {post_name}")
    
    # 分析 CONDITIONS 任务
    print(f"\n{'='*80}")
    print("分析 CONDITIONS 任务:")
    print(f"{'='*80}")
    
    for _, task in successful_tasks.iterrows():
        task_type = task.get('task_type', '')
        if task_type == 'CONDITIONS':
            code = task.get('task_code')
            name = task.get('name')
            print(f"\nCONDITIONS: {name}")
            
            # 找到这个任务依赖的前序任务
            print("  前序任务:")
            for _, dep in dependencies.iterrows():
                if dep.get('post_task_code') == code:
                    pre_code = dep.get('pre_task_code')
                    pre_name = task_code_to_name.get(pre_code, 'N/A')
                    pre_type = task_code_to_type.get(pre_code, 'N/A')
                    print(f"    <- [{pre_type}] {pre_name}")
            
            # 找到依赖于这个任务的后续任务
            print("  后续任务:")
            for _, dep in dependencies.iterrows():
                if dep.get('pre_task_code') == code:
                    post_code = dep.get('post_task_code')
                    post_name = task_code_to_name.get(post_code, 'N/A')
                    post_type = task_code_to_type.get(post_code, 'N/A')
                    print(f"    -> [{post_type}] {post_name}")

    return data, process, successful_tasks, dependencies

if __name__ == '__main__':
    analyze_workflow(293718)
