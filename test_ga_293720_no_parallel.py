#!/usr/bin/env python3
"""
不使用并行测试工作流293720的遗传算法性能
"""

import time
import sys
from pathlib import Path
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from data.mysql_data_loader import MySQLDataLoader
from baselines.meta_heuristics import GAScheduler

def load_workflow_293720_data():
    """加载工作流293720的数据"""
    print("正在加载工作流293720的数据...")
    
    # 创建数据加载器
    data_loader = MySQLDataLoader()
    data_loader.load_all_data()
    
    # 获取任务实例
    task_instances = data_loader.load_task_instances_by_workflow(293720)
    
    # 准备任务数据
    tasks = []
    for i, task in enumerate(task_instances):
        start_time = task.get('start_time')
        end_time = task.get('end_time')
        
        if start_time and end_time:
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            duration = (end_dt - start_dt).total_seconds()
        else:
            duration = 1.0  # 默认持续时间
        
        tasks.append({
            'id': task['id'],
            'name': task['name'],
            'duration': duration,
            'cpu_req': 1.0,
            'memory_req': 1.0
        })
    
    # 获取依赖关系
    dependencies = data_loader.get_process_dependencies(293720)
    print(f"依赖关系类型: {type(dependencies)}")
    print(f"依赖关系数量: {len(dependencies) if hasattr(dependencies, '__len__') else 'N/A'}")
    
    # 创建6个异构资源
    resources = [
        {'id': 0, 'name': 'Resource_0', 'cpu_capacity': 5.34, 'memory_capacity': 5.53},
        {'id': 1, 'name': 'Resource_1', 'cpu_capacity': 2.0, 'memory_capacity': 4.0},
        {'id': 2, 'name': 'Resource_2', 'cpu_capacity': 2.0, 'memory_capacity': 4.0},
        {'id': 3, 'name': 'Resource_3', 'cpu_capacity': 2.0, 'memory_capacity': 4.0},
        {'id': 4, 'name': 'Resource_4', 'cpu_capacity': 2.0, 'memory_capacity': 4.0},
        {'id': 5, 'name': 'Resource_5', 'cpu_capacity': 2.0, 'memory_capacity': 4.0}
    ]
    
    print(f"加载完成: {len(tasks)}个任务, {len(resources)}个资源")
    
    return tasks, resources, dependencies

def test_ga_performance():
    """测试遗传算法性能"""
    print("="*80)
    print("工作流293720 - 遗传算法性能测试（无并行）")
    print("="*80)
    
    # 加载数据
    tasks, resources, dependencies = load_workflow_293720_data()
    
    print(f"\n测试配置:")
    print(f"  任务数: {len(tasks)}")
    print(f"  资源数: {len(resources)}")
    print(f"  依赖关系数: {len(dependencies) if hasattr(dependencies, '__len__') else 'N/A'}")
    
    # 显示任务分布
    durations = [task['duration'] for task in tasks]
    print(f"\n任务持续时间分布:")
    print(f"  最短: {min(durations):.1f}秒")
    print(f"  最长: {max(durations):.1f}秒")
    print(f"  平均: {sum(durations)/len(durations):.1f}秒")
    print(f"  总计: {sum(durations):.1f}秒")
    
    # 测试串行版本（不使用并行）
    print(f"\n{'='*50}")
    print("测试遗传算法（无并行）...")
    print(f"{'='*50}")
    
    ga = GAScheduler(use_parallel=False)  # 明确禁用并行
    start_time = time.time()
    result = ga.schedule(tasks, resources, dependencies)
    execution_time = time.time() - start_time
    
    print(f"执行时间: {execution_time:.3f}秒")
    print(f"Makespan: {result.get('makespan', 0):.2f}秒")
    print(f"资源利用率: {result.get('resource_utilization', 0):.3f}")
    
    # 分析调度结果
    print(f"\n{'='*50}")
    print("调度结果分析")
    print(f"{'='*50}")
    
    if result.get('makespan', 0) != float('inf'):
        print(f"✅ 调度成功!")
        print(f"  最优makespan: {result.get('makespan', 0):.2f}秒")
        print(f"  资源利用率: {result.get('resource_utilization', 0):.3f}")
        
        # 显示任务分配
        task_assignments = result.get('task_assignments', {})
        if task_assignments:
            resource_usage = {}
            for task_id, resource_id in task_assignments.items():
                resource_usage[resource_id] = resource_usage.get(resource_id, 0) + 1
            
            print(f"  资源使用分布:")
            for resource_id in sorted(resource_usage.keys()):
                print(f"    资源{resource_id}: {resource_usage[resource_id]}个任务")
        
        # 显示任务时间信息
        task_start_times = result.get('task_start_times', {})
        task_end_times = result.get('task_end_times', {})
        if task_start_times and task_end_times:
            print(f"  任务执行时间:")
            for task_id in list(task_start_times.keys())[:5]:  # 只显示前5个任务
                start_time = task_start_times[task_id]
                end_time = task_end_times[task_id]
                print(f"    任务{task_id}: {start_time:.2f}s - {end_time:.2f}s")
    else:
        print("❌ 调度失败!")
        print("  可能的原因:")
        print("    1. 依赖关系解析错误")
        print("    2. 任务无法满足依赖约束")
        print("    3. 资源约束冲突")

def main():
    """主函数"""
    try:
        test_ga_performance()
        print(f"\n{'='*80}")
        print("测试完成!")
        print(f"{'='*80}")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

