#!/usr/bin/env python3
"""
简单测试遗传算法并行化性能
"""

import time
import sys
from pathlib import Path
import multiprocessing as mp

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from baselines.meta_heuristics import GAScheduler

def create_test_data():
    """创建测试数据"""
    # 创建7个任务
    tasks = [
        {'id': 'task_1', 'name': 'Task 1', 'duration': 20.0, 'cpu_req': 1.0, 'memory_req': 1.0},
        {'id': 'task_2', 'name': 'Task 2', 'duration': 100.0, 'cpu_req': 1.0, 'memory_req': 1.0},
        {'id': 'task_3', 'name': 'Task 3', 'duration': 20.0, 'cpu_req': 1.0, 'memory_req': 1.0},
        {'id': 'task_4', 'name': 'Task 4', 'duration': 30.0, 'cpu_req': 1.0, 'memory_req': 1.0},
        {'id': 'task_5', 'name': 'Task 5', 'duration': 20.0, 'cpu_req': 1.0, 'memory_req': 1.0},
        {'id': 'task_6', 'name': 'Task 6', 'duration': 30.0, 'cpu_req': 1.0, 'memory_req': 1.0},
        {'id': 'task_7', 'name': 'Task 7', 'duration': 30.0, 'cpu_req': 1.0, 'memory_req': 1.0}
    ]
    
    # 创建6个资源
    resources = [
        {'id': 0, 'name': 'Resource_0', 'cpu_capacity': 5.34, 'memory_capacity': 5.53},
        {'id': 1, 'name': 'Resource_1', 'cpu_capacity': 2.0, 'memory_capacity': 4.0},
        {'id': 2, 'name': 'Resource_2', 'cpu_capacity': 2.0, 'memory_capacity': 4.0},
        {'id': 3, 'name': 'Resource_3', 'cpu_capacity': 2.0, 'memory_capacity': 4.0},
        {'id': 4, 'name': 'Resource_4', 'cpu_capacity': 2.0, 'memory_capacity': 4.0},
        {'id': 5, 'name': 'Resource_5', 'cpu_capacity': 2.0, 'memory_capacity': 4.0}
    ]
    
    # 无依赖关系
    dependencies = []
    
    return tasks, resources, dependencies

def test_ga_performance():
    """测试遗传算法性能"""
    print("="*60)
    print("遗传算法并行化性能测试")
    print("="*60)
    
    # 创建测试数据
    tasks, resources, dependencies = create_test_data()
    
    print(f"任务数: {len(tasks)}")
    print(f"资源数: {len(resources)}")
    print(f"CPU核心数: {mp.cpu_count()}")
    
    # 测试串行版本
    print("\n测试串行版本...")
    ga_serial = GAScheduler(use_parallel=False)
    start_time = time.time()
    result_serial = ga_serial.schedule(tasks, resources, dependencies)
    serial_time = time.time() - start_time
    
    print(f"串行版本执行时间: {serial_time:.3f}秒")
    print(f"串行版本makespan: {result_serial.get('makespan', 0):.2f}")
    print(f"串行版本资源利用率: {result_serial.get('resource_utilization', 0):.3f}")
    
    # 测试并行版本
    print("\n测试并行版本...")
    ga_parallel = GAScheduler(use_parallel=True, max_workers=min(mp.cpu_count(), 8))
    start_time = time.time()
    result_parallel = ga_parallel.schedule(tasks, resources, dependencies)
    parallel_time = time.time() - start_time
    
    print(f"并行版本执行时间: {parallel_time:.3f}秒")
    print(f"并行版本makespan: {result_parallel.get('makespan', 0):.2f}")
    print(f"并行版本资源利用率: {result_parallel.get('resource_utilization', 0):.3f}")
    
    # 计算性能提升
    if parallel_time > 0:
        speedup = serial_time / parallel_time
        time_saved = ((serial_time - parallel_time) / serial_time * 100)
        print(f"\n性能提升: {speedup:.2f}x")
        print(f"时间节省: {time_saved:.1f}%")
        
        # 验证结果一致性
        makespan_diff = abs(result_serial.get('makespan', 0) - result_parallel.get('makespan', 0))
        if makespan_diff < 0.01:
            print("✅ 结果一致性验证通过")
        else:
            print(f"⚠️ 结果存在差异: {makespan_diff:.2f}")
    else:
        print("❌ 并行版本执行失败")

def main():
    """主函数"""
    try:
        test_ga_performance()
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

