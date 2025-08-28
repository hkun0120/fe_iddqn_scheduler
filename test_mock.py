#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.workflow_simulator import WorkflowSimulator

def test_step():
    # 创建测试任务
    tasks = [
        {'id': 'T1', 'name': 'Task1', 'duration': 10, 'cpu_req': 2, 'memory_req': 4, 'submit_time': 0},
        {'id': 'T2', 'name': 'Task2', 'duration': 15, 'cpu_req': 1, 'memory_req': 2, 'submit_time': 0},
        {'id': 'T3', 'name': 'Task3', 'duration': 8, 'cpu_req': 3, 'memory_req': 6, 'submit_time': 0}
    ]
    
    resources = [
        {'id': 1, 'cpu_capacity': 4, 'memory_capacity': 8},
        {'id': 2, 'cpu_capacity': 2, 'memory_capacity': 4}
    ]
    
    dependencies = [
        {'pre_task': 'T1', 'post_task': 'T2'},
        {'pre_task': 'T1', 'post_task': 'T3'}
    ]
    
    simulator = WorkflowSimulator(tasks, resources, dependencies)
    
    print("初始状态:")
    print(f"可调度任务: {simulator.ready_tasks}")
    print(f"已完成任务: {simulator.completed_tasks}")
    
    for step in range(5):
        print(f"\n--- Step {step + 1} ---")
        
        if simulator.is_done():
            print("所有任务完成！")
            break
            
        if not simulator.ready_tasks:
            print("等待任务完成...")
            if simulator.task_end_times:
                next_time = min(simulator.task_end_times.values())
                simulator.current_time = next_time
                simulator.ready_tasks = simulator._get_ready_tasks()
                print(f"时间更新到: {next_time}")
                print(f"新的可调度任务: {simulator.ready_tasks}")
        
        if simulator.ready_tasks:
            action = 0
            print(f"执行动作: 选择资源 {action}")
            
            next_state, reward, done, info = simulator.step(action)
            print(f"奖励: {reward}, 完成: {done}")
            print(f"任务分配: {simulator.task_assignments}")
            print(f"任务时间: {simulator.task_end_times}")
        else:
            print("没有可调度的任务")
    
    print(f"\n最终状态:")
    print(f"已完成: {simulator.completed_tasks}")
    print(f"Makespan: {simulator.get_makespan()}")

if __name__ == "__main__":
    test_step()
