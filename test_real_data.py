#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试真实数据的格式和结构
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import DataLoader
import pandas as pd

def test_real_data():
    """测试真实数据的加载和格式"""
    print("=" * 60)
    print("测试真实数据加载和格式")
    print("=" * 60)
    
    try:
        # 尝试加载数据
        data_loader = DataLoader("data/raw_data")
        data = data_loader.load_all_data()
        
        if not data:
            print("❌ 数据加载失败！")
            return
        
        print("✅ 数据加载成功！")
        
        # 检查各个数据表
        for table_name, table_data in data.items():
            print(f"\n📊 表: {table_name}")
            print(f"  行数: {len(table_data)}")
            print(f"  列数: {len(table_data.columns)}")
            print(f"  列名: {list(table_data.columns)}")
            
            if len(table_data) > 0:
                print(f"  数据类型:")
                for col, dtype in table_data.dtypes.items():
                    print(f"    {col}: {dtype}")
                
                print(f"  前3行数据:")
                print(table_data.head(3).to_string())
            
            print("-" * 40)
        
        # 检查任务实例数据
        if 'task_instance' in data:
            task_instances = data['task_instance']
            print(f"\n🔍 任务实例数据详细分析:")
            
            # 检查关键字段
            key_fields = ['id_instance', 'task_code', 'start_time', 'end_time', 'state', 'host']
            for field in key_fields:
                if field in task_instances.columns:
                    print(f"  {field}: 存在")
                    if field in ['start_time', 'end_time']:
                        print(f"    非空值数量: {task_instances[field].notna().sum()}")
                        print(f"    空值数量: {task_instances[field].isna().sum()}")
                        if task_instances[field].notna().sum() > 0:
                            sample_values = task_instances[field].dropna().head(3).tolist()
                            print(f"    样本值: {sample_values}")
                else:
                    print(f"  {field}: ❌ 缺失")
            
            # 检查状态值
            if 'state' in task_instances.columns:
                print(f"  状态值分布:")
                state_counts = task_instances['state'].value_counts()
                print(state_counts)
        
        # 检查任务定义数据
        if 'task_definition' in data:
            task_definitions = data['task_definition']
            print(f"\n🔍 任务定义数据详细分析:")
            print(f"  行数: {len(task_definitions)}")
            print(f"  列名: {list(task_definitions.columns)}")
            
            if len(task_definitions) > 0:
                print(f"  前3行数据:")
                print(task_definitions.head(3).to_string())
        
        # 检查进程任务关系数据
        if 'process_task_relation' in data:
            process_relations = data['process_task_relation']
            print(f"\n🔍 进程任务关系数据详细分析:")
            print(f"  行数: {len(process_relations)}")
            print(f"  列名: {list(process_relations.columns)}")
            
            if len(process_relations) > 0:
                print(f"  前3行数据:")
                print(process_relations.head(3).to_string())
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_data()
