#!/usr/bin/env python3
"""
诊断数据库连接问题
"""

import pandas as pd
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def diagnose_database_connection():
    """诊断数据库连接问题"""
    print("🔍 诊断数据库连接问题...")
    
    try:
        # 尝试导入MySQLDataLoader
        from data.mysql_data_loader import MySQLDataLoader
        print("✅ MySQLDataLoader导入成功")
        
        # 尝试创建数据加载器（使用默认配置：root用户，空密码）
        loader = MySQLDataLoader(host='localhost', user='root', password='', database='whalesb')
        print("✅ MySQLDataLoader实例创建成功")
        
        # 尝试加载数据
        print("📋 尝试加载数据库数据...")
        data = loader.load_all_data()
        
        # 检查各个表的数据
        tables = ['task_instance', 'process_instance', 'task_definition', 'process_task_relation']
        for table in tables:
            if table in data:
                df = data[table]
                print(f"✅ {table}: {len(df)} 条记录")
                if len(df) > 0:
                    print(f"   列名: {list(df.columns)}")
                else:
                    print(f"   ⚠️  {table} 表为空")
            else:
                print(f"❌ {table}: 未找到")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

def check_data_files():
    """检查数据文件"""
    print("\n🔍 检查数据文件...")
    
    data_dir = Path("fe_iddqn_training_system/data")
    required_files = [
        "dataset_info_20250930_120240.json",
        "train_data_20250930_120240.csv",
        "val_data_20250930_120240.csv"
    ]
    
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✅ {file}: 存在")
        else:
            print(f"❌ {file}: 不存在")

if __name__ == "__main__":
    print("🚀 开始诊断...")
    
    # 检查数据文件
    check_data_files()
    
    # 诊断数据库连接
    db_ok = diagnose_database_connection()
    
    if db_ok:
        print("\n✅ 数据库连接正常")
    else:
        print("\n❌ 数据库连接有问题")
        print("建议检查:")
        print("1. MySQL服务是否运行")
        print("2. 数据库连接配置是否正确")
        print("3. 网络连接是否正常")
