import os
import json
import pandas as pd
from sqlalchemy import create_engine

CONFIG_FILE = 'db_config.json'

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return None

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def main():
    # 加载配置
    config = load_config()
    if not config:
        # 如果没有配置文件，提示用户输入
        db_user = input("请输入数据库用户名: ")
        db_password = input("请输入数据库密码: ")
        db_host = input("请输入数据库主机地址: ")
        db_name = input("请输入数据库名称: ")
        config = {
            "db_user": db_user,
            "db_password": db_password,
            "db_host": db_host,
            "db_name": db_name
        }
        save_config(config)  # 保存到本地文件

    # CSV 文件绝对路径列表
    csv_files = input("请输入CSV文件的绝对路径，多个文件用逗号分隔: ").split(',')
    # 创建数据库连接
    engine = create_engine(f'mysql+pymysql://{config["db_user"]}:{config["db_password"]}@{config["db_host"]}/{config["db_name"]}')

    # 遍历每个 CSV 文件并导入到 MySQL
    for csv_file in csv_files:
        table_name = csv_file.split('/')[-1].split('.')[0]  # 假设表名与文件名相同
        print(f"正在导入 {csv_file} 到表 {table_name}...")

        # 分块读取 CSV 文件并写入 MySQL
        chunksize = 100000  # 每次处理 10 万行
        for chunk in pd.read_csv(csv_file.strip(), chunksize=chunksize):
            chunk.to_sql(table_name, con=engine, if_exists='append', index=False)

        print(f"{csv_file} 导入完成！")

if __name__ == "__main__":
    main()