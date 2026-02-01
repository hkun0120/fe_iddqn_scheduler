# FE-IDDQN Scheduler

基于深度强化学习的云工作流调度优化研究

## 项目结构

```
fe_iddqn_scheduler/
├── models/                    # 神经网络模型
│   ├── fe_iddqn.py            # 核心FE-IDDQN算法
│   ├── dual_stream_network.py # 双流网络架构
│   ├── graph_transformer.py   # DAG图编码器
│   └── prioritized_replay_buffer.py  # 优先经验回放
├── environment/               # 环境模拟
│   └── historical_replay_simulator.py # 基于历史数据的回放模拟器
├── experiments/               # 实验运行器
│   ├── experiment_runner.py   # 通用实验框架
│   └── fair_comparison_runner.py # 公平对比实验
├── baselines/                 # 基线算法
│   ├── heft.py                # HEFT算法
│   ├── peft.py                # PEFT算法
│   ├── ga.py                  # 遗传算法
│   └── round_robin.py         # 轮询算法
├── config/                    # 配置文件
├── data/                      # 数据集
├── evaluation/                # 评估工具
├── utils/                     # 工具函数
├── scripts/                   # 辅助脚本
├── results/                   # 训练结果
├── main.py                    # 主入口
├── run_fe_iddqn_training.py   # FE-IDDQN训练脚本
├── config.py                  # 全局配置
├── hyperparameters.py         # 超参数配置
└── _archive/                  # 归档的实验文件和文档
```

## 核心创新

FE-IDDQN (Feature-Enhanced Improved Double DQN) 的核心创新：

1. **双流网络架构**: 分离处理任务特征和资源特征
2. **Graph Transformer编码**: 对工作流DAG结构进行深度编码
3. **优先经验回放**: 基于TD误差的优先采样
4. **动作掩码机制**: 过滤不可行的调度决策

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行训练
python run_fe_iddqn_training.py

# 运行对比实验
python -m experiments.fair_comparison_runner
```

## 依赖

- Python 3.8+
- PyTorch 1.9+
- NetworkX
- NumPy
- Pandas

## 许可证

MIT License
