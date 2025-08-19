# FE-IDDQN云工作流调度算法

## 项目简介

本项目旨在实现一个基于特征工程的改进型双深度Q网络（FE-IDDQN）云工作流调度算法。该算法针对异构云环境下的工作流调度优化问题，通过深度强化学习方法，结合多维度特征工程，旨在最小化工作流完成时间（makespan）、最大化资源利用率并实现负载均衡。

项目数据来源于DolphinScheduler 3.0.0生产环境的真实工作流运行数据，任务结构为有向无环图（DAG），支持任务并行执行，并考虑任务和容器的资源限制。

## 主要特性

- **数据驱动的特征工程**：从DolphinScheduler生产环境数据中提取任务、工作流、Worker节点等多维度特征，构建丰富且具有代表性的状态表示。
- **先进的FE-IDDQN算法**：
  - **注意力增强的双流网络架构**：有效处理异构特征，并通过注意力机制聚焦关键信息。
  - **优先级经验回放缓冲区**：优化经验采样，加速模型收敛并提高学习效率。
  - **自适应探索策略**：平衡探索与利用，提高算法在复杂环境中的适应性。
- **全面的基线对比**：实现多种传统调度算法（FIFO、SJF、HEFT）、元启发式算法（GA、PSO、ACO）以及深度强化学习基线（DQN、DDQN、BF-DDQN），进行性能对比。
- **真实仿真环境**：高度还原实际工作流的DAG结构和并行执行特性，模拟资源限制和动态负载。
- **多目标奖励函数**：综合考虑makespan最小化、资源利用最大化和负载均衡，引导模型学习最优调度策略。
- **可复现的实验框架**：提供完整的实验运行、评估和结果分析工具，支持多次测试并输出对比图表和数据。

## 项目结构

```
fe_iddqn_scheduler/
├── README.md
├── requirements.txt
├── config/
│   ├── config.py
│   └── hyperparameters.py
├── data/
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   └── feature_engineer.py
├── models/
│   ├── fe_iddqn.py
│   ├── networks.py
│   ├── experience_replay.py
│   └── exploration_strategy.py
├── baselines/
│   ├── traditional_schedulers.py
│   ├── metaheuristic_schedulers.py
│   └── drl_baselines.py
├── environment/
│   ├── workflow_simulator.py
│   ├── dag_parser.py
│   └── resource_manager.py
├── evaluation/
│   ├── metrics.py
│   ├── experiment_runner.py
│   └── result_analyzer.py
├── utils/
│   ├── logger.py
│   ├── visualization.py
│   └── helpers.py
├── experiments/
│   ├── run_experiments.py
│   └── experiment_configs/
└── results/
    ├── logs/
    ├── models/
    ├── figures/
    └── tables/
```

## 安装与使用

### 1. 环境准备

本项目基于Python 3.8+开发，建议使用conda或venv创建虚拟环境。

```bash
# 创建虚拟环境
python -m venv venv_fe_iddqn
source venv_fe_iddqn/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将DolphinScheduler生产环境导出的以下五个CSV文件放置在 `fe_iddqn_scheduler/data/raw_data/` 目录下（需要手动创建 `raw_data` 目录）：

- `gaussdb_t_ds_process_instance_a.csv`
- `gaussdb_t_ds_task_instance_a.csv`
- `oceanbase_t_ds_process_definition.csv`
- `oceanbase_t_ds_process_task_relation.csv`
- `oceanbase_t_ds_task_definition.csv`

### 3. 运行实验

进入项目根目录，执行主实验脚本：

```bash
cd fe_iddqn_scheduler
python experiments/run_experiments.py
```

实验结果（日志、模型、图表、数据表）将输出到 `results/` 目录下。

## 配置与超参数

- `config/config.py`：包含数据路径、输出路径、日志级别等基本配置。
- `config/hyperparameters.py`：包含FE-IDDQN算法和基线算法的超参数设置。

用户可以根据需求修改这些配置文件。

## 模块详情

### `data/`
- `data_loader.py`：负责加载原始CSV文件，并建立DolphinScheduler表之间的关联。
- `data_preprocessor.py`：进行数据清洗、缺失值处理、异常值检测等预处理操作。
- `feature_engineer.py`：实现多维度特征提取，包括任务特征、工作流DAG特征、Worker节点状态特征等。

### `models/`
- `fe_iddqn.py`：FE-IDDQN算法的核心实现，包括训练循环、决策逻辑等。
- `networks.py`：定义注意力增强的双流神经网络结构。
- `experience_replay.py`：实现优先级经验回放缓冲区，支持关键路径感知采样。
- `exploration_strategy.py`：实现自适应探索策略，平衡探索与利用。

### `baselines/`
- `traditional_schedulers.py`：包含FIFO、SJF、HEFT等传统调度算法的实现。
- `metaheuristic_schedulers.py`：包含GA、PSO、ACO等元启发式调度算法的实现。
- `drl_baselines.py`：包含DQN、DDQN、BF-DDQN等深度强化学习基线算法的实现。

### `environment/`
- `workflow_simulator.py`：构建仿真环境，模拟工作流在资源受限下的执行过程。
- `dag_parser.py`：解析DolphinScheduler的工作流定义，构建DAG图。
- `resource_manager.py`：管理Worker节点资源，模拟资源分配和利用。

### `evaluation/`
- `metrics.py`：定义和计算makespan、资源利用率、负载均衡度等评估指标。
- `experiment_runner.py`：管理实验流程，运行不同算法，收集结果。
- `result_analyzer.py`：对实验结果进行统计分析和可视化，生成图表和表格。

### `utils/`
- `logger.py`：统一的日志记录工具。
- `visualization.py`：用于生成实验结果图表的工具。
- `helpers.py`：其他辅助函数。

## 贡献

欢迎对本项目进行贡献。如果您有任何建议或发现bug，请提交issue或pull request。

## 许可证

本项目采用MIT许可证。

