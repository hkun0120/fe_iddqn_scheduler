"""
增强版FE-IDDQN算法使用示例

本脚本演示如何使用增强后的FE-IDDQN调度算法，包括：
1. GNN网络处理DAG依赖
2. Transformer增强注意力机制
3. 多目标奖励函数
4. 高级探索策略
5. 增强经验回放
6. 课程学习训练
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入增强模块
from models import (
    # 网络模块
    EnhancedDualStreamNetwork,
    DAGAwareModule,
    MultiHeadGAT,
    DAGEncoder,
    
    # 探索策略
    AdaptiveEpsilonGreedy,
    NoisyLinear,
    IntrinsicCuriosityModule,
    CombinedExplorationStrategy,
    
    # 经验回放
    CombinedReplayBuffer,
    NStepReplayBuffer,
    HierarchicalReplayBuffer,
    
    # 奖励函数
    RewardConfig,
    EnhancedRewardCalculator,
    CurriculumRewardScheduler,
    
    # 主算法
    EnhancedFE_IDDQN,
    EnhancedFE_IDDQN_Config,
    DAGAwareActionMasker,
    LookaheadPlanner
)

from data import (
    EnhancedStateConfig,
    EnhancedStateEncoder,
    CriticalPathAnalyzer,
    StateToTensor
)

from environment import (
    EnhancedWorkflowSimulator,
    TaskState,
    ResourceState
)

from config import Hyperparameters


def create_sample_workflow():
    """创建示例工作流（DAG）"""
    # 任务定义（模拟典型的数据处理工作流）
    tasks = [
        {'id': 0, 'name': 'data_extract', 'duration': 10, 'resource_req': 2},
        {'id': 1, 'name': 'data_clean', 'duration': 15, 'resource_req': 3},
        {'id': 2, 'name': 'feature_eng', 'duration': 20, 'resource_req': 4},
        {'id': 3, 'name': 'model_train', 'duration': 30, 'resource_req': 6},
        {'id': 4, 'name': 'validation', 'duration': 10, 'resource_req': 2},
        {'id': 5, 'name': 'data_transform', 'duration': 12, 'resource_req': 3},
        {'id': 6, 'name': 'aggregation', 'duration': 8, 'resource_req': 2},
        {'id': 7, 'name': 'export', 'duration': 5, 'resource_req': 1},
    ]
    
    # DAG依赖关系
    dependencies = [
        (0, 1),  # extract -> clean
        (0, 5),  # extract -> transform (并行分支)
        (1, 2),  # clean -> feature_eng
        (5, 6),  # transform -> aggregation
        (2, 3),  # feature_eng -> model_train
        (6, 3),  # aggregation -> model_train (汇合)
        (3, 4),  # model_train -> validation
        (4, 7),  # validation -> export
    ]
    
    # 资源定义
    resources = [
        {'id': 0, 'name': 'worker_1', 'capacity': 4, 'speed': 1.0},
        {'id': 1, 'name': 'worker_2', 'capacity': 6, 'speed': 1.2},
        {'id': 2, 'name': 'worker_3', 'capacity': 8, 'speed': 0.9},
        {'id': 3, 'name': 'worker_4', 'capacity': 4, 'speed': 1.1},
        {'id': 4, 'name': 'gpu_worker', 'capacity': 10, 'speed': 2.0},
        {'id': 5, 'name': 'high_mem', 'capacity': 12, 'speed': 0.8},
    ]
    
    return tasks, resources, dependencies


def demo_network_architecture():
    """演示增强网络架构"""
    print("\n" + "="*60)
    print("1. 增强网络架构演示")
    print("="*60)
    
    # 网络配置
    task_input_dim = 64
    resource_input_dim = 32
    hidden_dim = 128
    output_dim = 6  # 动作空间大小
    
    # 创建增强双流网络
    network = EnhancedDualStreamNetwork(
        task_input_dim=task_input_dim,
        resource_input_dim=resource_input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=4,
        dropout=0.1,
        use_gnn=True
    )
    
    # 模拟输入
    batch_size = 4
    num_tasks = 8
    num_resources = 6
    
    task_features = torch.randn(batch_size, num_tasks, task_input_dim)
    resource_features = torch.randn(batch_size, num_resources, resource_input_dim)
    
    # 前向传播
    q_values = network(task_features, resource_features)
    
    print(f"任务特征维度: {task_features.shape}")
    print(f"资源特征维度: {resource_features.shape}")
    print(f"Q值输出维度: {q_values.shape}")
    print(f"网络参数量: {sum(p.numel() for p in network.parameters()):,}")
    
    return network


def demo_gnn_module():
    """演示GNN处理DAG"""
    print("\n" + "="*60)
    print("2. GNN处理DAG依赖演示")
    print("="*60)
    
    tasks, resources, dependencies = create_sample_workflow()
    num_tasks = len(tasks)
    
    # 创建邻接矩阵
    adj_matrix = np.zeros((num_tasks, num_tasks))
    for src, dst in dependencies:
        adj_matrix[src, dst] = 1
    
    # 创建DAG感知模块
    dag_module = DAGAwareModule(
        node_feature_dim=32,
        hidden_dim=64,
        output_dim=64,
        num_gnn_layers=2,
        num_heads=4
    )
    
    # 模拟节点特征
    batch_size = 2
    node_features = torch.randn(batch_size, num_tasks, 32)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    adj_tensor = adj_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # GNN处理
    dag_output = dag_module(node_features, adj_tensor)
    
    print(f"任务数量: {num_tasks}")
    print(f"DAG边数量: {len(dependencies)}")
    print(f"节点嵌入输出维度: {dag_output['node_embeddings'].shape}")
    print(f"图级嵌入维度: {dag_output['graph_embedding'].shape}")
    print(f"DAG表示维度: {dag_output['dag_representation'].shape}")


def demo_critical_path():
    """演示关键路径分析"""
    print("\n" + "="*60)
    print("3. 关键路径分析演示")
    print("="*60)
    
    tasks, resources, dependencies = create_sample_workflow()
    
    # 创建NetworkX DAG
    dag = nx.DiGraph()
    for task in tasks:
        dag.add_node(task['id'], **task)
    for src, dst in dependencies:
        dag.add_edge(src, dst)
    
    # 创建关键路径分析器
    analyzer = CriticalPathAnalyzer()
    
    # 创建任务持续时间映射
    task_durations = {task['id']: task['duration'] for task in tasks}
    
    # 分析关键路径
    critical_path, path_length = analyzer.find_critical_path(dag, task_durations)
    
    print(f"关键路径: {critical_path}")
    print(f"关键路径长度: {path_length}")
    
    # 找出关键路径上的任务
    critical_tasks = [tasks[i]['name'] for i in critical_path if i < len(tasks)]
    print(f"关键路径任务名称: {critical_tasks}")


def demo_reward_function():
    """演示多目标奖励函数"""
    print("\n" + "="*60)
    print("4. 多目标奖励函数演示")
    print("="*60)
    
    # 奖励配置
    config = RewardConfig()
    print(f"奖励权重配置:")
    print(f"  - Makespan: {config.makespan_weight}")
    print(f"  - 资源利用率: {config.resource_utilization_weight}")
    print(f"  - 负载均衡: {config.load_balance_weight}")
    print(f"  - 并行度: {config.parallelism_weight}")
    print(f"  - 关键路径: {config.critical_path_weight}")
    print(f"  - 等待时间: {config.waiting_time_weight}")
    
    # 创建奖励计算器
    calculator = EnhancedRewardCalculator(config)
    
    print(f"\n奖励计算器已创建")
    print(f"奖励归一化: {config.normalize_rewards}")
    print(f"奖励缩放: {config.reward_scale}")


def demo_exploration_strategy():
    """演示探索策略"""
    print("\n" + "="*60)
    print("5. 高级探索策略演示")
    print("="*60)
    
    # 创建自适应ε-greedy
    adaptive_epsilon = AdaptiveEpsilonGreedy(
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        adaptive_mode='performance'
    )
    
    # 模拟几个episode的性能
    performances = [0.5, 0.6, 0.55, 0.7, 0.65, 0.8, 0.75, 0.85]
    for perf in performances:
        adaptive_epsilon.update(reward=perf)
        print(f"性能: {perf:.2f}, 当前ε: {adaptive_epsilon.epsilon:.4f}")
    
    # 创建组合探索策略
    strategies = {
        'epsilon_greedy': (AdaptiveEpsilonGreedy(epsilon_start=1.0, epsilon_end=0.01), 0.7),
        'boltzmann': (AdaptiveEpsilonGreedy(epsilon_start=0.5, epsilon_end=0.05), 0.3)
    }
    combined_strategy = CombinedExplorationStrategy(strategies=strategies)
    
    # 模拟选择动作
    q_values = torch.randn(1, 6)
    
    print(f"\nQ值: {q_values.numpy().flatten()}")
    action = combined_strategy.select_action(q_values)
    print(f"选择的动作: {action}")


def demo_experience_replay():
    """演示增强经验回放"""
    print("\n" + "="*60)
    print("6. 增强经验回放演示")
    print("="*60)
    
    # 创建N-step回放缓冲区
    n_step_buffer = NStepReplayBuffer(
        capacity=1000,
        n_step=3,
        gamma=0.99
    )
    
    # 模拟存储经验
    state_dim = 64
    for i in range(100):
        state = np.random.randn(state_dim)
        action = np.random.randint(0, 6)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = i == 99
        
        n_step_buffer.add(state, action, reward, next_state, done)
    
    print(f"N-step缓冲区大小: {len(n_step_buffer)}")
    
    # 采样批次
    if len(n_step_buffer) >= 32:
        batch = n_step_buffer.sample(32)
        # batch返回: states, actions, n_step_rewards, nth_states, dones, gamma_ns
        print(f"采样批次大小: {batch[0].shape[0]}")
        print(f"N-step回报示例: {batch[2][:5].numpy()}")


def demo_enhanced_agent():
    """演示增强版FE-IDDQN Agent"""
    print("\n" + "="*60)
    print("7. 增强版FE-IDDQN Agent演示")
    print("="*60)
    
    # Agent配置 - 使用默认维度以保持一致性
    config = EnhancedFE_IDDQN_Config()
    # config.hidden_dim = 256  # 保持默认值
    config.use_gnn = True
    config.use_n_step = True
    config.n_step = 3
    config.use_per = True
    config.use_noisy_net = False
    config.learning_rate = 3e-4
    config.gamma = 0.99
    config.tau = 0.005
    config.batch_size = 32
    config.replay_buffer_size = 10000
    
    # 网络参数
    task_input_dim = 64
    resource_input_dim = 32
    action_dim = 6
    
    # 创建Agent
    agent = EnhancedFE_IDDQN(
        task_input_dim=task_input_dim,
        resource_input_dim=resource_input_dim,
        action_dim=action_dim,
        config=config
    )
    
    print(f"Agent配置:")
    print(f"  - 使用GNN: {config.use_gnn}")
    print(f"  - N-step: {config.n_step}")
    print(f"  - 使用PER: {config.use_per}")
    print(f"  - 隐藏层维度: {config.hidden_dim}")
    print(f"  - 动作空间: {action_dim}")
    
    # 模拟状态 - 使用分离的numpy数组
    task_features = np.random.randn(8, 64).astype(np.float32)  # 8个任务
    resource_features = np.random.randn(6, 32).astype(np.float32)  # 6个资源
    adj_matrix = np.zeros((8, 8), dtype=np.float32)  # DAG邻接矩阵
    
    # 选择动作
    action = agent.select_action(task_features, resource_features, adj_matrix)
    print(f"\n选择的动作: {action}")


def demo_action_masking():
    """演示DAG感知动作掩码"""
    print("\n" + "="*60)
    print("8. DAG感知动作掩码演示")
    print("="*60)
    
    tasks, resources, dependencies = create_sample_workflow()
    
    # 创建DAG
    dag = nx.DiGraph()
    for task in tasks:
        dag.add_node(task['id'], **task)
    for src, dst in dependencies:
        dag.add_edge(src, dst)
    
    # 创建动作掩码器
    masker = DAGAwareActionMasker()
    
    # 手动计算就绪任务（依赖已满足的任务）
    def get_ready_tasks(dag, completed_tasks):
        ready = []
        for node in dag.nodes():
            if node not in completed_tasks:
                predecessors = list(dag.predecessors(node))
                if all(pred in completed_tasks for pred in predecessors):
                    ready.append(node)
        return ready
    
    # 模拟不同任务完成状态
    completed_tasks = {0}  # 只有任务0完成
    ready_tasks = get_ready_tasks(dag, completed_tasks)
    print(f"已完成任务: {completed_tasks}")
    print(f"就绪可调度任务: {ready_tasks}")
    
    # 获取有效动作掩码
    mask = masker.get_valid_actions(ready_tasks, num_resources=len(resources))
    print(f"动作掩码维度: {mask.shape}")
    
    # 更多任务完成
    completed_tasks = {0, 1, 5}  # 任务0, 1, 5完成
    ready_tasks = get_ready_tasks(dag, completed_tasks)
    print(f"\n已完成任务: {completed_tasks}")
    print(f"就绪可调度任务: {ready_tasks}")


def demo_curriculum_learning():
    """演示课程学习"""
    print("\n" + "="*60)
    print("9. 课程学习奖励调度演示")
    print("="*60)
    
    # 创建课程学习调度器
    scheduler = CurriculumRewardScheduler(
        total_episodes=300
    )
    
    print("课程学习阶段配置:")
    
    # 模拟不同阶段
    for episode, stage_name in [(0, "阶段1 (初期)"), (100, "阶段2 (中期)"), (250, "阶段3 (后期)")]:
        scheduler.current_episode = episode
        config = scheduler.get_current_config()
        print(f"\n{stage_name} - Episode {episode}:")
        print(f"  - Makespan权重: {config.makespan_weight:.2f}")
        print(f"  - 资源利用率权重: {config.resource_utilization_weight:.2f}")
        print(f"  - 负载均衡权重: {config.load_balance_weight:.2f}")
        print(f"  - 并行度权重: {config.parallelism_weight:.2f}")


def main():
    """运行所有演示"""
    print("="*60)
    print("增强版FE-IDDQN调度算法演示")
    print("="*60)
    print("\n本演示展示了7个方面的改进:")
    print("1. 网络架构: GNN + Transformer + Cross-Attention")
    print("2. 奖励函数: 多目标奖励组件")
    print("3. 探索策略: 自适应ε + Noisy Networks + ICM")
    print("4. 经验回放: N-step + PER + 分层缓冲")
    print("5. 状态表示: 关键路径特征 + 图嵌入")
    print("6. 训练优化: 课程学习 + 梯度累积")
    print("7. DAG感知: 关键路径优先 + 动作掩码")
    
    try:
        # 1. 网络架构
        demo_network_architecture()
        
        # 2. GNN模块
        demo_gnn_module()
        
        # 3. 关键路径分析
        demo_critical_path()
        
        # 4. 奖励函数
        demo_reward_function()
        
        # 5. 探索策略
        demo_exploration_strategy()
        
        # 6. 经验回放
        demo_experience_replay()
        
        # 7. 增强Agent
        demo_enhanced_agent()
        
        # 8. 动作掩码
        demo_action_masking()
        
        # 9. 课程学习
        demo_curriculum_learning()
        
        print("\n" + "="*60)
        print("所有演示完成！")
        print("="*60)
        
    except ImportError as e:
        print(f"\n导入错误: {e}")
        print("请确保所有依赖已安装: pip install torch numpy networkx")
    except Exception as e:
        print(f"\n运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
