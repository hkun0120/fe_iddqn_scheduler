class Hyperparameters:
    """超参数配置类"""
    
    # FE-IDDQN算法超参数
    FE_IDDQN = {
        # 网络结构参数
        'task_stream_hidden_dims': [512, 256, 128],
        'resource_stream_hidden_dims': [512, 256, 128],
        'fusion_dim': 256,
        'output_dim': 6,  # 动作空间大小（6个资源）
        
        # 训练参数（降低学习率提高稳定性）
        'learning_rate': 3e-5,  # 从1e-4降低到3e-5
        'batch_size': 32,
        'gamma': 0.99,  # 折扣因子
        'tau': 0.005,  # 软更新参数
        'target_update_freq': 100,  # 目标网络更新频率
        
        # 经验回放参数
        'replay_buffer_size': 10000,
        'priority_alpha': 0.6,  # 优先级指数
        'priority_beta': 0.4,  # 重要性采样指数
        'priority_beta_increment': 0.001,
        
        # 【改进】探索策略参数 - 增强探索能力，鼓励发现并行调度策略
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,  # 从0.01增加到0.05，保持更多探索
        'epsilon_decay': 0.998,  # 从0.995改为0.998，更慢的衰减
        'temperature_start': 1.5,  # 从1.0增加到1.5，更强的初始探索
        'temperature_end': 0.2,   # 从0.1增加到0.2，保持更多随机性
        'temperature_decay': 0.998,  # 从0.99改为0.998，更慢的衰减
        
        # 训练控制参数
        'max_episodes': 1000,
        'max_steps_per_episode': 500,
        'warmup_steps': 100,  # 预热步数
        'train_freq': 4,  # 训练频率
        'eval_freq': 100,  # 评估频率
        
        # 注意力机制参数
        'attention_dim': 128,
        'attention_heads': 4,
        'dropout_rate': 0.1,
        
        # 【改进】奖励函数权重 - 增加负载均衡（并行度）的权重
        'reward_weights': {
            'makespan': 0.5,        # 时间优化
            'resource_utilization': 0.2,  # 从0.3减少到0.2
            'load_balance': 0.3      # 从0.2增加到0.3，更重视并行度
        }
    }
    
    # DQN算法超参数
    DQN = {
        'hidden_dims': [128, 64],
        'learning_rate': 1e-3,
        'batch_size': 64,
        'gamma': 0.99,
        'replay_buffer_size': 10000,
        'target_update_freq': 100,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995
    }
    
    # DDQN算法超参数
    DDQN = {
        'hidden_dims': [128, 64],
        'learning_rate': 1e-3,
        'batch_size': 64,
        'gamma': 0.99,
        'replay_buffer_size': 10000,
        'target_update_freq': 100,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995
    }
    
    # BF-DDQN算法超参数
    BF_DDQN = {
        'hidden_dims': [128, 64],
        'learning_rate': 1e-3,
        'batch_size': 64,
        'gamma': 0.99,
        'replay_buffer_size': 10000,
        'target_update_freq': 100,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995
    }
    
    # 遗传算法超参数 - 论文要求设置（测试时使用较小参数）
    GA = {
        'population_size': 20,   # 测试时减少到20
        'generations': 50,       # 测试时减少到50
        'crossover_rate': 0.8,   # 论文要求：交叉概率0.8
        'crossover_prob': 0.8,   # 别名
        'mutation_rate': 0.1,    # 论文要求：变异概率0.1
        'mutation_prob': 0.1,    # 别名
        'elite_size': 5,         # 精英个体数量
        'tournament_size': 3     # 锦标赛选择的大小
    }
    
    # 粒子群优化超参数 - 论文要求设置（测试时使用较小参数）
    PSO = {
        'swarm_size': 10,        # 测试时减少到10
        'iterations': 30,        # 测试时减少到30
        'max_iterations': 30,
        'inertia_weight': 0.7,   # 论文要求：惯性权重0.7
        'w_max': 0.9,            # 最大惯性权重
        'w_min': 0.4,            # 最小惯性权重
        'cognitive_weight': 1.5, # 论文要求：个体学习因子1.5
        'social_weight': 1.5,    # 论文要求：群体学习因子1.5
        'c1': 1.5,               # 认知权重别名
        'c2': 1.5                # 社会权重别名
    }
    
    # 蚁群优化超参数 - 论文要求设置（测试时使用较小参数）
    ACO = {
        'num_ants': 10,           # 测试时减少到10
        'n_ants': 10,             # 别名
        'iterations': 20,         # 测试时减少到20
        'max_iterations': 20,
        'alpha': 1.0,             # 论文要求：信息素重要性因子1.0
        'beta': 2.0,              # 论文要求：启发式信息重要性因子2.0
        'evaporation_rate': 0.5,  # 论文要求：信息素挥发率0.5
        'rho': 0.5,               # 信息素挥发率别名
        'pheromone_constant': 100,
        'q0': 0.9                 # 随机选择概率
    }
    
    @classmethod
    def get_algorithm_params(cls, algorithm_name):
        """获取指定算法的超参数"""
        algorithm_map = {
            'FE_IDDQN': cls.FE_IDDQN,
            'DQN': cls.DQN,
            'DDQN': cls.DDQN,
            'BF_DDQN': cls.BF_DDQN,
            'BF-DDQN': cls.BF_DDQN,  # 兼容两种写法
            'GA': cls.GA,
            'PSO': cls.PSO,
            'ACO': cls.ACO
        }
        
        if algorithm_name in algorithm_map:
            return algorithm_map[algorithm_name]
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(algorithm_map.keys())}")
