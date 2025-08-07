class Hyperparameters:
    """超参数配置类"""
    
    # FE-IDDQN算法超参数
    FE_IDDQN = {
        # 网络结构参数
        'task_stream_hidden_dims': [512, 256, 128],
        'resource_stream_hidden_dims': [512, 256, 128],
        'fusion_dim': 256,
        'output_dim': 64,  # 动作空间大小，根据实际调度动作数量调整
        
        # 训练参数
        'learning_rate': 1e-4,
        'batch_size': 64,
        'gamma': 0.99,  # 折扣因子
        'tau': 0.005,  # 软更新参数
        'target_update_freq': 100,  # 目标网络更新频率
        
        # 经验回放参数
        'replay_buffer_size': 100000,
        'priority_alpha': 0.6,  # 优先级指数
        'priority_beta': 0.4,  # 重要性采样指数
        'priority_beta_increment': 0.001,
        
        # 探索策略参数
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'temperature_start': 1.0,
        'temperature_end': 0.1,
        'temperature_decay': 0.99,
        
        # 训练控制参数
        'max_episodes': 1000,
        'max_steps_per_episode': 500,
        'warmup_steps': 1000,  # 预热步数
        'train_freq': 4,  # 训练频率
        'eval_freq': 100,  # 评估频率
        
        # 注意力机制参数
        'attention_dim': 64,
        'attention_heads': 4,
        'dropout_rate': 0.1,
        
        # 奖励函数权重
        'reward_weights': {
            'makespan': 0.5,
            'resource_utilization': 0.3,
            'load_balance': 0.2
        }
    }
    
    # 基线算法超参数
    BASELINES = {
        # DQN参数
        'DQN': {
            'learning_rate': 1e-4,
            'batch_size': 64,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'replay_buffer_size': 50000,
            'target_update_freq': 100,
            'hidden_dims': [256, 128, 64],
            'max_episodes': 1000
        },
        
        # DDQN参数
        'DDQN': {
            'learning_rate': 1e-4,
            'batch_size': 64,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'replay_buffer_size': 50000,
            'target_update_freq': 100,
            'hidden_dims': [256, 128, 64],
            'max_episodes': 1000
        },
        
        # BF-DDQN参数
        'BF_DDQN': {
            'learning_rate': 1e-4,
            'batch_size': 64,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'replay_buffer_size': 50000,
            'target_update_freq': 100,
            'hidden_dims': [256, 128, 64],
            'max_episodes': 1000,
            'batch_factor': 4  # 批处理因子
        },
        
        # 遗传算法参数
        'GA': {
            'population_size': 100,
            'generations': 500,
            'crossover_prob': 0.8,
            'mutation_prob': 0.1,
            'elite_size': 10,
            'tournament_size': 5
        },
        
        # 粒子群优化参数
        'PSO': {
            'swarm_size': 50,
            'max_iterations': 500,
            'w': 0.9,  # 惯性权重
            'c1': 2.0,  # 个体学习因子
            'c2': 2.0,  # 社会学习因子
            'w_min': 0.4,
            'w_max': 0.9
        },
        
        # 蚁群优化参数
        'ACO': {
            'n_ants': 50,
            'max_iterations': 500,
            'alpha': 1.0,  # 信息素重要程度
            'beta': 2.0,   # 启发式信息重要程度
            'rho': 0.1,    # 信息素挥发率
            'q0': 0.9      # 状态转移规则参数
        }
    }
    
    # 仿真环境参数
    SIMULATION = {
        # 资源配置
        'n_workers': 10,  # Worker节点数量
        'worker_cpu_capacity': [4, 8, 16],  # CPU核心数选项
        'worker_memory_capacity': [8, 16, 32],  # 内存容量选项（GB）
        'worker_network_bandwidth': [100, 1000],  # 网络带宽选项（Mbps）
        
        # 任务配置
        'task_cpu_requirement': [1, 2, 4],  # 任务CPU需求选项
        'task_memory_requirement': [1, 2, 4, 8],  # 任务内存需求选项（GB）
        'task_duration_range': [10, 3600],  # 任务执行时间范围（秒）
        
        # 工作流配置
        'max_workflow_size': 50,  # 最大工作流任务数
        'min_workflow_size': 5,   # 最小工作流任务数
        'dag_density': 0.3,       # DAG密度（边数/最大可能边数）
        
        # 调度配置
        'scheduling_interval': 1,  # 调度间隔（秒）
        'max_simulation_time': 86400,  # 最大仿真时间（秒）
        'arrival_rate': 0.1,      # 工作流到达率（个/秒）
        
        # 负载模拟
        'load_patterns': ['constant', 'peak', 'random'],
        'peak_hours': [9, 10, 11, 14, 15, 16],  # 高峰时段
        'peak_multiplier': 3.0,   # 高峰期负载倍数
    }
    
    # 评估指标配置
    METRICS = {
        'primary_metrics': [
            'makespan',
            'resource_utilization',
            'load_balance',
            'throughput'
        ],
        'secondary_metrics': [
            'energy_efficiency',
            'response_time',
            'queue_length',
            'failure_rate'
        ],
        'statistical_tests': [
            'wilcoxon',
            'mann_whitney',
            't_test'
        ]
    }
    
    @classmethod
    def get_algorithm_params(cls, algorithm_name):
        """获取指定算法的超参数"""
        if algorithm_name == 'FE_IDDQN':
            return cls.FE_IDDQN
        elif algorithm_name in cls.BASELINES:
            return cls.BASELINES[algorithm_name]
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    @classmethod
    def update_params(cls, algorithm_name, new_params):
        """更新算法超参数"""
        if algorithm_name == 'FE_IDDQN':
            cls.FE_IDDQN.update(new_params)
        elif algorithm_name in cls.BASELINES:
            cls.BASELINES[algorithm_name].update(new_params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    @classmethod
    def get_simulation_params(cls):
        """获取仿真环境参数"""
        return cls.SIMULATION
    
    @classmethod
    def get_metrics_config(cls):
        """获取评估指标配置"""
        return cls.METRICS

