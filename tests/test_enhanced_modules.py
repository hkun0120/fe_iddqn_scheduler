"""
单元测试套件 - 验证增强版FE-IDDQN的核心功能

运行命令: python -m pytest tests/test_enhanced_modules.py -v
"""

import unittest
import torch
import numpy as np
import networkx as nx
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入测试模块
from models import (
    MultiHeadGAT,
    DAGAwareModule,
    EnhancedDualStreamNetwork,
    RewardConfig,
    EnhancedRewardCalculator,
    AdaptiveEpsilonGreedy,
    NoisyLinear,
    NStepReplayBuffer,
    CombinedReplayBuffer,
    EnhancedFE_IDDQN,
    EnhancedFE_IDDQN_Config
)

from data import (
    EnhancedStateConfig,
    CriticalPathAnalyzer,
    EnhancedStateEncoder
)

from environment import EnhancedWorkflowSimulator


class TestGNNModule(unittest.TestCase):
    """测试GNN模块"""
    
    def setUp(self):
        self.device = 'cpu'
        self.batch_size = 2
        self.num_nodes = 8
        self.node_dim = 32
    
    def test_graph_attention_layer(self):
        """测试图注意力层"""
        from models.gnn_module import GraphAttentionLayer
        
        layer = GraphAttentionLayer(
            in_dim=self.node_dim,
            out_dim=64,
            num_heads=4,
            dropout=0.1
        ).to(self.device)
        
        node_features = torch.randn(self.batch_size, self.num_nodes, self.node_dim).to(self.device)
        adjacency = torch.ones(self.batch_size, self.num_nodes, self.num_nodes).to(self.device)
        
        output = layer(node_features, adjacency)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_nodes, 64))
    
    def test_multihead_gat(self):
        """测试多头GAT"""
        gat = MultiHeadGAT(
            node_dim=self.node_dim,
            hidden_dim=64,
            output_dim=128,
            num_heads=4,
            num_layers=2
        ).to(self.device)
        
        node_features = torch.randn(self.batch_size, self.num_nodes, self.node_dim).to(self.device)
        adjacency = torch.ones(self.batch_size, self.num_nodes, self.num_nodes).to(self.device)
        
        output = gat(node_features, adjacency)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_nodes, 128))
    
    def test_dag_aware_module(self):
        """测试DAG感知模块"""
        dag_module = DAGAwareModule(
            node_dim=self.node_dim,
            hidden_dim=64,
            output_dim=128
        ).to(self.device)
        
        node_features = torch.randn(self.batch_size, self.num_nodes, self.node_dim).to(self.device)
        adjacency = torch.eye(self.num_nodes).unsqueeze(0).repeat(self.batch_size, 1, 1).to(self.device)
        
        output, graph_embedding = dag_module(node_features, adjacency)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_nodes, 128))
        self.assertEqual(graph_embedding.shape, (self.batch_size, 128))


class TestEnhancedNetwork(unittest.TestCase):
    """测试增强网络"""
    
    def setUp(self):
        self.device = 'cpu'
        self.batch_size = 4
        self.num_tasks = 8
        self.num_resources = 6
        self.task_dim = 64
        self.resource_dim = 32
    
    def test_enhanced_dual_stream_network(self):
        """测试增强双流网络"""
        network = EnhancedDualStreamNetwork(
            task_input_dim=self.task_dim,
            resource_input_dim=self.resource_dim,
            hidden_dim=128,
            num_actions=6,
            num_heads=4,
            dropout=0.1,
            use_dueling=True
        ).to(self.device)
        
        task_features = torch.randn(self.batch_size, self.num_tasks, self.task_dim).to(self.device)
        resource_features = torch.randn(self.batch_size, self.num_resources, self.resource_dim).to(self.device)
        
        q_values = network(task_features, resource_features)
        
        self.assertEqual(q_values.shape, (self.batch_size, 6))


class TestRewardFunction(unittest.TestCase):
    """测试奖励函数"""
    
    def test_reward_config(self):
        """测试奖励配置"""
        config = RewardConfig()
        
        self.assertAlmostEqual(
            config.makespan_weight + 
            config.utilization_weight + 
            config.load_balance_weight +
            config.parallelism_weight +
            config.critical_path_weight +
            config.waiting_time_weight,
            1.0,
            places=6
        )
    
    def test_enhanced_reward_calculator(self):
        """测试增强奖励计算器"""
        config = RewardConfig()
        calculator = EnhancedRewardCalculator(config)
        
        current_state = {
            'completed_tasks': 2,
            'total_tasks': 8,
            'current_time': 20,
            'resource_utilization': [0.5, 0.6, 0.7, 0.4, 0.8, 0.3],
            'waiting_tasks': 2,
            'critical_path_remaining': 50
        }
        
        next_state = {
            'completed_tasks': 3,
            'total_tasks': 8,
            'current_time': 30,
            'resource_utilization': [0.6, 0.7, 0.8, 0.5, 0.7, 0.4],
            'waiting_tasks': 1,
            'critical_path_remaining': 40
        }
        
        reward, components = calculator.calculate(current_state, next_state, action=2, done=False)
        
        self.assertIsInstance(reward, float)
        self.assertIsInstance(components, dict)
        self.assertIn('makespan', components)
        self.assertIn('utilization', components)


class TestExplorationStrategy(unittest.TestCase):
    """测试探索策略"""
    
    def test_adaptive_epsilon_greedy(self):
        """测试自适应ε-greedy"""
        strategy = AdaptiveEpsilonGreedy(
            epsilon_start=1.0,
            epsilon_end=0.01,
            decay_rate=0.99
        )
        
        initial_epsilon = strategy.get_epsilon()
        self.assertAlmostEqual(initial_epsilon, 1.0, places=6)
        
        # 模拟更新
        for _ in range(100):
            strategy.update(0.5)
        
        updated_epsilon = strategy.get_epsilon()
        self.assertLess(updated_epsilon, initial_epsilon)
    
    def test_noisy_linear(self):
        """测试Noisy Linear层"""
        layer = NoisyLinear(10, 20)
        
        x = torch.randn(4, 10)
        output = layer(x)
        
        self.assertEqual(output.shape, (4, 20))
        
        # 重置噪声后应该产生不同的输出
        output1 = layer(x)
        layer.sample_noise()
        output2 = layer(x)
        
        self.assertFalse(torch.allclose(output1, output2))


class TestReplayBuffer(unittest.TestCase):
    """测试经验回放缓冲"""
    
    def test_nstep_replay_buffer(self):
        """测试N-step回放缓冲"""
        buffer = NStepReplayBuffer(
            capacity=100,
            n_step=3,
            gamma=0.99
        )
        
        # 添加经验
        state_dim = 64
        for i in range(50):
            state = np.random.randn(state_dim)
            action = np.random.randint(0, 6)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = (i == 49)
            
            buffer.push(state, action, reward, next_state, done)
        
        self.assertEqual(len(buffer), 50)
        
        # 采样
        if len(buffer) >= 32:
            batch = buffer.sample(32)
            self.assertEqual(len(batch[0]), 32)  # 状态
            self.assertEqual(len(batch[1]), 32)  # 动作
            self.assertEqual(len(batch[2]), 32)  # 回报
    
    def test_combined_replay_buffer(self):
        """测试组合回放缓冲"""
        buffer = CombinedReplayBuffer(
            capacity=100,
            use_per=True,
            use_n_step=True
        )
        
        state_dim = 64
        for i in range(50):
            state = np.random.randn(state_dim)
            action = np.random.randint(0, 6)
            reward = 1.0
            next_state = np.random.randn(state_dim)
            done = (i == 49)
            
            buffer.push(state, action, reward, next_state, done)
        
        self.assertGreater(len(buffer), 0)


class TestCriticalPathAnalyzer(unittest.TestCase):
    """测试关键路径分析器"""
    
    def setUp(self):
        # 创建简单的DAG
        self.tasks = [
            {'id': 0, 'duration': 10},
            {'id': 1, 'duration': 15},
            {'id': 2, 'duration': 20},
            {'id': 3, 'duration': 30},
        ]
        
        self.dependencies = [(0, 1), (1, 2), (2, 3)]
    
    def test_critical_path_analysis(self):
        """测试关键路径分析"""
        analyzer = CriticalPathAnalyzer(max_tasks=10)
        
        # 构建DAG
        dag = nx.DiGraph()
        for task in self.tasks:
            dag.add_node(task['id'], **task)
        for src, dst in self.dependencies:
            dag.add_edge(src, dst)
        
        # 分析
        critical_mask, lengths = analyzer.analyze(dag, self.tasks)
        
        self.assertEqual(len(critical_mask), len(self.tasks))
        self.assertEqual(len(lengths), len(self.tasks))
        
        # 所有任务都在关键路径上
        self.assertEqual(np.sum(critical_mask), len(self.tasks))


class TestEnhancedAgent(unittest.TestCase):
    """测试增强版Agent"""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_enhanced_fe_iddqn_config(self):
        """测试Agent配置"""
        config = EnhancedFE_IDDQN_Config(
            task_input_dim=64,
            resource_input_dim=32,
            hidden_dim=128,
            num_actions=6,
            use_gnn=True,
            use_transformer=True,
            use_n_step=True,
            learning_rate=3e-4
        )
        
        self.assertTrue(config.use_gnn)
        self.assertTrue(config.use_transformer)
        self.assertTrue(config.use_n_step)
    
    def test_enhanced_fe_iddqn_initialization(self):
        """测试Agent初始化"""
        config = EnhancedFE_IDDQN_Config(
            task_input_dim=64,
            resource_input_dim=32,
            hidden_dim=128,
            num_actions=6
        )
        
        agent = EnhancedFE_IDDQN(config).to(self.device)
        
        self.assertIsNotNone(agent.network)
        self.assertIsNotNone(agent.target_network)
        self.assertIsNotNone(agent.memory)


class TestWorkflowSimulation(unittest.TestCase):
    """测试工作流仿真"""
    
    def setUp(self):
        # 创建简单的工作流
        self.tasks = [
            {'id': 0, 'name': 'task_0', 'duration': 10, 'resource_req': 2},
            {'id': 1, 'name': 'task_1', 'duration': 15, 'resource_req': 3},
            {'id': 2, 'name': 'task_2', 'duration': 20, 'resource_req': 4},
        ]
        
        self.dependencies = [(0, 1), (1, 2)]
        
        self.resources = [
            {'id': 0, 'name': 'worker_0', 'capacity': 8, 'speed': 1.0},
            {'id': 1, 'name': 'worker_1', 'capacity': 8, 'speed': 1.0},
        ]
    
    def test_enhanced_workflow_simulator_reset(self):
        """测试仿真环境重置"""
        env = EnhancedWorkflowSimulator(
            self.tasks, 
            self.resources, 
            self.dependencies
        )
        
        state = env.reset()
        
        self.assertIsNotNone(state)
        self.assertIn('task_features', state)
        self.assertIn('resource_features', state)
    
    def test_enhanced_workflow_simulator_step(self):
        """测试仿真环境步骤"""
        env = EnhancedWorkflowSimulator(
            self.tasks, 
            self.resources, 
            self.dependencies
        )
        
        state = env.reset()
        
        # 执行一个动作
        action = 0  # 选择第一个资源
        next_state, reward, done, info = env.step(action)
        
        self.assertIsNotNone(next_state)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestGNNModule))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedNetwork))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestExplorationStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestReplayBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestCriticalPathAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowSimulation))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
