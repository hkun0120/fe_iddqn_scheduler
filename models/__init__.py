# Models module for FE-IDDQN scheduler

# 原始模块
from .fe_iddqn import FE_IDDQN
from .dual_stream_network import DualStreamNetwork
from .replay_buffer import PrioritizedReplayBuffer

# 增强版模块
from .gnn_module import DAGAwareModule, MultiHeadGAT, DAGEncoder, CriticalPathEncoder
from .enhanced_network import (
    EnhancedDualStreamNetwork, 
    EnhancedTaskStream, 
    EnhancedResourceStream,
    EnhancedFeatureFusion,
    TransformerEncoderBlock,
    CrossAttentionBlock
)
from .enhanced_replay_buffer import (
    CombinedReplayBuffer,
    NStepReplayBuffer,
    EnhancedPrioritizedReplayBuffer,
    HierarchicalReplayBuffer,
    HindsightExperienceReplay
)
from .exploration_strategies import (
    NoisyLinear,
    NoisyNetwork,
    AdaptiveEpsilonGreedy,
    BoltzmannExploration,
    UCBExploration,
    HeuristicGuidedExploration,
    IntrinsicCuriosityModule,
    CombinedExplorationStrategy
)
from .reward_functions import (
    RewardConfig,
    EnhancedRewardCalculator,
    AdaptiveRewardShaper,
    CurriculumRewardScheduler
)
from .enhanced_fe_iddqn import (
    EnhancedFE_IDDQN,
    EnhancedFE_IDDQN_Config,
    DAGAwareActionMasker,
    LookaheadPlanner
)

__all__ = [
    # 原始模块
    'FE_IDDQN', 
    'DualStreamNetwork', 
    'PrioritizedReplayBuffer',
    
    # GNN模块
    'DAGAwareModule',
    'MultiHeadGAT',
    'DAGEncoder',
    'CriticalPathEncoder',
    
    # 增强版网络
    'EnhancedDualStreamNetwork',
    'EnhancedTaskStream',
    'EnhancedResourceStream',
    'EnhancedFeatureFusion',
    'TransformerEncoderBlock',
    'CrossAttentionBlock',
    
    # 增强版经验回放
    'CombinedReplayBuffer',
    'NStepReplayBuffer',
    'EnhancedPrioritizedReplayBuffer',
    'HierarchicalReplayBuffer',
    'HindsightExperienceReplay',
    
    # 探索策略
    'NoisyLinear',
    'NoisyNetwork',
    'AdaptiveEpsilonGreedy',
    'BoltzmannExploration',
    'UCBExploration',
    'HeuristicGuidedExploration',
    'IntrinsicCuriosityModule',
    'CombinedExplorationStrategy',
    
    # 奖励函数
    'RewardConfig',
    'EnhancedRewardCalculator',
    'AdaptiveRewardShaper',
    'CurriculumRewardScheduler',
    
    # 增强版FE-IDDQN
    'EnhancedFE_IDDQN',
    'EnhancedFE_IDDQN_Config',
    'DAGAwareActionMasker',
    'LookaheadPlanner'
]

