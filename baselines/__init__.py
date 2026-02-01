# Baseline algorithms module for FE-IDDQN scheduler
from .traditional_schedulers import FIFOScheduler, SJFScheduler, HEFTScheduler
try:
    from .rl_baselines import DQNScheduler, DDQNScheduler, BF_DDQNScheduler
except Exception:  # 允许在未安装 torch 时按需使用非RL基线
    DQNScheduler = DDQNScheduler = BF_DDQNScheduler = None
from .meta_heuristics import GAScheduler, PSOScheduler, ACOScheduler

__all__ = [
    'FIFOScheduler', 'SJFScheduler', 'HEFTScheduler',
    'DQNScheduler', 'DDQNScheduler', 'BF_DDQNScheduler',
    'GAScheduler', 'PSOScheduler', 'ACOScheduler'
]

