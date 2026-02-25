# Baseline algorithms module for FE-IDDQN scheduler
from .traditional_schedulers import FIFOScheduler, SJFScheduler, HEFTScheduler
from .meta_heuristics import GAScheduler, PSOScheduler, ACOScheduler

# Try to import RL algorithms, skip if torch is not available
try:
    from .rl_baselines import DQNScheduler, DDQNScheduler, BF_DDQNScheduler
    _rl_available = True
except ImportError:
    _rl_available = False
    DQNScheduler = None
    DDQNScheduler = None
    BF_DDQNScheduler = None

__all__ = [
    'FIFOScheduler', 'SJFScheduler', 'HEFTScheduler',
    'GAScheduler', 'PSOScheduler', 'ACOScheduler'
]

# Add RL algorithms to __all__ only if available
if _rl_available:
    __all__.extend(['DQNScheduler', 'DDQNScheduler', 'BF_DDQNScheduler'])

