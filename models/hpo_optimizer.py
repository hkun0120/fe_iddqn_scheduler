# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Optuna HPO — 基于Optuna的PPO超参数优化

参考 GA-HPO PPO (Zhou et al., Sensors 2025):
  - 使用Optuna TPE (Tree-structured Parzen Estimator) 搜索PPO超参数
  - 在GA确定最优网络架构后运行
  - 搜索范围: learning_rate, gamma, eps_clip, K_epochs,
              GAE lambda, batch_size, entropy_coef 等

搜索策略:
  - TPE Sampler (默认)
  - MedianPruner 早停不佳的试验
  - 多目标: makespan + utilization (Pareto)
"""

import logging
import time
from typing import Dict, Optional, Callable, Any, List, Tuple
from dataclasses import dataclass

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import numpy as np


@dataclass
class HPOConfig:
    """HPO搜索配置"""
    n_trials: int = 50               # 搜索试验次数
    timeout: int = 3600              # 超时 (秒)
    n_startup_trials: int = 10       # TPE初始随机试验数
    
    # 评估设置
    eval_episodes: int = 30          # 每种参数组合的评估回合数
    
    # 搜索范围
    lr_range: Tuple[float, float] = (1e-5, 1e-3)
    gamma_range: Tuple[float, float] = (0.95, 0.999)
    eps_clip_range: Tuple[float, float] = (0.1, 0.3)
    k_epochs_range: Tuple[int, int] = (3, 15)
    gae_lambda_range: Tuple[float, float] = (0.9, 0.99)
    batch_size_choices: List[int] = None
    entropy_coef_range: Tuple[float, float] = (0.001, 0.05)
    value_loss_coef_range: Tuple[float, float] = (0.25, 1.0)
    
    seed: int = 42
    study_name: str = 'gds_ppo_hpo'
    
    def __post_init__(self):
        if self.batch_size_choices is None:
            self.batch_size_choices = [32, 64, 128, 256]


class OptunaHPO:
    """
    Optuna超参数优化器
    
    工作流程:
      1. 定义搜索空间 (PPO的核心超参数)
      2. Optuna TPE Sampler 逐次提出候选参数组合
      3. 对每组参数: 构建PPO → 短期训练 → 评估
      4. MedianPruner 裁剪不佳试验
      5. 返回最优超参数组合
    """
    
    def __init__(self, config: Optional[HPOConfig] = None):
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for HPO. "
                "Install it via: pip install optuna")
        
        self.config = config or HPOConfig()
        self.logger = logging.getLogger(__name__)
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
    
    def optimize(self,
                 objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
                 network_structure: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
        """
        执行超参数搜索
        
        Args:
            objective_fn: 接收超参数字典, 返回性能指标字典
                         e.g. {'makespan': 150, 'utilization': 0.85}
            network_structure: GA搜索得到的最优网络结构 (可选)
        
        Returns:
            最优超参数字典
        """
        cfg = self.config
        self.logger.info(
            f"Optuna HPO starting: {cfg.n_trials} trials, "
            f"timeout={cfg.timeout}s")
        
        # 创建Study
        sampler = TPESampler(
            seed=cfg.seed,
            n_startup_trials=cfg.n_startup_trials)
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5)
        
        self.study = optuna.create_study(
            study_name=cfg.study_name,
            direction='minimize',       # 最小化综合指标
            sampler=sampler,
            pruner=pruner,
        )
        
        # 封装objective函数
        def _objective(trial: optuna.Trial) -> float:
            return self._trial_objective(
                trial, objective_fn, network_structure)
        
        # 运行搜索
        self.study.optimize(
            _objective,
            n_trials=cfg.n_trials,
            timeout=cfg.timeout,
            show_progress_bar=True,
        )
        
        # 提取最优参数
        self.best_params = self.study.best_params
        
        self.logger.info(
            f"HPO complete. Best value: {self.study.best_value:.4f}")
        self.logger.info(f"Best params: {self.best_params}")
        
        return self._build_full_config(self.best_params, network_structure)
    
    def _trial_objective(self,
                         trial: optuna.Trial,
                         objective_fn: Callable,
                         network_structure: Optional[Dict]
                         ) -> float:
        """单次试验的目标函数"""
        cfg = self.config
        
        # 从搜索空间中采样超参数
        params = {
            'learning_rate': trial.suggest_float(
                'learning_rate', *cfg.lr_range, log=True),
            'gamma': trial.suggest_float(
                'gamma', *cfg.gamma_range),
            'eps_clip': trial.suggest_float(
                'eps_clip', *cfg.eps_clip_range),
            'k_epochs': trial.suggest_int(
                'k_epochs', *cfg.k_epochs_range),
            'gae_lambda': trial.suggest_float(
                'gae_lambda', *cfg.gae_lambda_range),
            'batch_size': trial.suggest_categorical(
                'batch_size', cfg.batch_size_choices),
            'entropy_coef': trial.suggest_float(
                'entropy_coef', *cfg.entropy_coef_range, log=True),
            'value_loss_coef': trial.suggest_float(
                'value_loss_coef', *cfg.value_loss_coef_range),
        }
        
        # 加入网络结构信息 (如果有)
        if network_structure:
            params['network_structure'] = network_structure
        
        self.logger.info(
            f"Trial {trial.number}: lr={params['learning_rate']:.2e}, "
            f"gamma={params['gamma']:.4f}, "
            f"eps_clip={params['eps_clip']:.3f}, "
            f"K={params['k_epochs']}")
        
        try:
            t0 = time.time()
            result = objective_fn(params)
            elapsed = time.time() - t0
            
            # 综合适应度 (越小越好)
            makespan = result.get('makespan', 1e6)
            utilization = result.get('utilization', 0.0)
            
            fitness = (0.6 * makespan / 1000.0
                       + 0.4 * (1.0 - utilization))
            
            self.logger.info(
                f"Trial {trial.number} done: "
                f"makespan={makespan:.2f}, util={utilization:.4f}, "
                f"fitness={fitness:.4f}, time={elapsed:.1f}s")
            
            # 记录中间指标 (用于Pruner)
            trial.report(fitness, step=0)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return fitness
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.warning(f"Trial {trial.number} failed: {e}")
            return float('inf')
    
    def _build_full_config(self,
                           best_params: Dict[str, Any],
                           network_structure: Optional[Dict]
                           ) -> Dict[str, Any]:
        """构建完整配置 (合并网络结构 + 最优超参数)"""
        full_config = {
            # PPO核心超参数 (Optuna搜索得到)
            'learning_rate': best_params['learning_rate'],
            'gamma': best_params['gamma'],
            'eps_clip': best_params['eps_clip'],
            'k_epochs': best_params['k_epochs'],
            'gae_lambda': best_params['gae_lambda'],
            'batch_size': best_params['batch_size'],
            'entropy_coef': best_params['entropy_coef'],
            'value_loss_coef': best_params['value_loss_coef'],
        }
        
        # 网络结构 (GA搜索得到)
        if network_structure:
            full_config['network_structure'] = network_structure
        
        return full_config
    
    def get_search_summary(self) -> Dict[str, Any]:
        """获取搜索摘要"""
        if self.study is None:
            return {'status': 'not_started'}
        
        return {
            'n_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'best_trial': self.study.best_trial.number,
            'n_pruned': len([
                t for t in self.study.trials
                if t.state == optuna.trial.TrialState.PRUNED]),
            'n_complete': len([
                t for t in self.study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]),
        }
    
    def get_param_importances(self) -> Dict[str, float]:
        """获取参数重要性分析"""
        if self.study is None:
            return {}
        try:
            importances = optuna.importance.get_param_importances(self.study)
            return dict(importances)
        except Exception:
            return {}
