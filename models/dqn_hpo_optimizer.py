# -*- coding: utf-8 -*-
from __future__ import annotations
"""
DQN HPO — 基于Optuna的FE-IDDQN超参数优化

与PPO版本的区别:
  PPO 搜索: eps_clip, k_epochs, gae_lambda, entropy_coef, value_loss_coef ...
  DQN 搜索: epsilon_decay, tau, n_step, per_alpha, per_beta, replay_buffer_size,
            target_update_freq, gradient_clip ...

参考 GA-HPO PPO (Zhou et al., Sensors 2025) 的混合优化思路，
本模块将同样的 Optuna TPE 搜索框架应用于 DQN 超参数空间。
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
class DQNHPOConfig:
    """DQN HPO搜索配置"""
    n_trials: int = 50               # 搜索试验次数
    timeout: int = 3600              # 超时 (秒)
    n_startup_trials: int = 10       # TPE初始随机试验数

    # 评估设置
    eval_episodes: int = 30          # 每种参数组合的评估回合数

    # ─── DQN 搜索范围 ───
    lr_range: Tuple[float, float] = (1e-5, 1e-3)
    gamma_range: Tuple[float, float] = (0.95, 0.999)
    tau_range: Tuple[float, float] = (0.001, 0.05)
    epsilon_decay_range: Tuple[float, float] = (0.990, 0.9999)
    n_step_choices: List[int] = None
    batch_size_choices: List[int] = None
    replay_buffer_size_choices: List[int] = None
    target_update_freq_choices: List[int] = None
    per_alpha_range: Tuple[float, float] = (0.4, 0.8)
    per_beta_start_range: Tuple[float, float] = (0.3, 0.6)
    gradient_clip_range: Tuple[float, float] = (0.5, 5.0)

    seed: int = 42
    study_name: str = 'fe_iddqn_hpo'

    def __post_init__(self):
        if self.n_step_choices is None:
            self.n_step_choices = [1, 3, 5]
        if self.batch_size_choices is None:
            self.batch_size_choices = [32, 64, 128, 256]
        if self.replay_buffer_size_choices is None:
            self.replay_buffer_size_choices = [10000, 50000, 100000]
        if self.target_update_freq_choices is None:
            self.target_update_freq_choices = [50, 100, 200, 500]


class DQNHPOptimizer:
    """
    FE-IDDQN 超参数优化器 (基于Optuna)

    工作流程:
      1. 定义搜索空间 (DQN特有的超参数)
      2. Optuna TPE Sampler 逐次提出候选参数组合
      3. 对每组参数: 构建FE-IDDQN → 短期训练 → 评估
      4. MedianPruner 裁剪不佳试验
      5. 返回最优超参数组合
    """

    def __init__(self, config: Optional[DQNHPOConfig] = None):
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for HPO. "
                "Install it via: pip install optuna")

        self.config = config or DQNHPOConfig()
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
            最优超参数字典 (合并了网络结构)
        """
        cfg = self.config
        self.logger.info(
            f"DQN HPO starting: {cfg.n_trials} trials, "
            f"timeout={cfg.timeout}s")

        sampler = TPESampler(
            seed=cfg.seed,
            n_startup_trials=cfg.n_startup_trials)
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5)

        self.study = optuna.create_study(
            study_name=cfg.study_name,
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
        )

        def _objective(trial: optuna.Trial) -> float:
            return self._trial_objective(
                trial, objective_fn, network_structure)

        self.study.optimize(
            _objective,
            n_trials=cfg.n_trials,
            timeout=cfg.timeout,
            show_progress_bar=True,
        )

        self.best_params = self.study.best_params
        self.logger.info(
            f"DQN HPO complete. Best value: {self.study.best_value:.4f}")
        self.logger.info(f"Best params: {self.best_params}")

        return self._build_full_config(self.best_params, network_structure)

    def _trial_objective(self,
                         trial: optuna.Trial,
                         objective_fn: Callable,
                         network_structure: Optional[Dict]
                         ) -> float:
        """单次试验的目标函数"""
        cfg = self.config

        # ─── 从搜索空间采样 DQN 超参数 ───
        params = {
            'learning_rate': trial.suggest_float(
                'learning_rate', *cfg.lr_range, log=True),
            'gamma': trial.suggest_float(
                'gamma', *cfg.gamma_range),
            'tau': trial.suggest_float(
                'tau', *cfg.tau_range, log=True),
            'epsilon_decay': trial.suggest_float(
                'epsilon_decay', *cfg.epsilon_decay_range),
            'n_step': trial.suggest_categorical(
                'n_step', cfg.n_step_choices),
            'batch_size': trial.suggest_categorical(
                'batch_size', cfg.batch_size_choices),
            'replay_buffer_size': trial.suggest_categorical(
                'replay_buffer_size', cfg.replay_buffer_size_choices),
            'target_update_freq': trial.suggest_categorical(
                'target_update_freq', cfg.target_update_freq_choices),
            'per_alpha': trial.suggest_float(
                'per_alpha', *cfg.per_alpha_range),
            'per_beta_start': trial.suggest_float(
                'per_beta_start', *cfg.per_beta_start_range),
            'gradient_clip': trial.suggest_float(
                'gradient_clip', *cfg.gradient_clip_range),
        }

        if network_structure:
            params['network_structure'] = network_structure

        self.logger.info(
            f"Trial {trial.number}: lr={params['learning_rate']:.2e}, "
            f"gamma={params['gamma']:.4f}, tau={params['tau']:.4f}, "
            f"eps_decay={params['epsilon_decay']:.4f}, "
            f"n_step={params['n_step']}, bs={params['batch_size']}")

        try:
            t0 = time.time()
            result = objective_fn(params)
            elapsed = time.time() - t0

            makespan = result.get('makespan', 1e6)
            utilization = result.get('utilization', 0.0)

            # 综合适应度 (越小越好)
            fitness = (0.6 * makespan / 1000.0
                       + 0.4 * (1.0 - utilization))

            self.logger.info(
                f"Trial {trial.number} done: "
                f"makespan={makespan:.2f}, util={utilization:.4f}, "
                f"fitness={fitness:.4f}, time={elapsed:.1f}s")

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
            'learning_rate': best_params['learning_rate'],
            'gamma': best_params['gamma'],
            'tau': best_params['tau'],
            'epsilon_decay': best_params['epsilon_decay'],
            'n_step': best_params['n_step'],
            'batch_size': best_params['batch_size'],
            'replay_buffer_size': best_params['replay_buffer_size'],
            'target_update_freq': best_params['target_update_freq'],
            'per_alpha': best_params['per_alpha'],
            'per_beta_start': best_params['per_beta_start'],
            'gradient_clip': best_params['gradient_clip'],
        }

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
