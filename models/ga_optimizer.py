# -*- coding: utf-8 -*-
"""
GA Architecture Optimizer — 遗传算法搜索PPO神经网络架构

参考 GA-HPO PPO (Zhou et al., Sensors 2025):
  - 用GA搜索最优神经网络结构 (层数、宽度、注意力头数等)
  - 每个个体 = 一种网络架构
  - 适应度 = 短期PPO训练后的调度性能 (makespan / 利用率)
  - Pareto-based 选择 (多目标)
  - 支持早停: 若连续N代无改善则终止搜索

搜索空间:
  - hidden_dim:                {64, 128, 256, 512}
  - fusion_dim:                {128, 256, 512}
  - num_transformer_layers:    {1, 2, 3, 4}
  - num_heads:                 {2, 4, 8}
  - dropout:                   {0.05, 0.1, 0.15, 0.2}
  - use_gnn:                   {True, False}
"""

import numpy as np
import random
import logging
import copy
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field


@dataclass
class GAConfig:
    """GA搜索配置"""
    population_size: int = 12        # 种群大小 (较小, 因为每次要训练网络)
    num_generations: int = 10        # 迭代代数
    
    crossover_rate: float = 0.8      # 交叉概率
    mutation_rate: float = 0.3       # 变异概率
    mutation_strength: float = 0.3   # 变异强度
    
    tournament_size: int = 3         # 锦标赛选择的参赛个体数
    elite_ratio: float = 0.2        # 精英保留比例
    
    early_stop_patience: int = 4    # 连续N代无改善则停止
    
    # 短期训练的步数 (用于评估)
    eval_episodes: int = 20
    
    seed: int = 42


# 搜索空间定义
SEARCH_SPACE = {
    'hidden_dim':               [64, 128, 256, 512],
    'fusion_dim':               [128, 256, 512],
    'num_transformer_layers':   [1, 2, 3, 4],
    'num_heads':                [2, 4, 8],
    'dropout':                  [0.05, 0.1, 0.15, 0.2],
    'use_gnn':                  [True, False],
}


class Individual:
    """GA个体 = 一种网络架构"""
    
    def __init__(self, genome: Optional[Dict[str, Any]] = None):
        self.genome: Dict[str, Any] = genome or self._random_genome()
        self.fitness: float = float('inf')
        self.objectives: Dict[str, float] = {}   # 多目标
        self.rank: int = 0                        # Pareto rank
        self.crowding_distance: float = 0.0
        self.eval_time: float = 0.0
    
    @staticmethod
    def _random_genome() -> Dict[str, Any]:
        """随机生成基因组"""
        genome = {}
        for key, choices in SEARCH_SPACE.items():
            genome[key] = random.choice(choices)
        # 约束: num_heads 必须能整除 hidden_dim
        while genome['hidden_dim'] % genome['num_heads'] != 0:
            genome['num_heads'] = random.choice(SEARCH_SPACE['num_heads'])
        return genome
    
    def to_network_structure(self) -> Dict[str, Any]:
        """转化为 GDS_PPO 可用的网络结构字典"""
        return {
            'hidden_dim': self.genome['hidden_dim'],
            'fusion_dim': self.genome['fusion_dim'],
            'num_transformer_layers': self.genome['num_transformer_layers'],
            'num_heads': self.genome['num_heads'],
            'dropout': self.genome['dropout'],
            'use_gnn': self.genome['use_gnn'],
        }
    
    def __repr__(self):
        h = self.genome.get('hidden_dim', '?')
        t = self.genome.get('num_transformer_layers', '?')
        return f"Individual(h={h}, T={t}, fit={self.fitness:.4f})"


class GAArchitectureOptimizer:
    """
    GA网络架构搜索器
    
    工作流程:
      1. 随机初始化种群
      2. 对每个个体: 用该架构构建PPO → 短期训练 → 评估适应度
      3. 多目标选择 (Pareto排序 + 拥挤距离)
      4. 交叉 + 变异 → 新一代
      5. 重复直到收敛或达到最大代数
    """
    
    def __init__(self, config: Optional[GAConfig] = None):
        self.config = config or GAConfig()
        self.logger = logging.getLogger(__name__)
        self._set_seed(self.config.seed)
        
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation_history: List[Dict] = []
    
    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
    
    # ──────── 搜索入口 ──────── #
    
    def search(self, fitness_fn: Callable[[Dict[str, Any]], Dict[str, float]]
               ) -> Dict[str, Any]:
        """
        执行GA架构搜索
        
        Args:
            fitness_fn: 接收网络结构字典, 返回适应度字典
                       e.g. {'makespan': 150.2, 'utilization': 0.85, 'params': 123456}
                       目标: makespan 最小化, utilization 最大化, params 最小化
        
        Returns:
            最佳网络结构字典
        """
        cfg = self.config
        self.logger.info(f"GA Architecture Search starting: "
                         f"pop={cfg.population_size}, gen={cfg.num_generations}")
        
        # 1. 初始化种群
        self.population = [Individual() for _ in range(cfg.population_size)]
        
        best_fitness = float('inf')
        no_improve_count = 0
        
        for gen in range(cfg.num_generations):
            gen_start = time.time()
            
            # 2. 评估适应度
            self._evaluate_population(fitness_fn)
            
            # 3. 非支配排序 (NSGA-II风格)
            fronts = self._non_dominated_sort()
            self._compute_crowding_distance(fronts)
            
            # 更新最优个体
            current_best = min(self.population, key=lambda x: x.fitness)
            if current_best.fitness < best_fitness:
                best_fitness = current_best.fitness
                self.best_individual = copy.deepcopy(current_best)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            gen_time = time.time() - gen_start
            self.generation_history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'current_best': current_best.fitness,
                'avg_fitness': np.mean([ind.fitness for ind in self.population]),
                'time': gen_time,
                'best_genome': self.best_individual.genome.copy(),
            })
            
            self.logger.info(
                f"Gen {gen+1}/{cfg.num_generations}: "
                f"best={best_fitness:.4f}, "
                f"avg={self.generation_history[-1]['avg_fitness']:.4f}, "
                f"time={gen_time:.1f}s")
            
            # 早停检查
            if no_improve_count >= cfg.early_stop_patience:
                self.logger.info(
                    f"Early stopping at generation {gen+1}, "
                    f"no improvement for {no_improve_count} generations")
                break
            
            # 最后一代不需要演化
            if gen == cfg.num_generations - 1:
                break
            
            # 4. 选择 + 交叉 + 变异
            new_population = self._evolve()
            self.population = new_population
        
        self.logger.info(f"GA Search complete. Best: {self.best_individual}")
        return self.best_individual.to_network_structure()
    
    # ──────── 适应度评估 ──────── #
    
    def _evaluate_population(self,
                              fitness_fn: Callable[[Dict], Dict[str, float]]):
        """评估整个种群的适应度"""
        for ind in self.population:
            if ind.fitness < float('inf') - 1:
                continue  # 跳过已评估的精英
            
            t0 = time.time()
            try:
                result = fitness_fn(ind.to_network_structure())
                ind.objectives = result
                # 主适应度: 综合指标 (越小越好)
                ind.fitness = self._aggregate_fitness(result)
                ind.eval_time = time.time() - t0
            except Exception as e:
                self.logger.warning(
                    f"Evaluation failed for {ind.genome}: {e}")
                ind.fitness = float('inf')
                ind.eval_time = time.time() - t0
    
    @staticmethod
    def _aggregate_fitness(objectives: Dict[str, float]) -> float:
        """
        多目标 → 单目标聚合
        
        目标:
          - makespan: 最小化 (权重0.5)
          - utilization: 最大化 (转为最小化) (权重0.3)
          - params: 最小化 — 偏好轻量模型 (权重0.2)
        """
        makespan = objectives.get('makespan', 1e6)
        utilization = objectives.get('utilization', 0.0)
        params = objectives.get('params', 0)
        
        # 归一化 (大致范围)
        makespan_norm = makespan / 1000.0
        util_penalty = 1.0 - utilization
        param_norm = params / 1e6
        
        return (0.5 * makespan_norm
                + 0.3 * util_penalty
                + 0.2 * param_norm)
    
    # ──────── NSGA-II 非支配排序 ──────── #
    
    def _non_dominated_sort(self) -> List[List[int]]:
        """非支配排序"""
        n = len(self.population)
        domination_count = [0] * n
        dominated_set: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(self.population[i], self.population[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(self.population[j], self.population[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                self.population[i].rank = 0
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        self.population[j].rank = k + 1
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        
        return [f for f in fronts if f]  # 去掉空front
    
    @staticmethod
    def _dominates(a: Individual, b: Individual) -> bool:
        """a是否支配b (所有目标不差且至少一个更好)"""
        objs_a = a.objectives
        objs_b = b.objectives
        if not objs_a or not objs_b:
            return a.fitness < b.fitness
        
        # 假设所有目标都是最小化 (utilization已取反)
        all_leq = True
        any_lt = False
        for key in objs_a:
            if key == 'utilization':
                va = -objs_a.get(key, 0)
                vb = -objs_b.get(key, 0)
            else:
                va = objs_a.get(key, float('inf'))
                vb = objs_b.get(key, float('inf'))
            if va > vb:
                all_leq = False
                break
            if va < vb:
                any_lt = True
        
        return all_leq and any_lt
    
    def _compute_crowding_distance(self, fronts: List[List[int]]):
        """计算拥挤距离"""
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    self.population[i].crowding_distance = float('inf')
                continue
            
            for i in front:
                self.population[i].crowding_distance = 0.0
            
            obj_keys = list(self.population[front[0]].objectives.keys()) or ['fitness']
            
            for key in obj_keys:
                sorted_front = sorted(
                    front,
                    key=lambda i: self.population[i].objectives.get(
                        key, self.population[i].fitness))
                
                self.population[sorted_front[0]].crowding_distance = float('inf')
                self.population[sorted_front[-1]].crowding_distance = float('inf')
                
                obj_min = self.population[sorted_front[0]].objectives.get(
                    key, self.population[sorted_front[0]].fitness)
                obj_max = self.population[sorted_front[-1]].objectives.get(
                    key, self.population[sorted_front[-1]].fitness)
                obj_range = obj_max - obj_min
                
                if obj_range < 1e-10:
                    continue
                
                for k in range(1, len(sorted_front) - 1):
                    prev_val = self.population[sorted_front[k-1]].objectives.get(
                        key, self.population[sorted_front[k-1]].fitness)
                    next_val = self.population[sorted_front[k+1]].objectives.get(
                        key, self.population[sorted_front[k+1]].fitness)
                    self.population[sorted_front[k]].crowding_distance += \
                        (next_val - prev_val) / obj_range
    
    # ──────── 演化算子 ──────── #
    
    def _evolve(self) -> List[Individual]:
        """选择 + 交叉 + 变异 → 新种群"""
        cfg = self.config
        
        # 精英保留
        elite_count = max(1, int(cfg.population_size * cfg.elite_ratio))
        sorted_pop = sorted(
            self.population,
            key=lambda x: (x.rank, -x.crowding_distance))
        new_pop = [copy.deepcopy(ind) for ind in sorted_pop[:elite_count]]
        
        # 生成后代填满种群
        while len(new_pop) < cfg.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            if random.random() < cfg.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = (
                    copy.deepcopy(parent1), copy.deepcopy(parent2))
            
            self._mutate(child1)
            self._mutate(child2)
            
            # 确保约束
            self._repair(child1)
            self._repair(child2)
            
            new_pop.append(child1)
            if len(new_pop) < cfg.population_size:
                new_pop.append(child2)
        
        return new_pop[:cfg.population_size]
    
    def _tournament_select(self) -> Individual:
        """锦标赛选择"""
        candidates = random.sample(
            self.population,
            min(self.config.tournament_size, len(self.population)))
        return min(candidates, key=lambda x: (x.rank, -x.crowding_distance))
    
    @staticmethod
    def _crossover(parent1: Individual,
                   parent2: Individual) -> Tuple[Individual, Individual]:
        """均匀交叉"""
        child1_genome = {}
        child2_genome = {}
        for key in SEARCH_SPACE:
            if random.random() < 0.5:
                child1_genome[key] = parent1.genome[key]
                child2_genome[key] = parent2.genome[key]
            else:
                child1_genome[key] = parent2.genome[key]
                child2_genome[key] = parent1.genome[key]
        return Individual(child1_genome), Individual(child2_genome)
    
    def _mutate(self, individual: Individual):
        """变异: 随机替换基因"""
        for key in SEARCH_SPACE:
            if random.random() < self.config.mutation_rate:
                individual.genome[key] = random.choice(SEARCH_SPACE[key])
    
    @staticmethod
    def _repair(individual: Individual):
        """修复约束: num_heads 必须整除 hidden_dim"""
        h = individual.genome['hidden_dim']
        nh = individual.genome['num_heads']
        if h % nh != 0:
            valid = [n for n in SEARCH_SPACE['num_heads'] if h % n == 0]
            individual.genome['num_heads'] = (
                random.choice(valid) if valid else 2)
    
    # ──────── 工具方法 ──────── #
    
    def get_search_summary(self) -> Dict[str, Any]:
        """获取搜索摘要"""
        return {
            'total_generations': len(self.generation_history),
            'best_fitness': self.best_individual.fitness if self.best_individual else None,
            'best_genome': self.best_individual.genome if self.best_individual else None,
            'history': self.generation_history,
        }
