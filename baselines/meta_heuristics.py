import numpy as np
import random
import logging
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from baselines.traditional_schedulers import BaseScheduler
from config.hyperparameters import Hyperparameters
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

class Individual:
    """遗传算法个体"""
    
    def __init__(self, chromosome: List[int], fitness: float = 0.0):
        self.chromosome = chromosome
        self.fitness = fitness
    
    def __lt__(self, other):
        return self.fitness < other.fitness

class Particle:
    """粒子群优化粒子"""
    
    def __init__(self, position: List[int], velocity: List[float]):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')

class Ant:
    """蚁群优化蚂蚁"""
    
    def __init__(self, num_tasks: int):
        self.path = []
        self.visited = set()
        self.fitness = float('inf')

class GAScheduler(BaseScheduler):
    """遗传算法调度器"""
    
    def __init__(self, use_parallel=True, max_workers=None):
        super().__init__("GA")
        self.params = Hyperparameters.get_algorithm_params("GA")
        self.use_parallel = use_parallel
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # 限制最大线程数
    
    def schedule(self, tasks: List[Dict], resources: List[Dict], 
                dependencies: List[Tuple[int, int]]) -> Dict:
        """遗传算法调度"""
        self.logger.info(f"Scheduling {len(tasks)} tasks using GA")
        
        num_tasks = len(tasks)
        num_resources = len(resources)
        
        # 初始化种群
        population = self._initialize_population(num_tasks, num_resources)
        
        best_individual = None
        best_fitness = float('inf')
        
        for generation in range(self.params["generations"]):
            # 并行评估适应度
            if self.use_parallel and len(population) > 4:  # 只有个体数较多时才使用并行
                population = self._evaluate_fitness_parallel(population, tasks, resources, dependencies)
            else:
                # 串行评估适应度
                for i, individual in enumerate(population):
                    individual.fitness = self._evaluate_fitness(
                        individual.chromosome, tasks, resources, dependencies
                    )
                    
                    if individual.fitness < best_fitness:
                        best_fitness = individual.fitness
                        best_individual = individual
                    
                    # 每10个个体输出一次进度
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"GA Generation {generation+1}/{self.params['generations']}: "
                                       f"Evaluated {i+1}/{len(population)} individuals, "
                                       f"Best fitness: {best_fitness:.2f}")
            
            # 更新最优解
            for individual in population:
                if individual.fitness < best_fitness:
                    best_fitness = individual.fitness
                    best_individual = individual
            
            # 每代输出一次进度
            self.logger.info(f"GA Generation {generation+1}/{self.params['generations']} completed, "
                           f"Best fitness: {best_fitness:.2f}")
            
            # 选择
            selected = self._selection(population)
            
            # 交叉
            offspring = self._crossover(selected)
            
            # 变异
            offspring = self._mutation(offspring, num_resources)
            
            # 精英保留
            population = self._elitism(population, offspring)
        
        # 构建调度结果
        if best_individual:
            return self._build_schedule_result(
                best_individual.chromosome, tasks, resources, dependencies
            )
        else:
            return {"algorithm": self.name, "makespan": float('inf'), "resource_utilization": 0}
    
    def _initialize_population(self, num_tasks: int, num_resources: int) -> List[Individual]:
        """初始化种群"""
        population = []
        for _ in range(self.params["population_size"]):
            chromosome = [random.randint(0, num_resources - 1) for _ in range(num_tasks)]
            population.append(Individual(chromosome))
        return population
    
    def _evaluate_fitness_parallel(self, population: List[Individual], tasks: List[Dict], 
                                  resources: List[Dict], dependencies: List[Tuple[int, int]]) -> List[Individual]:
        """并行评估种群适应度"""
        try:
            # 创建适应度评估函数
            evaluate_func = partial(self._evaluate_fitness, 
                                  tasks=tasks, resources=resources, dependencies=dependencies)
            
            # 提取所有染色体
            chromosomes = [individual.chromosome for individual in population]
            
            # 并行计算适应度
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                fitness_values = list(executor.map(evaluate_func, chromosomes))
            
            # 更新个体适应度
            for individual, fitness in zip(population, fitness_values):
                individual.fitness = fitness
            
            return population
            
        except Exception as e:
            self.logger.error(f"Parallel fitness evaluation failed: {e}")
            # 回退到串行评估
            for individual in population:
                individual.fitness = self._evaluate_fitness(
                    individual.chromosome, tasks, resources, dependencies
                )
            return population
    
    def _parse_dependencies(self, dependencies):
        """解析依赖关系，支持不同数据格式"""
        if isinstance(dependencies, list):
            return dependencies
        elif hasattr(dependencies, 'iterrows'):  # DataFrame
            return [(row[0], row[1]) for _, row in dependencies.iterrows()]
        else:
            return []
    
    def _evaluate_fitness(self, chromosome: List[int], tasks: List[Dict], 
                         resources: List[Dict], dependencies: List[Tuple[int, int]]) -> float:
        """评估个体适应度（makespan）"""
        try:
            COMM_RATIO = 0.1
            # 解析依赖关系
            dep_list = self._parse_dependencies(dependencies)
            # 构建任务分配
            task_assignments = {tasks[i]['id']: chromosome[i] for i in range(len(tasks))}
            
            # 计算makespan
            resource_end_times = [0] * len(resources)
            task_end_times = {}
            
            # 按拓扑顺序处理任务
            processed = set()
            max_iterations = len(tasks) * 2  # 防止无限循环
            iteration = 0
            
            while len(processed) < len(tasks) and iteration < max_iterations:
                iteration += 1
                tasks_processed_this_round = 0
                # 减少调试输出，只在每100个任务或每10次迭代输出一次
                if len(processed) % 100 == 0 or iteration % 10 == 0:
                    print(f"Processing {len(processed)} tasks, iteration {iteration}, total {len(tasks)}")
                
                for i, task in enumerate(tasks):
                    print(f"Processing task {task['id']}, iteration {iteration}, total {len(tasks)}")
                    if task['id'] in processed:
                        print(f"Task {task['id']} already processed")
                        continue
                    print(f"Task {task['id']} not processed")
                    # 检查依赖是否满足
                    dependencies_satisfied = True
                    max_dependency_end = 0
                    for pre_task, post_task in dep_list:
                        # 确保ID类型匹配
                        if str(post_task) == str(task['id']):
                            if str(pre_task) not in [str(tid) for tid in task_end_times.keys()]:
                                print(f"Task {task['id']} has no dependency")
                                dependencies_satisfied = False
                                break
                            # 找到对应的任务ID，含通信延迟
                            for tid, end_time in task_end_times.items():
                                if str(tid) == str(pre_task):
                                    comm = 0.0
                                    pre_res = task_assignments.get(tid)
                                    cur_res = task_assignments.get(task['id'])
                                    if pre_res is not None and cur_res is not None and pre_res != cur_res:
                                        pre_dur = [tt for tt in tasks if tt['id'] == tid][0]['duration']
                                        avg_dur = (pre_dur + task['duration']) / 2.0
                                        comm = COMM_RATIO * avg_dur
                                    max_dependency_end = max(max_dependency_end, end_time + comm)
                                    break
                    
                    if dependencies_satisfied:
                        resource_id = chromosome[i]
                        start_time = max(resource_end_times[resource_id], max_dependency_end)
                        speed = float(resources[resource_id].get('speed_factor', 1.0))
                        end_time = start_time + task['duration'] * speed
                        print(f"Task dependencies_satisfied {task['id']} processed, start_time {start_time}, end_time {end_time}")
                        task_end_times[task['id']] = end_time
                        resource_end_times[resource_id] = end_time
                        processed.add(task['id'])
                        tasks_processed_this_round += 1
                
                # 如果这一轮没有处理任何任务，说明有循环依赖或数据问题
                if tasks_processed_this_round == 0:
                    print(f"警告：检测到循环依赖或数据问题，剩余 {len(tasks) - len(processed)} 个任务无法处理")
                    print(f"已处理任务: {[task['id'] for task in tasks if task['id'] in processed]}")
                    print(f"未处理任务: {[task['id'] for task in tasks if task['id'] not in processed]}")
                    print(f"依赖关系: {dependencies[:5]}...")  # 只显示前5个依赖关系
                    # 强制处理剩余任务，忽略依赖关系
                    for i, task in enumerate(tasks):
                        if task['id'] not in processed:
                            resource_id = chromosome[i] % len(resources)
                            start_time = resource_end_times[resource_id]
                            end_time = start_time + task['duration']
                            
                            task_end_times[task['id']] = end_time
                            resource_end_times[resource_id] = end_time
                            processed.add(task['id'])
                    break  # 强制处理完所有任务后退出循环
            
            return max(resource_end_times) if resource_end_times else float('inf')
        except Exception as e:
            self.logger.error(f"Error evaluating fitness: {e}", exc_info=True)  # 打印堆栈
            return float('inf')
    
    def _selection(self, population: List[Individual]) -> List[Individual]:
        """锦标赛选择"""
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(population, self.params["tournament_size"])
            winner = min(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        return selected
    
    def _crossover(self, parents: List[Individual]) -> List[Individual]:
        """单点交叉"""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents) and random.random() < self.params["crossover_prob"]:
                parent1, parent2 = parents[i], parents[i + 1]
                
                # 处理只有1个任务的情况
                if len(parent1.chromosome) <= 1:
                    # 如果染色体长度<=1，直接复制
                    child1_chromosome = parent1.chromosome.copy()
                    child2_chromosome = parent2.chromosome.copy()
                else:
                    crossover_point = random.randint(1, len(parent1.chromosome) - 1)
                    
                    child1_chromosome = (parent1.chromosome[:crossover_point] + 
                                       parent2.chromosome[crossover_point:])
                    child2_chromosome = (parent2.chromosome[:crossover_point] + 
                                       parent1.chromosome[crossover_point:])
                
                offspring.extend([Individual(child1_chromosome), Individual(child2_chromosome)])
            else:
                offspring.extend([parents[i], parents[i + 1] if i + 1 < len(parents) else parents[i]])
        
        return offspring
    
    def _mutation(self, population: List[Individual], num_resources: int) -> List[Individual]:
        """随机变异"""
        for individual in population:
            for i in range(len(individual.chromosome)):
                if random.random() < self.params["mutation_prob"]:
                    individual.chromosome[i] = random.randint(0, num_resources - 1)
        return population
    
    def _elitism(self, population: List[Individual], offspring: List[Individual]) -> List[Individual]:
        """精英保留"""
        combined = population + offspring
        combined.sort(key=lambda x: x.fitness)
        return combined[:self.params["population_size"]]
    
    def _build_schedule_result(self, chromosome: List[int], tasks: List[Dict], 
                              resources: List[Dict], dependencies: List[Tuple[int, int]]) -> Dict:
        """构建调度结果"""
        # 解析依赖关系
        dep_list = self._parse_dependencies(dependencies)
        
        task_assignments = {tasks[i]['id']: chromosome[i] for i in range(len(tasks))}
        
        # 重新计算详细的调度信息
        resource_end_times = [0] * len(resources)
        task_start_times = {}
        task_end_times = {}
        processed = set()
        
        while len(processed) < len(tasks):
            for i, task in enumerate(tasks):
                if task['id'] in processed:
                    continue
                
                dependencies_satisfied = True
                max_dependency_end = 0
                for pre_task, post_task in dep_list:
                    if post_task == task['id']:
                        if pre_task not in task_end_times:
                            dependencies_satisfied = False
                            break
                        max_dependency_end = max(max_dependency_end, task_end_times[pre_task])
                
                if dependencies_satisfied:
                    resource_id = chromosome[i]
                    start_time = max(resource_end_times[resource_id], max_dependency_end)
                    speed = float(resources[resource_id].get('speed_factor', 1.0))
                    end_time = start_time + task['duration'] * speed
                    
                    task_start_times[task['id']] = start_time
                    task_end_times[task['id']] = end_time
                    resource_end_times[resource_id] = end_time
                    processed.add(task['id'])
        
        makespan = max(resource_end_times)
        total_work = sum(task['duration'] for task in tasks)
        total_capacity = makespan * len(resources)
        resource_utilization = total_work / total_capacity if total_capacity > 0 else 0
        
        return {
            'task_assignments': task_assignments,
            'task_start_times': task_start_times,
            'task_end_times': task_end_times,
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'algorithm': self.name
        }

class PSOScheduler(BaseScheduler):
    """粒子群优化调度器"""
    
    def __init__(self):
        super().__init__("PSO")
        self.params = Hyperparameters.get_algorithm_params("PSO")
    
    def schedule(self, tasks: List[Dict], resources: List[Dict], 
                dependencies: List[Tuple[int, int]]) -> Dict:
        """粒子群优化调度"""
        self.logger.info(f"Scheduling {len(tasks)} tasks using PSO")
        
        num_tasks = len(tasks)
        num_resources = len(resources)
        
        # 初始化粒子群
        swarm = self._initialize_swarm(num_tasks, num_resources)
        global_best_position = None
        global_best_fitness = float('inf')
        
        for iteration in range(self.params["max_iterations"]):
            # 更新惯性权重
            w = self.params["w_max"] - (self.params["w_max"] - self.params["w_min"]) * iteration / self.params["max_iterations"]
            
            for particle in swarm:
                # 评估适应度
                particle.fitness = self._evaluate_fitness(
                    particle.position, tasks, resources, dependencies
                )
                
                # 更新个体最优
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                
                # 更新全局最优
                if particle.fitness < global_best_fitness:
                    global_best_fitness = particle.fitness
                    global_best_position = particle.position.copy()
            
            # 更新粒子速度和位置
            for particle in swarm:
                self._update_particle(particle, global_best_position, w)
        
        # 构建调度结果
        if global_best_position:
            return self._build_schedule_result(
                global_best_position, tasks, resources, dependencies
            )
        else:
            return {"algorithm": self.name, "makespan": float('inf'), "resource_utilization": 0}
    
    def _initialize_swarm(self, num_tasks: int, num_resources: int) -> List[Particle]:
        """初始化粒子群"""
        swarm = []
        for _ in range(self.params["swarm_size"]):
            position = [random.randint(0, num_resources - 1) for _ in range(num_tasks)]
            velocity = [random.uniform(-1, 1) for _ in range(num_tasks)]
            swarm.append(Particle(position, velocity))
        return swarm
    
    def _evaluate_fitness(self, position: List[int], tasks: List[Dict], 
                         resources: List[Dict], dependencies: List[Tuple[int, int]]) -> float:
        """评估粒子适应度"""
        # 与GA相同的适应度评估方法
        try:
            # 解析依赖关系
            dep_list = self._parse_dependencies(dependencies)
            resource_end_times = [0] * len(resources)
            task_end_times = {}
            processed = set()
            
            max_iterations = len(tasks) * 2  # 防止无限循环
            iteration = 0
            
            while len(processed) < len(tasks) and iteration < max_iterations:
                iteration += 1
                tasks_processed_this_round = 0
                
                for i, task in enumerate(tasks):
                    if task['id'] in processed:
                        continue
                    
                    dependencies_satisfied = True
                    max_dependency_end = 0
                    for pre_task, post_task in dep_list:
                        # 确保ID类型匹配
                        if str(post_task) == str(task['id']):
                            if str(pre_task) not in [str(tid) for tid in task_end_times.keys()]:
                                dependencies_satisfied = False
                                break
                            # 找到对应的任务ID
                            for tid, end_time in task_end_times.items():
                                if str(tid) == str(pre_task):
                                    max_dependency_end = max(max_dependency_end, end_time)
                                    break
                    
                if dependencies_satisfied:
                    resource_id = position[i] % len(resources)
                    start_time = max(resource_end_times[resource_id], max_dependency_end)
                    speed = float(resources[resource_id].get('speed_factor', 1.0))
                    end_time = start_time + task['duration'] * speed
                    task_end_times[task['id']] = end_time
                    resource_end_times[resource_id] = end_time
                    processed.add(task['id'])
                    tasks_processed_this_round += 1
                
                # 如果这一轮没有处理任何任务，说明有循环依赖或数据问题
                if tasks_processed_this_round == 0:
                    print(f"PSO警告：检测到循环依赖或数据问题，剩余 {len(tasks) - len(processed)} 个任务无法处理")
                    return float('inf')
            
            return max(resource_end_times) if resource_end_times else float('inf')
        except Exception as e:
            self.logger.error(f"Error evaluating fitness: {e}")
            return float('inf')
    
    def _parse_dependencies(self, dependencies):
        """解析依赖关系，支持list或DataFrame"""
        if isinstance(dependencies, list):
            return dependencies
        elif hasattr(dependencies, 'iterrows'):
            return [(row[0], row[1]) for _, row in dependencies.iterrows()]
        else:
            return []
    
    def _update_particle(self, particle: Particle, global_best_position: List[int], w: float):
        """更新粒子速度和位置"""
        for i in range(len(particle.velocity)):
            r1, r2 = random.random(), random.random()
            
            # 更新速度
            particle.velocity[i] = (w * particle.velocity[i] + 
                                  self.params["c1"] * r1 * (particle.best_position[i] - particle.position[i]) +
                                  self.params["c2"] * r2 * (global_best_position[i] - particle.position[i]))
            
            # 更新位置
            particle.position[i] = int(particle.position[i] + particle.velocity[i])
            # 确保位置在有效范围内（0到资源数量-1）
            particle.position[i] = max(0, min(particle.position[i], 5))  # 假设有6个资源（0-5）
    
    def _build_schedule_result(self, position: List[int], tasks: List[Dict], 
                              resources: List[Dict], dependencies: List[Tuple[int, int]]) -> Dict:
        """构建调度结果"""
        # 解析依赖关系
        dep_list = self._parse_dependencies(dependencies)
        
        # 与GA相同的结果构建方法
        task_assignments = {tasks[i]['id']: position[i] for i in range(len(tasks))}
        
        resource_end_times = [0] * len(resources)
        task_start_times = {}
        task_end_times = {}
        processed = set()
        
        while len(processed) < len(tasks):
            for i, task in enumerate(tasks):
                if task['id'] in processed:
                    continue
                
                dependencies_satisfied = True
                max_dependency_end = 0
                for pre_task, post_task in dep_list:
                    if post_task == task['id']:
                        if pre_task not in task_end_times:
                            dependencies_satisfied = False
                            break
                        max_dependency_end = max(max_dependency_end, task_end_times[pre_task])
                
                if dependencies_satisfied:
                    resource_id = position[i]
                    start_time = max(resource_end_times[resource_id], max_dependency_end)
                    speed = float(resources[resource_id].get('speed_factor', 1.0))
                    end_time = start_time + task['duration'] * speed
                    
                    task_start_times[task['id']] = start_time
                    task_end_times[task['id']] = end_time
                    resource_end_times[resource_id] = end_time
                    processed.add(task['id'])
        
        makespan = max(resource_end_times)
        total_work = sum(task['duration'] for task in tasks)
        total_capacity = makespan * len(resources)
        resource_utilization = total_work / total_capacity if total_capacity > 0 else 0
        
        return {
            'task_assignments': task_assignments,
            'task_start_times': task_start_times,
            'task_end_times': task_end_times,
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'algorithm': self.name
        }

class ACOScheduler(BaseScheduler):
    """蚁群优化调度器"""
    
    def __init__(self):
        super().__init__("ACO")
        self.params = Hyperparameters.get_algorithm_params("ACO")
    
    def schedule(self, tasks: List[Dict], resources: List[Dict], 
                dependencies: List[Tuple[int, int]]) -> Dict:
        """蚁群优化调度"""
        self.logger.info(f"Scheduling {len(tasks)} tasks using ACO")
        
        num_tasks = len(tasks)
        num_resources = len(resources)
        
        # 初始化信息素矩阵
        pheromone = np.ones((num_tasks, num_resources)) * 0.1
        
        best_solution = None
        best_fitness = float('inf')
        
        for iteration in range(self.params["max_iterations"]):
            # 构建蚂蚁解
            ants = []
            for _ in range(self.params["n_ants"]):
                ant = self._construct_solution(pheromone, tasks, resources, dependencies)
                ants.append(ant)
                
                if ant.fitness < best_fitness:
                    best_fitness = ant.fitness
                    best_solution = ant.path.copy()
            
            # 更新信息素
            self._update_pheromone(pheromone, ants, num_tasks, num_resources)
        
        # 构建调度结果
        if best_solution:
            return self._build_schedule_result(
                best_solution, tasks, resources, dependencies
            )
        else:
            return {"algorithm": self.name, "makespan": float('inf'), "resource_utilization": 0}
    
    def _construct_solution(self, pheromone: np.ndarray, tasks: List[Dict], 
                           resources: List[Dict], dependencies: List[Tuple[int, int]]) -> Ant:
        """构建蚂蚁解"""
        num_tasks = len(tasks)
        ant = Ant(num_tasks)
        
        # 为每个任务选择资源
        for i in range(num_tasks):
            probabilities = self._calculate_probabilities(pheromone[i], i, tasks, resources)
            resource_id = self._roulette_wheel_selection(probabilities)
            ant.path.append(resource_id)
        
        # 评估解的质量
        ant.fitness = self._evaluate_fitness(ant.path, tasks, resources, dependencies)
        
        return ant
    
    def _calculate_probabilities(self, pheromone_row: np.ndarray, task_idx: int, 
                                tasks: List[Dict], resources: List[Dict]) -> np.ndarray:
        """计算选择概率"""
        task = tasks[task_idx]
        heuristic = np.zeros(len(resources))
        
        # 启发式信息：基于资源适合度
        for j, resource in enumerate(resources):
            if (resource['cpu_capacity'] >= task['cpu_req'] and 
                resource['memory_capacity'] >= task['memory_req']):
                # 资源越空闲，启发式值越高
                heuristic[j] = 1.0 / (task['duration'] + 1)
            else:
                heuristic[j] = 0.0
        
        # 计算概率
        numerator = (pheromone_row ** self.params["alpha"]) * (heuristic ** self.params["beta"])
        denominator = np.sum(numerator)
        
        if denominator == 0:
            return np.ones(len(resources)) / len(resources)
        
        return numerator / denominator
    
    def _roulette_wheel_selection(self, probabilities: np.ndarray) -> int:
        """轮盘赌选择"""
        if random.random() < self.params["q0"]:
            # 贪婪选择
            return np.argmax(probabilities)
        else:
            # 随机选择
            cumsum = np.cumsum(probabilities)
            r = random.random()
            for i, cum_prob in enumerate(cumsum):
                if r <= cum_prob:
                    return i
            return len(probabilities) - 1
    
    def _evaluate_fitness(self, path: List[int], tasks: List[Dict], 
                         resources: List[Dict], dependencies: List[Tuple[int, int]]) -> float:
        """评估解的适应度"""
        try:
            # 解析依赖关系
            dep_list = self._parse_dependencies(dependencies)
            resource_end_times = [0] * len(resources)
            task_end_times = {}
            processed = set()
            
            max_iterations = len(tasks) * 2  # 防止无限循环
            iteration = 0
            
            while len(processed) < len(tasks) and iteration < max_iterations:
                iteration += 1
                tasks_processed_this_round = 0
                
                for i, task in enumerate(tasks):
                    if task['id'] in processed:
                        continue
                    
                    dependencies_satisfied = True
                    max_dependency_end = 0
                    for pre_task, post_task in dep_list:
                        # 确保ID类型匹配
                        if str(post_task) == str(task['id']):
                            if str(pre_task) not in [str(tid) for tid in task_end_times.keys()]:
                                dependencies_satisfied = False
                                break
                            # 找到对应的任务ID
                            for tid, end_time in task_end_times.items():
                                if str(tid) == str(pre_task):
                                    max_dependency_end = max(max_dependency_end, end_time)
                                    break
                    
                    if dependencies_satisfied:
                        resource_id = path[i] % len(resources)
                        start_time = max(resource_end_times[resource_id], max_dependency_end)
                        end_time = start_time + task['duration']
                        
                        task_end_times[task['id']] = end_time
                        resource_end_times[resource_id] = end_time
                        processed.add(task['id'])
                        tasks_processed_this_round += 1
                
                # 如果这一轮没有处理任何任务，说明有循环依赖或数据问题
                if tasks_processed_this_round == 0:
                    print(f"ACO警告：检测到循环依赖或数据问题，剩余 {len(tasks) - len(processed)} 个任务无法处理")
                    return float('inf')
            
            return max(resource_end_times) if resource_end_times else float('inf')
        except Exception as e:
            self.logger.error(f"Error evaluating fitness: {e}")
            return float('inf')
    
    def _update_pheromone(self, pheromone: np.ndarray, ants: List[Ant], 
                         num_tasks: int, num_resources: int):
        """更新信息素"""
        # 信息素挥发
        pheromone *= (1 - self.params["rho"])
        
        # 信息素增强
        for ant in ants:
            if ant.fitness < float('inf'):
                delta_pheromone = 1.0 / ant.fitness
                for i, resource_id in enumerate(ant.path):
                    pheromone[i][resource_id] += delta_pheromone
    
    def _build_schedule_result(self, path: List[int], tasks: List[Dict], 
                              resources: List[Dict], dependencies: List[Tuple[int, int]]) -> Dict:
        """构建调度结果"""
        # 解析依赖关系
        dep_list = self._parse_dependencies(dependencies)
        
        task_assignments = {tasks[i]['id']: path[i] for i in range(len(tasks))}
        
        resource_end_times = [0] * len(resources)
        task_start_times = {}
        task_end_times = {}
        processed = set()
        
        while len(processed) < len(tasks):
            for i, task in enumerate(tasks):
                if task['id'] in processed:
                    continue
                
                dependencies_satisfied = True
                max_dependency_end = 0
                for pre_task, post_task in dep_list:
                    if post_task == task['id']:
                        if pre_task not in task_end_times:
                            dependencies_satisfied = False
                            break
                        max_dependency_end = max(max_dependency_end, task_end_times[pre_task])
                
                if dependencies_satisfied:
                    resource_id = path[i]
                    start_time = max(resource_end_times[resource_id], max_dependency_end)
                    end_time = start_time + task['duration']
                    
                    task_start_times[task['id']] = start_time
                    task_end_times[task['id']] = end_time
                    resource_end_times[resource_id] = end_time
                    processed.add(task['id'])
        
        makespan = max(resource_end_times)
        total_work = sum(task['duration'] for task in tasks)
        total_capacity = makespan * len(resources)
        resource_utilization = total_work / total_capacity if total_capacity > 0 else 0
        
        return {
            'task_assignments': task_assignments,
            'task_start_times': task_start_times,
            'task_end_times': task_end_times,
            'makespan': makespan,
            'resource_utilization': resource_utilization,
            'algorithm': self.name
        }

