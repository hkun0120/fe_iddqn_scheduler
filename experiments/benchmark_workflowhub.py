# -*- coding: utf-8 -*-
"""
WfCommons 基准测试脚本
生成标准科学工作流并对比调度算法效果。
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from data.workflowhub_adapter import (
    wfcommons_available,
    get_available_recipes,
    build_environment_from_recipe,
)
from environment import EnhancedWorkflowSimulator
from evaluation.metrics import Evaluator
from baselines.traditional_schedulers import FIFOScheduler, SJFScheduler, HEFTScheduler
from experiments.experiment_runner_enhanced import ExperimentRunner


def run_traditional_schedulers(tasks, resources, dependencies):
    schedulers = [FIFOScheduler(), SJFScheduler(), HEFTScheduler()]
    evaluator = Evaluator()
    results = {}

    for scheduler in schedulers:
        schedule_result = scheduler.schedule(tasks, resources, dependencies)
        metrics = evaluator.evaluate(schedule_result)
        results[scheduler.name] = {
            "makespan": metrics.get("makespan", 0),
            "resource_utilization": metrics.get("resource_utilization", 0),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="WfCommons 工作流调度基准测试")
    parser.add_argument("--recipe", type=str, default="montage",
                        help="工作流配方名称")
    parser.add_argument("--tasks", type=int, default=200, help="任务数量")
    parser.add_argument("--resources", type=int, default=6, help="资源数量")
    parser.add_argument("--episodes", type=int, default=20, help="训练episodes")
    parser.add_argument("--runs", type=int, default=3, help="重复运行次数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="wfcommons_results")

    args = parser.parse_args()

    if not wfcommons_available():
        raise RuntimeError("wfcommons 未安装，请先安装后再运行。")

    if args.recipe not in get_available_recipes():
        raise ValueError(
            f"不支持的配方: {args.recipe}. 可用: {', '.join(get_available_recipes())}"
        )

    tasks, resources, dependencies = build_environment_from_recipe(
        recipe_name=args.recipe,
        num_tasks=args.tasks,
        num_resources=args.resources,
    )

    env = EnhancedWorkflowSimulator(tasks, resources, dependencies)

    # 传统调度器结果
    traditional_results = run_traditional_schedulers(tasks, resources, dependencies)

    # 强化学习对比（原始 vs 增强）
    runner = ExperimentRunner(env, num_runs=args.runs, device=args.device)
    baseline_results = runner.run_baseline(args.episodes)
    enhanced_results = runner.run_enhanced(args.episodes)
    improvements = runner.calculate_improvements(baseline_results, enhanced_results)

    results = {
        "recipe": args.recipe,
        "num_tasks": args.tasks,
        "num_resources": args.resources,
        "traditional": traditional_results,
        "baseline": baseline_results,
        "enhanced": enhanced_results,
        "improvements": improvements,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"wfcommons_{args.recipe}_{args.tasks}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {out_file}")


if __name__ == "__main__":
    main()
