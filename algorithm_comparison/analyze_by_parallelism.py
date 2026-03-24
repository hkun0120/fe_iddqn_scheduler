#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于已有实验结果，按工作流依赖复杂度（并行度）重新分组分析

并行度分类方法：
  - 依赖密度 = num_deps / num_tasks
  - 高并行度：密度 < 0.5（依赖少，任务可大量并行执行）
  - 中等并行度：0.5 <= 密度 < 0.9
  - 低并行度/串行：密度 >= 0.9（依赖多，几乎串行执行）
"""

import json
import sys
import os
import numpy as np
from pathlib import Path

# 加载已有结果
RESULT_FILE = Path(__file__).parent / "results" / "fe_iddqn_v1_comparison" / "summary_20260228_150805.json"

def main():
    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取各算法结果
    alg_map = {}
    for r in data["all_results"]:
        alg_map[r["algorithm"]] = r

    sjf = alg_map["SJF"]
    orig = alg_map.get("\u539f\u59cb FE-IDDQN")  # 原始 FE-IDDQN
    gahpo = alg_map.get("GA-HPO FE-IDDQN")
    random_r = alg_map.get("Random")
    rr = alg_map.get("RoundRobin")
    cpop = alg_map.get("CPOP")

    num_wf = len(sjf["test_results"])

    # ============================================================
    # 1. 逐工作流分析：增加并行度指标
    # ============================================================
    print("=" * 130)
    print(f"{'#':>3} {'name':32s} {'tasks':>5} {'deps':>5} {'density':>7} {'category':>12} | {'SJF':>10} {'Orig':>10} {'GA-HPO':>10} {'Improve':>8}")
    print("-" * 130)

    workflow_info = []  # (index, density, category)

    for i in range(num_wf):
        wf = sjf["test_results"][i]
        n_tasks = wf["num_tasks"]
        n_deps = wf["num_deps"]
        density = n_deps / n_tasks if n_tasks > 0 else 0

        if density < 0.5:
            cat = "HIGH_PAR"
        elif density < 0.9:
            cat = "MEDIUM"
        else:
            cat = "LOW_PAR"

        workflow_info.append((i, density, cat, n_tasks, n_deps))

        sjf_ms = sjf["test_results"][i]["makespan"]
        orig_ms = orig["test_results"][i]["makespan"]
        gahpo_ms = gahpo["test_results"][i]["makespan"]
        improvement = (orig_ms - gahpo_ms) / orig_ms * 100 if orig_ms > 0 else 0

        name = wf["name"][:30]
        print(f"{i+1:>3} {name:32s} {n_tasks:>5} {n_deps:>5} {density:>7.2f} {cat:>12} | {sjf_ms:>10.1f} {orig_ms:>10.1f} {gahpo_ms:>10.1f} {improvement:>+7.1f}%")

    # ============================================================
    # 2. 按并行度分组汇总
    # ============================================================
    print()
    print("=" * 100)
    print("                    Analysis by Parallelism Level")
    print("=" * 100)

    algorithms = ["Random", "RoundRobin", "SJF", "EFT", "CPOP",
                   "\u539f\u59cb FE-IDDQN", "GA-HPO FE-IDDQN"]

    groups = [
        ("HIGH_PAR", "high parallelism (density < 0.5)"),
        ("MEDIUM", "medium parallelism (0.5 <= density < 0.9)"),
        ("LOW_PAR", "low parallelism / serial (density >= 0.9)"),
    ]

    group_results = {}

    for group_key, group_label in groups:
        indices = [wi[0] for wi in workflow_info if wi[2] == group_key]
        if not indices:
            continue

        print(f"\n  [{group_label}] ({len(indices)} workflows)")
        densities = [workflow_info[i][1] for i in range(len(workflow_info)) if workflow_info[i][2] == group_key]
        tasks_list = [workflow_info[i][3] for i in range(len(workflow_info)) if workflow_info[i][2] == group_key]
        print(f"    Avg density: {np.mean(densities):.3f}, Avg tasks: {np.mean(tasks_list):.1f}")
        print()

        header = f"    {'Algorithm':<22s} | {'Makespan':>10s} | {'Util':>7s} | {'LoadBal':>7s}"
        print(header)
        print("    " + "-" * 60)

        group_data = {}
        for alg_name in algorithms:
            alg_data = alg_map.get(alg_name)
            if not alg_data:
                continue
            ms_vals = [alg_data["test_results"][i]["makespan"] for i in indices]
            ut_vals = [alg_data["test_results"][i]["utilization"] for i in indices]
            lb_vals = [alg_data["test_results"][i]["load_balance"] for i in indices]
            avg_ms = np.mean(ms_vals)
            avg_ut = np.mean(ut_vals)
            avg_lb = np.mean(lb_vals)
            group_data[alg_name] = {"makespan": avg_ms, "util": avg_ut, "lb": avg_lb}
            print(f"    {alg_name:<22s} | {avg_ms:>10.2f} | {avg_ut:>7.4f} | {avg_lb:>7.4f}")

        # Compute improvements
        if "\u539f\u59cb FE-IDDQN" in group_data and "GA-HPO FE-IDDQN" in group_data and "SJF" in group_data:
            orig_avg = group_data["\u539f\u59cb FE-IDDQN"]["makespan"]
            gahpo_avg = group_data["GA-HPO FE-IDDQN"]["makespan"]
            sjf_avg = group_data["SJF"]["makespan"]
            print()
            if gahpo_avg < orig_avg:
                print(f"    => GA-HPO vs Orig: improved {(orig_avg - gahpo_avg) / orig_avg * 100:.1f}%")
            else:
                print(f"    => GA-HPO vs Orig: worse by {(gahpo_avg - orig_avg) / orig_avg * 100:.1f}%")
            print(f"    => GA-HPO vs SJF: {(gahpo_avg - sjf_avg) / sjf_avg * 100:+.1f}%")
            print(f"    => Orig vs SJF: {(orig_avg - sjf_avg) / sjf_avg * 100:+.1f}%")

        group_results[group_key] = group_data

    # ============================================================
    # 3. Key Conclusions
    # ============================================================
    print()
    print("=" * 100)
    print("                         Key Conclusions")
    print("=" * 100)

    for group_key, group_label in groups:
        if group_key not in group_results:
            continue
        gd = group_results[group_key]
        if "GA-HPO FE-IDDQN" not in gd or "SJF" not in gd:
            continue

        gahpo_ms = gd["GA-HPO FE-IDDQN"]["makespan"]
        sjf_ms = gd["SJF"]["makespan"]
        orig_ms = gd.get("\u539f\u59cb FE-IDDQN", {}).get("makespan", 0)
        ratio = (gahpo_ms - sjf_ms) / sjf_ms * 100 if sjf_ms > 0 else 0
        indices = [wi[0] for wi in workflow_info if wi[2] == group_key]

        print(f"\n  {group_label} ({len(indices)} wf):")
        print(f"    SJF={sjf_ms:.1f}, Orig={orig_ms:.1f}, GA-HPO={gahpo_ms:.1f}")
        print(f"    GA-HPO vs SJF: {ratio:+.1f}%")
        if orig_ms > 0:
            print(f"    GA-HPO vs Orig: {(orig_ms - gahpo_ms) / orig_ms * 100:+.1f}% improvement")

    # ============================================================
    # 4. Per-workflow win analysis by parallelism group
    # ============================================================
    print()
    print("=" * 100)
    print("                    Win/Loss Analysis by Group")
    print("=" * 100)

    for group_key, group_label in groups:
        indices = [wi[0] for wi in workflow_info if wi[2] == group_key]
        if not indices:
            continue

        ga_wins = 0
        ga_beats_sjf = 0
        for i in indices:
            orig_ms = orig["test_results"][i]["makespan"]
            gahpo_ms = gahpo["test_results"][i]["makespan"]
            sjf_ms = sjf["test_results"][i]["makespan"]
            if gahpo_ms < orig_ms * 0.99:
                ga_wins += 1
            if gahpo_ms <= sjf_ms * 1.01:
                ga_beats_sjf += 1

        print(f"\n  {group_label} ({len(indices)} wf):")
        print(f"    GA-HPO wins vs Orig: {ga_wins}/{len(indices)}")
        print(f"    GA-HPO matches/beats SJF: {ga_beats_sjf}/{len(indices)}")

    # ============================================================
    # 5. Save analysis results
    # ============================================================
    out = {
        "analysis_type": "parallelism_based",
        "classification": {
            "HIGH_PAR": {"label": "density < 0.5", "workflows": [wi[0] for wi in workflow_info if wi[2] == "HIGH_PAR"]},
            "MEDIUM": {"label": "0.5 <= density < 0.9", "workflows": [wi[0] for wi in workflow_info if wi[2] == "MEDIUM"]},
            "LOW_PAR": {"label": "density >= 0.9", "workflows": [wi[0] for wi in workflow_info if wi[2] == "LOW_PAR"]},
        },
        "per_workflow": [{
            "index": wi[0],
            "density": round(wi[1], 4),
            "category": wi[2],
            "num_tasks": wi[3],
            "num_deps": wi[4],
            "name": sjf["test_results"][wi[0]]["name"],
            "sjf_makespan": sjf["test_results"][wi[0]]["makespan"],
            "orig_makespan": orig["test_results"][wi[0]]["makespan"],
            "gahpo_makespan": gahpo["test_results"][wi[0]]["makespan"],
        } for wi in workflow_info],
        "group_summary": {k: v for k, v in group_results.items()},
    }

    out_path = Path(__file__).parent / "results" / "parallelism_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
