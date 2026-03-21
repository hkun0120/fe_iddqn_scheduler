#!/usr/bin/env python3
"""
为工作流 291298 和 291632 生成多算法甘特图，并检查依赖约束是否被真正满足。

输出：
- analysis_outputs/gantt_dependency_analysis/gantt_<workflow_id>.png
- analysis_outputs/gantt_dependency_analysis/dependency_parallelism_report.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.mysql_data_loader import MySQLDataLoader
from environment.historical_replay_simulator import HistoricalReplaySimulator
from models.fe_iddqn import FE_IDDQN
from test_long_workflow import SimpleBaselineScheduler


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


WORKFLOW_IDS = [291298, 291632]
ALGORITHMS = ["FE-IDDQN", "FIFO", "SJF", "RoundRobin", "LoadBalance", "Random"]
OUT_DIR = Path("analysis_outputs/gantt_dependency_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Analyzer:
    def __init__(self):
        self.loader = MySQLDataLoader(host="localhost", user="root", password="", database="whalesb", port=3306)
        self.data = self.loader.load_all_data()

        self.task_input_dim = 16
        self.resource_input_dim = 7
        self.action_dim = 5

        self.fe_agent = FE_IDDQN(
            self.task_input_dim,
            self.resource_input_dim,
            self.action_dim,
            max_tasks=5,
            max_resources=5,
            enable_graph_encoder=True,
        )

        self.baselines = {
            "FIFO": SimpleBaselineScheduler("FIFO", self.action_dim),
            "SJF": SimpleBaselineScheduler("SJF", self.action_dim),
            "RoundRobin": SimpleBaselineScheduler("RoundRobin", self.action_dim),
            "LoadBalance": SimpleBaselineScheduler("LoadBalance", self.action_dim),
            "Random": SimpleBaselineScheduler("Random", self.action_dim),
        }

    def _create_simulator(self, process_id: int) -> HistoricalReplaySimulator:
        process_instance = self.data["process_instance"][self.data["process_instance"]["id"] == process_id]
        task_instances = self.data["task_instance"][self.data["task_instance"]["process_instance_id"] == process_id]
        successful_tasks = task_instances[task_instances["state"] == 7]

        return HistoricalReplaySimulator(
            process_instance,
            successful_tasks,
            self.data["task_definition"],
            self.data["process_task_relation"],
        )

    @staticmethod
    def _task_code(task_row: pd.Series) -> Any:
        return task_row.get("task_code", task_row.get("task_definition_code", task_row.get("id")))

    @staticmethod
    def _duration(task_row: pd.Series) -> float:
        if pd.notna(task_row.get("start_time")) and pd.notna(task_row.get("end_time")):
            try:
                return max(1.0, float((pd.to_datetime(task_row["end_time"]) - pd.to_datetime(task_row["start_time"])).total_seconds()))
            except Exception:
                return 10.0
        return 10.0

    def run_single_algorithm(self, process_id: int, algorithm: str) -> Dict[str, Any]:
        sim = self._create_simulator(process_id)
        sim.reset()

        # 任务信息映射（用于重建时长/名称）
        task_map: Dict[Any, Dict[str, Any]] = {}
        if hasattr(sim, "current_process_tasks") and len(sim.current_process_tasks) > 0:
            for _, row in sim.current_process_tasks.iterrows():
                tcode = self._task_code(row)
                task_map[tcode] = {
                    "task_name": row.get("name", str(tcode)),
                    "duration": self._duration(row),
                    "task_type": row.get("task_type", "N/A"),
                }

        schedule_history: List[Dict[str, Any]] = []
        done = False
        step = 0
        while not done and step < 20000:
            state = sim.get_state()
            if state is None:
                break
            task_features, resource_features = state

            if algorithm == "FE-IDDQN":
                action = self.fe_agent.select_action(task_features, resource_features, graph_adj=sim.get_graph_adj())
            else:
                action = self.baselines[algorithm].select_action(task_features, resource_features)

            # 记录“调度顺序”
            if hasattr(sim, "current_process_tasks") and sim.current_task_idx < len(sim.current_process_tasks):
                current_task = sim.current_process_tasks.iloc[sim.current_task_idx]
                task_code = self._task_code(current_task)
                available_hosts = list(sim.available_resources.keys())
                selected_host = available_hosts[action % len(available_hosts)] if available_hosts else "N/A"
                schedule_history.append(
                    {
                        "step": step,
                        "task_code": task_code,
                        "task_name": current_task.get("name", str(task_code)),
                        "selected_resource": selected_host,
                    }
                )

            _, _, done, _ = sim.step(int(action))
            step += 1

        # 重建可视化时间线（按资源累计执行）
        host_clock: Dict[str, float] = {}
        timeline: List[Dict[str, Any]] = []
        for rec in schedule_history:
            host = rec["selected_resource"]
            host_clock.setdefault(host, 0.0)

            info = task_map.get(rec["task_code"], {"duration": 10.0, "task_name": rec["task_name"], "task_type": "N/A"})
            duration = float(info["duration"])
            start = host_clock[host]
            finish = start + duration
            host_clock[host] = finish

            timeline.append(
                {
                    "order": rec["step"] + 1,
                    "task_code": rec["task_code"],
                    "task_name": info.get("task_name", rec["task_name"]),
                    "task_type": info.get("task_type", "N/A"),
                    "resource": host,
                    "start": start,
                    "finish": finish,
                    "duration": duration,
                }
            )

        makespan = max((x["finish"] for x in timeline), default=0.0)
        total_work = float(sum(x["duration"] for x in timeline))
        num_resources = max(1, len(set(x["resource"] for x in timeline)))
        util = total_work / (makespan * num_resources) if makespan > 0 else 0.0

        # 依赖检查
        dependencies = getattr(sim, "current_process_dependencies", []) or []
        order_idx = {x["task_code"]: i for i, x in enumerate(timeline)}
        start_by_task = {x["task_code"]: x["start"] for x in timeline}
        finish_by_task = {x["task_code"]: x["finish"] for x in timeline}

        order_violations: List[Dict[str, Any]] = []
        temporal_violations: List[Dict[str, Any]] = []

        for dep in dependencies:
            pre = dep.get("pre_task_code")
            post = dep.get("post_task_code")
            if pre not in order_idx or post not in order_idx:
                continue

            if order_idx[pre] >= order_idx[post]:
                order_violations.append({"pre": pre, "post": post, "pre_order": order_idx[pre], "post_order": order_idx[post]})

            # 真正执行时序约束：post.start >= pre.finish
            if start_by_task[post] < finish_by_task[pre]:
                temporal_violations.append(
                    {
                        "pre": pre,
                        "post": post,
                        "pre_finish": round(float(finish_by_task[pre]), 3),
                        "post_start": round(float(start_by_task[post]), 3),
                        "gap": round(float(start_by_task[post] - finish_by_task[pre]), 3),
                    }
                )

        return {
            "algorithm": algorithm,
            "process_id": process_id,
            "schedule": timeline,
            "metrics": {
                "makespan": makespan,
                "resource_utilization": util,
                "total_work": total_work,
                "effective_parallelism": (total_work / makespan) if makespan > 0 else 0.0,
                "num_resources_used": len(set(x["resource"] for x in timeline)),
            },
            "dependency": {
                "dependency_count": len(dependencies),
                "order_violation_count": len(order_violations),
                "temporal_violation_count": len(temporal_violations),
                "order_violations_sample": order_violations[:20],
                "temporal_violations_sample": temporal_violations[:20],
            },
            "schedule_order_first20": [x["task_code"] for x in timeline[:20]],
        }

    @staticmethod
    def _draw_workflow_figure(process_id: int, results: List[Dict[str, Any]]) -> Path:
        # 统一资源编号顺序
        all_resources = []
        for r in results:
            all_resources.extend([x["resource"] for x in r["schedule"]])
        resources = sorted(list(set(all_resources)), key=lambda x: str(x))
        res_to_y = {r: i for i, r in enumerate(resources)}

        n = len(results)
        fig, axes = plt.subplots(n, 1, figsize=(18, max(3 * n, 8)), sharex=True)
        if n == 1:
            axes = [axes]

        cmap = plt.cm.tab20

        for ax, r in zip(axes, results):
            sched = r["schedule"]
            for i, item in enumerate(sched):
                y = res_to_y[item["resource"]]
                color = cmap(i % 20)
                ax.barh(y, item["duration"], left=item["start"], height=0.6, color=color, edgecolor="black", linewidth=0.3)
                if item["duration"] >= max(1.0, r["metrics"]["makespan"] * 0.015):
                    ax.text(item["start"] + item["duration"] / 2, y, str(item["order"]), ha="center", va="center", fontsize=6)

            dep = r["dependency"]
            title = (
                f"{r['algorithm']} | makespan={r['metrics']['makespan']:.1f} | util={r['metrics']['resource_utilization']:.2%} "
                f"| eff_parallel={r['metrics']['effective_parallelism']:.2f} "
                f"| dep(order={dep['order_violation_count']}, temporal={dep['temporal_violation_count']})"
            )
            ax.set_title(title, fontsize=10)
            ax.set_yticks(list(res_to_y.values()))
            ax.set_yticklabels([str(x) for x in resources], fontsize=7)
            ax.grid(axis="x", alpha=0.25)

        axes[-1].set_xlabel("Time (seconds, reconstructed)")
        plt.suptitle(f"Workflow {process_id}: Multi-algorithm Gantt + Dependency Checks", fontsize=13, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        out = OUT_DIR / f"gantt_{process_id}_all_algorithms.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        return out

    def run(self):
        report: Dict[str, Any] = {"workflows": {}}

        for wid in WORKFLOW_IDS:
            logger.info("Analyzing workflow %s ...", wid)
            wf_results: List[Dict[str, Any]] = []
            for algo in ALGORITHMS:
                # 重置基线内部计数器
                if algo in self.baselines and hasattr(self.baselines[algo], "call_count"):
                    self.baselines[algo].call_count = 0
                res = self.run_single_algorithm(wid, algo)
                wf_results.append(res)

            # 顺序一致性：与 FE-IDDQN 的任务顺序是否一致
            fe_order = next((x["schedule_order_first20"] for x in wf_results if x["algorithm"] == "FE-IDDQN"), [])
            order_compare = {}
            for r in wf_results:
                order_compare[r["algorithm"]] = (r["schedule_order_first20"] == fe_order)

            fig_path = self._draw_workflow_figure(wid, wf_results)

            report["workflows"][str(wid)] = {
                "figure": str(fig_path).replace('\\\\', '/'),
                "results": [
                    {
                        "algorithm": r["algorithm"],
                        "metrics": r["metrics"],
                        "dependency": r["dependency"],
                    }
                    for r in wf_results
                ],
                "task_order_same_as_fe_iddqn_first20": order_compare,
            }

        out_json = OUT_DIR / "dependency_parallelism_report.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info("Saved report to %s", out_json)
        for wid in WORKFLOW_IDS:
            logger.info("Saved gantt to %s", OUT_DIR / f"gantt_{wid}_all_algorithms.png")


if __name__ == "__main__":
    np.random.seed(42)
    analyzer = Analyzer()
    analyzer.run()
