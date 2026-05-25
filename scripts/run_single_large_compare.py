#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.traditional_schedulers import FIFOScheduler, SJFScheduler, HEFTScheduler
from environment.historical_replay_simulator import HistoricalReplaySimulator as LogReplaySimulator
from models.enhanced_fe_iddqn import EnhancedFE_IDDQN, EnhancedFE_IDDQN_Config
from train_fe_iddqn_ga_hpo import (
    load_replay_dataframes,
    _extract_replay_snapshot_for_baselines,
    _get_valid_action_count,
    _normalize_state_for_agent,
    _reset_env_and_get_state,
    evaluate_dqn_agent,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Single large workflow algorithm comparison")
    parser.add_argument("--process_id", type=int, default=293712)
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/real_data_validation/full_smoke_whalesb/models/final_model.pt",
        help="Trained FE-IDDQN checkpoint path",
    )
    parser.add_argument("--fe_episodes", type=int, default=5)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/real_data_validation/single_instance_compare",
        help="Directory to save gantt figure and result json",
    )
    return parser.parse_args()


def plot_gantt(ax, schedule, title):
    if not schedule:
        ax.set_title(f"{title} (no schedule)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Resource")
        return

    resources = sorted({str(item.get("resource", item.get("host", "unknown"))) for item in schedule})
    y_map = {res: idx for idx, res in enumerate(resources)}

    for item in schedule:
        res = str(item.get("resource", item.get("host", "unknown")))
        start = float(item.get("start_time", item.get("timestamp", 0.0)))
        end = float(item.get("finish_time", item.get("end_time", start)))
        width = max(0.01, end - start)
        y = y_map[res]
        ax.barh(y, width, left=start, height=0.7, alpha=0.9)

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Resource")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)


def collect_fe_schedule(agent, env):
    state = _reset_env_and_get_state(env)
    done = False
    schedule = []

    while not done:
        _, task_feats, res_feats, adj, node_depths, critical_mask = _normalize_state_for_agent(
            state,
            task_input_dim=agent.task_input_dim,
            resource_input_dim=agent.resource_input_dim,
        )

        action = agent.select_action(
            task_feats,
            res_feats,
            adj_matrix=adj,
            node_depths=node_depths,
            critical_path_mask=critical_mask,
            valid_action_count=_get_valid_action_count(env, res_feats, agent.action_dim),
            training=False,
        )

        state, _, done, info = env.step(action)
        if info.get("task_scheduled", False):
            schedule.append(
                {
                    "task_id": info.get("task_id"),
                    "task_name": info.get("task_name", "task"),
                    "resource": info.get("host", "unknown"),
                    "start_time": float(info.get("start_time", 0.0)),
                    "finish_time": float(info.get("end_time", 0.0)),
                }
            )

    return schedule


def normalize_columns(process_instances, task_instances, process_task_relations):
    if "process_definition_code" not in process_instances.columns:
        if "process_definition_id" in process_instances.columns:
            process_instances["process_definition_code"] = process_instances["process_definition_id"]
        else:
            process_instances["process_definition_code"] = 0

    if "process_definition_code" not in process_task_relations.columns:
        if "process_definition_id" in process_task_relations.columns:
            process_task_relations["process_definition_code"] = process_task_relations["process_definition_id"]
        else:
            process_task_relations["process_definition_code"] = 0

    if "process_definition_code" not in task_instances.columns:
        proc_code_map = process_instances[["id", "process_definition_code"]].drop_duplicates()
        task_instances = task_instances.merge(
            proc_code_map,
            left_on="process_instance_id",
            right_on="id",
            how="left",
            suffixes=("", "_proc"),
        )
        if "id_proc" in task_instances.columns:
            task_instances = task_instances.drop(columns=["id_proc"])

    if "host" not in task_instances.columns:
        if "worker_group" in task_instances.columns:
            task_instances["host"] = task_instances["worker_group"].fillna("default_host")
        else:
            task_instances["host"] = "default_host"

    return process_instances, task_instances, process_task_relations


def main():
    args = parse_args()
    process_id = args.process_id
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_replay_dataframes(Path("data/raw_data"))

    process_instances = data["process_instance"].copy()
    task_instances = data["task_instance"].copy()
    task_definitions = data["task_definition"].copy()
    process_task_relations = data["process_task_relation"].copy()

    process_instances, task_instances, process_task_relations = normalize_columns(
        process_instances, task_instances, process_task_relations
    )

    proc_subset = process_instances[
        (process_instances["id"] == process_id) & (process_instances["state"] == 7)
    ].copy()
    task_subset = task_instances[task_instances["process_instance_id"] == process_id].copy()

    if proc_subset.empty or task_subset.empty:
        raise SystemExit("Target process_id not found or has no tasks")

    env = LogReplaySimulator(
        process_instances=proc_subset,
        task_instances=task_subset,
        task_definitions=task_definitions,
        process_task_relations=process_task_relations,
    )

    env.reset()
    tasks, resources, dependencies = _extract_replay_snapshot_for_baselines(env)

    schedulers = {
        "FIFO": FIFOScheduler(),
        "SJF": SJFScheduler(),
        "HEFT": HEFTScheduler(),
    }

    results = {
        "process_id": process_id,
        "task_count": len(tasks),
        "resource_count": len(resources),
        "dependency_count": len(dependencies),
        "results": {},
    }

    for name, scheduler in schedulers.items():
        out = scheduler.schedule(tasks, resources, dependencies)
        results["results"][name] = {
            "makespan": float(out.get("makespan", 0.0)),
            "resource_utilization": float(out.get("resource_utilization", 0.0)),
            "load_balance": float(out.get("load_balance", 0.0)) if out.get("load_balance") is not None else None,
        }
        results["results"][name]["schedule"] = out.get("schedule", [])

    if model_path.exists():
        state = _reset_env_and_get_state(env)
        _, task_feats, res_feats, _, _, _ = _normalize_state_for_agent(
            state,
            task_input_dim=16,
            resource_input_dim=7,
        )
        task_input_dim = task_feats.shape[-1] if len(task_feats.shape) >= 2 else 16
        resource_input_dim = res_feats.shape[-1] if len(res_feats.shape) >= 2 else 7
        action_dim = int(max(getattr(env, "num_resources", 1) or 1, res_feats.shape[0] if len(res_feats.shape) >= 2 else 1))

        checkpoint = torch.load(str(model_path), map_location="cpu")
        q_state = checkpoint.get("q_network_state_dict", {})
        output_weight = q_state.get("feature_fusion.advantage_stream.2.weight")
        if output_weight is not None:
            action_dim = max(action_dim, int(output_weight.shape[0]))

        cfg = checkpoint.get("config")
        if isinstance(cfg, EnhancedFE_IDDQN_Config):
            dqn_cfg = cfg
        elif isinstance(cfg, dict):
            dqn_cfg = EnhancedFE_IDDQN_Config(**cfg)
        else:
            dqn_cfg = EnhancedFE_IDDQN_Config()
        dqn_cfg.device = "cpu"

        agent = EnhancedFE_IDDQN(task_input_dim, resource_input_dim, action_dim, dqn_cfg)
        agent.load(str(model_path))
        fe_eval = evaluate_dqn_agent(agent, env, num_episodes=args.fe_episodes)
        fe_env = LogReplaySimulator(
            process_instances=proc_subset.copy(),
            task_instances=task_subset.copy(),
            task_definitions=task_definitions.copy(),
            process_task_relations=process_task_relations.copy(),
        )
        fe_schedule = collect_fe_schedule(agent, fe_env)

        results["results"]["FE-IDDQN"] = {
            "makespan": float(fe_eval.get("makespan", 0.0)),
            "makespan_std": float(fe_eval.get("makespan_std", 0.0)),
            "resource_utilization": float(fe_eval.get("utilization", 0.0)),
            "load_balance": float(fe_eval.get("load_balance", 0.0)),
            "episodes": int(fe_eval.get("episodes", args.fe_episodes)),
            "model_path": str(model_path),
            "schedule": fe_schedule,
        }
    else:
        results["results"]["FE-IDDQN"] = {
            "error": f"model file not found: {model_path}"
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    algo_order = ["FIFO", "SJF", "HEFT", "FE-IDDQN"]
    for ax, algo in zip(axes.flatten(), algo_order):
        algo_result = results["results"].get(algo, {})
        schedule = algo_result.get("schedule", [])
        title = f"{algo} | makespan={algo_result.get('makespan', 0.0):.1f}s"
        plot_gantt(ax, schedule, title)

    gantt_path = output_dir / f"gantt_process_{process_id}.png"
    fig.suptitle(f"Workflow {process_id} Scheduling Gantt Comparison", fontsize=14)
    fig.savefig(gantt_path, dpi=200)
    plt.close(fig)

    json_path = output_dir / f"single_compare_process_{process_id}.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    results["artifacts"] = {
        "gantt_png": str(gantt_path),
        "result_json": str(json_path),
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
