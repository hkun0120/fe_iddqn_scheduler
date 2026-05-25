#!/usr/bin/env python3
"""Export task-level schedules and Gantt charts for representative large workflows."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.historical_replay_simulator import HistoricalReplaySimulator as LogReplaySimulator
from models.enhanced_fe_iddqn import EnhancedFE_IDDQN, EnhancedFE_IDDQN_Config
from train_fe_iddqn_ga_hpo import (
    _clone_replay_env_for_process_ids,
    _get_valid_action_count,
    _normalize_state_for_agent,
    _reset_env_and_get_state,
    make_replay_envs,
)


ALGORITHMS = {
    "ga_hpo_dqn": "GA-HPO DQN",
    "ga_hpo_fe_dqn": "GA-HPO FE-DQN",
}


plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Songti SC",
    "Heiti SC",
    "Arial Unicode MS",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def build_agent_config(run_dir: Path, force_cpu: bool = True) -> Tuple[EnhancedFE_IDDQN_Config, Dict[str, Any]]:
    args = read_json(run_dir / "config.json")
    dqn_config = EnhancedFE_IDDQN_Config(
        learning_rate=float(args.get("lr", 3e-4)),
        batch_size=int(args.get("batch_size", 64)),
        n_step=int(args.get("n_step", 3)),
        use_n_step=(int(args.get("n_step", 3)) > 1) and not bool(args.get("disable_nstep", False)),
        use_per=not bool(args.get("disable_per", False)),
        use_feature_engineering=not bool(args.get("disable_fe", False)),
        max_episodes=int(args.get("max_episodes", 120)),
        max_steps_per_episode=int(args.get("max_steps_per_episode", 1000)),
        use_gnn=not bool(args.get("no_gnn", False)),
        device="cpu" if force_cpu else args.get("device", "auto"),
    )

    hpo_path = run_dir / "hpo_result.json"
    if hpo_path.exists():
        best_params = read_json(hpo_path).get("best_params", {})
        dqn_config.learning_rate = best_params.get("learning_rate", dqn_config.learning_rate)
        dqn_config.gamma = best_params.get("gamma", dqn_config.gamma)
        dqn_config.tau = best_params.get("tau", dqn_config.tau)
        dqn_config.epsilon_decay = best_params.get("epsilon_decay", dqn_config.epsilon_decay)
        dqn_config.n_step = best_params.get("n_step", dqn_config.n_step)
        dqn_config.use_n_step = (dqn_config.n_step > 1) and not bool(args.get("disable_nstep", False))
        dqn_config.batch_size = best_params.get("batch_size", dqn_config.batch_size)
        dqn_config.replay_buffer_size = best_params.get("replay_buffer_size", dqn_config.replay_buffer_size)
        dqn_config.target_update_freq = best_params.get("target_update_freq", dqn_config.target_update_freq)
        dqn_config.per_alpha = best_params.get("per_alpha", dqn_config.per_alpha)
        dqn_config.per_beta_start = best_params.get("per_beta_start", dqn_config.per_beta_start)
        dqn_config.gradient_clip = best_params.get("gradient_clip", dqn_config.gradient_clip)

    ga_path = run_dir / "ga_search_result.json"
    if ga_path.exists():
        genome = read_json(ga_path).get("best_genome", {})
        dqn_config.hidden_dim = genome.get("hidden_dim", dqn_config.hidden_dim)
        dqn_config.fusion_dim = genome.get("fusion_dim", dqn_config.fusion_dim)
        dqn_config.num_transformer_layers = genome.get(
            "num_transformer_layers", dqn_config.num_transformer_layers
        )
        dqn_config.num_heads = genome.get("num_heads", dqn_config.num_heads)
        dqn_config.dropout = genome.get("dropout", dqn_config.dropout)
        dqn_config.use_gnn = genome.get("use_gnn", dqn_config.use_gnn) and not bool(args.get("no_gnn", False))

    return dqn_config, args


def load_agent(run_dir: Path, model_name: str = "final_model.pt") -> EnhancedFE_IDDQN:
    dqn_config, args = build_agent_config(run_dir)
    agent = EnhancedFE_IDDQN(
        task_input_dim=int(args["task_input_dim"]),
        resource_input_dim=int(args["resource_input_dim"]),
        action_dim=int(args["action_dim"]),
        config=dqn_config,
    )
    model_path = run_dir / "models" / model_name
    agent.load(str(model_path))
    agent.q_network.eval()
    agent.target_network.eval()
    return agent


def collect_final_workflow_rows(root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for group in ALGORITHMS:
        for path in sorted((root / group).glob("seed_*/final_eval_workflows.csv")):
            seed = int(path.parent.name.split("_")[1])
            frame = pd.read_csv(path)
            frame["group"] = group
            frame["seed"] = seed
            rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No final_eval_workflows.csv files found under {root}")
    return pd.concat(rows, ignore_index=True)


def representative_large_cases(root: Path, limit: int = 2) -> List[Dict[str, Any]]:
    workflow_rows = collect_final_workflow_rows(root)
    means = (
        workflow_rows.groupby(["group", "process_id"], as_index=False)["makespan"]
        .mean()
        .pivot(index="process_id", columns="group", values="makespan")
        .reset_index()
    )
    metadata_columns = [
        "process_id",
        "process_name",
        "task_count",
        "duration_seconds",
        "dag_depth",
        "dag_width",
        "dag_edge_count",
        "dag_complexity_score",
        "workflow_size",
        "duration_bin",
        "dag_complexity_bin",
    ]
    metadata = workflow_rows.drop_duplicates("process_id")[metadata_columns]
    merged = means.merge(metadata, on="process_id", how="left")
    merged = merged[
        (merged["workflow_size"] == "large")
        & merged["ga_hpo_dqn"].notna()
        & merged["ga_hpo_fe_dqn"].notna()
    ].copy()
    merged["abs_gain"] = merged["ga_hpo_dqn"] - merged["ga_hpo_fe_dqn"]
    merged["rel_gain_pct"] = merged["abs_gain"] / merged["ga_hpo_dqn"].replace(0, np.nan) * 100.0

    candidates: List[pd.Series] = []
    chain_like = merged[merged["dag_width"].fillna(0) <= 2].sort_values("abs_gain", ascending=False)
    if not chain_like.empty:
        candidates.append(chain_like.iloc[0])

    parallel_like = merged[merged["dag_width"].fillna(0) >= 5].sort_values("abs_gain", ascending=False)
    if not parallel_like.empty:
        candidates.append(parallel_like.iloc[0])

    if len(candidates) < limit:
        for _, row in merged.sort_values("abs_gain", ascending=False).iterrows():
            if not any(int(row["process_id"]) == int(item["process_id"]) for item in candidates):
                candidates.append(row)
            if len(candidates) >= limit:
                break

    cases: List[Dict[str, Any]] = []
    for row in candidates[:limit]:
        process_id = int(row["process_id"])
        seed = closest_seed_to_mean_gain(workflow_rows, process_id)
        cases.append({
            "process_id": process_id,
            "seed": seed,
            "selection_reason": "chain_like_top_gain" if len(cases) == 0 else "parallel_like_top_gain",
            **{key: row.get(key) for key in row.index},
        })
    return cases


def closest_seed_to_mean_gain(workflow_rows: pd.DataFrame, process_id: int) -> int:
    subset = workflow_rows[workflow_rows["process_id"] == process_id]
    wide = subset.pivot(index="seed", columns="group", values="makespan").dropna()
    if wide.empty:
        return 42
    gains = wide["ga_hpo_dqn"] - wide["ga_hpo_fe_dqn"]
    return int((gains - gains.mean()).abs().sort_values().index[0])


def task_code_value(row: pd.Series) -> Any:
    if "task_code" in row and pd.notna(row["task_code"]):
        return row["task_code"]
    if "task_definition_code" in row and pd.notna(row["task_definition_code"]):
        return row["task_definition_code"]
    return None


def run_agent_on_workflow(agent: EnhancedFE_IDDQN,
                          base_env: LogReplaySimulator,
                          process_id: int,
                          algorithm: str,
                          seed: int,
                          max_steps: int = 10000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    env = _clone_replay_env_for_process_ids(base_env, [process_id], episode_seed=seed)
    state = _reset_env_and_get_state(env)
    process_tasks = env.current_process_tasks.copy()
    dependencies = list(getattr(env, "current_process_dependencies", []) or [])

    task_lookup: Dict[int, pd.Series] = {
        int(row["id"]): row
        for _, row in process_tasks.iterrows()
        if pd.notna(row.get("id"))
    }
    code_to_task_id: Dict[str, int] = {}
    for task_id, row in task_lookup.items():
        code = task_code_value(row)
        if code is not None and pd.notna(code):
            code_to_task_id[str(code)] = task_id

    records: List[Dict[str, Any]] = []
    done = False
    step = 0
    while not done and step < max_steps:
        current_row: Optional[pd.Series] = None
        if hasattr(env, "current_process_tasks") and env.current_task_idx < len(env.current_process_tasks):
            current_row = env.current_process_tasks.iloc[env.current_task_idx].copy()

        _, task_feats, res_feats, adj, node_depths, critical_mask = _normalize_state_for_agent(
            state,
            task_input_dim=agent.task_input_dim,
            resource_input_dim=agent.resource_input_dim,
            disable_fe=not agent.config.use_feature_engineering,
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
        state, reward, done, info = env.step(action)

        if info.get("task_scheduled") and current_row is not None:
            task_id = int(info.get("task_id", current_row.get("id")))
            records.append({
                "algorithm": algorithm,
                "seed": int(seed),
                "process_id": int(process_id),
                "step": int(step),
                "action": int(action),
                "task_id": task_id,
                "task_code": task_code_value(current_row),
                "task_name": str(info.get("task_name", current_row.get("name", ""))),
                "task_type": str(current_row.get("task_type", "")),
                "original_host": str(current_row.get("host", "")),
                "assigned_host": str(info.get("host", "")),
                "start_time": float(info.get("start_time", 0.0)),
                "end_time": float(info.get("end_time", 0.0)),
                "duration": float(info.get("duration", 0.0)),
                "reward": float(reward),
                "cpu_req": float(info.get("cpu_req", 0.0)),
                "memory_req": float(info.get("memory_req", 0.0)),
            })
        step += 1

    schedule = pd.DataFrame(records)
    schedule = add_dependency_corrected_times(schedule, dependencies, code_to_task_id)
    validation = validate_dependencies(schedule, dependencies, code_to_task_id)
    validation.update({
        "algorithm": algorithm,
        "seed": int(seed),
        "process_id": int(process_id),
        "tasks": int(len(schedule)),
        "raw_makespan": float(schedule["end_time"].max()) if not schedule.empty else 0.0,
        "strict_makespan": float(schedule["strict_end_time"].max()) if not schedule.empty else 0.0,
        "env_makespan": float(env.get_makespan()),
        "utilization": float(env.get_resource_utilization()),
        "load_balance": float(env.get_load_balance_score()),
        "done": bool(done),
        "steps": int(step),
    })
    return schedule, validation


def add_dependency_corrected_times(schedule: pd.DataFrame,
                                   dependencies: List[Dict[str, Any]],
                                   code_to_task_id: Dict[str, int]) -> pd.DataFrame:
    if schedule.empty:
        return schedule

    schedule = schedule.copy()
    task_to_preds: Dict[int, List[int]] = {}
    for dep in dependencies:
        pre_id = code_to_task_id.get(str(dep.get("pre_task_code")))
        post_id = code_to_task_id.get(str(dep.get("post_task_code")))
        if pre_id is not None and post_id is not None:
            task_to_preds.setdefault(post_id, []).append(pre_id)

    machine_available: Dict[str, float] = {}
    strict_end_by_task: Dict[int, float] = {}
    strict_starts: List[float] = []
    strict_ends: List[float] = []
    waits: List[float] = []

    for _, row in schedule.sort_values("step").iterrows():
        task_id = int(row["task_id"])
        host = str(row["assigned_host"])
        machine_ready = machine_available.get(host, 0.0)
        pred_ready = max([strict_end_by_task.get(pred, 0.0) for pred in task_to_preds.get(task_id, [])] or [0.0])
        strict_start = max(machine_ready, pred_ready)
        duration = float(row["duration"])
        strict_end = strict_start + duration
        machine_available[host] = strict_end
        strict_end_by_task[task_id] = strict_end
        strict_starts.append(strict_start)
        strict_ends.append(strict_end)
        waits.append(max(0.0, strict_start - float(row["start_time"])))

    schedule["strict_start_time"] = strict_starts
    schedule["strict_end_time"] = strict_ends
    schedule["dependency_wait"] = waits
    return schedule


def validate_dependencies(schedule: pd.DataFrame,
                          dependencies: List[Dict[str, Any]],
                          code_to_task_id: Dict[str, int]) -> Dict[str, Any]:
    by_task = {
        int(row["task_id"]): row
        for _, row in schedule.iterrows()
    }
    order_violations = []
    raw_time_violations = []
    strict_time_violations = []
    mapped = 0
    for dep in dependencies:
        pre_code = dep.get("pre_task_code")
        post_code = dep.get("post_task_code")
        pre_id = code_to_task_id.get(str(pre_code))
        post_id = code_to_task_id.get(str(post_code))
        if pre_id is None or post_id is None or pre_id not in by_task or post_id not in by_task:
            continue
        mapped += 1
        pre = by_task[pre_id]
        post = by_task[post_id]
        if int(pre["step"]) >= int(post["step"]):
            order_violations.append([pre_code, post_code])
        if float(pre["end_time"]) > float(post["start_time"]) + 1e-9:
            raw_time_violations.append({
                "pre_task_code": pre_code,
                "post_task_code": post_code,
                "pre_end": float(pre["end_time"]),
                "post_start": float(post["start_time"]),
            })
        if float(pre["strict_end_time"]) > float(post["strict_start_time"]) + 1e-9:
            strict_time_violations.append([pre_code, post_code])

    return {
        "dependencies_total": int(len(dependencies)),
        "dependencies_mapped": int(mapped),
        "order_violations": int(len(order_violations)),
        "raw_time_violations": int(len(raw_time_violations)),
        "strict_time_violations": int(len(strict_time_violations)),
        "raw_time_violation_examples": raw_time_violations[:10],
    }


def short_label(name: str, fallback: str, max_len: int = 20) -> str:
    text = str(name or fallback)
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


def plot_case_gantt(schedule: pd.DataFrame,
                    validation_rows: List[Dict[str, Any]],
                    process_meta: Dict[str, Any],
                    out_path: Path,
                    use_strict_time: bool = True) -> None:
    if schedule.empty:
        return

    start_col = "strict_start_time" if use_strict_time else "start_time"
    end_col = "strict_end_time" if use_strict_time else "end_time"
    schedule = schedule.copy()
    schedule["plot_duration"] = schedule[end_col] - schedule[start_col]

    hosts = sorted(schedule["assigned_host"].astype(str).unique())
    host_to_y = {host: idx for idx, host in enumerate(hosts)}
    algos = list(ALGORITHMS.keys())

    fig, axes = plt.subplots(len(algos), 1, figsize=(15, max(7, 3.8 * len(algos))), dpi=180, sharex=True)
    if len(algos) == 1:
        axes = [axes]

    colors = {
        "ga_hpo_dqn": "#8da0cb",
        "ga_hpo_fe_dqn": "#66c2a5",
    }
    for ax, algo in zip(axes, algos):
        sub = schedule[schedule["algorithm"] == algo].sort_values("step")
        for _, row in sub.iterrows():
            y = host_to_y[str(row["assigned_host"])]
            ax.broken_barh(
                [(float(row[start_col]), float(row["plot_duration"]))],
                (y - 0.35, 0.7),
                facecolors=colors.get(algo, "#6baed6"),
                edgecolors="white",
                linewidth=0.4,
                alpha=0.86,
            )
            if len(sub) <= 70 and float(row["plot_duration"]) > 0:
                ax.text(
                    float(row[start_col]) + float(row["plot_duration"]) / 2,
                    y,
                    str(int(row["step"]) + 1),
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="#1f2933",
                )

        metrics = next((item for item in validation_rows if item["algorithm"] == algo), {})
        ax.set_title(
            f"{ALGORITHMS[algo]} | makespan={metrics.get('strict_makespan', 0):.1f}s "
            f"(raw={metrics.get('raw_makespan', 0):.1f}s) | "
            f"raw dep violations={metrics.get('raw_time_violations', 0)}",
            loc="left",
            fontsize=10,
        )
        ax.set_yticks(range(len(hosts)))
        ax.set_yticklabels(hosts, fontsize=7)
        ax.grid(axis="x", linestyle="--", alpha=0.28)

    title = (
        f"Workflow {process_meta.get('process_id')} | "
        f"tasks={process_meta.get('task_count')} | "
        f"duration={process_meta.get('duration_seconds')}s | "
        f"DAG depth={process_meta.get('dag_depth')} width={process_meta.get('dag_width')} "
        f"edges={process_meta.get('dag_edge_count')}"
    )
    fig.suptitle(title, fontsize=11)
    axes[-1].set_xlabel("Time (seconds, dependency-corrected)" if use_strict_time else "Time (seconds, raw env)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def process_meta_from_rows(rows: pd.DataFrame, process_id: int) -> Dict[str, Any]:
    row = rows[rows["process_id"] == process_id].iloc[0]
    keys = [
        "process_id",
        "process_name",
        "task_count",
        "duration_seconds",
        "dag_depth",
        "dag_width",
        "dag_edge_count",
        "dag_complexity_score",
        "workflow_size",
        "duration_bin",
        "dag_complexity_bin",
    ]
    return {key: row.get(key) for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/ablation_stratified_paper_20260516")
    parser.add_argument("--data-dir", default="data/raw_data")
    parser.add_argument("--output-dir", default="results/figures/large_workflow_gantt")
    parser.add_argument("--process-id", action="append", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Use one seed for all selected workflows.")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--model-name", default="final_model.pt")
    parser.add_argument("--plot-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    root = Path(args.root)
    out_dir = Path(args.output_dir)

    workflow_rows = collect_final_workflow_rows(root)
    if args.process_id:
        cases = []
        for process_id in args.process_id:
            cases.append({
                "process_id": process_id,
                "seed": args.seed if args.seed is not None else closest_seed_to_mean_gain(workflow_rows, process_id),
                "selection_reason": "manual",
            })
    else:
        cases = representative_large_cases(root, limit=args.limit)
        if args.seed is not None:
            for case in cases:
                case["seed"] = args.seed

    _, _, final_env, _ = make_replay_envs(
        Path(args.data_dir),
        train_ratio=0.8,
        logger=logging.getLogger("large_workflow_gantt"),
        eval_split="test",
        train_workflows_per_episode=3,
        eval_workflows_per_episode=5,
    )

    summary_rows: List[Dict[str, Any]] = []
    for case in cases:
        process_id = int(case["process_id"])
        seed = int(case["seed"])
        meta = process_meta_from_rows(workflow_rows, process_id)
        case_dir = out_dir / f"workflow_{process_id}_seed_{seed}"
        case_dir.mkdir(parents=True, exist_ok=True)
        all_schedules: List[pd.DataFrame] = []
        validations: List[Dict[str, Any]] = []

        for group, label in ALGORITHMS.items():
            run_dir = root / group / f"seed_{seed}"
            agent = load_agent(run_dir, model_name=args.model_name)
            schedule, validation = run_agent_on_workflow(
                agent,
                final_env,
                process_id=process_id,
                algorithm=group,
                seed=seed,
            )
            schedule_path = case_dir / f"{group}_schedule.csv"
            schedule.to_csv(schedule_path, index=False)
            validation["schedule_csv"] = str(schedule_path)
            validations.append(validation)
            all_schedules.append(schedule)

        combined = pd.concat(all_schedules, ignore_index=True)
        combined_path = case_dir / "combined_schedule.csv"
        combined.to_csv(combined_path, index=False)

        validation_payload = {
            "case": {
                **meta,
                "seed": seed,
                "selection_reason": case.get("selection_reason"),
            },
            "validations": validations,
        }
        validation_path = case_dir / "dependency_check.json"
        write_json(validation_path, validation_payload)

        gantt_path = case_dir / "gantt_dependency_corrected.svg"
        plot_case_gantt(
            combined,
            validations,
            process_meta={**meta, "process_id": process_id},
            out_path=gantt_path,
            use_strict_time=not args.plot_raw,
        )

        dqn = next(item for item in validations if item["algorithm"] == "ga_hpo_dqn")
        fe = next(item for item in validations if item["algorithm"] == "ga_hpo_fe_dqn")
        summary_rows.append({
            **meta,
            "seed": seed,
            "selection_reason": case.get("selection_reason"),
            "dqn_raw_makespan": dqn["raw_makespan"],
            "fe_raw_makespan": fe["raw_makespan"],
            "raw_abs_gain": dqn["raw_makespan"] - fe["raw_makespan"],
            "raw_rel_gain_pct": (
                (dqn["raw_makespan"] - fe["raw_makespan"]) / dqn["raw_makespan"] * 100.0
                if dqn["raw_makespan"] else math.nan
            ),
            "dqn_strict_makespan": dqn["strict_makespan"],
            "fe_strict_makespan": fe["strict_makespan"],
            "strict_abs_gain": dqn["strict_makespan"] - fe["strict_makespan"],
            "strict_rel_gain_pct": (
                (dqn["strict_makespan"] - fe["strict_makespan"]) / dqn["strict_makespan"] * 100.0
                if dqn["strict_makespan"] else math.nan
            ),
            "dqn_raw_dep_violations": dqn["raw_time_violations"],
            "fe_raw_dep_violations": fe["raw_time_violations"],
            "gantt_svg": str(gantt_path),
            "combined_schedule_csv": str(combined_path),
            "dependency_check_json": str(validation_path),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
