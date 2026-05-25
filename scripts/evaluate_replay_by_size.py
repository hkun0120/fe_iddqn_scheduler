    #!/usr/bin/env python3
"""Evaluate replay scheduling performance by workflow size bucket.

This script compares FE-IDDQN against FIFO/SJF/HEFT on small/medium/large
workflow subsets using the same replay simulator and metrics pipeline.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.historical_replay_simulator import HistoricalReplaySimulator as LogReplaySimulator
from models.enhanced_fe_iddqn import EnhancedFE_IDDQN
from train_fe_iddqn_ga_hpo import (
    EnhancedFE_IDDQN_Config,
    _normalize_state_for_agent,
    _reset_env_and_get_state,
    evaluate_dqn_agent,
    load_replay_dataframes,
    run_replay_baseline_comparison,
)


LOGGER = logging.getLogger("evaluate_replay_by_size")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare replay results by workflow size")
    parser.add_argument("--replay_data_dir", type=str, default="data/raw_data")
    parser.add_argument("--split_csv", type=str, default="data/raw_data/splits/val_data.csv")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to FE-IDDQN checkpoint, e.g. results/.../models/final_model.pt")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per algorithm for each size bucket")
    parser.add_argument("--output_dir", type=str,
                        default="results/real_data_validation/by_size_comparison")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _normalize_replay_columns(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    process_instances = data["process_instance"].copy()
    task_instances = data["task_instance"].copy()
    task_definitions = data["task_definition"].copy()
    process_task_relations = data["process_task_relation"].copy()

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

    return {
        "process_instance": process_instances,
        "task_instance": task_instances,
        "task_definition": task_definitions,
        "process_task_relation": process_task_relations,
    }


def _build_env_for_process_ids(
    data: Dict[str, pd.DataFrame],
    process_ids: Set[int],
) -> LogReplaySimulator:
    process_instances = data["process_instance"]
    task_instances = data["task_instance"]

    proc_subset = process_instances[
        (process_instances["id"].isin(process_ids)) & (process_instances["state"] == 7)
    ].copy()
    task_subset = task_instances[
        task_instances["process_instance_id"].isin(set(proc_subset["id"].tolist()))
    ].copy()

    return LogReplaySimulator(
        process_instances=proc_subset,
        task_instances=task_subset,
        task_definitions=data["task_definition"].copy(),
        process_task_relations=data["process_task_relation"].copy(),
    )


def _load_agent(model_path: Path, env: LogReplaySimulator) -> EnhancedFE_IDDQN:
    state = _reset_env_and_get_state(env)
    _, task_feats, res_feats, _, _, _ = _normalize_state_for_agent(
        state,
        task_input_dim=16,
        resource_input_dim=7,
    )

    task_input_dim = task_feats.shape[-1] if len(task_feats.shape) >= 2 else 16
    resource_input_dim = res_feats.shape[-1] if len(res_feats.shape) >= 2 else 7
    action_dim = int(max(
        getattr(env, "num_resources", 1) or 1,
        res_feats.shape[0] if len(res_feats.shape) >= 2 else 1,
    ))
    action_dim = max(action_dim, 1)

    checkpoint = torch.load(str(model_path), map_location="cpu")
    q_state = checkpoint.get("q_network_state_dict", {})
    output_weight = q_state.get("feature_fusion.advantage_stream.2.weight")
    if output_weight is not None:
        ckpt_action_dim = int(output_weight.shape[0])
        action_dim = max(action_dim, ckpt_action_dim)

    cfg = checkpoint.get("config", None)
    if isinstance(cfg, EnhancedFE_IDDQN_Config):
        dqn_cfg = cfg
    elif isinstance(cfg, dict):
        dqn_cfg = EnhancedFE_IDDQN_Config(**cfg)
    else:
        dqn_cfg = EnhancedFE_IDDQN_Config(device="cpu")

    dqn_cfg.device = "cpu"
    agent = EnhancedFE_IDDQN(task_input_dim, resource_input_dim, action_dim, dqn_cfg)
    agent.load(str(model_path))
    return agent


def main() -> None:
    args = parse_args()
    setup_logging()

    replay_data_dir = Path(args.replay_data_dir)
    split_csv = Path(args.split_csv)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(split_csv)
    required_cols = {"process_id", "workflow_size"}
    if not required_cols.issubset(split_df.columns):
        raise ValueError(f"split csv must contain columns: {required_cols}")

    raw_data = load_replay_dataframes(replay_data_dir)
    replay_data = _normalize_replay_columns(raw_data)

    # Build one seed env for shape inference and model loading.
    all_ids = set(split_df["process_id"].astype(int).tolist())
    seed_env = _build_env_for_process_ids(replay_data, all_ids)
    agent = _load_agent(model_path, seed_env)

    rows: List[Dict] = []
    sizes = [s for s in ["small", "medium", "large"] if s in set(split_df["workflow_size"].astype(str))]

    for size in sizes:
        size_ids = set(split_df[split_df["workflow_size"] == size]["process_id"].astype(int).tolist())
        if not size_ids:
            continue

        env = _build_env_for_process_ids(replay_data, size_ids)

        fe_result = evaluate_dqn_agent(agent, env, num_episodes=args.episodes)
        rows.append({
            "Workflow Size": size,
            "Algorithm": "FE-IDDQN",
            "Avg Makespan": float(fe_result.get("makespan", np.nan)),
            "Std Makespan": float(fe_result.get("makespan_std", 0.0)),
            "Avg Utilization": float(fe_result.get("utilization", 0.0)),
            "Episodes": int(fe_result.get("episodes", args.episodes)),
        })

        baselines = run_replay_baseline_comparison(env, LOGGER, num_episodes=args.episodes)
        for alg in ["HEFT", "FIFO", "SJF"]:
            result = baselines.get(alg, {})
            rows.append({
                "Workflow Size": size,
                "Algorithm": alg,
                "Avg Makespan": float(result.get("makespan", np.nan)),
                "Std Makespan": float(result.get("makespan_std", 0.0)),
                "Avg Utilization": float(result.get("utilization", 0.0)),
                "Episodes": int(result.get("episodes", 0)),
            })

        LOGGER.info("Finished size=%s with %d process instances", size, len(size_ids))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Workflow Size", "Avg Makespan"], ascending=[True, True]).reset_index(drop=True)

    csv_path = output_dir / "paper_results_by_size.csv"
    md_path = output_dir / "paper_results_by_size.md"
    summary_path = output_dir / "paper_results_by_size_summary.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_lines = [
        "| Workflow Size | Algorithm | Avg Makespan | Std Makespan | Avg Utilization | Episodes |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['Workflow Size']} | {row['Algorithm']} | {row['Avg Makespan']:.4f} | {row['Std Makespan']:.4f} | {row['Avg Utilization']:.4f} | {int(row['Episodes'])} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary = {}
    if not df.empty:
        for size, g in df.groupby("Workflow Size"):
            g_sorted = g.sort_values("Avg Makespan", ascending=True)
            summary[size] = {
                "best_algorithm": str(g_sorted.iloc[0]["Algorithm"]),
                "best_makespan": float(g_sorted.iloc[0]["Avg Makespan"]),
                "rows": int(len(g_sorted)),
            }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info("By-size comparison saved to %s", csv_path)
    LOGGER.info("Markdown table saved to %s", md_path)
    LOGGER.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
