#!/usr/bin/env python3
"""Aggregate per-workflow final_eval_workflows.csv files into stratified tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ALGORITHM_NAMES: Dict[str, str] = {
    "ga_hpo_dqn": "GA-HPO DQN",
    "ga_hpo_fe_dqn": "GA-HPO FE-DQN",
    "fe_iddqn": "FE-IDDQN",
    "iddqn": "IDDQN",
    "ga_hpo_iddqn": "GA-HPO FE-IDDQN",
}

DEFAULT_STRATA = [
    "workflow_size",
    "duration_bin",
    "dag_complexity_bin",
    "workflow_stratum",
    "balanced_workflow_stratum",
]


def _ci95(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) <= 1:
        return 0.0
    return float(1.96 * clean.std(ddof=1) / np.sqrt(len(clean)))


def collect_records(root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for group_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        group = group_dir.name
        algorithm = ALGORITHM_NAMES.get(group, group)
        for seed_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            csv_path = seed_dir / "final_eval_workflows.csv"
            if not csv_path.exists():
                continue
            try:
                seed = int(seed_dir.name.replace("seed_", ""))
            except ValueError:
                seed = seed_dir.name
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df.insert(0, "seed", seed)
            df.insert(0, "algorithm", algorithm)
            df.insert(0, "group", group)
            df.insert(0, "run_dir", str(seed_dir))
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize(records: pd.DataFrame, strata_columns: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for stratum_type in strata_columns:
        if stratum_type not in records.columns:
            continue
        grouped = records.groupby(["group", "algorithm", stratum_type], dropna=True)
        for (group, algorithm, stratum_value), df in grouped:
            makespan = df["makespan"].astype(float)
            utilization = df["utilization"].astype(float)
            load_balance = df["load_balance"].astype(float)
            rows.append({
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "group": group,
                "algorithm": algorithm,
                "samples": int(len(df)),
                "seeds": int(df["seed"].nunique()),
                "workflows": int(df["process_id"].nunique()) if "process_id" in df.columns else int(len(df)),
                "makespan_mean": float(makespan.mean()),
                "makespan_std": float(makespan.std(ddof=1)) if len(makespan) > 1 else 0.0,
                "makespan_ci95": _ci95(makespan),
                "utilization_mean": float(utilization.mean()),
                "utilization_std": float(utilization.std(ddof=1)) if len(utilization) > 1 else 0.0,
                "utilization_ci95": _ci95(utilization),
                "load_balance_mean": float(load_balance.mean()),
                "load_balance_std": float(load_balance.std(ddof=1)) if len(load_balance) > 1 else 0.0,
                "load_balance_ci95": _ci95(load_balance),
                "truncated_samples": int(df.get("truncated", pd.Series(dtype=bool)).sum()),
            })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    best = (
        summary.groupby(["stratum_type", "stratum_value"])["makespan_mean"]
        .min()
        .rename("best_makespan")
        .reset_index()
    )
    summary = summary.merge(best, on=["stratum_type", "stratum_value"], how="left")
    summary["makespan_gap_to_best_pct"] = (
        (summary["makespan_mean"] - summary["best_makespan"])
        / summary["best_makespan"].replace(0, np.nan)
        * 100.0
    )
    return summary.sort_values(
        ["stratum_type", "stratum_value", "makespan_mean", "algorithm"]
    ).reset_index(drop=True)


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        path.write_text("No stratified workflow evaluation files found.\n", encoding="utf-8")
        return

    lines: List[str] = ["# Stratified Final Evaluation", ""]
    for stratum_type, block in summary.groupby("stratum_type"):
        lines.extend([f"## {stratum_type}", ""])
        cols = [
            "stratum_value",
            "algorithm",
            "seeds",
            "workflows",
            "makespan_mean",
            "makespan_ci95",
            "utilization_mean",
            "load_balance_mean",
            "makespan_gap_to_best_pct",
        ]
        lines.append(block[cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--strata", nargs="*", default=DEFAULT_STRATA)
    args = parser.parse_args()

    records = collect_records(args.root)
    records_path = args.root / "ablation_workflow_eval.records.csv"
    records.to_csv(records_path, index=False)

    summary = summarize(records, args.strata)
    summary_path = args.root / "ablation_stratified_summary.csv"
    summary.to_csv(summary_path, index=False)

    markdown_path = args.root / "ablation_stratified_summary.md"
    write_markdown(summary, markdown_path)

    if summary.empty:
        print("No stratified workflow evaluation files found.")
    else:
        print(summary.to_string(index=False))
        print(f"Wrote {summary_path}")
        print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
