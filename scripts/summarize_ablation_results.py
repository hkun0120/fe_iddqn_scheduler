#!/usr/bin/env python3
"""Summarize ablation final_eval.json files into CSV/Markdown tables."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


GROUP_LABELS: Dict[str, str] = {
    "ga_hpo_dqn": "GA-HPO DQN",
    "ga_hpo_fe_dqn": "GA-HPO FE-DQN",
    "fe_iddqn": "FE-IDDQN",
    "iddqn": "IDDQN",
    "ga_hpo_iddqn": "GA-HPO FE-IDDQN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation results")
    parser.add_argument("--root", default="results/ablation")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def metric_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ci95": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "ci95": float(1.96 * std / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def collect_rows(root: Path) -> pd.DataFrame:
    rows = []
    for group_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        group = group_dir.name
        for seed_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            if not seed_dir.name.startswith("seed_"):
                continue
            final_eval = read_json(seed_dir / "final_eval.json")
            failure = read_json(seed_dir / "failure_info.json")
            status = read_json(seed_dir / "run_status.json")
            seed = int(seed_dir.name.replace("seed_", ""))
            row = {
                "group": group,
                "algorithm": GROUP_LABELS.get(group, group),
                "seed": seed,
                "status": "completed" if final_eval else "failed" if failure else "running",
                "output_dir": str(seed_dir),
            }
            if final_eval:
                row.update({
                    "makespan": float(final_eval.get("makespan", np.nan)),
                    "makespan_std": float(final_eval.get("makespan_std", np.nan)),
                    "utilization": float(final_eval.get("utilization", np.nan)),
                    "load_balance": float(final_eval.get("load_balance", np.nan)),
                    "episodes": int(final_eval.get("episodes", 0)),
                })
            if failure:
                row["error_type"] = failure.get("error_type")
                row["error_message"] = failure.get("error_message")
            if status:
                row["stage"] = status.get("stage")
                row["updated_at"] = status.get("updated_at")
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for group, group_df in df.groupby("group", sort=False):
        done = group_df[group_df["status"] == "completed"].copy()
        row = {
            "group": group,
            "algorithm": GROUP_LABELS.get(group, group),
            "completed": int(len(done)),
            "total": int(len(group_df)),
            "failed": int((group_df["status"] == "failed").sum()),
            "running": int((group_df["status"] == "running").sum()),
        }
        for metric in ["makespan", "utilization", "load_balance"]:
            values = done[metric].dropna().astype(float).tolist() if metric in done else []
            stats = metric_stats(values)
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty and "makespan_mean" in summary.columns:
        summary = summary.sort_values("makespan_mean", ascending=True, na_position="last").reset_index(drop=True)
        best_makespan = float(summary["makespan_mean"].dropna().min()) if summary["makespan_mean"].notna().any() else np.nan
        if np.isfinite(best_makespan):
            summary["makespan_gap_to_best_pct"] = (
                (summary["makespan_mean"] - best_makespan) / best_makespan * 100.0
            )
    return summary


def write_outputs(root: Path, prefix: Optional[str], df: pd.DataFrame, summary: pd.DataFrame) -> None:
    if prefix:
        out_prefix = Path(prefix)
    else:
        out_prefix = root / "ablation_summary"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_prefix.with_suffix(".per_seed.csv"), index=False)
    summary.to_csv(out_prefix.with_suffix(".csv"), index=False)

    md_lines = [
        "| Algorithm | Completed | Makespan mean | Makespan std | Utilization mean | Load balance mean | Gap to best |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        gap = row.get("makespan_gap_to_best_pct", np.nan)
        md_lines.append(
            "| {algorithm} | {completed}/{total} | {makespan:.4f} | {ms_std:.4f} | "
            "{util:.4f} | {lb:.4f} | {gap:.2f}% |".format(
                algorithm=row["algorithm"],
                completed=int(row["completed"]),
                total=int(row["total"]),
                makespan=row.get("makespan_mean", np.nan),
                ms_std=row.get("makespan_std", np.nan),
                util=row.get("utilization_mean", np.nan),
                lb=row.get("load_balance_mean", np.nan),
                gap=gap,
            )
        )
    out_prefix.with_suffix(".md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    df = collect_rows(root)
    summary = summarize(df)
    write_outputs(root, args.output_prefix, df, summary)
    if summary.empty:
        print(f"No result rows found under {root}")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
