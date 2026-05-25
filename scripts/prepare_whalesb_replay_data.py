#!/usr/bin/env python3
"""Prepare replay training data directly from whalesb MySQL.

This script exports required tables for train_fe_iddqn_ga_hpo.py, validates
dependency integrity, and generates fixed process-id splits for reproducibility.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine


LOGGER = logging.getLogger("prepare_whalesb_replay_data")


TABLE_SQL: Dict[str, str] = {
    "process_definition": "SELECT * FROM t_ds_process_definition",
    "process_instance": "SELECT * FROM t_ds_process_instance",
    "task_definition": "SELECT * FROM t_ds_task_definition",
    "task_instance": "SELECT * FROM t_ds_task_instance",
    "process_task_relation": "SELECT * FROM t_ds_process_task_relation",
}


OUTPUT_FILES: Dict[str, str] = {
    "process_definition": "t_ds_process_definition.csv",
    "process_instance": "t_ds_process_instance.csv",
    "task_definition": "t_ds_task_definition.csv",
    "task_instance": "t_ds_task_instance.csv",
    "process_task_relation": "t_ds_process_task_relation.csv",
}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export whalesb data for FE-IDDQN replay training"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default="")
    parser.add_argument("--database", default="whalesb")
    parser.add_argument(
        "--output-dir",
        default="data/raw_data",
        help="Directory to write exported CSV files",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional lower bound for process_instance.start_time (inclusive)",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional upper bound for process_instance.start_time (inclusive)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cycle-check-limit",
        type=int,
        default=1000,
        help="Max process_definition_code groups to run full DAG cycle checks",
    )
    parser.add_argument(
        "--skip-cycle-check",
        action="store_true",
        help="Skip DAG cycle checks to speed up export",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extraction and validation without writing files",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def mysql_engine(args: argparse.Namespace):
    conn = f"mysql+pymysql://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}"
    return create_engine(conn)


def load_tables(engine, args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    tables = {}
    for key, sql in TABLE_SQL.items():
        LOGGER.info("Loading %s ...", key)
        tables[key] = pd.read_sql(sql, engine)
        LOGGER.info("Loaded %s rows=%d", key, len(tables[key]))

    if args.start_time or args.end_time:
        proc = tables["process_instance"]
        if "start_time" not in proc.columns:
            raise ValueError("process_instance has no start_time column; cannot apply time window")

        mask = pd.Series([True] * len(proc), index=proc.index)
        start_ts = pd.to_datetime(args.start_time) if args.start_time else None
        end_ts = pd.to_datetime(args.end_time) if args.end_time else None
        proc_start = pd.to_datetime(proc["start_time"], errors="coerce")

        if start_ts is not None:
            mask &= proc_start >= start_ts
        if end_ts is not None:
            mask &= proc_start <= end_ts

        tables["process_instance"] = proc[mask].copy().reset_index(drop=True)
        LOGGER.info(
            "Applied start_time window: process_instance rows -> %d",
            len(tables["process_instance"]),
        )

    return tables


def normalize_columns(tables: Dict[str, pd.DataFrame]) -> None:
    proc = tables["process_instance"]
    rel = tables["process_task_relation"]
    tasks = tables["task_instance"]

    if "process_definition_code" not in proc.columns and "process_definition_id" in proc.columns:
        proc["process_definition_code"] = proc["process_definition_id"]

    if "process_definition_code" not in rel.columns and "process_definition_id" in rel.columns:
        rel["process_definition_code"] = rel["process_definition_id"]

    if "process_definition_code" not in tasks.columns and "process_instance_id" in tasks.columns:
        proc_map = proc[["id", "process_definition_code"]].drop_duplicates()
        tasks2 = tasks.merge(
            proc_map,
            left_on="process_instance_id",
            right_on="id",
            how="left",
            suffixes=("", "_proc"),
        )
        if "id_proc" in tasks2.columns:
            tasks2 = tasks2.drop(columns=["id_proc"])
        tables["task_instance"] = tasks2
        tasks = tasks2

    if "task_code" not in tasks.columns and "task_definition_code" in tasks.columns:
        tasks["task_code"] = tasks["task_definition_code"]


def filter_to_successful_scope(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    proc = tables["process_instance"].copy()
    tasks = tables["task_instance"].copy()
    task_def = tables["task_definition"].copy()
    proc_def = tables["process_definition"].copy()
    rel = tables["process_task_relation"].copy()

    if "state" not in proc.columns:
        raise ValueError("process_instance missing required column: state")
    if "id" not in proc.columns:
        raise ValueError("process_instance missing required column: id")
    if "process_instance_id" not in tasks.columns:
        raise ValueError("task_instance missing required column: process_instance_id")

    successful_proc = proc[proc["state"] == 7].copy()
    successful_ids = set(successful_proc["id"].tolist())

    tasks = tasks[tasks["process_instance_id"].isin(successful_ids)].copy()
    non_empty_proc_ids = set(tasks["process_instance_id"].unique().tolist())
    successful_proc = successful_proc[successful_proc["id"].isin(non_empty_proc_ids)].copy()

    proc_codes = set(successful_proc["process_definition_code"].dropna().unique().tolist())
    rel = rel[rel["process_definition_code"].isin(proc_codes)].copy()

    if "code" in proc_def.columns:
        proc_def = proc_def[proc_def["code"].isin(proc_codes)].copy()

    task_codes = set()
    if "task_code" in tasks.columns:
        task_codes |= set(tasks["task_code"].dropna().unique().tolist())
    if "task_definition_code" in tasks.columns:
        task_codes |= set(tasks["task_definition_code"].dropna().unique().tolist())

    if "code" in task_def.columns and task_codes:
        task_def = task_def[task_def["code"].isin(task_codes)].copy()

    # Keep only dependency edges that map to actual task nodes in selected scope.
    task_codes_by_proc: Dict[object, set] = {}
    for p_code, g in tasks.groupby("process_definition_code"):
        codes = set(g.get("task_code", pd.Series(dtype=object)).dropna().tolist())
        if not codes and "task_definition_code" in g.columns:
            codes = set(g["task_definition_code"].dropna().tolist())
        task_codes_by_proc[p_code] = codes

    if not rel.empty:
        keep_mask = []
        for _, row in rel.iterrows():
            p_code = row.get("process_definition_code")
            pre = row.get("pre_task_code")
            post = row.get("post_task_code")
            nodes = task_codes_by_proc.get(p_code, set())
            keep_mask.append(pd.notna(pre) and pd.notna(post) and pre in nodes and post in nodes)
        rel = rel[pd.Series(keep_mask, index=rel.index)].copy()

    return {
        "process_definition": proc_def.reset_index(drop=True),
        "process_instance": successful_proc.reset_index(drop=True),
        "task_definition": task_def.reset_index(drop=True),
        "task_instance": tasks.reset_index(drop=True),
        "process_task_relation": rel.reset_index(drop=True),
    }


def _rank_bins(series: pd.Series, labels: List[str]) -> pd.Series:
    """按秩分位数分箱，避免重复值导致 qcut 失败。"""
    if series.empty:
        return pd.Series(dtype=object)

    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if numeric.nunique(dropna=False) <= 1:
        return pd.Series([labels[0]] * len(numeric), index=series.index, dtype=object)

    pct = numeric.rank(method="first", pct=True)
    bins = np.linspace(0.0, 1.0, num=len(labels) + 1)
    bins[0] = -0.001
    return pd.cut(pct, bins=bins, labels=labels, include_lowest=True).astype(str)


def _merge_sparse_strata(stats: pd.DataFrame,
                         primary_col: str,
                         fallback_cols: List[str],
                         min_count: int = 5) -> pd.Series:
    """保留主分层，稀疏桶自动合并到更粗粒度分层。"""
    labels = ("full|" + stats[primary_col].astype(str)).astype(object)

    for col in fallback_cols:
        if col not in stats.columns:
            continue

        counts = labels.value_counts(dropna=False)
        sparse_labels = set(counts[counts < min_count].index.tolist())
        if not sparse_labels:
            break

        sparse_values = set(stats.loc[labels.isin(sparse_labels), col].astype(str).tolist())
        if not sparse_values:
            continue

        merge_mask = stats[col].astype(str).isin(sparse_values)
        labels.loc[merge_mask] = col + "|" + stats.loc[merge_mask, col].astype(str)

    counts = labels.value_counts(dropna=False)
    sparse_labels = set(counts[counts < min_count].index.tolist())
    if sparse_labels:
        sparse_sizes = set(stats.loc[labels.isin(sparse_labels), "workflow_size"].astype(str).tolist())
        merge_mask = stats["workflow_size"].astype(str).isin(sparse_sizes)
        labels.loc[merge_mask] = "workflow_size|" + stats.loc[merge_mask, "workflow_size"].astype(str)

    counts = labels.value_counts(dropna=False)
    sparse_labels = set(counts[counts < min_count].index.tolist())
    if sparse_labels:
        labels.loc[labels.isin(sparse_labels)] = "all|merged_sparse"

    return labels


def _build_definition_dag_stats(rel_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if rel_df is None or rel_df.empty:
        return pd.DataFrame(columns=[
            "process_definition_code",
            "dag_node_count",
            "dag_edge_count",
            "dag_depth",
            "dag_width",
        ])

    rows = []
    for process_code, group in rel_df.groupby("process_definition_code"):
        graph = nx.DiGraph()
        for _, row in group.iterrows():
            pre = row.get("pre_task_code")
            post = row.get("post_task_code")
            if pd.notna(pre) and pd.notna(post):
                graph.add_edge(pre, post)

        node_count = int(graph.number_of_nodes())
        edge_count = int(graph.number_of_edges())
        if node_count == 0:
            depth = 0
            width = 0
        elif nx.is_directed_acyclic_graph(graph):
            depth = int(nx.dag_longest_path_length(graph) + 1)
            try:
                width = max(len(level) for level in nx.topological_generations(graph))
            except Exception:
                width = 1
        else:
            LOGGER.warning("Cycle detected in process_definition_code=%s; using fallback DAG stats", process_code)
            depth = 0
            width = node_count

        rows.append({
            "process_definition_code": process_code,
            "dag_node_count": node_count,
            "dag_edge_count": edge_count,
            "dag_depth": depth,
            "dag_width": width,
        })

    return pd.DataFrame(rows)


def build_process_stats(process_df: pd.DataFrame,
                        task_df: pd.DataFrame,
                        rel_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    task_counts = (
        task_df.groupby("process_instance_id").size().rename("task_count").reset_index()
    )
    stats = process_df.merge(task_counts, left_on="id", right_on="process_instance_id", how="inner")
    stats = stats.rename(columns={"id": "process_id", "name": "process_name"})
    if "process_definition_code" not in stats.columns:
        stats["process_definition_code"] = np.nan

    stats = stats[[
        "process_id",
        "process_definition_code",
        "task_count",
        "process_name",
        "start_time",
        "end_time",
    ]].copy()
    start_time = pd.to_datetime(stats["start_time"], errors="coerce")
    end_time = pd.to_datetime(stats["end_time"], errors="coerce")
    stats["duration_seconds"] = (
        end_time - start_time
    ).dt.total_seconds().clip(lower=0).fillna(0.0)

    dag_stats = _build_definition_dag_stats(rel_df)
    if not dag_stats.empty:
        stats = stats.merge(dag_stats, on="process_definition_code", how="left")
    for column in ["dag_node_count", "dag_edge_count", "dag_depth", "dag_width"]:
        if column not in stats.columns:
            stats[column] = 0
        stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(0.0)

    max_possible_edges = (stats["task_count"] * (stats["task_count"] - 1) / 2).replace(0, np.nan)
    stats["dag_density"] = (
        stats["dag_edge_count"] / max_possible_edges
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(lower=0.0, upper=1.0)
    stats["parallelism_ratio"] = (
        stats["task_count"] / stats["dag_depth"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    stats["dag_complexity_score"] = (
        np.log1p(stats["task_count"]) +
        0.7 * np.log1p(stats["dag_edge_count"]) +
        0.3 * stats["dag_depth"] +
        2.0 * stats["dag_density"]
    )

    stats["workflow_size"] = pd.cut(
        stats["task_count"],
        bins=[0, 10, 30, float("inf")],
        labels=["small", "medium", "large"],
    )
    stats["duration_bin"] = _rank_bins(stats["duration_seconds"], ["short", "medium", "long"])
    stats["dag_complexity_bin"] = _rank_bins(stats["dag_complexity_score"], ["simple", "moderate", "complex"])
    stats["workflow_stratum"] = (
        stats["workflow_size"].astype(str) +
        "|" + stats["duration_bin"].astype(str) +
        "|" + stats["dag_complexity_bin"].astype(str)
    )
    stats["size_duration_stratum"] = (
        stats["workflow_size"].astype(str) +
        "|" + stats["duration_bin"].astype(str)
    )
    stats["size_complexity_stratum"] = (
        stats["workflow_size"].astype(str) +
        "|" + stats["dag_complexity_bin"].astype(str)
    )
    stats["duration_complexity_stratum"] = (
        stats["duration_bin"].astype(str) +
        "|" + stats["dag_complexity_bin"].astype(str)
    )
    stats["balanced_workflow_stratum"] = _merge_sparse_strata(
        stats,
        primary_col="workflow_stratum",
        fallback_cols=[
            "size_duration_stratum",
            "size_complexity_stratum",
            "duration_complexity_stratum",
            "workflow_size",
        ],
        min_count=5,
    )
    return stats


def make_splits(process_stats: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify_name = None
    stratify_col = None
    for candidate in [
        "balanced_workflow_stratum",
        "workflow_stratum",
        "size_duration_stratum",
        "size_complexity_stratum",
        "workflow_size",
    ]:
        value_counts = process_stats[candidate].value_counts(dropna=False)
        can_stratify = bool((value_counts >= 5).all()) and len(value_counts) > 1
        if can_stratify:
            stratify_name = candidate
            stratify_col = process_stats[candidate]
            break

    if stratify_col is None:
        LOGGER.warning("Workflow strata are imbalanced; using non-stratified split")
        train_df, temp_df = train_test_split(
            process_stats,
            test_size=0.4,
            random_state=seed,
            shuffle=True,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=seed,
            shuffle=True,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    LOGGER.info("Using stratified replay split column: %s", stratify_name)
    value_counts = stratify_col.value_counts(dropna=False)
    can_stratify = bool((value_counts >= 2).all()) and len(value_counts) > 1

    if can_stratify:
        train_df, temp_df = train_test_split(
            process_stats,
            test_size=0.4,
            random_state=seed,
            stratify=stratify_col,
        )
        temp_counts = temp_df[stratify_name].value_counts(dropna=False)
        temp_stratify = (
            temp_df[stratify_name]
            if bool((temp_counts >= 2).all()) and len(temp_counts) > 1
            else None
        )
        if temp_stratify is None:
            LOGGER.warning("Temp split strata are sparse; val/test split is non-stratified")
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=seed,
            stratify=temp_stratify,
        )
    else:
        LOGGER.warning("Workflow strata are imbalanced after fallback; using non-stratified split")
        train_df, temp_df = train_test_split(
            process_stats,
            test_size=0.4,
            random_state=seed,
            shuffle=True,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=seed,
            shuffle=True,
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def dependency_integrity_report(
    proc_df: pd.DataFrame,
    task_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    skip_cycle_check: bool,
    cycle_check_limit: int,
) -> Dict[str, object]:
    report: Dict[str, object] = {}

    codes_in_proc = set(proc_df["process_definition_code"].dropna().unique().tolist())
    codes_in_rel = set(rel_df["process_definition_code"].dropna().unique().tolist())
    codes_missing_rel = sorted(list(codes_in_proc - codes_in_rel))
    report["process_codes_without_relations"] = len(codes_missing_rel)

    task_by_code: Dict[object, set] = {}
    for p_code, g in task_df.groupby("process_definition_code"):
        task_codes = set(g.get("task_code", pd.Series(dtype=object)).dropna().tolist())
        if not task_codes and "task_definition_code" in g.columns:
            task_codes = set(g["task_definition_code"].dropna().tolist())
        task_by_code[p_code] = task_codes

    missing_edge_count = 0
    checked_edges = 0
    for _, row in rel_df.iterrows():
        code = row.get("process_definition_code")
        pre = row.get("pre_task_code")
        post = row.get("post_task_code")
        if pd.isna(code) or pd.isna(pre) or pd.isna(post):
            continue
        checked_edges += 1
        nodes = task_by_code.get(code, set())
        if pre not in nodes or post not in nodes:
            missing_edge_count += 1

    report["checked_relation_edges"] = int(checked_edges)
    report["relation_edges_missing_task_nodes"] = int(missing_edge_count)

    if skip_cycle_check:
        report["cycle_check_skipped"] = True
        return report

    cycle_detected = 0
    checked_codes = 0
    for idx, (code, g) in enumerate(rel_df.groupby("process_definition_code")):
        if idx >= cycle_check_limit:
            break
        G = nx.DiGraph()
        for _, row in g.iterrows():
            pre = row.get("pre_task_code")
            post = row.get("post_task_code")
            if pd.notna(pre) and pd.notna(post):
                G.add_edge(pre, post)
        if len(G.nodes) > 0:
            checked_codes += 1
            if not nx.is_directed_acyclic_graph(G):
                cycle_detected += 1

    report["cycle_check_skipped"] = False
    report["cycle_check_codes_checked"] = int(checked_codes)
    report["cycle_detected_codes"] = int(cycle_detected)
    report["cycle_check_limit"] = int(cycle_check_limit)
    return report


def dataset_info(process_stats: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, object]:
    stratum_col = "balanced_workflow_stratum" if "balanced_workflow_stratum" in process_stats.columns else "workflow_size"
    return {
        "total_processes": int(len(process_stats)),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "workflow_size_distribution": process_stats["workflow_size"].value_counts().to_dict(),
        "train_workflow_distribution": train_df["workflow_size"].value_counts().to_dict(),
        "val_workflow_distribution": val_df["workflow_size"].value_counts().to_dict(),
        "test_workflow_distribution": test_df["workflow_size"].value_counts().to_dict(),
        "duration_bin_distribution": process_stats.get("duration_bin", pd.Series(dtype=object)).value_counts().to_dict(),
        "dag_complexity_distribution": process_stats.get("dag_complexity_bin", pd.Series(dtype=object)).value_counts().to_dict(),
        "train_duration_distribution": train_df.get("duration_bin", pd.Series(dtype=object)).value_counts().to_dict(),
        "val_duration_distribution": val_df.get("duration_bin", pd.Series(dtype=object)).value_counts().to_dict(),
        "test_duration_distribution": test_df.get("duration_bin", pd.Series(dtype=object)).value_counts().to_dict(),
        "train_dag_complexity_distribution": train_df.get("dag_complexity_bin", pd.Series(dtype=object)).value_counts().to_dict(),
        "val_dag_complexity_distribution": val_df.get("dag_complexity_bin", pd.Series(dtype=object)).value_counts().to_dict(),
        "test_dag_complexity_distribution": test_df.get("dag_complexity_bin", pd.Series(dtype=object)).value_counts().to_dict(),
        "stratum_distribution": process_stats[stratum_col].value_counts().to_dict(),
        "train_stratum_distribution": train_df[stratum_col].value_counts().to_dict(),
        "val_stratum_distribution": val_df[stratum_col].value_counts().to_dict(),
        "test_stratum_distribution": test_df[stratum_col].value_counts().to_dict(),
    }


def write_outputs(
    output_dir: Path,
    tables: Dict[str, pd.DataFrame],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    info: Dict[str, object],
    validation: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    for key, df in tables.items():
        out_file = output_dir / OUTPUT_FILES[key]
        df.to_csv(out_file, index=False)
        LOGGER.info("Wrote %s rows=%d", out_file, len(df))

    train_df.to_csv(split_dir / "train_data.csv", index=False)
    val_df.to_csv(split_dir / "val_data.csv", index=False)
    test_df.to_csv(split_dir / "test_data.csv", index=False)
    train_df[["process_id"]].to_csv(split_dir / "train_process_ids.csv", index=False)
    val_df[["process_id"]].to_csv(split_dir / "val_process_ids.csv", index=False)
    test_df[["process_id"]].to_csv(split_dir / "test_process_ids.csv", index=False)

    manifest = {
        "source": {
            "host": args.host,
            "port": args.port,
            "database": args.database,
            "start_time": args.start_time,
            "end_time": args.end_time,
        },
        "dataset_info": info,
        "validation": validation,
        "row_counts": {k: int(len(v)) for k, v in tables.items()},
    }

    with open(split_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)

    with open(output_dir / "provenance_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    LOGGER.info("Wrote split and manifest files under %s", output_dir)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    engine = mysql_engine(args)
    try:
        tables = load_tables(engine, args)
    finally:
        engine.dispose()

    normalize_columns(tables)
    scoped = filter_to_successful_scope(tables)

    process_stats = build_process_stats(
        scoped["process_instance"],
        scoped["task_instance"],
        scoped["process_task_relation"],
    )
    train_df, val_df, test_df = make_splits(process_stats, args.seed)

    validation = dependency_integrity_report(
        proc_df=scoped["process_instance"],
        task_df=scoped["task_instance"],
        rel_df=scoped["process_task_relation"],
        skip_cycle_check=args.skip_cycle_check,
        cycle_check_limit=args.cycle_check_limit,
    )
    info = dataset_info(process_stats, train_df, val_df, test_df)

    LOGGER.info("Dataset summary: %s", json.dumps(info, ensure_ascii=False))
    LOGGER.info("Validation summary: %s", json.dumps(validation, ensure_ascii=False))

    if args.dry_run:
        LOGGER.info("Dry run enabled; no files written.")
        return

    write_outputs(Path(args.output_dir), scoped, train_df, val_df, test_df, info, validation, args)


if __name__ == "__main__":
    main()
