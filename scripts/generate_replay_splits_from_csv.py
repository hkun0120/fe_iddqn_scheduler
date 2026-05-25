#!/usr/bin/env python3
"""Generate fixed replay workflow splits from existing CSV exports.

This is the local-CSV companion to prepare_whalesb_replay_data.py. It rebuilds
only data_dir/splits without requiring a live MySQL connection.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_whalesb_replay_data import (
    build_process_stats,
    dataset_info,
    dependency_integrity_report,
    filter_to_successful_scope,
    make_splits,
    normalize_columns,
)


LOGGER = logging.getLogger("generate_replay_splits_from_csv")


CSV_CANDIDATES: Dict[str, List[str]] = {
    "process_definition": [
        "Commercial_t_ds_process_definition.csv",
        "Commercial_B_t_ds_process_definition.csv",
        "t_ds_process_definition.csv",
        "oceanbase_t_ds_process_definition.csv",
        "__B_t_ds_process_definition.csv",
    ],
    "process_instance": [
        "Commercial_t_ds_process_instance.csv",
        "Commercial_B_t_ds_process_instance.csv",
        "t_ds_process_instance.csv",
        "gaussdb_t_ds_process_instance_a.csv",
        "__B_t_ds_process_instance.csv",
    ],
    "task_definition": [
        "Commercial_t_ds_task_definition.csv",
        "Commercial_B_t_ds_task_definition.csv",
        "t_ds_task_definition.csv",
        "oceanbase_t_ds_task_definition.csv",
        "__B_t_ds_task_definition.csv",
    ],
    "task_instance": [
        "Commercial_t_ds_task_instance.csv",
        "Commercial_B_t_ds_task_instance.csv",
        "t_ds_task_instance.csv",
        "gaussdb_t_ds_task_instance_a.csv",
        "__B_t_ds_task_instance.csv",
    ],
    "process_task_relation": [
        "Commercial_t_ds_process_task_relation.csv",
        "Commercial_B_t_ds_process_task_relation.csv",
        "t_ds_process_task_relation.csv",
        "oceanbase_t_ds_process_task_relation.csv",
        "__B_t_ds_process_task_relation.csv",
    ],
}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stratified replay workflow splits from local CSV files"
    )
    parser.add_argument("--data-dir", default="data/raw_data")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Split output directory. Default: <data-dir>/splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-cycle-check",
        action="store_true",
        help="Skip dependency cycle checks in the validation summary",
    )
    parser.add_argument("--cycle-check-limit", type=int, default=1000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build split stats without writing files",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def read_first_existing_csv(data_dir: Path,
                            logical_name: str,
                            candidates: List[str]) -> pd.DataFrame:
    for name in candidates:
        path = data_dir / name
        if path.exists():
            LOGGER.info("Loading %s from %s", logical_name, path)
            df = pd.read_csv(path)
            LOGGER.info("Loaded %s rows=%d cols=%d", logical_name, len(df), len(df.columns))
            return df
    raise FileNotFoundError(f"No CSV found for {logical_name} in {data_dir}")


def load_tables(data_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        key: read_first_existing_csv(data_dir, key, candidates)
        for key, candidates in CSV_CANDIDATES.items()
    }


def write_split_outputs(output_dir: Path,
                        train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        info: Dict[str, object],
                        validation: Dict[str, object],
                        args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train_data.csv", index=False)
    val_df.to_csv(output_dir / "val_data.csv", index=False)
    test_df.to_csv(output_dir / "test_data.csv", index=False)
    train_df[["process_id"]].to_csv(output_dir / "train_process_ids.csv", index=False)
    val_df[["process_id"]].to_csv(output_dir / "val_process_ids.csv", index=False)
    test_df[["process_id"]].to_csv(output_dir / "test_process_ids.csv", index=False)

    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)

    manifest = {
        "source": {
            "data_dir": str(Path(args.data_dir)),
            "seed": args.seed,
            "generator": Path(__file__).name,
        },
        "dataset_info": info,
        "validation": validation,
    }
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    LOGGER.info("Wrote stratified workflow splits to %s", output_dir)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "splits"

    tables = load_tables(data_dir)
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

    write_split_outputs(output_dir, train_df, val_df, test_df, info, validation, args)


if __name__ == "__main__":
    main()
