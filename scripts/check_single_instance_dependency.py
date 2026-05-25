#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_fe_iddqn_ga_hpo import load_replay_dataframes


def main():
    parser = argparse.ArgumentParser(description="Check dependency feasibility for single-instance schedules")
    parser.add_argument("--process_id", type=int, default=293712)
    parser.add_argument(
        "--compare_json",
        type=str,
        default="results/real_data_validation/full_smoke_whalesb/single_instance_gantt/single_compare_process_293712.json",
    )
    parser.add_argument("--replay_data_dir", type=str, default="data/raw_data")
    args = parser.parse_args()

    compare_obj = json.loads(Path(args.compare_json).read_text(encoding="utf-8"))
    frames = load_replay_dataframes(Path(args.replay_data_dir))

    proc = frames["process_instance"]
    rel = frames["process_task_relation"]
    tasks = frames["task_instance"]

    a = proc[proc["id"] == args.process_id]
    if a.empty:
        raise SystemExit("process not found")

    if "process_definition_code" in a.columns:
        pcode = a.iloc[0]["process_definition_code"]
    elif "process_definition_id" in a.columns:
        pcode = a.iloc[0]["process_definition_id"]
    else:
        raise SystemExit("no process_definition_code/process_definition_id in process_instance")

    if "process_definition_code" not in rel.columns and "process_definition_id" in rel.columns:
        rel = rel.copy()
        rel["process_definition_code"] = rel["process_definition_id"]

    relp = rel[rel["process_definition_code"] == pcode]

    subset = tasks[tasks["process_instance_id"] == args.process_id].copy()
    if "task_code" not in subset.columns and "task_definition_code" in subset.columns:
        subset["task_code"] = subset["task_definition_code"]

    code_to_id = {}
    for _, r in subset.iterrows():
        code = r.get("task_code")
        if str(code) == "nan":
            continue
        code_to_id[str(code)] = int(r["id"])

    edges = []
    for _, r in relp.iterrows():
        pre = r.get("pre_task_code")
        post = r.get("post_task_code")
        if str(pre) == "nan" or str(post) == "nan":
            continue
        u = code_to_id.get(str(pre))
        v = code_to_id.get(str(post))
        if u is not None and v is not None:
            edges.append((u, v))

    print(f"process_id={args.process_id}, dependency_edges={len(edges)}")

    for algo, info in compare_obj.get("results", {}).items():
        sched = info.get("schedule", [])
        if not sched:
            print(f"{algo}: no schedule found")
            continue

        start = {int(x["task_id"]): float(x.get("start_time", x.get("timestamp", 0.0))) for x in sched}
        finish = {int(x["task_id"]): float(x.get("finish_time", x.get("end_time", x.get("start_time", 0.0)))) for x in sched}

        violations = []
        missing = 0
        for u, v in edges:
            if u not in finish or v not in start:
                missing += 1
                continue
            if start[v] + 1e-9 < finish[u]:
                violations.append((u, v, finish[u], start[v]))

        print(f"{algo}: violations={len(violations)}, missing_edges={missing}")
        if violations:
            print("  sample_violations:")
            for item in violations[:3]:
                print(f"    pre={item[0]} post={item[1]} pre_end={item[2]:.2f} post_start={item[3]:.2f}")


if __name__ == "__main__":
    main()
