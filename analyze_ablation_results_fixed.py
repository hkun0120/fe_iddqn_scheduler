import json
import glob
import os
import re
import numpy as np


def extract_makespan_from_logs(log_paths):
    if not log_paths:
        return None
    pat = re.compile(r"Makespan:\s*([0-9]+(?:\.[0-9]+)?)")
    # scan logs from the end to find final eval lines faster
    for lp in log_paths:
        try:
            with open(lp, "r") as f:
                lines = f.readlines()
            for line in reversed(lines):
                m = pat.search(line)
                if m:
                    return float(m.group(1))
        except Exception:
            continue
    return None


def get_seed_makespan(seed_dir):
    # 1) try final_eval.json
    final_path = os.path.join(seed_dir, "final_eval.json")
    if os.path.exists(final_path):
        try:
            with open(final_path, "r") as f:
                data = json.load(f)
            return data.get("makespan") or data.get("mean_makespan")
        except Exception:
            pass

    # 2) try training_log.json
    train_path = os.path.join(seed_dir, "training_log.json")
    if os.path.exists(train_path):
        try:
            with open(train_path, "r") as f:
                data = json.load(f)
            # common keys
            for k in ("makespan", "mean_makespan", "val_makespan"):
                if k in data:
                    return data[k]
            # if it's a list of records, take last record's makespan
            if isinstance(data, list) and data:
                last = data[-1]
                for k in ("makespan", "val_makespan", "mean_makespan"):
                    if k in last:
                        return last[k]
        except Exception:
            pass

    # 3) scan .log files under seed_dir/logs and seed_dir
    log_files = sorted(glob.glob(os.path.join(seed_dir, "logs", "*.log")))
    log_files += sorted(glob.glob(os.path.join(seed_dir, "*.log")))
    val = extract_makespan_from_logs(log_files)
    return val


def analyze_all(root="results/ablation"):
    if not os.path.exists(root):
        print(f"No ablation results directory: {root}")
        return

    groups = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if not groups:
        print(f"No groups found under {root}")
        return

    print("=" * 80)
    print("Ablation Study Results - aggregate across available seeds")
    print("=" * 80)

    for group in groups:
        group_path = os.path.join(root, group)
        seed_dirs = sorted([os.path.join(group_path, d) for d in os.listdir(group_path)
                            if os.path.isdir(os.path.join(group_path, d)) and d.startswith("seed_")])
        makespans = []
        missing = 0
        for sd in seed_dirs:
            val = get_seed_makespan(sd)
            if val is None:
                missing += 1
            else:
                try:
                    makespans.append(float(val))
                except Exception:
                    missing += 1

        n = len(makespans)
        if n > 0:
            mean_val = np.mean(makespans)
            std_val = np.std(makespans, ddof=1) if n > 1 else 0.0
            ci95 = 1.96 * (std_val / np.sqrt(n)) if n > 1 else 0.0
            print(f"{group:<30s}: Mean {mean_val:.2f} | Std {std_val:.2f} | 95% CI +/-{ci95:.2f} (n={n}) | missing={missing}")
        else:
            print(f"{group:<30s}: No extracted makespan values (missing={missing})")


if __name__ == "__main__":
    analyze_all()
