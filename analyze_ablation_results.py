import json
import glob
import numpy as np

def print_result(group_name, path_pattern):
    files = glob.glob(path_pattern)
    makespans = []
    
    for file in files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                val = data.get("makespan", data.get("mean_makespan"))
                if val is not None:
                    makespans.append(val)
        except Exception: pass
    
    if makespans:
        mean_val = np.mean(makespans)
        std_val = np.std(makespans, ddof=1) # 样本标准差
        n = len(makespans)
        # 计算 95% 置信区间 (95% CI) = 1.96 * (std / sqrt(n))
        ci95 = 1.96 * (std_val / np.sqrt(n)) if n > 1 else 0.0
        print(f"{group_name:<30s}: Mean {mean_val:.2f} | Std {std_val:.2f} | 95% CI +/-{ci95:.2f} (n={n})")
    else:
        print(f"{group_name:<30s}: Pending / No Data")

def analyze_all():
    print("="*80)
    print("Ablation Study Results - 20 Seeds Avera    print("Ablation Study Results - 20 Seerint_result("1. IDDQN (Bas    print("Ab/ablation/iddqn/seed_*/final_eval.json")
    print_result("2. FE-IDDQN (No GA)", "results/ablation/fe_iddqn/seed_*/final_eval.json")
    print_result("3. GA-HPO DQN", "results/ablation/ga_hpo_dqn/seed_*/final_eval.json")
    print_result("4. GA-HPO FE-DQN", "results/ablation/ga_hpo_fe_dqn/se    print_result("4. GA-HPO FE-DQNes    print_result("4. GA-HPO FE-DQ"results/    print_result("4. GA-HPO Ffinal_e    print_resultri    print_result("4. GA-HPO FE-Din__":
    analyze_all()
