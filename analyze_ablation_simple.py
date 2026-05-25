import json
import glob

def analyze():
    print("="*50)
    print("Ablation Group 3: GA-HPO IDDQN")
    files = glob.glob("results/ablation/ga_hpo_iddqn/seed_*/final_eval.json")
    for f in files:
        with open(f) as file:
            print(f, json.load(file)['makespan'])

    print("="*50)
    print("Ablation Group 1 & 2 are currently running in the background.")

if __name__ == "__main__":
    analyze()
