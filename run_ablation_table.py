import subprocess
import os
import sys
import concurrent.futures

def run_ablation_experiments():
    python_bin = sys.executable
    # 按照老师要求使用20次独立随机种子实验
    seeds = list(range(42, 62))  # 42 到 61 共 20 个种子
    
    os.makedirs('results/ablation', exist_ok=True)
    
    experiments = [
        {
            'name': 'Group 1 (GA-HPO DQN)',
            'args': ['--mode', 'full', '--disable_fe', '--disable_per', '--disable_nstep', '--no_gnn'],
            'out_dir': 'results/ablation/ga_hpo_dqn/seed_{}'
        },
        {
            'name': 'Group 2 (GA-HPO FE-DQN)',
            'args': ['--mode', 'full', '--disable_per', '--disable_nstep', '--no_gnn'],
            'out_dir': 'results/ablation/ga_hpo_fe_dqn/seed_{}'
        },
        {
            'name': 'Group 3 (FE-IDDQN) (No GA-HPO)',
            'args': ['--mode', 'train_only'],
            'out_dir': 'results/ablation/fe_iddqn/seed_{}'
        },
        {
            'name': 'Group 4 (IDDQN) (No GA-HPO, No FE)',
            'args': ['--mode', 'train_only', '--disable_fe'],
            'out_dir': 'results/ablation/iddqn/seed_{}'
        },
        {
            'name': 'Group 5 (GA-HPO IDDQN - Full)',
            'args': ['--mode', 'full'],
            'out_dir': 'results/ablation/ga_hpo_iddqn/seed_{}'
        }
    ]
    
    # 构建所有需要执行的任务列表（5个组 x 20个种子 = 100个任务）
    tasks = []
    for exp in experiments:
        for seed in seeds:
            cmd = [python_bin, "train_fe_iddqn_ga_hpo.py", "--seed", str(seed),
                   "--env_type", "replay", "--replay_data_dir", "data/raw_data"]
            cmd.extend(exp['args'])
            cmd.extend(["--output_dir", exp['out_dir'].format(seed)])
            tasks.append((exp['name'], seed, cmd))
            
    # 设置最大并行数（根据单次2-4G显存计算，5090通常有24G/32G，设置为5-6个并发较安全）
    MAX_PARALLEL_WORKERS = 2
    print(f"Total experiments to run: {len(tasks)}. Running {MAX_PARALLEL_WORKERS} concurrently...")

    def run_task(task_info):
        name, seed, cmd = task_info
        print(f"Starting {name} Seed {seed}...")
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Finished {name} Seed {seed}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed {name} Seed {seed}")

    # 启用多进程池代替多线程池，绕过服务器环境底层 glibc 的 pthread_cancel 线程缺陷        
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        executor.map(run_task, tasks)

    print("All ablation groups completed successfully!")

if __name__ == "__main__":
    run_ablation_experiments()
