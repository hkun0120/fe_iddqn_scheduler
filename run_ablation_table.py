import subprocess
import os
import sys
import json
import signal
import concurrent.futures
from datetime import datetime


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _signal_name(returncode):
    if returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"SIG{-returncode}"


def _summarize_failure(output_dir, returncode):
    failure_info = _read_json(os.path.join(output_dir, 'failure_info.json'))
    run_status = _read_json(os.path.join(output_dir, 'run_status.json'))

    summary = {
        'returncode': returncode,
        'signal': _signal_name(returncode),
        'run_status': run_status,
        'failure_info': failure_info,
    }

    if failure_info:
        stage = failure_info.get('stage', 'unknown')
        err_type = failure_info.get('error_type', 'Error')
        err_msg = failure_info.get('error_message', '')
        summary['message'] = f"{err_type} at stage '{stage}': {err_msg}"
    elif run_status:
        stage = run_status.get('stage', 'unknown')
        status = run_status.get('status', 'unknown')
        if summary['signal']:
            summary['message'] = (
                f"Process terminated by {summary['signal']} while status='{status}' "
                f"at stage '{stage}'"
            )
        else:
            summary['message'] = f"Process exited with return code {returncode} at stage '{stage}'"
    elif summary['signal']:
        summary['message'] = f"Process terminated by {summary['signal']} before status file was written"
    else:
        summary['message'] = f"Process exited with return code {returncode} before writing failure metadata"

    return summary


def _is_completed_output(output_dir):
    return os.path.exists(os.path.join(output_dir, 'final_eval.json'))

def _env_truthy(name, default='0'):
    value = os.getenv(name, default).strip().lower()
    return value in {'1', 'true', 'yes', 'y', 'on'}

def run_ablation_experiments():
    python_bin = sys.executable
    # 按照老师要求使用20次独立随机种子实验
    seed_env = os.getenv('ABLATION_SEEDS', '').strip()
    if seed_env:
        seeds = [int(item) for item in seed_env.split(',') if item.strip()]
    else:
        seeds = list(range(42, 62))  # 42 到 61 共 20 个种子

    output_root = os.getenv('ABLATION_OUTPUT_ROOT', 'results/ablation')
    os.makedirs(output_root, exist_ok=True)
    
    experiments = [
        {
            'key': 'ga_hpo_dqn',
            'name': 'Group 1 (GA-HPO DQN)',
            'args': ['--mode', 'full', '--disable_fe', '--disable_per', '--disable_nstep', '--no_gnn'],
            'out_dir': os.path.join(output_root, 'ga_hpo_dqn', 'seed_{}')
        },
        {
            'key': 'ga_hpo_fe_dqn',
            'name': 'Group 2 (GA-HPO FE-DQN)',
            'args': ['--mode', 'full', '--disable_per', '--disable_nstep', '--no_gnn'],
            'out_dir': os.path.join(output_root, 'ga_hpo_fe_dqn', 'seed_{}')
        },
        {
            'key': 'fe_iddqn',
            'name': 'Group 3 (FE-IDDQN) (No GA-HPO)',
            'args': ['--mode', 'train_only'],
            'out_dir': os.path.join(output_root, 'fe_iddqn', 'seed_{}')
        },
        {
            'key': 'iddqn',
            'name': 'Group 4 (IDDQN) (No GA-HPO, No FE)',
            'args': ['--mode', 'train_only', '--disable_fe'],
            'out_dir': os.path.join(output_root, 'iddqn', 'seed_{}')
        },
        {
            'key': 'ga_hpo_iddqn',
            'name': 'Group 5 (GA-HPO FE-IDDQN - Full)',
            'args': ['--mode', 'full'],
            'out_dir': os.path.join(output_root, 'ga_hpo_iddqn', 'seed_{}')
        }
    ]
    group_env = os.getenv('ABLATION_GROUPS', '').strip()
    if group_env:
        selected_groups = {item.strip() for item in group_env.split(',') if item.strip()}
        valid_groups = {exp['key'] for exp in experiments}
        unknown_groups = sorted(selected_groups - valid_groups)
        if unknown_groups:
            raise ValueError(
                f"Unknown ABLATION_GROUPS values: {unknown_groups}. "
                f"Valid values: {sorted(valid_groups)}"
            )
        experiments = [exp for exp in experiments if exp['key'] in selected_groups]
        print(f"Selected ablation groups: {', '.join(exp['key'] for exp in experiments)}")
    
    default_workers = int(os.getenv('ABLATION_MAX_WORKERS', '1'))
    common_args = [
        '--eval_split', os.getenv('ABLATION_EVAL_SPLIT', 'val'),
        '--max_episodes', os.getenv('ABLATION_MAX_EPISODES', '80'),
        '--max_steps_per_episode', os.getenv('ABLATION_MAX_STEPS_PER_EPISODE', '500'),
        '--val_eval_interval', os.getenv('ABLATION_VAL_EVAL_INTERVAL', '0'),
        '--val_eval_episodes', os.getenv('ABLATION_VAL_EVAL_EPISODES', '2'),
        '--final_eval_episodes', os.getenv('ABLATION_FINAL_EVAL_EPISODES', '5'),
        '--full_eval_max_steps', os.getenv('ABLATION_FULL_EVAL_MAX_STEPS', '0'),
        '--paper_eval_episodes', os.getenv('ABLATION_PAPER_EVAL_EPISODES', '0'),
        '--ga_population_size', os.getenv('ABLATION_GA_POPULATION_SIZE', '4'),
        '--ga_generations', os.getenv('ABLATION_GA_GENERATIONS', '3'),
        '--ga_eval_episodes', os.getenv('ABLATION_GA_EVAL_EPISODES', '2'),
        '--ga_max_workers', os.getenv('ABLATION_GA_MAX_WORKERS', '1'),
        '--hpo_trials', os.getenv('ABLATION_HPO_TRIALS', '5'),
        '--hpo_timeout', os.getenv('ABLATION_HPO_TIMEOUT', '600'),
        '--hpo_eval_episodes', os.getenv('ABLATION_HPO_EVAL_EPISODES', '2'),
        '--search_max_processes', os.getenv('ABLATION_SEARCH_MAX_PROCESSES', '6'),
        '--search_max_tasks', os.getenv('ABLATION_SEARCH_MAX_TASKS', '80'),
        '--search_max_steps', os.getenv('ABLATION_SEARCH_MAX_STEPS', '80'),
        '--search_eval_episodes', os.getenv('ABLATION_SEARCH_EVAL_EPISODES', '2'),
        '--search_split', os.getenv('ABLATION_SEARCH_SPLIT', 'val'),
        '--search_strata_column', os.getenv('ABLATION_SEARCH_STRATA_COLUMN', 'balanced_workflow_stratum'),
        '--torch_num_threads', os.getenv('ABLATION_TORCH_NUM_THREADS', '4'),
        '--skip_baselines',
    ]
    if _env_truthy('ABLATION_FULL_TEST_EVAL'):
        common_args.append('--full_test_eval')

    # 构建所有需要执行的任务列表（5个组 x 20个种子 = 100个任务）
    tasks = []
    for exp in experiments:
        for seed in seeds:
            output_dir = exp['out_dir'].format(seed)
            if _is_completed_output(output_dir):
                print(f"⏭️  Skipping completed {exp['name']} Seed {seed}")
                continue
            cmd = [python_bin, "train_fe_iddqn_ga_hpo.py", "--seed", str(seed),
                   "--env_type", "replay", "--replay_data_dir", "data/raw_data"]
            cmd.extend(common_args)
            cmd.extend(exp['args'])
            cmd.extend(["--output_dir", output_dir])
            tasks.append({
                'name': exp['name'],
                'seed': seed,
                'cmd': cmd,
                'output_dir': output_dir,
            })

    max_parallel_workers = max(1, default_workers)
    print(f"Total pending experiments: {len(tasks)}. Running {max_parallel_workers} concurrently...")

    run_started_at = datetime.now().isoformat(timespec='seconds')
    summary = {
        'started_at': run_started_at,
        'python': python_bin,
        'output_root': output_root,
        'max_parallel_workers': max_parallel_workers,
        'completed': [],
        'failed': [],
    }

    def run_task(task_info):
        name = task_info['name']
        seed = task_info['seed']
        cmd = task_info['cmd']
        output_dir = task_info['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, 'process.log')
        _write_json(os.path.join(output_dir, 'command.json'), {
            'name': name,
            'seed': seed,
            'cmd': cmd,
            'started_at': datetime.now().isoformat(timespec='seconds'),
        })
        print(f"Starting {name} Seed {seed}...")
        try:
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            with open(log_path, 'w', encoding='utf-8') as log_file:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            if not _is_completed_output(output_dir):
                failure_summary = _summarize_failure(output_dir, 0)
                failure_summary['message'] = (
                    "Process exited with return code 0 but final_eval.json is missing. "
                    f"{failure_summary['message']}"
                )
                print(f"❌ Incomplete {name} Seed {seed}: {failure_summary['message']}")
                return {
                    'name': name,
                    'seed': seed,
                    'output_dir': output_dir,
                    'log_path': log_path,
                    'status': 'failed',
                    **failure_summary,
                }
            print(f"✅ Finished {name} Seed {seed}")
            return {
                'name': name,
                'seed': seed,
                'output_dir': output_dir,
                'log_path': log_path,
                'status': 'completed',
            }
        except subprocess.CalledProcessError as exc:
            failure_summary = _summarize_failure(output_dir, exc.returncode)
            print(f"❌ Failed {name} Seed {seed}: {failure_summary['message']}")
            return {
                'name': name,
                'seed': seed,
                'output_dir': output_dir,
                'log_path': log_path,
                'status': 'failed',
                **failure_summary,
            }

    # 使用线程池调度多个独立子进程；默认串行，避免 replay 数据加载阶段内存压力过大。
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_workers) as executor:
        for result in executor.map(run_task, tasks):
            if result is None:
                continue
            if result['status'] == 'completed':
                summary['completed'].append(result)
            else:
                summary['failed'].append(result)

    summary['finished_at'] = datetime.now().isoformat(timespec='seconds')
    summary['completed_count'] = len(summary['completed'])
    summary['failed_count'] = len(summary['failed'])
    summary_path = os.path.join(output_root, 'ablation_run_summary.json')
    _write_json(summary_path, summary)

    if summary['failed']:
        print(f"Ablation run finished with failures: {summary['completed_count']} completed, {summary['failed_count']} failed.")
        print(f"See {summary_path} for details.")
        sys.exit(1)
    else:
        print(f"All pending ablation tasks completed successfully ({summary['completed_count']} total).")

if __name__ == "__main__":
    run_ablation_experiments()
