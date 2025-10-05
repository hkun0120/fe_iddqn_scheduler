#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

sys.path.append(str(Path(__file__).parent))

from create_gantt_chart_293719 import GanttChartGenerator
from baselines.traditional_schedulers import HEFTScheduler
from models.fe_iddqn import FE_IDDQN


def plot_gantt(schedule_result, tasks, resources, out_path, title):
    task_start_times = schedule_result.get('task_start_times', {})
    task_end_times = schedule_result.get('task_end_times', {})
    task_assignments = schedule_result.get('task_assignments', {})
    if not task_start_times or not task_end_times:
        return None

    gantt_data = []
    for t in tasks:
        tid = int(t['id'])
        if tid in task_start_times:
            s = float(task_start_times[tid])
            e = float(task_end_times[tid])
            rid = int(task_assignments.get(tid, 0))
            gantt_data.append({
                'task_id': tid,
                'start_time': s,
                'end_time': e,
                'duration': e - s,
                'resource_id': rid,
                'task_name': str(t.get('name', tid))[:15]
            })
    if not gantt_data:
        return None
    gantt_data.sort(key=lambda x: x['start_time'])

    fig, ax = plt.subplots(figsize=(16, 9))
    colors = plt.cm.Set3(np.linspace(0, 1, len(resources)))
    resource_colors = {i: colors[i] for i in range(len(resources))}
    y_positions = {}
    y = 0
    for item in gantt_data:
        rid = item['resource_id']
        if rid not in y_positions:
            y_positions[rid] = y
            y += 1
        ypos = y_positions[rid]
        rect = patches.Rectangle((item['start_time'], ypos - 0.4), item['duration'], 0.8,
                                 linewidth=1, edgecolor='black', facecolor=resource_colors[rid], alpha=0.7)
        ax.add_patch(rect)
        ax.text(item['start_time'] + item['duration']/2.0, ypos, f"{item['task_id']}",
                ha='center', va='center', fontsize=6, fontweight='bold')

    max_end = max(d['end_time'] for d in gantt_data)
    ax.set_xlim(0, max_end * 1.05)
    ax.set_ylim(-0.5, len(y_positions) - 0.5)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f"R{rid}" for rid in sorted(y_positions.keys())])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('resource')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    legend_elems = [patches.Patch(color=resource_colors[i], label=f'R{i}') for i in range(len(resources))]
    ax.legend(handles=legend_elems, loc='upper right')

    # stats box: makespan and utilization (align with previous charts)
    makespan = max_end
    total_work = sum(float(t.get('duration', 0.0)) for t in tasks)
    total_capacity = makespan * len(resources) if makespan > 0 else 1.0
    resource_utilization = total_work / total_capacity if total_capacity > 0 else 0.0
    stats_text = f"Makespan: {makespan:.1f}秒\n资源利用率: {resource_utilization:.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


def run_heft(tasks, resources, deps):
    dep_list = [(int(d['pre_task']), int(d['post_task'])) for d in deps]
    scheduler = HEFTScheduler()
    return scheduler.schedule(tasks, resources, dep_list)


def run_fe(tasks, resources, deps, generator):
    import numpy as _np
    dep_list = [(int(d['pre_task']), int(d['post_task'])) for d in deps]
    incoming = {int(t['id']): 0 for t in tasks}
    for pre, post in dep_list:
        if post in incoming:
            incoming[post] += 1

    # HEFT upward rank
    try:
        import networkx as _nx
        heft = HEFTScheduler()
        _dag = _nx.DiGraph()
        _task_dict = {int(t['id']): t for t in tasks}
        for t in tasks:
            _dag.add_node(int(t['id']))
        for _pre, _post in dep_list:
            _dag.add_edge(int(_pre), int(_post))
        upward_rank = heft._calculate_upward_rank(_dag, _task_dict, resources)
    except Exception:
        upward_rank = {int(t['id']): float(t.get('duration', 0.0)) for t in tasks}

    # predictor + host speeds
    duration_predictor = None
    host_speed_map = {}
    try:
        from sklearn.linear_model import LinearRegression
        ti_all = generator.data['task_instance']
        ti_all = ti_all[(ti_all['state'] == 7) & pd.notna(ti_all['start_time']) & pd.notna(ti_all['end_time'])].copy()
        if not ti_all.empty:
            ti_all['duration_hist'] = (pd.to_datetime(ti_all['end_time']) - pd.to_datetime(ti_all['start_time'])).dt.total_seconds()
            host_mean = ti_all.groupby('host')['duration_hist'].mean()
            gmean = float(ti_all['duration_hist'].mean()) if not pd.isna(ti_all['duration_hist'].mean()) else 1.0
            if gmean <= 0:
                gmean = 1.0
            for h, m in host_mean.items():
                if isinstance(h, str):
                    host_speed_map[h] = max(0.5, min(2.0, float(m / gmean)))
            X, y = [], []
            def _encode_task_type(tt: str):
                kinds = ['SHELL', 'SQL', 'PYTHON', 'PROCEDURE', 'DATAX', 'SPARK', 'MR']
                onehot = [0]*len(kinds)
                if isinstance(tt, str):
                    up = tt.upper()
                    for i, k in enumerate(kinds):
                        if k in up:
                            onehot[i] = 1
                            break
                else:
                    onehot[0] = 1
                return onehot
            for _, r in ti_all.iterrows():
                tt_onehot = _encode_task_type(str(r.get('task_type', 'SHELL')))
                prio = float(r.get('task_instance_priority', 2) or 2)
                retry = float(r.get('retry_times', 0) or 0)
                h = r.get('host')
                hspd = float(host_speed_map.get(h, 1.0))
                X.append(tt_onehot + [prio, retry, hspd, 1.0])
                y.append(float(r['duration_hist']))
            if len(X) >= 20:
                duration_predictor = LinearRegression()
                duration_predictor.fit(np.array(X), np.array(y))
    except Exception:
        duration_predictor = None

    # strict historical duration toggle
    STRICT_HIST = int(os.getenv('STRICT_HIST_DUR', '1')) == 1
    if STRICT_HIST:
        duration_predictor = None
        host_speed_map = {}

    # load FE model
    model_path = 'fe_iddqn_training_system/fe_iddqn_training_system/models/fe_iddqn_epoch_9_20251002_165447.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')
    agent = FE_IDDQN(task_input_dim=16, resource_input_dim=7, action_dim=len(resources), device=torch.device('cpu'))
    try:
        with open(model_path, 'rb') as f:
            ckpt = pickle.load(f)
    except Exception:
        ckpt = torch.load(model_path, map_location='cpu')
    def _try_keys(d, keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                return d[k]
        return None
    q_state = _try_keys(ckpt, ['q_network_state_dict', 'model_state_dict', 'q_net_state'])
    t_state = _try_keys(ckpt, ['target_network_state_dict', 'model_state_dict', 'tgt_net_state'])
    if q_state is None and isinstance(ckpt, dict):
        q_state = ckpt
        t_state = ckpt
    agent.q_network.load_state_dict(q_state)
    agent.target_network.load_state_dict(t_state if t_state is not None else q_state)
    agent.q_network.eval()

    resource_end_times = [0.0] * len(resources)
    current_time = 0.0
    task_start_times, task_end_times, task_assignment = {}, {}, {}
    processed = set()

    def encode_task_type(tt: str):
        kinds = ['SHELL', 'SQL', 'PYTHON', 'PROCEDURE', 'DATAX', 'SPARK', 'MR']
        onehot = [0]*len(kinds)
        if isinstance(tt, str):
            up = tt.upper()
            for i, k in enumerate(kinds):
                if k in up:
                    onehot[i] = 1
                    break
        else:
            onehot[0] = 1
        return onehot

    while len(processed) < len(tasks):
        ready = []
        for t in tasks:
            tid = int(t['id'])
            if tid in processed:
                continue
            ok = True
            max_dep_end = 0.0
            for pre, post in dep_list:
                if post == tid:
                    if pre not in task_end_times:
                        ok = False
                        break
                    max_dep_end = max(max_dep_end, task_end_times[pre])
            if ok:
                ready.append((tid, max_dep_end))
        if not ready:
            next_t = min(resource_end_times) if resource_end_times else current_time
            current_time = max(current_time, next_t)
            continue

        task_feats = []
        for t in tasks:
            tid = int(t['id'])
            ttype = encode_task_type(t.get('task_type', 'SHELL'))
            duration = float(t.get('duration', 0.0))
            priority = int(t.get('priority', t.get('task_instance_priority', 2)) or 2)
            retry = int(t.get('retry_times', 0) or 0)
            dep_cnt = int(incoming.get(tid, 0))
            completed_flag = 1 if tid in processed else 0
            is_ready = 1
            for pre, post in dep_list:
                if post == tid and pre not in task_end_times:
                    is_ready = 0
                    break
            other = [float(t.get('cpu_req', 1.0)), float(t.get('memory_req', 1.0)), duration,
                     float(priority), float(retry), float(t.get('complexity_score', 1.0)),
                     float(dep_cnt), float(completed_flag), float(is_ready)]
            feats = (ttype + other)[:16]
            task_feats.append(feats)

        res_feats = []
        for rid, r in enumerate(resources):
            avail_time = float(resource_end_times[rid])
            util = avail_time / (current_time + 1.0)
            cpu_cap = float(r.get('cpu_capacity', 2.0))
            mem_cap = float(r.get('memory_capacity', 4.0))
            res_feats.append([cpu_cap, mem_cap, avail_time, util, cpu_cap, mem_cap, float(current_time)])

        t_tensor = torch.FloatTensor(np.array(task_feats, dtype=np.float32)).unsqueeze(0)
        r_tensor = torch.FloatTensor(np.array(res_feats, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.q_network(t_tensor, r_tensor).squeeze(0).cpu().numpy()

        EF_COEF = float(os.getenv('EF_COEF', '1e-4'))
        HEFT_COEF = float(os.getenv('HEFT_COEF', '1e-2'))
        BASE_DISP = float(os.getenv('BASE_DISP', '1e-6'))
        DISP_SCALE = float(os.getenv('DISP_SCALE', '0.05'))
        ALPHA = float(os.getenv('PRED_ALPHA', '0.7'))
        load_std = float(np.std(resource_end_times)) if resource_end_times else 0.0
        DISP_COEF = BASE_DISP * (1.0 + DISP_SCALE * load_std)

        best_tuple = None
        for tid, dep_end in ready:
            scores = q_values.copy()
            tinfo = next(t for t in tasks if int(t['id']) == tid)
            need_cpu = float(tinfo.get('cpu_req', 1.0))
            need_mem = float(tinfo.get('memory_req', 1.0))
            for rrid, r in enumerate(resources):
                cpu_cap = float(r.get('cpu_capacity', 2.0))
                mem_cap = float(r.get('memory_capacity', 4.0))
                if cpu_cap < need_cpu or mem_cap < need_mem:
                    scores[rrid] = -1e12
                    continue
                est_start = max(float(resource_end_times[rrid]), float(dep_end))
                if (not STRICT_HIST) and (duration_predictor is not None):
                    kinds = ['SHELL', 'SQL', 'PYTHON', 'PROCEDURE', 'DATAX', 'SPARK', 'MR']
                    onehot = [0]*len(kinds)
                    ttype = str(tinfo.get('task_type', 'SHELL')).upper()
                    for i, k in enumerate(kinds):
                        if k in ttype:
                            onehot[i] = 1
                            break
                    prio = float(tinfo.get('priority', tinfo.get('task_instance_priority', 2)) or 2)
                    retry = float(tinfo.get('retry_times', 0) or 0)
                    host_name = r.get('host')
                    hspd = float(host_speed_map.get(host_name, r.get('speed_factor', 1.0))) if not STRICT_HIST else 1.0
                    feat = np.array(onehot + [prio, retry, hspd, 1.0]).reshape(1, -1)
                    try:
                        pred = float(duration_predictor.predict(feat)[0])
                    except Exception:
                        pred = float(tinfo['duration']) * (1.0 if STRICT_HIST else float(resources[rrid].get('speed_factor', 1.0)))
                    base_dur = float(tinfo['duration']) * (1.0 if STRICT_HIST else float(resources[rrid].get('speed_factor', 1.0)))
                    dur_adj = max(0.1, ALPHA * pred + (1.0 - ALPHA) * base_dur)
                else:
                    base = float(tinfo['duration'])
                    dur_adj = base if STRICT_HIST else base * float(resources[rrid].get('speed_factor', 1.0))
                est_end = est_start + dur_adj
                scores[rrid] -= EF_COEF * est_end
                scores[rrid] -= DISP_COEF * float(resource_end_times[rrid])
            rid_cand = int(np.argmax(scores))
            cand_start = max(float(resource_end_times[rid_cand]), float(dep_end))
            if (not STRICT_HIST) and (duration_predictor is not None):
                kinds = ['SHELL', 'SQL', 'PYTHON', 'PROCEDURE', 'DATAX', 'SPARK', 'MR']
                onehot = [0]*len(kinds)
                ttype = str(tinfo.get('task_type', 'SHELL')).upper()
                for i, k in enumerate(kinds):
                    if k in ttype:
                        onehot[i] = 1
                        break
                prio = float(tinfo.get('priority', tinfo.get('task_instance_priority', 2)) or 2)
                retry = float(tinfo.get('retry_times', 0) or 0)
                host_name = resources[rid_cand].get('host') if not STRICT_HIST else None
                hspd = float(host_speed_map.get(host_name, resources[rid_cand].get('speed_factor', 1.0))) if not STRICT_HIST else 1.0
                feat = np.array(onehot + [prio, retry, hspd, 1.0]).reshape(1, -1)
                try:
                    pred_cand = float(duration_predictor.predict(feat)[0])
                except Exception:
                    pred_cand = float(tinfo['duration']) * (1.0 if STRICT_HIST else float(resources[rid_cand].get('speed_factor', 1.0)))
                base_cand = float(tinfo['duration']) * (1.0 if STRICT_HIST else float(resources[rid_cand].get('speed_factor', 1.0)))
                cand_dur = max(0.1, ALPHA * pred_cand + (1.0 - ALPHA) * base_cand)
            else:
                base = float(tinfo['duration'])
                cand_dur = base if STRICT_HIST else base * float(resources[rid_cand].get('speed_factor', 1.0))
            cand_end = cand_start + cand_dur
            hrank = float(upward_rank.get(int(tid), 0.0))
            task_metric = cand_end - HEFT_COEF * hrank
            if (best_tuple is None) or (task_metric < best_tuple[0]):
                best_tuple = (task_metric, int(tid), float(dep_end), rid_cand, cand_start, cand_end)

        _, chosen_tid, _, rid, start, end = best_tuple
        task_start_times[chosen_tid] = start
        task_end_times[chosen_tid] = end
        task_assignment[chosen_tid] = rid
        resource_end_times[rid] = end
        processed.add(chosen_tid)

    makespan = max(resource_end_times) if resource_end_times else 0.0
    total_work = sum(float(t['duration']) for t in tasks)
    util = total_work / (makespan * len(resources)) if makespan > 0 else 0.0
    return {
        'task_assignments': task_assignment,
        'task_start_times': task_start_times,
        'task_end_times': task_end_times,
        'makespan': makespan,
        'resource_utilization': util,
        'algorithm': 'FE_IDDQN'
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pids', nargs='+', type=int, help='process ids')
    parser.add_argument('--out', type=str, default='showcase_outputs', help='output folder')
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    algorithms = ["SJF", "GA", "PSO", "HEFT", "FE_IDDQN"]
    all_rows = []

    gen = GanttChartGenerator()
    for pid in args.pids:
        pid_dir = out_root / str(pid)
        pid_dir.mkdir(parents=True, exist_ok=True)
        tasks, resources, deps = gen.build_tasks_and_resources(int(pid))
        if not tasks:
            print(f"{pid}: no tasks, skip")
            continue

        # non-FE algorithms via generator
        for alg in ["SJF", "GA", "PSO", "HEFT"]:
            try:
                result = gen.run_scheduler(alg, tasks, resources, deps)
                if result and 'task_end_times' in result:
                    out_img = pid_dir / f"gantt_chart_{pid}_{alg.lower()}.png"
                    plot_gantt(result, tasks, resources, str(out_img), f"Process {pid} - {alg}")
                    mk = result.get('makespan')
                    if mk is None:
                        mk = max(result['task_end_times'].values()) if result['task_end_times'] else 0.0
                    total_work = sum(float(t['duration']) for t in tasks)
                    util = total_work / (mk * len(resources)) if mk > 0 else 0.0
                    all_rows.append({'process_id': pid, 'algorithm': alg, 'makespan': mk, 'utilization': util})
            except Exception as e:
                print(f"{pid} {alg} failed: {e}")

        # FE
        try:
            fe_res = run_fe(tasks, resources, deps, gen)
            if fe_res and 'task_end_times' in fe_res:
                out_img = pid_dir / f"gantt_chart_{pid}_fe_iddqn.png"
                plot_gantt(fe_res, tasks, resources, str(out_img), f"Process {pid} - FE_IDDQN")
                all_rows.append({'process_id': pid, 'algorithm': 'FE_IDDQN', 'makespan': fe_res.get('makespan'), 'utilization': fe_res.get('resource_utilization')})
        except Exception as e:
            print(f"{pid} FE_IDDQN failed: {e}")

        # per-pid metrics file
        pid_rows = [r for r in all_rows if r['process_id'] == pid]
        if pid_rows:
            pd.DataFrame(pid_rows).to_csv(pid_dir / f"metrics_{pid}.csv", index=False)

    # combined metrics
    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_root / 'metrics_comparison.csv', index=False)
        print(f"Saved {out_root / 'metrics_comparison.csv'}")


if __name__ == '__main__':
    main()


