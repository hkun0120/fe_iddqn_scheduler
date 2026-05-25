#!/usr/bin/env python3
"""分析对比实验结果"""
import json

with open('comparison_results/summary_20260227_222512.json') as f:
    data = json.load(f)

print('=' * 110)
print(f"{'工作流':<42s} {'任务':>4s} {'SJF':>8s} {'CPOP':>8s} {'PPO':>8s} {'IDDQN':>8s} {'PPO%':>7s} {'IDDQN%':>7s}")
print('-' * 110)

sjf_res = data['all_results'][2]
cpop_res = data['all_results'][4]
ppo_res = data['all_results'][5]
iddqn_res = data['all_results'][6]

ppo_wins_sjf = 0
iddqn_wins_sjf = 0
iddqn_wins_cpop = 0
ppo_wins_cpop = 0
iddqn_wins_rr = 0

for i in range(len(sjf_res['test_results'])):
    sjf = sjf_res['test_results'][i]
    cpop = cpop_res['test_results'][i]
    ppo = ppo_res['test_results'][i]
    iddqn = iddqn_res['test_results'][i]
    rr = data['all_results'][1]['test_results'][i]

    ppo_pct = (ppo['makespan'] - sjf['makespan']) / sjf['makespan'] * 100
    iddqn_pct = (iddqn['makespan'] - sjf['makespan']) / sjf['makespan'] * 100

    mark_p = ' '
    mark_i = ' '
    if ppo['makespan'] <= sjf['makespan']:
        ppo_wins_sjf += 1
        mark_p = '*'
    if iddqn['makespan'] <= sjf['makespan']:
        iddqn_wins_sjf += 1
        mark_i = '*'
    if ppo['makespan'] <= cpop['makespan']:
        ppo_wins_cpop += 1
    if iddqn['makespan'] <= cpop['makespan']:
        iddqn_wins_cpop += 1
    if iddqn['makespan'] <= rr['makespan']:
        iddqn_wins_rr += 1

    name = sjf['name'][:42]
    print(f"{name:<42s} {sjf['num_tasks']:>4d} {sjf['makespan']:>8.1f} {cpop['makespan']:>8.1f} "
          f"{ppo['makespan']:>8.1f}{mark_p}{iddqn['makespan']:>8.1f}{mark_i} {ppo_pct:>+6.1f}% {iddqn_pct:>+6.1f}%")

print('=' * 110)
print(f"\nPPO  wins vs SJF:  {ppo_wins_sjf}/20 workflows")
print(f"IDDQN wins vs SJF:  {iddqn_wins_sjf}/20 workflows")
print(f"PPO  wins vs CPOP: {ppo_wins_cpop}/20 workflows")
print(f"IDDQN wins vs CPOP: {iddqn_wins_cpop}/20 workflows")
print(f"IDDQN wins vs RR:   {iddqn_wins_rr}/20 workflows")

print("\n" + "=" * 80)
print("                          总体对比结果")
print("=" * 80)
print(f"{'算法':<20s} {'Makespan':>10s} {'Std':>10s} {'利用率':>8s} {'负载均衡':>8s} {'训练':>8s}")
print("-" * 80)
for res in data['summary']:
    print(f"{res['algorithm']:<20s} {res['avg_makespan']:>10.2f} {res['std_makespan']:>10.2f} "
          f"{res['avg_utilization']:>8.4f} {res['avg_load_balance']:>8.4f} {res['train_time']:>7.1f}s")
print("=" * 80)

# vs 上一轮对比
print("\n" + "=" * 80)
print("         改进效果 (v1: 1149维/200ep → v2: 128维/1000ep+BC)")
print("=" * 80)
v1_ppo = 1838.17
v1_iddqn = 1604.14
v2_ppo = data['summary'][5]['avg_makespan']
v2_iddqn = data['summary'][6]['avg_makespan']
print(f"GDS-PPO:        {v1_ppo:.2f} → {v2_ppo:.2f}  ({(v1_ppo - v2_ppo)/v1_ppo*100:+.1f}% makespan 降低)")
print(f"GA-HPO FE-IDDQN: {v1_iddqn:.2f} → {v2_iddqn:.2f}  ({(v1_iddqn - v2_iddqn)/v1_iddqn*100:+.1f}% makespan 降低)")
print(f"SJF baseline:    {1044.50:.2f}")
print(f"PPO vs SJF gap:  {(v2_ppo - 1044.5)/1044.5*100:.1f}%  (v1: {(v1_ppo - 1044.5)/1044.5*100:.1f}%)")
print(f"IDDQN vs SJF gap: {(v2_iddqn - 1044.5)/1044.5*100:.1f}%  (v1: {(v1_iddqn - 1044.5)/1044.5*100:.1f}%)")
