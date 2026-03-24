#!/usr/bin/env python3
"""验证数据库中的依赖关系是否与可视化一致"""
import pymysql
import pandas as pd

conn = pymysql.connect(host='localhost', user='root', password='',
                       database='whalesb', port=3306, charset='utf8mb4')

# 取实验中用到的候选工作流
q = """
SELECT pi.id, pi.name, pi.process_definition_code, COUNT(ti.id) AS task_count
FROM t_ds_process_instance pi
JOIN t_ds_task_instance ti ON ti.process_instance_id = pi.id
WHERE pi.state = 7 AND ti.state = 7
GROUP BY pi.id
HAVING task_count BETWEEN 10 AND 60
ORDER BY RAND(42)
LIMIT 5
"""
procs = pd.read_sql(q, conn)
print("=== 候选工作流 ===")
print(procs[['id','name','process_definition_code','task_count']].to_string())

# 查看前2个工作流的详细依赖
for row_i in range(min(2, len(procs))):
    pid = int(procs.iloc[row_i]['id'])
    def_code = int(procs.iloc[row_i]['process_definition_code'])
    name = procs.iloc[row_i]['name']
    print(f"\n{'='*80}")
    print(f"PID={pid}, def_code={def_code}")
    print(f"Name: {name}")

    # 任务列表
    tasks = pd.read_sql(f"""
        SELECT id, name, task_code, start_time, end_time
        FROM t_ds_task_instance
        WHERE process_instance_id = {pid} AND state = 7
        ORDER BY start_time
    """, conn)
    print(f"\n任务数: {len(tasks)}")
    task_codes = set()
    for _, t in tasks.iterrows():
        task_codes.add(int(t['task_code']))
        print(f"  [{t['task_code']}] {t['name']}")

    # 全部关系
    deps_all = pd.read_sql(f"""
        SELECT pre_task_code, post_task_code
        FROM t_ds_process_task_relation
        WHERE process_definition_code = {def_code}
    """, conn)
    print(f"\n关系表记录总数: {len(deps_all)}")

    # pre_task_code=0 的记录 (根节点标记)
    roots = deps_all[deps_all['pre_task_code'] == 0]
    print(f"  pre_task_code=0 的记录(根节点): {len(roots)}")
    for _, r in roots.iterrows():
        print(f"    0 -> {r['post_task_code']}")

    # 真实依赖
    deps_real = deps_all[deps_all['pre_task_code'] != 0]
    print(f"  真实依赖 (pre!=0): {len(deps_real)}")
    matched = 0
    unmatched = 0
    for _, d in deps_real.iterrows():
        pre = int(d['pre_task_code'])
        post = int(d['post_task_code'])
        pre_ok = pre in task_codes
        post_ok = post in task_codes
        status = "OK" if (pre_ok and post_ok) else f"MISS(pre={pre_ok},post={post_ok})"
        if pre_ok and post_ok:
            matched += 1
        else:
            unmatched += 1
        print(f"    {pre} -> {post}  [{status}]")

    print(f"\n  匹配率: {matched}/{matched+unmatched} ({matched/(matched+unmatched)*100:.0f}%)")

    # 检查 relation 表是否有多个版本
    deps_versions = pd.read_sql(f"""
        SELECT pre_task_code, post_task_code, pre_task_version, post_task_version,
               COUNT(*) as cnt
        FROM t_ds_process_task_relation
        WHERE process_definition_code = {def_code}
        GROUP BY pre_task_code, post_task_code, pre_task_version, post_task_version
        HAVING cnt > 1
    """, conn)
    if len(deps_versions) > 0:
        print(f"\n  ⚠ 有重复的依赖记录:")
        print(deps_versions.to_string())

    # 看看 relation 表的完整结构
    if row_i == 0:
        sample = pd.read_sql(f"""
            SELECT *
            FROM t_ds_process_task_relation
            WHERE process_definition_code = {def_code}
            LIMIT 3
        """, conn)
        print(f"\n  relation 表样例 (前3条):")
        print(sample.to_string())

conn.close()
