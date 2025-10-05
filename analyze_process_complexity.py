import pandas as pd
import networkx as nx
from data.mysql_data_loader import MySQLDataLoader


def analyze_process_complexity(loader: MySQLDataLoader, limit: int = 300):
    # Load all tables via loader API
    data = loader.load_all_data()
    df_proc_def = data.get('process_definition')
    df_rel = data.get('process_task_relation')
    df_proc_inst = data.get('process_instance')
    df_task_inst = data.get('task_instance')

    if any(x is None or x.empty for x in [df_proc_def, df_rel]):
        return []

    # recent definitions (robust to mixed types)
    if 'update_time' in df_proc_def.columns:
        try:
            df_proc_def = df_proc_def.copy()
            df_proc_def['__ut'] = pd.to_datetime(df_proc_def['update_time'], errors='coerce')
            df_proc_def = df_proc_def.sort_values('__ut', ascending=False).drop(columns=['__ut']).head(limit)
        except Exception:
            df_proc_def = df_proc_def.head(limit)
    else:
        df_proc_def = df_proc_def.head(limit)

    results = []
    for _, p in df_proc_def.iterrows():
        try:
            code = int(p['code'])
        except Exception:
            continue

        vers = df_rel[df_rel['process_definition_code'] == code]['process_definition_version'].unique()
        if len(vers) == 0:
            continue
        ver = max(vers)

        rel_v = df_rel[(df_rel['process_definition_code'] == code) & (df_rel['process_definition_version'] == ver)]
        if rel_v is None or rel_v.empty:
            continue

        # Build DAG on task definition codes
        edges = rel_v[['pre_task_code', 'post_task_code']].dropna()
        if edges.empty:
            continue
        try:
            edges = edges.astype(int).values.tolist()
        except Exception:
            continue

        G = nx.DiGraph()
        G.add_edges_from([(int(a), int(b)) for a, b in edges])
        # drop isolated nodes
        for n in list(G.nodes()):
            if G.in_degree(n) == 0 and G.out_degree(n) == 0:
                G.remove_node(n)
        if len(G.nodes()) == 0:
            continue

        try:
            depth = nx.dag_longest_path_length(G) + 1
            crit_len = len(nx.dag_longest_path(G))
        except nx.NetworkXUnfeasible:
            # not a DAG; skip
            continue
        width = 0
        try:
            for level in nx.topological_generations(G):
                width = max(width, sum(1 for _ in level))
        except Exception:
            width = 0
        dep_cnt = G.number_of_edges()

        # Estimate average and top peak concurrency from historical instances
        avg_concurrency = 0.0
        top_instance_id = None
        top_instance_peak = -1
        insts = df_proc_inst[df_proc_inst['process_definition_code'] == code]
        if insts is not None and not insts.empty and df_task_inst is not None and not df_task_inst.empty:
            tids = df_task_inst[df_task_inst['process_instance_id'].isin(insts['id'])]
            if not tids.empty and {'start_time', 'end_time'}.issubset(tids.columns):
                peaks = []
                for pid, grp in tids.groupby('process_instance_id'):
                    s = grp[['start_time', 'end_time']].dropna()
                    if s.empty:
                        continue
                    s = s.sort_values('start_time')
                    times = []
                    for _, r in s.iterrows():
                        try:
                            st = pd.to_datetime(r['start_time'])
                            et = pd.to_datetime(r['end_time'])
                        except Exception:
                            continue
                        times.append((st, 1))
                        times.append((et, -1))
                    if not times:
                        continue
                    times.sort()
                    cur = 0
                    peak = 0
                    for _, v in times:
                        cur += v
                        if cur > peak:
                            peak = cur
                    peaks.append(peak)
                    if peak > top_instance_peak:
                        top_instance_peak = peak
                        top_instance_id = int(pid)
                if peaks:
                    avg_concurrency = float(sum(peaks)) / len(peaks)

        # Host heterogeneity stats for this process (host count and speed std)
        host_count = 0
        host_speed_std = 0.0
        try:
            if insts is not None and not insts.empty and df_task_inst is not None and not df_task_inst.empty:
                tsub = df_task_inst[df_task_inst['process_instance_id'].isin(insts['id'])].copy()
                tsub = tsub[(tsub['state'] == 7) & pd.notna(tsub['start_time']) & pd.notna(tsub['end_time'])]
                if not tsub.empty:
                    tsub['duration_hist'] = (pd.to_datetime(tsub['end_time']) - pd.to_datetime(tsub['start_time'])).dt.total_seconds()
                    hmean = tsub.groupby('host')['duration_hist'].mean()
                    # normalize by global mean to approximate speed factor dispersion
                    gmean = float(tsub['duration_hist'].mean()) if not pd.isna(tsub['duration_hist'].mean()) else 1.0
                    if gmean <= 0:
                        gmean = 1.0
                    norm = hmean / gmean
                    host_count = int((~hmean.index.isna()).sum())
                    if len(norm) >= 2:
                        host_speed_std = float(norm.std())
        except Exception:
            pass

        # simple composite score
        score = 0.5 * depth + 0.3 * crit_len + 0.2 * avg_concurrency + 0.1 * dep_cnt
        results.append({
            'process_code': code,
            'version': ver,
            'depth': depth,
            'width': width,
            'critical_path_len': crit_len,
            'dependency_count': dep_cnt,
            'avg_concurrency_est': avg_concurrency,
            'host_count': host_count,
            'host_speed_std': host_speed_std,
            'top_instance_id': top_instance_id,
            'score': score,
        })

    return results


def main():
    loader = MySQLDataLoader(host='127.0.0.1', user='root', password='', database='whalesb')
    results = analyze_process_complexity(loader, limit=500)
    df = pd.DataFrame(results).sort_values('score', ascending=False)
    print(df.head(20))
    out = 'process_complexity_top20.csv'
    df.head(20).to_csv(out, index=False)
    print(f"已输出: {out}")
    # 选择大小三档（small/medium/large）各2例
    try:
        if not df.empty:
            n = len(df)
            # constraints
            cand = df.copy()
            # prefer stronger concurrency and heterogeneity
            cand_strict = cand[(cand['host_count'] >= 4) & (cand['host_speed_std'] >= 0.25) & (cand['avg_concurrency_est'] >= 6)]
            if len(cand_strict) < 6:
                cand_strict = cand[(cand['host_count'] >= 3) & (cand['host_speed_std'] >= 0.2) & (cand['avg_concurrency_est'] >= 4)]
            # small/medium/large bands by score quantiles
            q33 = cand_strict['score'].quantile(0.33) if not cand_strict.empty else 0
            q66 = cand_strict['score'].quantile(0.66) if not cand_strict.empty else 0
            small = cand_strict[cand_strict['score'] <= q33].head(2)
            medium = cand_strict[(cand_strict['score'] > q33) & (cand_strict['score'] <= q66)].head(2)
            large = cand_strict[cand_strict['score'] > q66].head(2)
            pick = pd.concat([small, medium, large], ignore_index=True)
            pick[['process_code','version','top_instance_id','score','host_count','host_speed_std','avg_concurrency_est']].to_csv('picked_instances_strict.csv', index=False)
            print('已输出: picked_instances_strict.csv')
    except Exception as _:
        pass


if __name__ == '__main__':
    main()


