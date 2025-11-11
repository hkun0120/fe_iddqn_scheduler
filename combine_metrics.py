#!/usr/bin/env python3
import os
import glob
import pandas as pd

def main():
    files = glob.glob('metrics_*.csv')
    if not files:
        print('No metrics_*.csv files found')
        return
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # normalize columns
            if 'process_id' not in df.columns and 'process' in df.columns:
                df = df.rename(columns={'process': 'process_id'})
            dfs.append(df[['process_id','algorithm','makespan','utilization']])
        except Exception as e:
            print(f'Skip {f}: {e}')
    if not dfs:
        print('No valid metrics loaded')
        return
    all_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    all_df.to_csv('metrics_combined.csv', index=False)
    print('Saved metrics_combined.csv')

    # Build wide table for quick compare
    wide_ms = all_df.pivot_table(index='process_id', columns='algorithm', values='makespan', aggfunc='min')
    wide_ut = all_df.pivot_table(index='process_id', columns='algorithm', values='utilization', aggfunc='max')
    wide_ms.to_csv('metrics_makespan_wide.csv')
    wide_ut.to_csv('metrics_utilization_wide.csv')
    print('Saved metrics_makespan_wide.csv, metrics_utilization_wide.csv')

    # FE-IDDQN vs HEFT summary
    summary_rows = []
    for pid, row in wide_ms.iterrows():
        fe = row.get('FE_IDDQN')
        heft = row.get('HEFT')
        sjf = row.get('SJF') if 'SJF' in row.index else None
        ga = row.get('GA') if 'GA' in row.index else None
        pso = row.get('PSO') if 'PSO' in row.index else None
        better = None
        delta = None
        pct = None
        if pd.notna(fe) and pd.notna(heft) and heft > 0:
            better = fe < heft
            delta = fe - heft
            pct = delta / heft
        summary_rows.append({
            'process_id': pid,
            'fe_makespan': fe,
            'heft_makespan': heft,
            'sjf_makespan': sjf,
            'ga_makespan': ga,
            'pso_makespan': pso,
            'fe_better_than_heft': better,
            'fe_minus_heft': delta,
            'fe_vs_heft_pct': pct
        })
    pd.DataFrame(summary_rows).to_csv('metrics_fe_vs_heft.csv', index=False)
    print('Saved metrics_fe_vs_heft.csv')

if __name__ == '__main__':
    main()



