#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from create_gantt_chart_293719 import GanttChartGenerator


def main():
    if len(sys.argv) < 3:
        print('Usage: python diagnose_task_duration.py <process_id> <task_instance_id>')
        sys.exit(1)
    pid = int(sys.argv[1])
    tid = int(sys.argv[2])

    gen = GanttChartGenerator()
    tasks, resources, deps = gen.build_tasks_and_resources(pid)
    if not tasks:
        print(f'No tasks for process {pid}')
        return

    # duration used by scheduler
    tinfo = next((t for t in tasks if int(t['id']) == tid), None)
    if not tinfo:
        print(f'Task {tid} not found in built tasks for process {pid}')
        return
    used_duration = float(tinfo.get('duration', 0.0))

    # historical from task_instance table
    ti = gen.data['task_instance']
    row = ti[ti['id'] == tid]
    hist_duration = None
    if not row.empty and pd.notna(row.iloc[0].get('start_time')) and pd.notna(row.iloc[0].get('end_time')):
        st = pd.to_datetime(row.iloc[0]['start_time'])
        et = pd.to_datetime(row.iloc[0]['end_time'])
        hist_duration = (et - st).total_seconds()

    print('Process:', pid)
    print('TaskInstance:', tid)
    print('Scheduler used duration (s):', used_duration)
    print('Historical duration (s):', hist_duration)
    print('Raw task row:', tinfo)

if __name__ == '__main__':
    main()


