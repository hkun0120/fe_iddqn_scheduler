# Remote Server Run Notes

This package contains the current code, replay data, split files, and scripts needed
to continue the stratified paper ablation on a 32C/128G server.

## 1. Prepare Python

Python 3.8+ is recommended. The training script is compatible with Python 3.7,
but some third-party packages may install older wheels on Python 3.7.

```bash
cd fe_iddqn_scheduler_remote_*
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

If the default PyTorch wheel is slow or unsuitable for the server, install a CPU wheel
from the official PyTorch index before running `pip install -r requirements.txt`.

## 2. Run the complementary paper shard

The provided remote script defaults to the groups that were still pending locally:

- `fe_iddqn`
- `iddqn`
- `ga_hpo_iddqn`

Run it in the background:

```bash
nohup bash scripts/run_remote_paper_ablation_32c.sh > remote_paper_run.log 2>&1 &
echo $! > remote_paper_run.pid
```

The default resource profile is:

- `ABLATION_MAX_WORKERS=8`
- `ABLATION_TORCH_NUM_THREADS=4`
- `TRAIN_WORKFLOWS_PER_EPISODE=3`
- `EVAL_WORKFLOWS_PER_EPISODE=5`

That maps cleanly to about 32 CPU threads. If memory pressure appears, reduce
`ABLATION_MAX_WORKERS` to `6` or `4`.

## 3. Run all groups instead

To run a complete standalone copy of all five groups:

```bash
ABLATION_GROUPS=ga_hpo_dqn,ga_hpo_fe_dqn,fe_iddqn,iddqn,ga_hpo_iddqn \
nohup bash scripts/run_remote_paper_ablation_32c.sh > remote_paper_run_all.log 2>&1 &
```

## 4. Monitor

```bash
tail -f remote_paper_run.log
ps -p "$(cat remote_paper_run.pid)" -o pid,etime,stat,command
find results -path '*final_eval.json' | wc -l
python3 scripts/summarize_ablation_results.py --root results/ablation_stratified_paper_remote_*
```

Each seed also writes its own `process.log` under:

```text
results/<run_name>/<group>/seed_<seed>/process.log
```

## 5. Merge back with the local run

After the server shard finishes, copy only the completed group directories back into
the local paper result root. Example:

```bash
rsync -av server:/path/to/fe_iddqn_scheduler_remote_*/results/ablation_stratified_paper_remote_*/fe_iddqn \
  results/ablation_stratified_paper_20260516/
rsync -av server:/path/to/fe_iddqn_scheduler_remote_*/results/ablation_stratified_paper_remote_*/iddqn \
  results/ablation_stratified_paper_20260516/
rsync -av server:/path/to/fe_iddqn_scheduler_remote_*/results/ablation_stratified_paper_remote_*/ga_hpo_iddqn \
  results/ablation_stratified_paper_20260516/
```

Then regenerate summaries locally:

```bash
python3 scripts/summarize_ablation_results.py --root results/ablation_stratified_paper_20260516
python3 scripts/summarize_stratified_final_eval.py --root results/ablation_stratified_paper_20260516
```
