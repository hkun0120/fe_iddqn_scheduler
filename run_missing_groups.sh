#!/bin/bash
PYTHON_BIN="python3"
SEEDS=(42 43 44)
for seed in "${SEEDS[@]}"; do
    echo "Running Seed $seed Group 1: GA-HPO DQN"
    $PYTHON_BIN train_fe_iddqn_ga_hpo.py --seed $seed \
        --disable_per --disable_nstep --no_gnn \
        --output_dir results/ablation/ga_hpo_dqn/seed_$seed

    echo "Running Seed $seed Group 2: GA-HPO FE-DQN"
    $PYTHON_BIN train_fe_iddqn_ga_hpo.py --seed $seed \
        --disable_per --disable_nstep --no_gnn \
        --output_dir results/ablation/ga_hpo_fe_dqn/seed_$seed
done
