#!/bin/bash
# 自动生成所有消融实验组脚本

PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
SEEDS=(42 43 44)

# 确保输出总目录存在
mkdir -p results/ablation

##########################################
# Group 1: GA-HPO DQN
# (排除 FE, PER, N-step, GNN)
##########################################
for seed in "${SEEDS[@]}"; do
    echo "Starting Group 1 (GA-HPO DQN) Seed $seed..."
    $PYTHON_BIN train_fe_iddqn_ga_hpo.py --seed $seed \
        --disable_fe --disable_per --disable_nstep --no_gnn \
        --output_dir results/ablation/ga_hpo_dqn/seed_$seed
done

##########################################
# Group 2: GA-HPO FE-DQN
# (保留 FE, 排除 PER, N-step, GNN)
##########################################
for seed in "${SEEDS[@]}"; do
    echo "Starting Group 2 (GA-HPO FE-DQN) Seed $seed..."
    $PYTHON_BIN train_fe_iddqn_ga_hpo.py --seed $seed \
        --disable_per --disable_nstep --no_gnn \
        --output_dir results/ablat        --output_dir results/aon        --##        --output_dir results/ablat      p 3        --output_dir results/ablat        --output_dir results/aon       ���        --o#####################################        --output_dEEDS[@]}"        --output_dir results/ablat       )         --output_di$PY        --output_dir resga_hpo.py --seed $seed \
        --mode train_only         --mode train_only   ts/ablation/fe_iddqn/seed_$seed
done

####################################################################� FE 且 排除 GA-HPO, 固定特性的纯IDDQN)
############################################################################ echo "Sta###############################d..."
    $P    $P    $P    $P_iddqn_    $P    $P    $P    $P_iddqn_  --mode train_only --disable_fe \
        --output        --output        --ouseed_$seed
done

##########################################
##########################################
useed_$seedHPuseed_$seedHPuseed_$seedHPuseed_$seedHPuseed_$s#########################################
for seed in "${SEEDS[@]}"; do
    echo "Starting Group 5 (GA-HPO IDDQN - Full) Seed $seed..."
    $PYTHON_BIN train_fe_iddqn_ga_hpo.py --seed $seed \
        --output_dir results/ablation/ga_hpo_iddqn/seed_$seed
done

echo "All ablation groups completed successfully!"
